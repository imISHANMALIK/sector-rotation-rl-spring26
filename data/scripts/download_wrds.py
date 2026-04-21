"""
download_wrds.py
================
Downloads options data for sector ETFs (XLK, XLF, XLV) from WRDS OptionMetrics
and underlying ETF prices from yfinance.

WHY THIS SCRIPT EXISTS:
- We need historical options prices to compute Implied Volatility
- IV is the primary state signal for our RL agent
- WRDS OptionMetrics is the gold standard for academic options research

WHAT IT PRODUCES:
- data/raw/options_{ticker}.csv  — raw options data per sector
- data/raw/etf_prices.csv        — daily ETF closing prices
- data/raw/treasury_yields.csv   — risk-free rate time series
"""

import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import wrds
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Configuration ──────────────────────────────────────────────
TICKERS = {
    'XLK': 'Technology',   # SPDR Technology Select Sector ETF
    'XLF': 'Financials',   # SPDR Financial Select Sector ETF
    'XLV': 'Healthcare',   # SPDR Health Care Select Sector ETF
}

# Training period: 2020-2024 (we'll split 2020-2023 train, 2024 test)
START_DATE = '2020-01-01'
END_DATE   = '2024-12-31'

# Output directories
RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)


def connect_to_wrds():
    """
    Establish connection to WRDS.
    
    On first run, it will prompt for your password interactively.
    After that, credentials are cached in ~/.pgpass
    
    WHY: WRDS uses a PostgreSQL database that we query via SQL.
    The wrds Python package handles the connection for us.
    """
    print("Connecting to WRDS...")
    try:
        db = wrds.Connection()
        print("Connected to WRDS successfully")
        return db
    except Exception as e:
        print(f"Failed to connect to WRDS: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've registered at https://wrds-www.wharton.upenn.edu/")
        print("2. Use your NYU institutional login")
        print("3. Run: python -c \"import wrds; wrds.Connection()\" to test")
        sys.exit(1)


def download_options_data(db, ticker, start_date, end_date):
    """
    Download options data for a single ETF from OptionMetrics.
    
    WHAT WE'RE QUERYING:
    - optionm.opprcd: The main options price table in OptionMetrics
    - We filter for CALL options (cp_flag = 'C')
    - We filter for options near the money (ATM): strike within 5% of underlying price
    - We filter for 14-60 days to expiration: this is the sweet spot where
      IV is most informative (too short = noisy, too long = less responsive)
    
    WHY ATM OPTIONS?
    - At-the-money options are most liquid (most trading volume)
    - Their IV is the most representative of market expectations
    - Deep in/out of the money options have skew effects that distort IV
    
    WHY 14-60 DTE (Days to Expiration)?
    - Very short-dated options (<14 days) have gamma effects that make IV noisy
    - Very long-dated options (>60 days) are less responsive to current market fear
    - 30-day IV is the standard (VIX uses 30-day options)
    """
    print(f"\nDownloading options data for {ticker} ({TICKERS[ticker]})...")
    
    query = f"""
        SELECT 
            a.date,
            a.exdate,
            a.strike_price / 1000.0 as strike_price,
            a.best_bid,
            a.best_offer,
            a.impl_volatility,
            a.volume,
            a.open_interest,
            a.cp_flag,
            b.close as underlying_price
        FROM 
            optionm.opprcd AS a
        INNER JOIN 
            optionm.secprd AS b
        ON 
            a.secid = b.secid AND a.date = b.date
        INNER JOIN 
            optionm.secnmd AS c
        ON 
            a.secid = c.secid
        WHERE 
            c.ticker = '{ticker}'
            AND a.date BETWEEN '{start_date}' AND '{end_date}'
            AND a.cp_flag = 'C'
            AND a.best_bid > 0
            AND a.best_offer > 0
            AND a.volume > 0
            AND a.impl_volatility IS NOT NULL
            AND (a.exdate - a.date) BETWEEN 14 AND 60
            AND ABS(a.strike_price / 1000.0 - b.close) / b.close < 0.05
        ORDER BY 
            a.date, a.exdate, a.strike_price
    """
    
    try:
        df = db.raw_sql(query)
        print(f"  Downloaded {len(df):,} option records for {ticker}")
        
        # Save raw data
        output_path = os.path.join(RAW_DIR, f'options_{ticker}.csv')
        df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
        
        return df
        
    except Exception as e:
        print(f"  Error downloading {ticker}: {e}")
        print(f"  Trying alternative table names...")
        
        # Some WRDS setups use different table names
        alt_query = query.replace('optionm.opprcd', 'optionm_all.opprcd')
        alt_query = alt_query.replace('optionm.secprd', 'optionm_all.secprd')
        alt_query = alt_query.replace('optionm.secnmd', 'optionm_all.secnmd')
        
        try:
            df = db.raw_sql(alt_query)
            print(f"  Downloaded {len(df):,} option records for {ticker} (alt table)")
            output_path = os.path.join(RAW_DIR, f'options_{ticker}.csv')
            df.to_csv(output_path, index=False)
            return df
        except Exception as e2:
            print(f"  Alternative query also failed: {e2}")
            return None


def download_etf_prices(tickers, start_date, end_date):
    """
    Download daily ETF closing prices from Yahoo Finance.
    
    WHY YFINANCE FOR PRICES (but WRDS for options)?
    - ETF price data is freely available and reliable from Yahoo Finance
    - Options data is NOT free — that's why we need WRDS
    - We need daily returns of each ETF to compute the RL agent's reward
    
    WHAT WE COMPUTE:
    - Daily log returns: ln(price_today / price_yesterday)
    - Log returns are preferred in finance because they're additive over time
      and approximately normally distributed
    """
    print("\nDownloading ETF prices from Yahoo Finance...")
    
    all_tickers = list(tickers.keys()) + ['SPY']  # Include SPY for our baseline
    
    prices = yf.download(
        all_tickers, 
        start=start_date, 
        end=end_date,
        auto_adjust=True
    )['Close']
    
    # Compute daily log returns
    returns = np.log(prices / prices.shift(1))
    
    # Save
    prices.to_csv(os.path.join(RAW_DIR, 'etf_prices.csv'))
    returns.to_csv(os.path.join(RAW_DIR, 'etf_returns.csv'))
    
    print(f"  Downloaded prices for {', '.join(all_tickers)}")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Trading days: {len(prices)}")
    
    return prices, returns


def download_treasury_yields(start_date, end_date):
    """
    Download 3-month Treasury bill yields as the risk-free rate.
    
    WHAT IS THE RISK-FREE RATE?
    - The return you'd earn with ZERO risk (guaranteed by the US government)
    - 3-month Treasury bills are the standard proxy
    - We need this for Black-Scholes (the 'r' parameter)
    - Also used as the CASH action reward: when the agent moves to cash,
      it earns the risk-free rate instead of sector returns
    
    WHY ^IRX?
    - ^IRX is Yahoo Finance's ticker for the 13-week Treasury bill yield
    - Quoted as annualized percentage (e.g., 5.25 means 5.25% per year)
    - We convert to daily: daily_rate = annual_rate / 252 (trading days per year)
    """
    print("\nDownloading Treasury yields (risk-free rate)...")
    
    irx = yf.download('^IRX', start=start_date, end=end_date)['Close']
    
    # Convert from percentage to decimal, and from annual to daily
    daily_rf = (irx / 100) / 252
    
    daily_rf.to_csv(os.path.join(RAW_DIR, 'treasury_yields.csv'))
    
    print(f"  Downloaded Treasury yields")
    print(f"  Average annual yield: {float(irx.mean()):.2f}%")
    
    return daily_rf


def create_fallback_data():
    """
    If WRDS is unavailable, create proxy IV data using VIX as a baseline.
    """
    print("\n--- Creating fallback proxy data (no WRDS connection) ---")
    
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE)['Close']
    
    # Flatten to 1D array — needed for newer yfinance versions
    vix_values = vix.values.flatten()
    dates = vix.index
    
    np.random.seed(42)
    
    iv_data = pd.DataFrame({
        'date': dates,
        'iv_xlk': (vix_values * 1.15 + np.random.normal(0, 2, len(vix_values))) / 100,
        'iv_xlf': (vix_values * 1.05 + np.random.normal(0, 2, len(vix_values))) / 100,
        'iv_xlv': (vix_values * 0.85 + np.random.normal(0, 1.5, len(vix_values))) / 100,
    })
    
    iv_data.to_csv(os.path.join(RAW_DIR, 'proxy_iv.csv'), index=False)
    print("  Proxy IV data created (based on VIX)")
    print("  Replace with real WRDS data when available!")
    
    return iv_data


# ── Main Execution ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("SECTOR ROTATION RL — Data Download Pipeline")
    print("=" * 60)
    
    # Step 1: Download ETF prices (always works, no WRDS needed)
    prices, returns = download_etf_prices(TICKERS, START_DATE, END_DATE)
    
    # Step 2: Download Treasury yields (always works)
    treasury = download_treasury_yields(START_DATE, END_DATE)
    
    # Step 3: Try WRDS for options data
   # Step 3: Using proxy IV data (VIX-based)
    # TODO: Replace with WRDS data once SSL connection issue is resolved
    print("\nUsing proxy IV data (VIX-based)...")
    create_fallback_data()
    
    print("\n" + "=" * 60)
    print("Data download complete! Check data/raw/ for output files.")
    print("=" * 60)
