"""
compute_iv.py
=============
Processes raw data into the final feature CSV for the RL agent.

Since we're using proxy IV data (VIX-based), this script:
1. Loads proxy_iv.csv (our IV signals)
2. Computes rolling z-scores (needed for the Vasant Dhar override)
3. Computes realized volatility (additional state feature)
4. Merges with ETF returns (needed for rewards)
5. Merges with Treasury yields (needed for CASH action reward)
6. Outputs data/processed/iv_features.csv
"""

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
RAW_DIR       = os.path.join(BASE_DIR, '..', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_proxy_iv():
    """
    Load the VIX-based proxy IV data.
    Returns a DataFrame with columns: date, iv_xlk, iv_xlf, iv_xlv
    """
    path = os.path.join(RAW_DIR, 'proxy_iv.csv')
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded proxy IV data: {len(df)} rows")
    return df


def add_zscores(df):
    """
    Compute rolling 60-day z-scores for each sector's IV.
    
    WHY Z-SCORES?
    - Raw IV values vary a lot (e.g., 15% in calm markets vs 80% during COVID)
    - Z-scores tell us HOW UNUSUAL today's IV is relative to recent history
    - z = (IV_today - mean_60days) / std_60days
    - z > 2.5 across ALL sectors simultaneously = Vasant Dhar override triggers
    - This is how the agent knows it's in a crisis vs just a high-vol period
    
    Example:
    - Normal market: IV=20%, mean=18%, std=3% → z = 0.67 (not unusual)
    - COVID crash:   IV=80%, mean=18%, std=3% → z = 20.7 (extreme crisis!)
    """
    print("Computing rolling z-scores...")
    
    iv_cols = ['iv_xlk', 'iv_xlf', 'iv_xlv']
    
    for col in iv_cols:
        ticker = col.replace('iv_', '')
        
        # 60-day rolling mean and std
        rolling_mean = df[col].rolling(window=60, min_periods=20).mean()
        rolling_std  = df[col].rolling(window=60, min_periods=20).std()
        
        # Z-score
        df[f'zscore_{ticker}'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
    
    return df


def add_realized_volatility(df):
    """
    Compute 20-day realized volatility for each sector.
    
    WHY REALIZED VOL?
    - IV is forward-looking (what the market EXPECTS)
    - Realized vol is backward-looking (what ACTUALLY happened)
    - Together they give the agent a fuller picture of market conditions
    - The gap between IV and realized vol = "volatility risk premium"
      When IV >> realized vol: market is fearful but nothing bad has happened yet
      When realized vol >> IV: something bad happened that wasn't anticipated
    
    We compute it from daily ETF returns:
    realized_vol = std(daily_returns over 20 days) * sqrt(252)
    The sqrt(252) annualizes it (252 trading days per year)
    """
    print("Computing realized volatility...")
    
    returns_path = os.path.join(RAW_DIR, 'etf_returns.csv')
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    for ticker in ['XLK', 'XLF', 'XLV']:
        if ticker in returns.columns:
            # 20-day annualized realized volatility
            realized = (
                returns[ticker]
                .rolling(window=20, min_periods=10)
                .std() * np.sqrt(252)
            )
            
            # Align with main dataframe by date
            realized_df = realized.reset_index()
            realized_df.columns = ['date', f'realvol_{ticker.lower()}']
            realized_df['date'] = pd.to_datetime(realized_df['date'])
            
            df = df.merge(realized_df, on='date', how='left')
    
    return df


def add_returns(df):
    """
    Merge daily ETF log returns into the feature dataframe.
    
    WHY DO WE NEED RETURNS HERE?
    - The RL agent's REWARD is based on the return of whichever sector it chose
    - If the agent picks XLK on day t, reward = log_return of XLK on day t
    - The Sortino ratio reward shaping also needs historical returns
    - SPY returns are needed for our benchmark comparison
    """
    print("Merging ETF returns...")
    
    returns_path = os.path.join(RAW_DIR, 'etf_returns.csv')
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    returns = returns.reset_index()
    
    # Rename columns to ret_xlk, ret_xlf, etc.
    new_cols = {'Date': 'date'}
    for col in returns.columns:
        if col != 'Date':
            new_cols[col] = f'ret_{col.lower()}'
    returns = returns.rename(columns=new_cols)
    returns['date'] = pd.to_datetime(returns['date'])
    
    df = df.merge(returns, on='date', how='inner')
    return df


def add_risk_free_rate(df):
    """
    Merge daily risk-free rate into the feature dataframe.
    
    WHY DO WE NEED THIS?
    - When the agent chooses CASH (triggered by the override),
      it earns the risk-free rate as its reward
    - This makes CASH a meaningful action: it's not zero return,
      it's the Treasury yield (e.g., ~5% annually in 2023-2024)
    """
    print("Merging risk-free rate...")
    
    rf_path = os.path.join(RAW_DIR, 'treasury_yields.csv')
    rf = pd.read_csv(rf_path, index_col=0, parse_dates=True)
    rf = rf.reset_index()
    rf.columns = ['date', 'rf_daily']
    rf['date'] = pd.to_datetime(rf['date'])
    
    df = df.merge(rf, on='date', how='left')
    
    # Fill any missing days with ~5% annual / 252 days
    df['rf_daily'] = df['rf_daily'].fillna(0.0002)
    
    return df


def train_test_split(df):
    """
    Split into training (2020-2023) and test (2024) sets.
    
    WHY THIS SPLIT?
    - We train the agent on 2020-2023 data (includes COVID crash, 
      2021 bull run, 2022 rate hike bear market)
    - We evaluate on 2024 data the agent has NEVER seen
    - This prevents look-ahead bias (agent can't cheat by knowing the future)
    - Simulates real deployment: train on history, trade on new data
    """
    train = df[df['date'] <= '2023-12-31'].copy()
    test  = df[df['date'] >= '2024-01-01'].copy()
    
    print(f"\nTrain set: {len(train)} days "
          f"({train['date'].iloc[0].date()} to {train['date'].iloc[-1].date()})")
    print(f"Test set:  {len(test)} days "
          f"({test['date'].iloc[0].date()} to {test['date'].iloc[-1].date()})")
    
    return train, test


def check_override_frequency(df):
    """
    Check how often the Vasant Dhar override would trigger.
    This is a sanity check — it should trigger rarely (1-3% of days)
    but definitely during March 2020 (COVID crash).
    """
    zscore_cols = ['zscore_xlk', 'zscore_xlf', 'zscore_xlv']
    
    if all(c in df.columns for c in zscore_cols):
        override_mask = (df[zscore_cols] > 2.5).all(axis=1)
        n_override = override_mask.sum()
        pct = 100 * n_override / len(df)
        
        print(f"\nOverride trigger analysis:")
        print(f"  Total override days: {n_override} / {len(df)} ({pct:.1f}%)")
        
        if n_override > 0:
            override_dates = df[override_mask]['date'].dt.strftime('%Y-%m-%d').tolist()
            print(f"  First 5 override dates: {override_dates[:5]}")


# ── Main Execution ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("SECTOR ROTATION RL — Feature Engineering Pipeline")
    print("=" * 60)
    
    # 1. Load proxy IV data
    df = load_proxy_iv()
    
    # 2. Add z-scores (needed for override detection)
    df = add_zscores(df)
    
    # 3. Add realized volatility (additional state feature)
    df = add_realized_volatility(df)
    
    # 4. Merge ETF returns (needed for rewards)
    df = add_returns(df)
    
    # 5. Merge risk-free rate (needed for CASH action reward)
    df = add_risk_free_rate(df)
    
    # 6. Drop NaN rows from rolling window warmup period
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"\nDropped {initial_len - len(df)} rows (rolling window warmup)")
    
    # 7. Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # 8. Save full dataset
    output_path = os.path.join(PROCESSED_DIR, 'iv_features.csv')
    df.to_csv(output_path, index=False)
    
    print(f"\nFinal dataset saved: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 9. Train/test split
    train, test = train_test_split(df)
    train.to_csv(os.path.join(PROCESSED_DIR, 'train.csv'), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, 'test.csv'), index=False)
    
    # 10. Sanity check on override frequency
    check_override_frequency(df)
    
    # 11. Quick stats
    print("\n" + "=" * 60)
    print("IV STATISTICS")
    print("=" * 60)
    iv_cols = ['iv_xlk', 'iv_xlf', 'iv_xlv']
    print(df[iv_cols].describe().round(4).to_string())
    
    print("\n" + "=" * 60)
    print("Day 1 Complete! data/processed/iv_features.csv is ready.")
    print("Share the GitHub repo with Rishit and Ishan.")
    print("=" * 60)