"""
generate_data.py — One-shot data pipeline (no WRDS required).

Downloads XLK/XLF/XLV/SPY prices, VIX, and T-bill yields via yfinance.
Creates VIX-based proxy IV (same approach as the original fallback in
download_wrds.py).  Falls back to purely synthetic GBM data if yfinance
is unavailable.

Outputs:
  data/processed/iv_features.csv   — full 2020-2024 feature set
  data/processed/train.csv         — 2020-2023 slice
  data/processed/test.csv          — 2024 slice
  data/processed/norm_stats.csv    — per-feature mean/std for the train set
"""
import os
import numpy as np
import pandas as pd

START    = "2020-01-01"
END      = "2024-12-31"
SECTORS  = ["XLK", "XLF", "XLV"]
RAW_DIR  = "data/raw"
PROC_DIR = "data/processed"

os.makedirs(RAW_DIR,  exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)


# ── Download helpers ────────────────────────────────────────────

def _dl_single(ticker: str, start: str, end: str) -> pd.Series:
    import yfinance as yf
    raw   = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = ticker
    return close


def _dl_multi(tickers: list, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    raw   = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    return close[tickers]


# ── Feature engineering ─────────────────────────────────────────

def build_features(etf_close: pd.DataFrame,
                   vix: pd.Series,
                   irx: pd.Series) -> pd.DataFrame:
    """
    Combine raw price data into the 15-column feature CSV the RL env expects.
    """
    np.random.seed(42)
    n   = len(vix)
    idx = vix.index

    # --- Proxy IV from VIX (same formula as download_wrds.py fallback) ---
    iv = pd.DataFrame({
        "iv_xlk": np.clip((vix.values * 1.15 + np.random.normal(0, 2, n)) / 100, 0.05, 2.0),
        "iv_xlf": np.clip((vix.values * 1.05 + np.random.normal(0, 2, n)) / 100, 0.05, 2.0),
        "iv_xlv": np.clip((vix.values * 0.85 + np.random.normal(0, 1.5, n)) / 100, 0.05, 2.0),
    }, index=idx)

    # --- Log returns for ETFs + SPY ---
    rets_raw = np.log(etf_close / etf_close.shift(1))
    rets = rets_raw.rename(columns={s: f"ret_{s.lower()}" for s in SECTORS + ["SPY"]})
    rets = rets.rename(columns={"ret_spy": "ret_spy"})   # keep SPY lower-case

    # --- Daily risk-free rate: annualised % → decimal / 252 ---
    rf = ((irx / 100) / 252).rename("rf_daily")

    # --- Merge ---
    df = (iv
          .join(rets, how="inner")
          .join(rf,   how="left"))
    df.index.name = "date"
    df = df.reset_index()
    df["rf_daily"] = df["rf_daily"].fillna(0.0002)

    # --- Rolling z-scores for IV (60-day window) ---
    for s in [s.lower() for s in SECTORS]:
        col      = f"iv_{s}"
        mean60   = df[col].rolling(60, min_periods=20).mean()
        std60    = df[col].rolling(60, min_periods=20).std() + 1e-8
        df[f"zscore_{s}"] = (df[col] - mean60) / std60

    # --- 20-day annualised realised vol ---
    for s in SECTORS:
        col = f"ret_{s.lower()}"
        df[f"realvol_{s.lower()}"] = (
            df[col].rolling(20, min_periods=10).std() * np.sqrt(252)
        )

    df = df.dropna().reset_index(drop=True)
    return df


# ── Synthetic fallback ──────────────────────────────────────────

def _synthetic_close() -> tuple:
    """
    GBM + Ornstein-Uhlenbeck VIX.  Includes a COVID-style shock in early 2020.
    Returns (etf_close, vix_series, irx_series) all on the same index.
    """
    dates = pd.bdate_range(start=START, end=END)
    n     = len(dates)
    dt    = 1.0 / 252
    np.random.seed(0)

    spike_s, spike_e = 40, 70        # ~March 2020 (trading day 40-70)

    # VIX — mean-reverting Ornstein-Uhlenbeck
    vix    = np.zeros(n);  vix[0] = 20.0
    th, mv, sv = 2.0, 20.0, 10.0
    for i in range(1, n):
        vix[i] = vix[i-1] + th*(mv - vix[i-1])*dt + sv*np.sqrt(dt)*np.random.randn()
    vix[spike_s:spike_e] += np.linspace(0, 60, spike_e - spike_s)
    vix[spike_e:spike_e+30] = np.linspace(vix[spike_e-1], 25, 30)
    vix = np.clip(vix, 10, 90)

    # T-bill: near-zero 2020-2021, rising to ~5% by 2024
    irx = np.zeros(n)
    half = n // 2
    irx[:half] = np.linspace(0.08, 0.5, half)
    irx[half:]  = np.linspace(0.5, 5.5, n - half)

    # ETF prices — GBM with COVID shock
    params = {"XLK": (0.20, 0.25), "XLF": (0.15, 0.22),
              "XLV": (0.12, 0.18), "SPY": (0.14, 0.18)}
    prices = {}
    for tkr, (mu, sig) in params.items():
        lr  = (mu - 0.5*sig**2)*dt + sig*np.sqrt(dt)*np.random.randn(n)
        lr[spike_s:spike_e] += np.random.normal(-0.008, 0.025, spike_e - spike_s)
        prices[tkr] = 100 * np.exp(np.cumsum(lr))

    etf_close = pd.DataFrame(prices, index=dates)
    vix_s     = pd.Series(vix, index=dates, name="^VIX")
    irx_s     = pd.Series(irx, index=dates, name="^IRX")
    return etf_close, vix_s, irx_s


# ── Main ────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Sector Rotation RL — Data Generation Pipeline")
    print("=" * 60)

    # --- Download or fall back to synthetic ---
    try:
        print(f"Downloading ETF prices ({START} → {END})...")
        etf_close = _dl_multi(SECTORS + ["SPY"], START, END)
        print(f"  Prices: {len(etf_close)} rows   ({etf_close.index[0].date()} – {etf_close.index[-1].date()})")

        print("Downloading VIX...")
        vix = _dl_single("^VIX", START, END)
        print(f"  VIX:   {len(vix)} rows")

        print("Downloading T-bill yield (^IRX)...")
        irx = _dl_single("^IRX", START, END)
        print(f"  IRX:   {len(irx)} rows")

        # Align all series to VIX index (yfinance sometimes returns different sets of dates)
        common = etf_close.index.intersection(vix.index)
        etf_close = etf_close.loc[common]
        vix       = vix.loc[common]
        irx       = irx.reindex(common).ffill().fillna(0.5)

        using_synthetic = False

    except Exception as exc:
        print(f"\nyfinance failed: {exc}")
        print("Falling back to synthetic GBM data.\n")
        etf_close, vix, irx = _synthetic_close()
        using_synthetic = True

    # --- Feature engineering ---
    print("\nBuilding features (z-scores, realised vol, returns, rf)...")
    df = build_features(etf_close, vix, irx)
    print(f"  Final dataset: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Date range: {df['date'].iloc[0].date()} – {df['date'].iloc[-1].date()}")

    # --- Save full dataset ---
    full_path = f"{PROC_DIR}/iv_features.csv"
    df.to_csv(full_path, index=False)
    print(f"  Saved: {full_path}")

    # --- Train / test split ---
    train = df[df["date"] <= "2023-12-31"].reset_index(drop=True)
    test  = df[df["date"] >= "2024-01-01"].reset_index(drop=True)
    train.to_csv(f"{PROC_DIR}/train.csv", index=False)
    test.to_csv(f"{PROC_DIR}/test.csv",  index=False)
    print(f"  Train: {len(train)} rows  |  Test: {len(test)} rows")

    # --- Normalization stats (state features only) ---
    state_cols = (
        [f"iv_{s.lower()}"      for s in SECTORS] +
        [f"zscore_{s.lower()}"  for s in SECTORS] +
        [f"realvol_{s.lower()}" for s in SECTORS]
    )
    norm = pd.DataFrame({
        "mean": train[state_cols].mean(),
        "std":  train[state_cols].std().clip(lower=1e-8),
    })
    norm.to_csv(f"{PROC_DIR}/norm_stats.csv")
    print(f"  Saved: {PROC_DIR}/norm_stats.csv")

    print("\n" + "=" * 60)
    data_source = "SYNTHETIC (GBM)" if using_synthetic else "yfinance (real prices + VIX proxy IV)"
    print(f"Data source: {data_source}")
    print("=" * 60)
    return train, test


if __name__ == "__main__":
    main()
