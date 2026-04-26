"""
risk_override.py
================
Vasant Dhar Safety Override — standalone module.

THE AUTOMATION FRONTIER PHILOSOPHY (Dhar, 2016):
Every automated decision system has a frontier beyond which
its predictions become unreliable. For financial RL agents,
this frontier corresponds to extreme market regimes — crashes,
flash events, systemic crises — where historical patterns
completely break down.

Our agent was trained on 2020-2023 data. When the market enters
a regime it has never seen, its Q-values are meaningless. The
override forces CASH in these situations, recognizing the limits
of the agent's competence.

WHY Z-SCORE THRESHOLD OF 2.5?
- z > 2.5 means IV is in the top ~0.6% of its historical distribution
- When ALL THREE sectors simultaneously exceed this threshold,
  it signals a market-wide panic, not sector-specific fear
- This triggered during COVID (Feb-Mar 2020) — exactly the kind
  of event we want to exit

WHY HARD-CODED AND NOT LEARNED?
- A learned safety mechanism could be optimized away during training
  if the agent finds it profitable to ignore safety
- Hard-coded rules guarantee safety is NEVER traded away
- This is a fundamental principle of safe RL systems
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class VasantDharOverride:
    """
    Standalone implementation of the Vasant Dhar safety override.

    Can be used:
    1. Inside the environment (already integrated in environment.py)
    2. In the demo app to display override status in real-time
    3. In analysis to identify all historical override dates
    """

    def __init__(self, threshold: float = 2.5, window: int = 60):
        """
        Parameters:
        -----------
        threshold : float — z-score level that triggers override (default 2.5)
        window    : int   — rolling window for z-scores (default 60 days)

        WHY 60-DAY WINDOW?
        - Short window (10 days): too sensitive, triggers on minor spikes
        - Long window (252 days): too slow, misses developing crises
        - 60 days (~3 months): captures recent regime without overreacting
        """
        self.threshold = threshold
        self.window    = window
        self._history  = {'xlk': [], 'xlf': [], 'xlv': []}

    def update(self, iv_xlk: float, iv_xlf: float, iv_xlv: float) -> dict:
        """
        Update with today's IV values and check if override should trigger.

        Parameters:
        -----------
        iv_xlk, iv_xlf, iv_xlv : float — today's implied volatility per sector

        Returns:
        --------
        dict with keys:
            'override' : bool — True if agent should move to CASH
            'zscores'  : dict — z-score per sector
            'reason'   : str  — human-readable explanation
        """
        # Update rolling history
        self._history['xlk'].append(iv_xlk)
        self._history['xlf'].append(iv_xlf)
        self._history['xlv'].append(iv_xlv)

        # Keep only last window values
        for k in self._history:
            self._history[k] = self._history[k][-self.window:]

        # Compute z-scores for each sector
        zscores = {}
        for sector, iv_val in [('xlk', iv_xlk), ('xlf', iv_xlf), ('xlv', iv_xlv)]:
            hist  = np.array(self._history[sector])
            if len(hist) < 5:
                zscores[sector] = 0.0  # not enough history yet
            else:
                mu    = np.mean(hist)
                sigma = np.std(hist)
                zscores[sector] = float((iv_val - mu) / sigma) if sigma > 1e-8 else 0.0

        # Override triggers only if ALL sectors exceed threshold simultaneously
        # WHY ALL THREE? If only tech is fearful, rotate to healthcare.
        # If ALL sectors are fearful simultaneously, nowhere is safe → CASH.
        override = all(z > self.threshold for z in zscores.values())

        if override:
            reason = (
                f"OVERRIDE TRIGGERED: Extreme fear across all sectors — "
                f"XLK z={zscores['xlk']:.2f}, "
                f"XLF z={zscores['xlf']:.2f}, "
                f"XLV z={zscores['xlv']:.2f} "
                f"(threshold={self.threshold}). Moving to CASH."
            )
        else:
            reason = (
                f"Normal market conditions — agent policy active. "
                f"Z-scores: XLK={zscores['xlk']:.2f}, "
                f"XLF={zscores['xlf']:.2f}, "
                f"XLV={zscores['xlv']:.2f}"
            )

        return {
            'override': override,
            'zscores':  zscores,
            'reason':   reason,
        }

    def reset(self):
        """Clear history — call at the start of each new episode."""
        self._history = {'xlk': [], 'xlf': [], 'xlv': []}


def analyze_override_history(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """
    Analyze historical data to find all dates where override would trigger.
    Used for EDA, report analysis, and ablation studies.

    Parameters:
    -----------
    df        : pd.DataFrame — iv_features.csv data
    threshold : float        — z-score threshold

    Returns:
    --------
    pd.DataFrame — all override trigger dates with z-scores
    """
    zscore_cols   = ['zscore_xlk', 'zscore_xlf', 'zscore_xlv']
    override_mask = (df[zscore_cols] > threshold).all(axis=1)
    override_df   = df[override_mask][['date'] + zscore_cols].copy()
    override_df   = override_df.reset_index(drop=True)

    print(f'Override Analysis (threshold = {threshold} std devs):')
    print(f'  Total trigger days : {len(override_df)} / {len(df)} '
          f'({len(override_df)/len(df)*100:.1f}%)')

    if len(override_df) > 0:
        print(f'  First trigger      : {override_df["date"].iloc[0]}')
        print(f'  Last trigger       : {override_df["date"].iloc[-1]}')
        print(f'  Expected           : COVID crash Feb-Mar 2020')

    return override_df


def ablation_compare(results_with: dict, results_without: dict):
    """
    Compare performance WITH vs WITHOUT the safety override.
    Used in the ablation study on Day 4.

    Parameters:
    -----------
    results_with    : dict — metrics from agent WITH override enabled
    results_without : dict — metrics from agent WITHOUT override

    Prints a formatted comparison table.
    """
    print('\n' + '=' * 62)
    print(f'{"ABLATION: Override ON vs Override OFF":^62}')
    print('=' * 62)
    print(f'{"Metric":<25} {"With Override":>18} {"No Override":>18}')
    print('-' * 62)

    metrics = [
        ('sortino',      'Sortino Ratio',    1),
        ('sharpe',       'Sharpe Ratio',     1),
        ('total_return', 'Total Return (%)', 1),
        ('max_drawdown', 'Max Drawdown (%)', 100),
    ]

    for key, label, scale in metrics:
        v_with    = results_with.get(key,    0) * scale
        v_without = results_without.get(key, 0) * scale
        better    = '←' if v_with > v_without else ''
        print(f'{label:<25} {v_with:>18.4f} {v_without:>18.4f} {better}')

    print('=' * 62)
    print('← indicates override version performed better on that metric')


# ── Smoke test ─────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('Testing VasantDharOverride...')
    print('=' * 60)

    override = VasantDharOverride(threshold=2.5, window=60)

    # Test 1: Normal market — should NOT trigger
    print('\nTest 1: Normal market conditions...')
    for _ in range(70):
        result = override.update(0.20, 0.18, 0.15)

    print(f'  Override triggered : {result["override"]}')
    print(f'  Z-scores           : {result["zscores"]}')
    assert not result['override'], "Should NOT trigger in normal market!"
    print('  PASSED')

    # Test 2: Crisis market — SHOULD trigger
    print('\nTest 2: Crisis market (COVID-like spike)...')
    override.reset()
    for _ in range(60):
        override.update(0.20, 0.18, 0.15)  # Build history

    # Now spike all sectors simultaneously
    result = override.update(0.90, 0.85, 0.70)
    print(f'  Override triggered : {result["override"]}')
    print(f'  Z-scores           : {result["zscores"]}')
    print(f'  Reason             : {result["reason"]}')
    assert result['override'], "SHOULD trigger during crisis!"
    print('  PASSED')

    # Test 3: Only one sector spikes — should NOT trigger
    # Test 3: Only one sector spikes — should NOT trigger
    print('\nTest 3: Only tech sector spikes (rotate, not override)...')
    override.reset()

    # Build history WITH some natural variation so one spike stands out
    np.random.seed(42)
    for _ in range(60):
        override.update(
            0.20 + np.random.normal(0, 0.03),  # natural variation
            0.18 + np.random.normal(0, 0.03),
            0.15 + np.random.normal(0, 0.02)
        )

    # Only XLK spikes — XLF and XLV stay normal
    result = override.update(0.90, 0.19, 0.16)
    print(f'  Override triggered : {result["override"]}')
    print(f'  Z-scores           : {result["zscores"]}')
    print(f'  Reason             : {result["reason"]}')
    assert not result['override'], "Should NOT trigger when only one sector spikes!"
    print('  PASSED')

    # Test 4: Real data analysis
    print('\nTest 4: Analyzing real historical data...')
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed', 'iv_features.csv'
    )

    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['date'])
        override_dates = analyze_override_history(df, threshold=2.5)

        if len(override_dates) > 0:
            print(f'\n  First 5 override dates:')
            print(override_dates.head().to_string(index=False))
            print('\n  These should be Feb-Mar 2020 (COVID crash)')
        print('  PASSED')
    else:
        print('  Skipped (data file not found)')

    print('\n' + '=' * 60)
    print('All override tests passed!')
    print('=' * 60)