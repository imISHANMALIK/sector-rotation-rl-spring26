"""
stress_test.py
==============
Stress testing the sector rotation RL agent on known crisis periods.

WHY STRESS TESTING?
A strategy that works in normal markets but fails during crises
is dangerous and undeployable. We need to verify:
1. The override activates during genuine market panics
2. The agent rotates defensively before full crises develop
3. The safety mechanism actually protects capital

CRISIS PERIODS:
1. COVID Crash (Feb-Apr 2020): Fastest 30% market drop in history
2. 2022 Bear Market (Jan-Oct 2022): Rate hike driven tech selloff
3. Aug 2024 Flash Crash: Japan carry trade unwind
4. 2024 Full Year: The headline evaluation result
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import SectorRotationEnv
from src.dqn_agent   import DQNAgent
from src.evaluate    import (sortino_ratio, sharpe_ratio,
                              max_drawdown, total_return,
                              cumulative_returns)

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'iv_features.csv')
PLOT_DIR  = os.path.join(ROOT_DIR, 'notebooks')
CKPT_DIR  = os.path.join(ROOT_DIR, 'checkpoints')

# Crisis periods to analyze
CRISIS_PERIODS = {
    'COVID Crash': {
        'start': '2020-02-01',
        'end':   '2020-04-30',
        'description': 'Fastest 30% market drop in history.',
    },
    '2022 Bear Market': {
        'start': '2022-01-01',
        'end':   '2022-10-31',
        'description': 'Fed rate hike driven selloff.',
    },
    'Aug 2024 Flash': {
        'start': '2024-07-15',
        'end':   '2024-09-15',
        'description': 'Japan carry trade unwind.',
    },
    '2024 Full Year': {
        'start': '2024-01-01',
        'end':   '2024-12-31',
        'description': 'Full evaluation period.',
    },
}


def analyze_crisis_period(df, crisis_name, start, end):
    """Analyze market conditions during a crisis period."""
    mask      = (df['date'] >= start) & (df['date'] <= end)
    period_df = df[mask].copy().reset_index(drop=True)

    if len(period_df) == 0:
        print(f'  No data for {crisis_name}')
        return None

    print(f'\n{"="*60}')
    print(f'CRISIS: {crisis_name}')
    print(f'Period: {start} to {end} ({len(period_df)} trading days)')
    print(f'{"="*60}')

    for sector, col in [('SPY', 'ret_spy'), ('XLK', 'ret_xlk'),
                        ('XLF', 'ret_xlf'), ('XLV', 'ret_xlv')]:
        if col in period_df.columns:
            rets = period_df[col].dropna().values
            print(f'  {sector:<6}: Return={total_return(rets):+.2f}% | '
                  f'MaxDD={max_drawdown(rets)*100:.2f}% | '
                  f'Sortino={sortino_ratio(rets):.3f}')

    zscore_cols = ['zscore_xlk', 'zscore_xlf', 'zscore_xlv']
    if all(c in period_df.columns for c in zscore_cols):
        override_mask = (period_df[zscore_cols] > 2.5).all(axis=1)
        n_override    = override_mask.sum()
        print(f'\n  Override triggers: {n_override} / {len(period_df)} '
              f'({n_override/len(period_df)*100:.1f}%)')

    return period_df


def run_period_backtest(agent, df, start, end, period_name):
    """
    Run the agent on a specific time period.

    We write the period data to a temp file, create a temporary
    environment, run the agent through it, then clean up.
    This lets us test any arbitrary date range.
    """
    mask      = (df['date'] >= start) & (df['date'] <= end)
    period_df = df[mask].copy().reset_index(drop=True)

    if len(period_df) < 5:
        print(f'  Not enough data for {period_name}')
        return None

    # Write temp CSV for this period
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False
    ) as f:
        period_df.to_csv(f.name, index=False)
        temp_path = f.name

    original_eps  = agent.epsilon
    agent.epsilon = 0.0

    try:
        env      = SectorRotationEnv(temp_path, mode='all')
        state, _ = env.reset()

        records         = []
        portfolio_value = 1.0
        action_names    = {0: 'XLK', 1: 'XLF', 2: 'XLV', 3: 'CASH'}

        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)

            portfolio_value *= np.exp(info['daily_return'])
            records.append({
                'date':            info['date'],
                'action':          action_names[info['action_executed']],
                'override':        info['override_triggered'],
                'daily_return':    info['daily_return'],
                'portfolio_value': portfolio_value,
            })

            state = next_state
            if terminated or truncated:
                break

    finally:
        agent.epsilon = original_eps
        os.unlink(temp_path)

    results    = pd.DataFrame(records)
    results['date'] = pd.to_datetime(results['date'])
    returns    = results['daily_return'].values
    spy_rets   = period_df['ret_spy'].dropna().values

    return {
        'period':        period_name,
        'n_days':        len(returns),
        'agent_sortino': sortino_ratio(returns),
        'spy_sortino':   sortino_ratio(spy_rets),
        'agent_return':  total_return(returns),
        'spy_return':    total_return(spy_rets),
        'agent_maxdd':   max_drawdown(returns),
        'spy_maxdd':     max_drawdown(spy_rets),
        'n_overrides':   int(results['override'].sum()),
        'action_counts': results['action'].value_counts().to_dict(),
        'results_df':    results,
    }


def plot_stress_results(all_metrics, df):
    """Generate stress test visualization."""
    valid = [m for m in all_metrics if m is not None]
    if not valid:
        print('No metrics to plot')
        return

    n = min(len(valid), 4)
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        'Stress Test — Agent vs SPY Across Market Regimes',
        fontsize=15, fontweight='bold'
    )

    # One subplot per crisis period
    for i, metrics in enumerate(valid[:4]):
        ax        = fig.add_subplot(3, 2, i + 1)
        res_df    = metrics['results_df']
        agent_cum = res_df['portfolio_value'].values

        ax.plot(res_df['date'], agent_cum,
                label=f'Agent ({metrics["agent_return"]:+.1f}%)',
                color='#2196F3', linewidth=2)

        # SPY for same period
        start = str(res_df['date'].iloc[0])[:10]
        end   = str(res_df['date'].iloc[-1])[:10]
        p_df  = df[(df['date'] >= start) & (df['date'] <= end)]
        spy_r = p_df['ret_spy'].dropna().values
        if len(spy_r) > 0:
            spy_c = np.exp(np.cumsum(spy_r))
            ax.plot(p_df.dropna(subset=['ret_spy'])['date'].values, spy_c,
                    label=f'SPY ({metrics["spy_return"]:+.1f}%)',
                    color='#FF5722', linewidth=2, linestyle='--')

        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)

        # Override shading
        for d in res_df[res_df['override']]['date']:
            ax.axvspan(d, d + pd.Timedelta(days=1),
                       alpha=0.3, color='red', linewidth=0)

        ax.set_title(
            f'{metrics["period"]}\n'
            f'Sortino: Agent={metrics["agent_sortino"]:.2f} '
            f'SPY={metrics["spy_sortino"]:.2f}',
            fontweight='bold', fontsize=10
        )
        ax.set_ylabel('Portfolio Value')
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Summary bar chart
    ax_bar = fig.add_subplot(3, 2, 5)
    names  = [m['period'][:15] for m in valid]
    a_sort = [m['agent_sortino'] for m in valid]
    s_sort = [m['spy_sortino']   for m in valid]
    x      = np.arange(len(names))
    w      = 0.35

    ax_bar.bar(x - w/2, a_sort, w, label='DQN Agent',
               color='#2196F3', alpha=0.85)
    ax_bar.bar(x + w/2, s_sort, w, label='SPY',
               color='#FF5722', alpha=0.85)
    ax_bar.set_title('Sortino Ratio by Period', fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=15, fontsize=8)
    ax_bar.legend(fontsize=9)
    ax_bar.axhline(0, color='black', linewidth=0.8)

    # Override timeline
    ax_ov = fig.add_subplot(3, 2, 6)
    zcols = ['zscore_xlk', 'zscore_xlf', 'zscore_xlv']
    zcols_present = [c for c in zcols if c in df.columns]

    if zcols_present:
        colors_z = {'zscore_xlk': '#2196F3',
                    'zscore_xlf': '#4CAF50',
                    'zscore_xlv': '#FF9800'}
        for col in zcols_present:
            ax_ov.plot(df['date'], df[col],
                       color=colors_z.get(col, 'gray'),
                       linewidth=0.8, alpha=0.7,
                       label=col.replace('zscore_', '').upper())

        ax_ov.axhline(2.5, color='red', linestyle='--',
                      linewidth=1.5, label='Override (2.5)')

        if len(zcols_present) == 3:
            ov_mask = (df[zcols_present] > 2.5).all(axis=1)
            for d in df[ov_mask]['date']:
                ax_ov.axvspan(d, d + pd.Timedelta(days=1),
                              alpha=0.3, color='red', linewidth=0)

    ax_ov.set_title('IV Z-Scores: Override Timeline (2020-2024)',
                    fontweight='bold')
    ax_ov.set_ylabel('Z-Score')
    ax_ov.legend(fontsize=7)
    ax_ov.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, 'stress_test_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved → {path}')
    plt.show()


def print_stress_report(all_metrics):
    """Print the complete stress test report."""
    valid = [m for m in all_metrics if m is not None]

    print('\n' + '=' * 75)
    print(f'{"STRESS TEST REPORT":^75}')
    print('=' * 75)
    print(f'{"Period":<22} {"Agt Sort":>9} {"SPY Sort":>9} '
          f'{"Agt Ret%":>9} {"SPY Ret%":>9} '
          f'{"Agt DD%":>8} {"Overrides":>10}')
    print('-' * 75)

    for m in valid:
        beat = '✓' if m['agent_sortino'] > m['spy_sortino'] else '✗'
        print(f'{m["period"]:<22} '
              f'{m["agent_sortino"]:>9.4f} '
              f'{m["spy_sortino"]:>9.4f} '
              f'{m["agent_return"]:>9.2f} '
              f'{m["spy_return"]:>9.2f} '
              f'{m["agent_maxdd"]*100:>8.2f} '
              f'{m["n_overrides"]:>10} {beat}')

    print('=' * 75)
    wins = sum(1 for m in valid if m['agent_sortino'] > m['spy_sortino'])
    print(f'\nAgent beats SPY: {wins}/{len(valid)} periods')
    print()

    for m in valid:
        print(f'{m["period"]} — Sector allocation:')
        total = m["n_days"]
        for sector in ['XLK', 'XLF', 'XLV', 'CASH']:
            count = m['action_counts'].get(sector, 0)
            pct   = 100 * count / max(total, 1)
            bar   = '█' * int(pct / 5)
            print(f'  {sector:<6}: {count:>4}d ({pct:5.1f}%) {bar}')
        print()


# ── Entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('STRESS TEST — Sector Rotation DQN')
    print('=' * 60)

    ckpt_path = os.path.join(CKPT_DIR, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        print(f'Checkpoint not found: {ckpt_path}')
        print('Run: python src/train.py')
        exit(1)

    with open(os.path.join(ROOT_DIR, 'configs', 'hyperparams.yaml')) as f:
        config = yaml.safe_load(f)

    agent = DQNAgent(
        state_dim  = config['state_dim'],
        action_dim = config['action_dim'],
        hidden     = config['hidden_dim'],
    )
    agent.load(ckpt_path)

    df = pd.read_csv(DATA_PATH, parse_dates=['date'])

    # Market analysis
    print('\n--- Market Analysis ---')
    for name, period in CRISIS_PERIODS.items():
        analyze_crisis_period(df, name, period['start'], period['end'])

    # Agent performance on each period
    print('\n--- Agent Performance ---')
    all_metrics = []
    for name, period in CRISIS_PERIODS.items():
        print(f'\nRunning: {name}...')
        metrics = run_period_backtest(
            agent, df,
            period['start'], period['end'], name
        )
        if metrics:
            all_metrics.append(metrics)
            print(f'  Agent Sortino: {metrics["agent_sortino"]:.4f} '
                  f'vs SPY: {metrics["spy_sortino"]:.4f}')
            print(f'  Agent Return:  {metrics["agent_return"]:+.2f}% '
                  f'vs SPY: {metrics["spy_return"]:+.2f}%')
            print(f'  Overrides:     {metrics["n_overrides"]}')

    print_stress_report(all_metrics)
    plot_stress_results(all_metrics, df)