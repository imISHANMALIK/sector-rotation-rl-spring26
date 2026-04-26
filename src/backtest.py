"""
backtest.py
===========
Backtesting engine for the sector rotation RL agent.

WHAT IS BACKTESTING?
Running a trading strategy on historical data as if you were
trading in real-time — you only use information available at
each point in time (no look-ahead bias).

WHY A SEPARATE BACKTEST FILE?
- evaluate.py has pure metric functions (math only)
- backtest.py runs the full simulation and produces detailed results
- Separation makes code cleaner and easier to debug

WHAT THIS PRODUCES:
- Day-by-day portfolio value tracking
- Every action the agent took and why
- Comparison charts vs SPY baseline
- Full metrics table for the report
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import SectorRotationEnv
from src.dqn_agent   import DQNAgent
from src.evaluate    import (sortino_ratio, sharpe_ratio,
                              max_drawdown, cumulative_returns,
                              total_return)

# ── Paths ──────────────────────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'iv_features.csv')
PLOT_DIR  = os.path.join(ROOT_DIR, 'notebooks')
CKPT_DIR  = os.path.join(ROOT_DIR, 'checkpoints')
os.makedirs(PLOT_DIR, exist_ok=True)


def run_backtest(agent, data_path=DATA_PATH, mode='test'):
    """
    Run the full backtesting simulation day by day.

    Parameters:
    -----------
    agent     : DQNAgent — trained agent loaded from checkpoint
    data_path : str      — path to iv_features.csv
    mode      : str      — 'test' for 2024, 'train' for in-sample check

    Returns:
    --------
    pd.DataFrame — detailed day-by-day trading log
    dict         — summary performance metrics

    THE SIMULATION:
    For each trading day in the evaluation period:
    1. Agent observes current IV state
    2. Override check runs inside environment.step()
    3. Agent picks sector (or override forces CASH)
    4. Record: date, action, return, portfolio value
    5. Move to next day

    WHY NO LOOK-AHEAD BIAS?
    The agent only sees data up to the current day.
    We never feed it future prices or future IV values.
    This simulates real deployment conditions accurately.
    """
    # Force greedy policy — no random exploration during backtesting
    original_eps  = agent.epsilon
    agent.epsilon = 0.0

    env      = SectorRotationEnv(data_path, mode=mode)
    state, _ = env.reset()

    records         = []
    portfolio_value = 1.0  # Start with $1.00 normalized

    action_names = {0: 'XLK', 1: 'XLF', 2: 'XLV', 3: 'CASH'}

    while True:
        # Agent selects best known action (greedy)
        action = agent.select_action(state, training=False)

        # Environment executes action (override may change it)
        next_state, reward, terminated, truncated, info = env.step(action)

        # Update portfolio value
        # WHY EXP? We use log returns, so exp(log_return) converts back
        # to a portfolio multiplier. e.g. log_return=0.01 → multiply by 1.01005
        portfolio_value *= np.exp(info['daily_return'])

        records.append({
            'date':            info['date'],
            'action_requested': action_names[info['action_requested']],
            'action_executed':  action_names[info['action_executed']],
            'override':         info['override_triggered'],
            'daily_return':     info['daily_return'],
            'reward':           info['reward'],
            'portfolio_value':  portfolio_value,
        })

        state = next_state
        if terminated or truncated:
            break

    # Restore epsilon
    agent.epsilon = original_eps

    # Build results DataFrame
    results_df           = pd.DataFrame(records)
    results_df['date']   = pd.to_datetime(results_df['date'])

    # Compute summary metrics
    returns  = results_df['daily_return'].values
    metrics  = {
        'strategy':      'DQN Agent',
        'sortino':       sortino_ratio(returns),
        'sharpe':        sharpe_ratio(returns),
        'max_drawdown':  max_drawdown(returns),
        'total_return':  total_return(returns),
        'n_days':        len(returns),
        'n_overrides':   int(results_df['override'].sum()),
        'final_value':   portfolio_value,
        'action_counts': results_df['action_executed'].value_counts().to_dict(),
    }

    return results_df, metrics


def compute_spy_metrics(data_path=DATA_PATH, mode='test'):
    """
    Compute SPY buy-and-hold metrics for the same evaluation period.

    WHY SPY AS BASELINE?
    SPY tracks the S&P 500 — the simplest possible investing strategy.
    If our complex RL system can't beat passive S&P 500 investing,
    there's no justification for using RL at all.
    This is the standard benchmark in quantitative finance.
    """
    df = pd.read_csv(data_path, parse_dates=['date'])

    if mode == 'test':
        df = df[df['date'] >= '2024-01-01']
    elif mode == 'train':
        df = df[df['date'] <= '2023-12-31']

    returns       = df['ret_spy'].dropna().values
    spy_portfolio = np.exp(np.cumsum(returns))

    return {
        'strategy':      'SPY Buy-and-Hold',
        'sortino':       sortino_ratio(returns),
        'sharpe':        sharpe_ratio(returns),
        'max_drawdown':  max_drawdown(returns),
        'total_return':  total_return(returns),
        'n_days':        len(returns),
        'returns':       returns,
        'portfolio':     spy_portfolio,
        'dates':         df.dropna(subset=['ret_spy'])['date'].values,
    }


def plot_results(results_df, agent_metrics, spy_metrics, save=True):
    """
    Generate all result plots for the report and demo.

    PLOTS GENERATED:
    1. Cumulative returns: agent vs SPY — the headline chart
    2. Drawdown curve: worst losses over time
    3. Sector allocation: which sector agent picked each day
    4. Daily returns: with override days highlighted
    5. Metrics comparison: bar chart side by side
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        'DQN Agent vs SPY Buy-and-Hold — 2024 Test Period',
        fontsize=16, fontweight='bold', y=0.98
    )

    # ── Plot 1: Cumulative Returns ─────────────────────────────
    ax1 = fig.add_subplot(3, 2, (1, 2))

    ax1.plot(
        results_df['date'], results_df['portfolio_value'],
        label=f'DQN Agent (Return: {agent_metrics["total_return"]:+.1f}%)',
        color='#2196F3', linewidth=2
    )
    ax1.plot(
        spy_metrics['dates'], spy_metrics['portfolio'],
        label=f'SPY Buy-Hold (Return: {spy_metrics["total_return"]:+.1f}%)',
        color='#FF5722', linewidth=2, linestyle='--'
    )
    ax1.axhline(1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    # Shade override periods in red
    override_days = results_df[results_df['override']]['date']
    for d in override_days:
        ax1.axvspan(d, d + pd.Timedelta(days=1),
                    alpha=0.3, color='red', linewidth=0)

    ax1.set_title('Cumulative Portfolio Value (Red = Override/CASH days)',
                  fontweight='bold')
    ax1.set_ylabel('Portfolio Value (start = 1.0)')
    ax1.legend(fontsize=10)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # ── Plot 2: Drawdown ───────────────────────────────────────
    ax2 = fig.add_subplot(3, 2, 3)

    agent_cum  = results_df['portfolio_value'].values
    agent_peak = np.maximum.accumulate(agent_cum)
    agent_dd   = (agent_cum - agent_peak) / agent_peak * 100

    spy_cum    = spy_metrics['portfolio']
    spy_peak   = np.maximum.accumulate(spy_cum)
    spy_dd     = (spy_cum - spy_peak) / spy_peak * 100

    ax2.fill_between(results_df['date'], agent_dd,
                     alpha=0.4, color='#2196F3', label='DQN Agent')
    ax2.fill_between(
        spy_metrics['dates'][-len(spy_dd):], spy_dd,
        alpha=0.4, color='#FF5722', label='SPY'
    )
    ax2.plot(results_df['date'], agent_dd,
             color='#2196F3', linewidth=1)

    ax2.set_title('Drawdown (%)', fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # ── Plot 3: Sector Allocation ──────────────────────────────
    ax3 = fig.add_subplot(3, 2, 4)

    action_color  = {
        'XLK': '#2196F3', 'XLF': '#4CAF50',
        'XLV': '#FF9800', 'CASH': '#9E9E9E'
    }
    action_counts = agent_metrics.get('action_counts', {})

    # Ensure consistent order
    sectors = ['XLK', 'XLF', 'XLV', 'CASH']
    counts  = [action_counts.get(s, 0) for s in sectors]
    colors  = [action_color[s] for s in sectors]

    bars = ax3.bar(sectors, counts, color=colors,
                   edgecolor='white', linewidth=1.5)
    ax3.set_title('Sector Allocation (Days)', fontweight='bold')
    ax3.set_ylabel('Number of Days')

    for bar, count in zip(bars, counts):
        pct = 100 * count / max(sum(counts), 1)
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=9
        )

    # ── Plot 4: Daily Returns ──────────────────────────────────
    ax4 = fig.add_subplot(3, 2, 5)

    ax4.plot(
        results_df['date'], results_df['daily_return'],
        alpha=0.6, color='#2196F3', linewidth=0.8
    )
    ax4.axhline(0, color='black', linestyle='--',
                linewidth=1, alpha=0.5)

    # Highlight override days
    for d in override_days:
        ax4.axvline(d, color='red', alpha=0.4, linewidth=1.5)

    ax4.set_title('Daily Returns (Red lines = Override days)',
                  fontweight='bold')
    ax4.set_ylabel('Daily Log Return')
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # ── Plot 5: Metrics Comparison ─────────────────────────────
    ax5 = fig.add_subplot(3, 2, 6)

    metric_labels = ['Sortino\nRatio', 'Sharpe\nRatio', 'Total\nReturn %']
    agent_vals    = [
        agent_metrics['sortino'],
        agent_metrics['sharpe'],
        agent_metrics['total_return'],
    ]
    spy_vals = [
        spy_metrics['sortino'],
        spy_metrics['sharpe'],
        spy_metrics['total_return'],
    ]

    x     = np.arange(len(metric_labels))
    width = 0.35

    ax5.bar(x - width/2, agent_vals, width,
            label='DQN Agent', color='#2196F3',
            alpha=0.85, edgecolor='white')
    ax5.bar(x + width/2, spy_vals, width,
            label='SPY Buy-Hold', color='#FF5722',
            alpha=0.85, edgecolor='white')

    ax5.set_title('Key Metrics Comparison', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metric_labels)
    ax5.legend(fontsize=9)
    ax5.axhline(0, color='black', linewidth=0.8, alpha=0.5)

    # Add value labels on bars
    for bars, vals in [(ax5.containers[0], agent_vals),
                       (ax5.containers[1], spy_vals)]:
        for bar, val in zip(bars, vals):
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=8
            )

    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, 'backtest_results.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'\nPlot saved → {path}')

    plt.show()
    return fig


def print_full_report(agent_metrics, spy_metrics):
    """
    Print the full performance comparison table.
    This exact table goes in your report.
    """
    print('\n' + '=' * 65)
    print(f'{"BACKTEST RESULTS — 2024 TEST PERIOD":^65}')
    print('=' * 65)
    print(f'{"Metric":<28} {"DQN Agent":>17} {"SPY Buy-Hold":>17}')
    print('-' * 65)

    rows = [
        ('Sortino Ratio',       agent_metrics['sortino'],
                                spy_metrics['sortino'],       1),
        ('Sharpe Ratio',        agent_metrics['sharpe'],
                                spy_metrics['sharpe'],        1),
        ('Total Return (%)',    agent_metrics['total_return'],
                                spy_metrics['total_return'],  1),
        ('Max Drawdown (%)',    agent_metrics['max_drawdown'] * 100,
                                spy_metrics['max_drawdown'] * 100, 1),
        ('Final Portfolio Val', agent_metrics['final_value'],
                                spy_metrics['portfolio'][-1], 1),
    ]

    for label, agent_val, spy_val, _ in rows:
        print(f'{label:<28} {agent_val:>17.4f} {spy_val:>17.4f}')

    print(f'{"Trading Days":<28} '
          f'{agent_metrics["n_days"]:>17} '
          f'{spy_metrics["n_days"]:>17}')
    print(f'{"Override Triggers":<28} '
          f'{agent_metrics["n_overrides"]:>17} '
          f'{"N/A":>17}')
    print('=' * 65)

    print(f'\nAgent Sector Allocation (2024):')
    total = agent_metrics['n_days']
    for sector in ['XLK', 'XLF', 'XLV', 'CASH']:
        count = agent_metrics['action_counts'].get(sector, 0)
        pct   = 100 * count / max(total, 1)
        bar   = '█' * int(pct / 2)
        print(f'  {sector:<6}: {count:>4} days ({pct:5.1f}%) {bar}')

    print()
    beat = agent_metrics['sortino'] > spy_metrics['sortino']
    print(f'  RESULT: Agent {"BEATS" if beat else "UNDERPERFORMS"} '
          f'SPY on Sortino ratio')
    print(f'  Sortino gap: '
          f'{agent_metrics["sortino"] - spy_metrics["sortino"]:+.4f}')
    print('=' * 65)


# ── Entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Backtest Sector Rotation DQN')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test'],
                        help='Evaluation mode: train=2020-2023, test=2024')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')
    args = parser.parse_args()

    ckpt_path = os.path.join(ROOT_DIR, args.checkpoint)

    if not os.path.exists(ckpt_path):
        print(f'Checkpoint not found: {ckpt_path}')
        print('Run training first: python src/train.py')
        exit(1)

    print(f'Loading checkpoint: {ckpt_path}')

    # Load config to recreate agent with same architecture
    config_path = os.path.join(ROOT_DIR, 'configs', 'hyperparams.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    agent = DQNAgent(
        state_dim  = config['state_dim'],
        action_dim = config['action_dim'],
        hidden     = config['hidden_dim'],
    )
    agent.load(ckpt_path)

    # Run backtest
    print(f'\nRunning backtest on {args.mode} data...')
    results_df, agent_metrics = run_backtest(agent, DATA_PATH, mode=args.mode)

    # SPY baseline
    spy_metrics = compute_spy_metrics(DATA_PATH, mode=args.mode)

    # Print full report
    print_full_report(agent_metrics, spy_metrics)

    # Generate and save plots
    if not args.no_plot:
        plot_results(results_df, agent_metrics, spy_metrics, save=True)

    # Save results CSV for further analysis
    results_path = os.path.join(ROOT_DIR, 'notebooks', 'backtest_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f'Results saved → {results_path}')