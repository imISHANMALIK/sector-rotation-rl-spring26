"""
ablation.py
===========
Ablation studies for the sector rotation RL agent.

WHAT IS AN ABLATION STUDY?
Remove one component at a time and measure the performance drop.
If performance drops significantly → that component matters.
If performance stays the same → that component is unnecessary.

OUR THREE ABLATIONS:
1. Override ON vs OFF
   - Train with override disabled
   - Shows: does the safety layer actually protect capital?

2. Sortino vs Raw Return reward
   - Train with raw daily return as reward (no Sortino shaping)
   - Shows: does risk-adjusted reward shaping improve behavior?

3. DQN vs Random Policy
   - Random policy: pick a random sector every day
   - Shows: did the agent actually learn, or is it just lucky?

WHY THESE THREE?
They directly validate the three key design decisions:
1. The Vasant Dhar override
2. The Sortino reward shaping
3. The DQN learning algorithm itself
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import SectorRotationEnv
from src.dqn_agent   import DQNAgent
from src.evaluate    import (sortino_ratio, sharpe_ratio,
                              max_drawdown, total_return,
                              cumulative_returns)

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'processed', 'iv_features.csv')
PLOT_DIR  = os.path.join(ROOT_DIR, 'notebooks')


def run_training_variant(
    config,
    variant_name,
    n_episodes=500,
    disable_override=False,
    use_raw_reward=False
):
    """
    Train a variant of the agent for ablation comparison.

    Parameters:
    -----------
    config           : dict — base hyperparameters
    variant_name     : str  — name for logging
    n_episodes       : int  — training episodes (500 for speed)
    disable_override : bool — if True, disable Vasant Dhar override
    use_raw_reward   : bool — if True, use raw return not Sortino

    Returns:
    --------
    dict — performance metrics on 2024 test data

    WHY 500 EPISODES (not 2000)?
    Ablation needs 3 training runs. 500 each = ~15-20 mins total.
    Enough signal to compare components without taking all day.
    """
    print(f'\nTraining: {variant_name}')
    print(f'  Override : {"DISABLED" if disable_override else "ENABLED"}')
    print(f'  Reward   : {"RAW RETURN" if use_raw_reward else "SORTINO"}')
    print(f'  Episodes : {n_episodes}')

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_env = SectorRotationEnv(DATA_PATH, mode='train')
    val_env   = SectorRotationEnv(DATA_PATH, mode='test')

    # Disable override by monkey-patching the check method
    if disable_override:
        train_env._check_override = lambda t: False
        val_env._check_override   = lambda t: False

    steps_per_ep = len(train_env._df)
    total_steps  = n_episodes * steps_per_ep
    eps_decay    = 1.0 - (1.0 - config['eps_end']) / (total_steps * 0.8)

    agent = DQNAgent(
        state_dim          = config['state_dim'],
        action_dim         = config['action_dim'],
        hidden             = config['hidden_dim'],
        lr                 = config['learning_rate'],
        gamma              = config['gamma'],
        epsilon            = config['eps_start'],
        epsilon_min        = config['eps_end'],
        epsilon_decay      = eps_decay,
        buffer_capacity    = config['buffer_size'],
        batch_size         = config['batch_size'],
        target_update_freq = config['target_update'],
        grad_clip          = config['grad_clip'],
    )

    # Training loop
    for episode in range(1, n_episodes + 1):
        state, _ = train_env.reset()

        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = \
                train_env.step(action)

            # Use raw return for ablation variant
            if use_raw_reward:
                reward = info['daily_return']

            agent.buffer.push(
                state, action, reward, next_state,
                terminated or truncated
            )
            agent.train_step()
            state = next_state

            if terminated or truncated:
                break

        if episode % 100 == 0:
            print(f'  Episode {episode}/{n_episodes}...')

    # Evaluate on 2024 test data
    original_eps  = agent.epsilon
    agent.epsilon = 0.0

    returns   = []
    actions   = []
    overrides = 0

    state, _ = val_env.reset()
    while True:
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, info = \
            val_env.step(action)
        returns.append(info['daily_return'])
        actions.append(info['action_executed'])
        if info['override_triggered']:
            overrides += 1
        state = next_state
        if terminated or truncated:
            break

    agent.epsilon = original_eps
    returns_arr   = np.array(returns)

    action_names  = {0: 'XLK', 1: 'XLF', 2: 'XLV', 3: 'CASH'}
    action_counts = {
        action_names[i]: int(np.sum(np.array(actions) == i))
        for i in range(4)
    }

    metrics = {
        'variant':       variant_name,
        'sortino':       sortino_ratio(returns_arr),
        'sharpe':        sharpe_ratio(returns_arr),
        'total_return':  total_return(returns_arr),
        'max_drawdown':  max_drawdown(returns_arr),
        'n_overrides':   overrides,
        'returns':       returns_arr,
        'action_counts': action_counts,
    }

    print(f'  Sortino: {metrics["sortino"]:.4f} | '
          f'Return: {metrics["total_return"]:+.2f}% | '
          f'MaxDD: {metrics["max_drawdown"]*100:.2f}%')

    return metrics


def random_baseline():
    """
    Random policy: pick a random sector every day.

    WHY THIS BASELINE?
    If our DQN doesn't beat random selection, the agent
    hasn't learned anything useful. A strong RL agent
    should significantly outperform random.

    We run 10 seeds and average to reduce variance.
    """
    print('\nRunning random baseline (10 seeds averaged)...')

    df      = pd.read_csv(DATA_PATH, parse_dates=['date'])
    test_df = df[df['date'] >= '2024-01-01'].reset_index(drop=True)
    ret_cols = ['ret_xlk', 'ret_xlf', 'ret_xlv']

    all_sortinos = []
    all_returns  = []
    all_maxdds   = []
    all_rets_arr = []

    for seed in range(10):
        np.random.seed(seed)
        returns = []

        for _, row in test_df.iterrows():
            action = np.random.randint(0, 3)
            col    = ret_cols[action]
            ret    = float(row[col]) if not pd.isna(row[col]) else 0.0
            returns.append(ret)

        arr = np.array(returns)
        all_sortinos.append(sortino_ratio(arr))
        all_returns.append(total_return(arr))
        all_maxdds.append(max_drawdown(arr))
        all_rets_arr.append(arr)

    # Average across seeds
    avg_returns = np.mean(all_rets_arr, axis=0)

    metrics = {
        'variant':      'Random Policy',
        'sortino':      float(np.mean(all_sortinos)),
        'sharpe':       sharpe_ratio(avg_returns),
        'total_return': float(np.mean(all_returns)),
        'max_drawdown': float(np.mean(all_maxdds)),
        'n_overrides':  0,
        'returns':      avg_returns,
        'action_counts': {'XLK': 0, 'XLF': 0, 'XLV': 0, 'CASH': 0},
    }

    print(f'  Sortino: {metrics["sortino"]:.4f} | '
          f'Return: {metrics["total_return"]:+.2f}% | '
          f'MaxDD: {metrics["max_drawdown"]*100:.2f}%')

    return metrics


def spy_metrics():
    """SPY buy-and-hold for 2024 test period."""
    df      = pd.read_csv(DATA_PATH, parse_dates=['date'])
    test_df = df[df['date'] >= '2024-01-01']
    rets    = test_df['ret_spy'].dropna().values

    metrics = {
        'variant':      'SPY Buy-and-Hold',
        'sortino':      sortino_ratio(rets),
        'sharpe':       sharpe_ratio(rets),
        'total_return': total_return(rets),
        'max_drawdown': max_drawdown(rets),
        'n_overrides':  0,
        'returns':      rets,
        'action_counts': {},
    }

    print(f'\nSPY baseline:'
          f'  Sortino: {metrics["sortino"]:.4f} | '
          f'Return: {metrics["total_return"]:+.2f}% | '
          f'MaxDD: {metrics["max_drawdown"]*100:.2f}%')

    return metrics


def plot_ablation(all_metrics, save=True):
    """
    Generate ablation comparison plots.

    WHAT GETS PLOTTED:
    1. Sortino ratio comparison (bar chart)
    2. Total return comparison (bar chart)
    3. Max drawdown comparison (bar chart)
    4. Cumulative returns over time (line chart)
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        'Ablation Study — Component Contribution Analysis\n2024 Test Period',
        fontsize=14, fontweight='bold'
    )

    variants = [m['variant'] for m in all_metrics]
    colors   = ['#2196F3', '#F44336', '#FF9800',
                 '#9E9E9E', '#FF5722']

    # ── Sortino comparison ─────────────────────────────────────
    ax1      = axes[0, 0]
    sortinos = [m['sortino'] for m in all_metrics]
    bars     = ax1.bar(range(len(variants)), sortinos,
                       color=colors[:len(variants)],
                       alpha=0.85, edgecolor='white')
    ax1.set_title('Sortino Ratio', fontweight='bold')
    ax1.set_ylabel('Sortino Ratio (higher = better)')
    ax1.set_xticks(range(len(variants)))
    ax1.set_xticklabels(variants, rotation=15,
                         ha='right', fontsize=8)
    ax1.axhline(0, color='black', linewidth=0.8)
    for bar, val in zip(bars, sortinos):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 max(bar.get_height(), 0) + 0.05,
                 f'{val:.3f}', ha='center',
                 va='bottom', fontsize=9, fontweight='bold')

    # ── Total return comparison ────────────────────────────────
    ax2     = axes[0, 1]
    returns = [m['total_return'] for m in all_metrics]
    bars    = ax2.bar(range(len(variants)), returns,
                      color=colors[:len(variants)],
                      alpha=0.85, edgecolor='white')
    ax2.set_title('Total Return (%)', fontweight='bold')
    ax2.set_ylabel('Total Return % (higher = better)')
    ax2.set_xticks(range(len(variants)))
    ax2.set_xticklabels(variants, rotation=15,
                         ha='right', fontsize=8)
    ax2.axhline(0, color='black', linewidth=0.8)
    for bar, val in zip(bars, returns):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 max(bar.get_height(), 0) + 0.3,
                 f'{val:.1f}%', ha='center',
                 va='bottom', fontsize=9, fontweight='bold')

    # ── Max drawdown comparison ────────────────────────────────
    ax3       = axes[1, 0]
    drawdowns = [m['max_drawdown']*100 for m in all_metrics]
    bars      = ax3.bar(range(len(variants)), drawdowns,
                        color=colors[:len(variants)],
                        alpha=0.85, edgecolor='white')
    ax3.set_title('Max Drawdown (%)', fontweight='bold')
    ax3.set_ylabel('Max Drawdown % (less negative = better)')
    ax3.set_xticks(range(len(variants)))
    ax3.set_xticklabels(variants, rotation=15,
                         ha='right', fontsize=8)
    for bar, val in zip(bars, drawdowns):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() - 0.3,
                 f'{val:.1f}%', ha='center',
                 va='top', fontsize=9, fontweight='bold')

    # ── Cumulative returns ─────────────────────────────────────
    ax4     = axes[1, 1]
    df      = pd.read_csv(DATA_PATH, parse_dates=['date'])
    test_df = df[df['date'] >= '2024-01-01'].reset_index(drop=True)
    dates   = test_df['date'].values

    for i, m in enumerate(all_metrics):
        rets = m['returns']
        n    = min(len(rets), len(dates))
        cum  = np.exp(np.cumsum(rets[:n]))
        ax4.plot(dates[:n], cum,
                 label=m['variant'],
                 color=colors[i % len(colors)],
                 linewidth=2)

    ax4.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax4.set_title('Cumulative Returns by Variant',
                  fontweight='bold')
    ax4.set_ylabel('Portfolio Value (start=1.0)')
    ax4.legend(fontsize=8)

    plt.tight_layout()

    if save:
        path = os.path.join(PLOT_DIR, 'ablation_results.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'\nPlot saved → {path}')

    plt.show()


def print_ablation_report(all_metrics):
    """Print the complete ablation study report."""
    print('\n' + '=' * 72)
    print(f'{"ABLATION STUDY REPORT — 2024 TEST PERIOD":^72}')
    print('=' * 72)
    print(f'{"Variant":<28} {"Sortino":>8} {"Sharpe":>8} '
          f'{"Ret%":>8} {"MaxDD%":>8} {"Overrides":>10}')
    print('-' * 72)

    for m in all_metrics:
        print(f'{m["variant"]:<28} '
              f'{m["sortino"]:>8.4f} '
              f'{m["sharpe"]:>8.4f} '
              f'{m["total_return"]:>8.2f} '
              f'{m["max_drawdown"]*100:>8.2f} '
              f'{m["n_overrides"]:>10}')

    print('=' * 72)

    # Component contribution analysis
    base = next(
        (m for m in all_metrics
         if 'Override + Sortino' in m['variant']),
        None
    )

    if base:
        print(f'\nComponent Contribution (vs Full Model):')
        for m in all_metrics:
            if m['variant'] == base['variant']:
                continue
            if m['sortino'] <= -900:
                continue
            delta_sort = base['sortino'] - m['sortino']
            delta_ret  = base['total_return'] - m['total_return']
            delta_dd   = (base['max_drawdown'] -
                          m['max_drawdown']) * 100
            print(f'\n  vs {m["variant"]}:')
            print(f'    Sortino improvement : {delta_sort:+.4f}')
            print(f'    Return improvement  : {delta_ret:+.2f}%')
            print(f'    Drawdown reduction  : {delta_dd:+.2f}%')


# ── Entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 65)
    print('ABLATION STUDY — Sector Rotation DQN')
    print('=' * 65)

    with open(os.path.join(
        ROOT_DIR, 'configs', 'hyperparams.yaml'
    )) as f:
        config = yaml.safe_load(f)

    N_EPISODES = 500  # Fast enough for ablation

    all_metrics = []

    # 1. Full model
    print('\n[1/5] Full model: DQN + Override + Sortino reward')
    m1 = run_training_variant(
        config,
        'DQN + Override + Sortino',
        n_episodes=N_EPISODES,
        disable_override=False,
        use_raw_reward=False
    )
    all_metrics.append(m1)

    # 2. No override
    print('\n[2/5] Ablation: DQN + Sortino (NO override)')
    m2 = run_training_variant(
        config,
        'DQN - No Override',
        n_episodes=N_EPISODES,
        disable_override=True,
        use_raw_reward=False
    )
    all_metrics.append(m2)

    # 3. Raw return reward
    print('\n[3/5] Ablation: DQN + Override (RAW return reward)')
    m3 = run_training_variant(
        config,
        'DQN - Raw Reward',
        n_episodes=N_EPISODES,
        disable_override=False,
        use_raw_reward=True
    )
    all_metrics.append(m3)

    # 4. Random baseline
    print('\n[4/5] Random policy baseline')
    m4 = random_baseline()
    all_metrics.append(m4)

    # 5. SPY baseline
    print('\n[5/5] SPY buy-and-hold baseline')
    m5 = spy_metrics()
    all_metrics.append(m5)

    # Print report
    print_ablation_report(all_metrics)

    # Generate plots
    plot_ablation(all_metrics)

    # Save results CSV
    results_path = os.path.join(PLOT_DIR, 'ablation_results.csv')
    pd.DataFrame([
        {k: v for k, v in m.items()
         if k not in ('returns', 'action_counts')}
        for m in all_metrics
    ]).to_csv(results_path, index=False)
    print(f'Results saved → {results_path}')
    print('\nAblation study complete!')