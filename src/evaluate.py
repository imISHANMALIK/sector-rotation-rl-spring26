"""
evaluate.py
===========
Evaluation metrics for the sector rotation RL agent.
"""

import numpy as np
import pandas as pd
import os


def sortino_ratio(returns, target_return=0.0, annualize=True):
    """
    Compute the Sortino ratio.
    Only penalizes returns BELOW target (downside risk).
    Upside volatility is NOT penalized.

    INTERPRETATION:
    - Sortino > 1.0 : Good
    - Sortino > 2.0 : Excellent
    - SPY typically : 0.5 - 1.0
    """
    returns = np.array(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    mean_ret = np.mean(returns)
    downside = returns[returns < target_return]

    if len(downside) == 0:
        return float(np.clip(mean_ret * np.sqrt(252), 0, 10))

    downside_dev = np.std(downside)

    if downside_dev < 1e-8:
        return 0.0

    ratio = (mean_ret - target_return) / downside_dev

    if annualize:
        ratio = ratio * np.sqrt(252)

    return float(ratio)


def sharpe_ratio(returns, risk_free_rate=0.0, annualize=True):
    """
    Compute the Sharpe ratio.
    Penalizes ALL volatility including upside.
    Reported for comparability with literature.

    INTERPRETATION:
    - Sharpe > 1.0 : Good
    - Sharpe > 2.0 : Very good
    - SPY typically: 0.5 - 0.8
    """
    returns = np.array(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    std = np.std(returns)

    if std < 1e-8:
        return 0.0

    ratio = (np.mean(returns) - risk_free_rate) / std

    if annualize:
        ratio = ratio * np.sqrt(252)

    return float(ratio)


def max_drawdown(returns):
    """
    Compute the maximum drawdown.

    WHAT IS MAX DRAWDOWN?
    The largest peak-to-trough decline in portfolio value.
    Example: portfolio grows to $150 then falls to $105
    Drawdown = (150 - 105) / 150 = 30%

    WHY IT MATTERS:
    Even a profitable strategy is unacceptable if investors
    face a 60% loss along the way. Max drawdown captures
    the worst-case investor experience.

    Returns negative number e.g. -0.35 means 35% max loss.
    """
    returns = np.array(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    cumulative  = np.exp(np.cumsum(returns))
    running_max = np.maximum.accumulate(cumulative)
    drawdowns   = (cumulative - running_max) / running_max

    return float(np.min(drawdowns))


def cumulative_returns(returns):
    """
    Compute cumulative portfolio value over time.
    Starting value = 1.0

    Returns np.array of portfolio value at each time step.
    """
    returns = np.array(returns, dtype=np.float64)
    returns = np.where(np.isnan(returns), 0, returns)
    return np.exp(np.cumsum(returns))


def total_return(returns):
    """
    Total return over the full period as a percentage.
    Example: 35.0 means the portfolio grew 35% overall.
    """
    cum = cumulative_returns(returns)
    if len(cum) == 0:
        return 0.0
    return float((cum[-1] - 1.0) * 100)


def spy_baseline(df):
    """
    Compute SPY buy-and-hold performance over the dataset period.

    WHY THIS BASELINE?
    SPY tracks the S&P 500 — the simplest possible strategy.
    If our RL agent can't beat passive S&P 500 investing,
    there's no justification for using RL at all.
    """
    spy_returns = df['ret_spy'].dropna().values

    return {
        'strategy':     'SPY Buy-and-Hold',
        'sortino':      sortino_ratio(spy_returns),
        'sharpe':       sharpe_ratio(spy_returns),
        'max_drawdown': max_drawdown(spy_returns),
        'total_return': total_return(spy_returns),
        'n_days':       len(spy_returns),
        'returns':      spy_returns,
        'cumulative':   cumulative_returns(spy_returns),
    }


def evaluate_agent(agent, env):
    """
    Run the trained agent through the test environment.

    Sets epsilon=0 so agent always picks best known action.
    Steps through every day in the test set.
    Records returns, actions, override triggers.

    Returns dict of comprehensive performance metrics.
    """
    original_epsilon = agent.epsilon
    agent.epsilon    = 0.0

    all_returns   = []
    all_actions   = []
    all_dates     = []
    override_days = []

    state, _ = env.reset()

    while True:
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, info = env.step(action)

        all_returns.append(info['daily_return'])
        all_actions.append(info['action_executed'])
        all_dates.append(info['date'])

        if info['override_triggered']:
            override_days.append(info['date'])

        state = next_state

        if terminated or truncated:
            break

    agent.epsilon = original_epsilon

    returns_array = np.array(all_returns)
    action_names  = {0: 'XLK', 1: 'XLF', 2: 'XLV', 3: 'CASH'}
    action_counts = {
        action_names[i]: int(np.sum(np.array(all_actions) == i))
        for i in range(4)
    }

    return {
        'strategy':       'DQN Agent',
        'sortino':        sortino_ratio(returns_array),
        'sharpe':         sharpe_ratio(returns_array),
        'max_drawdown':   max_drawdown(returns_array),
        'total_return':   total_return(returns_array),
        'n_days':         len(returns_array),
        'n_overrides':    len(override_days),
        'override_dates': override_days,
        'returns':        returns_array,
        'cumulative':     cumulative_returns(returns_array),
        'actions':        all_actions,
        'dates':          all_dates,
        'action_counts':  action_counts,
    }


def print_comparison_table(agent_metrics, baseline_metrics):
    """Print a clean formatted comparison table."""
    print('\n' + '=' * 62)
    print(f'{"PERFORMANCE COMPARISON — 2024 TEST SET":^62}')
    print('=' * 62)
    print(f'{"Metric":<25} {"DQN Agent":>17} {"SPY Buy-Hold":>17}')
    print('-' * 62)
    print(f'{"Sortino Ratio":<25} '
          f'{agent_metrics["sortino"]:>17.4f} '
          f'{baseline_metrics["sortino"]:>17.4f}')
    print(f'{"Sharpe Ratio":<25} '
          f'{agent_metrics["sharpe"]:>17.4f} '
          f'{baseline_metrics["sharpe"]:>17.4f}')
    print(f'{"Total Return (%)":<25} '
          f'{agent_metrics["total_return"]:>17.2f} '
          f'{baseline_metrics["total_return"]:>17.2f}')
    print(f'{"Max Drawdown (%)":<25} '
          f'{agent_metrics["max_drawdown"]*100:>17.2f} '
          f'{baseline_metrics["max_drawdown"]*100:>17.2f}')
    print(f'{"Trading Days":<25} '
          f'{agent_metrics["n_days"]:>17} '
          f'{baseline_metrics["n_days"]:>17}')
    print('=' * 62)


# ── Smoke test ─────────────────────────────────────────────────
if __name__ == '__main__':
    print('Testing evaluation metrics...')

    np.random.seed(42)
    good = np.random.normal(0.001,  0.010, 252)
    bad  = np.random.normal(-0.001, 0.020, 252)

    print(f'Good strategy — Sortino: {sortino_ratio(good):.4f}, '
          f'Sharpe: {sharpe_ratio(good):.4f}, '
          f'MaxDD: {max_drawdown(good)*100:.2f}%')

    print(f'Bad  strategy — Sortino: {sortino_ratio(bad):.4f}, '
          f'Sharpe: {sharpe_ratio(bad):.4f}, '
          f'MaxDD: {max_drawdown(bad)*100:.2f}%')

    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'data', 'processed', 'iv_features.csv'
    )

    if os.path.exists(data_path):
        df       = pd.read_csv(data_path)
        baseline = spy_baseline(df)
        print(f'\nSPY 2020-2024 — Sortino: {baseline["sortino"]:.4f}, '
              f'Return: {baseline["total_return"]:.2f}%')

    print('\nAll tests passed!')