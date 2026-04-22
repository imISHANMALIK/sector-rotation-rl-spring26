import numpy as np


def sortino_ratio(returns: np.ndarray, window: int = 20, target: float = 0.0) -> float:
    """Rolling-window Sortino ratio; penalizes only downside deviations below target."""
    if len(returns) < 2:
        return 0.0

    recent = returns[-window:] if len(returns) >= window else returns
    excess = recent - target
    mean_excess = float(np.mean(excess))
    downside = excess[excess < 0]

    if len(downside) == 0:
        return mean_excess / 1e-8 if mean_excess > 0 else 0.0

    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if downside_std == 0:
        return 0.0

    return mean_excess / downside_std


def compute_reward(
    returns_history: np.ndarray,
    window: int = 20,
    target: float = 0.0,
) -> float:
    """Reward shaping using rolling Sortino ratio."""
    return sortino_ratio(returns_history, window=window, target=target)
