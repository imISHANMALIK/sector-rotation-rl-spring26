"""
evaluate.py — Evaluate trained DQN on 2024 holdout.

Reports: total return, annualised return, Sortino ratio, vs SPY benchmark.
Saves: models/equity_curve_2024.png
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.environment import SectorRotationEnv
from src.dqn_agent import DQNAgent

DATA_PATH  = "data/processed/iv_features.csv"
NORM_PATH  = "data/processed/norm_stats.csv"
MODEL_PATH = "models/dqn_final.pth"
MODEL_DIR  = "models"


def load_normalizer():
    if not os.path.exists(NORM_PATH):
        return None, None
    norm = pd.read_csv(NORM_PATH, index_col=0)
    return (norm["mean"].values.astype(np.float32),
            norm["std"].values.astype(np.float32))


def normalize(obs, mean, std):
    if mean is None:
        return obs
    return (obs - mean) / (std + 1e-8)


def sortino(rets: np.ndarray, ann=252) -> float:
    downside = rets[rets < 0]
    if len(downside) == 0:
        return np.inf
    return float(np.mean(rets) / (np.std(downside) + 1e-8) * np.sqrt(ann))


def evaluate(model_path: str = MODEL_PATH):
    env = SectorRotationEnv(features_path=DATA_PATH, mode="test")
    n_test = len(env._df)
    print(f"Test set: {n_test} trading days (2024 holdout)")

    agent = DQNAgent(state_dim=9, action_dim=4)
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"WARNING: {model_path} not found — using untrained (random) policy")

    norm_mean, norm_std = load_normalizer()

    obs, _ = env.reset()
    obs    = normalize(obs, norm_mean, norm_std)

    log = []
    done = False
    while not done:
        action = agent.select_action(obs, training=False)
        next_obs, _, done, _, info = env.step(action)
        log.append(info)
        obs = normalize(next_obs, norm_mean, norm_std)

    # ── Metrics ─────────────────────────────────────────────────
    rets    = np.array([e["daily_return"] for e in log])
    cumlog  = np.cumsum(rets)
    total_r = float(np.expm1(cumlog[-1]))            # exp(sum(log_rets)) - 1
    n_days  = len(rets)
    ann_r   = float((1 + total_r) ** (252 / n_days) - 1)
    sort    = sortino(rets)

    df_test = env._df
    spy_rets   = None
    spy_total  = None
    if "ret_spy" in df_test.columns:
        spy_raw  = df_test["ret_spy"].dropna().values[:n_days]
        spy_rets = spy_raw
        spy_total = float(np.expm1(np.sum(spy_raw)))

    # Action distribution
    action_counts: dict = {}
    for e in log:
        a = e["action_name"]
        action_counts[a] = action_counts.get(a, 0) + 1
    n_override = sum(1 for e in log if e["override_triggered"])

    # ── Print results ────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS — 2024 Holdout")
    print("=" * 50)
    print(f"Total Return:      {total_r:+.2%}")
    print(f"Annualised Return: {ann_r:+.2%}")
    print(f"Sortino Ratio:     {sort:.3f}")
    if spy_total is not None:
        print(f"SPY Benchmark:     {spy_total:+.2%}")
        print(f"Alpha vs SPY:      {total_r - spy_total:+.2%}")
    print(f"\nAction Distribution ({n_days} days):")
    for a, c in sorted(action_counts.items()):
        print(f"  {a:<6}: {c:>4} days  ({100*c/n_days:.1f}%)")
    print(f"\nOverride triggers: {n_override}")

    # ── Equity curve plot ────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    _plot_equity(cumlog, spy_rets, log)

    results = {
        "total_return": total_r,
        "ann_return":   ann_r,
        "sortino":      sort,
        "spy_return":   spy_total,
        "action_distribution": action_counts,
        "n_override":   n_override,
    }
    with open(f"{MODEL_DIR}/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {MODEL_DIR}/eval_results.json")
    return results


def _plot_equity(agent_cumlog, spy_rets, log):
    agent_cumret = np.expm1(agent_cumlog)

    plt.figure(figsize=(12, 6))
    plt.plot(agent_cumret, color="steelblue", linewidth=2, label="DQN Agent")

    if spy_rets is not None:
        n = len(agent_cumret)
        spy_cum = np.expm1(np.cumsum(spy_rets[:n]))
        plt.plot(spy_cum, color="darkorange", linewidth=2,
                 linestyle="--", label="SPY Benchmark")

    # Mark override events
    override_days = [i for i, e in enumerate(log) if e["override_triggered"]]
    if override_days:
        plt.scatter(override_days,
                    agent_cumret[override_days],
                    color="red", s=20, zorder=5,
                    label=f"Override → CASH ({len(override_days)}x)")

    plt.title("Equity Curve — 2024 Holdout (DQN vs SPY)")
    plt.xlabel("Trading Day")
    plt.ylabel("Cumulative Return")
    plt.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = f"{MODEL_DIR}/equity_curve_2024.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Equity curve saved → {out}")


if __name__ == "__main__":
    evaluate()
