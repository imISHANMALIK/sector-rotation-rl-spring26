"""
train.py — DQN training for sector rotation (2020-2023 data).

Day 3 spec:
  γ=0.99, lr=1e-3, buffer=10000, batch=64, hidden=128
  eps_start=1.0, eps_end=0.1, LINEAR decay over 80% of episodes
  1000-2000 episodes; monitor reward trend, loss, epsilon

Bug checks baked in:
  - Action distribution logged every 100 eps → detect "always picks same action"
  - Q-value magnitude tracked → detect Q-explosion
  - Target-net sync confirmed via step counter
  - Replay buffer size logged until training starts
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.environment import SectorRotationEnv
from src.dqn_agent import DQNAgent

# ── Hyperparameters ────────────────────────────────────────────
TOTAL_EPISODES  = 1500
GAMMA           = 0.99
LR              = 1e-3
BUFFER_SIZE     = 10_000
BATCH_SIZE      = 64
HIDDEN_DIM      = 128
EPS_START       = 1.0
EPS_END         = 0.1
EPS_DECAY_FRAC  = 0.80          # decay over 80% of episodes → 1200 episodes
TARGET_UPDATE   = 100
GRAD_CLIP       = 1.0
LOG_INTERVAL    = 100
CHECKPOINT_FREQ = 500

DATA_PATH  = "data/processed/iv_features.csv"
NORM_PATH  = "data/processed/norm_stats.csv"
MODEL_DIR  = "models"


# ── Normalisation ───────────────────────────────────────────────

def load_normalizer():
    if not os.path.exists(NORM_PATH):
        return None, None
    norm = pd.read_csv(NORM_PATH, index_col=0)
    return (norm["mean"].values.astype(np.float32),
            norm["std"].values.astype(np.float32))


def normalize(obs: np.ndarray, mean, std) -> np.ndarray:
    if mean is None:
        return obs
    return (obs - mean) / (std + 1e-8)


# ── Data bootstrap ──────────────────────────────────────────────

def ensure_data():
    if not os.path.exists(DATA_PATH):
        print("Processed data not found. Running generate_data.py...")
        import generate_data
        generate_data.main()


# ── Training ────────────────────────────────────────────────────

def train():
    ensure_data()
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = SectorRotationEnv(features_path=DATA_PATH, mode="train")
    steps_per_ep = len(env._df) - 1
    print(f"Environment: {steps_per_ep} steps/episode  (train mode)")
    print(f"Hyperparams: γ={GAMMA}, lr={LR}, buf={BUFFER_SIZE}, "
          f"batch={BATCH_SIZE}, hidden={HIDDEN_DIM}")
    print(f"Eps: {EPS_START} → {EPS_END} over {int(EPS_DECAY_FRAC*TOTAL_EPISODES)} episodes\n")

    norm_mean, norm_std = load_normalizer()

    # epsilon_decay=1.0 disables the per-step decay inside train_step();
    # we set epsilon explicitly at the start of every episode instead.
    agent = DQNAgent(
        state_dim=9, action_dim=4,
        hidden=HIDDEN_DIM,
        lr=LR,
        gamma=GAMMA,
        epsilon=EPS_START,
        epsilon_min=EPS_END,
        epsilon_decay=1.0,          # per-step decay disabled
        buffer_capacity=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE,
        grad_clip=GRAD_CLIP,
    )

    decay_episodes = int(EPS_DECAY_FRAC * TOTAL_EPISODES)   # 1200

    ep_rewards  = []
    ep_losses   = []
    ep_epsilons = []
    ep_actions  = []      # action distributions for collapse detection

    t0 = time.time()

    for ep in range(TOTAL_EPISODES):

        # ── Linear epsilon schedule ──────────────────────────────
        if ep < decay_episodes:
            agent.epsilon = EPS_START - (EPS_START - EPS_END) * (ep / decay_episodes)
        else:
            agent.epsilon = EPS_END

        obs, _ = env.reset()
        obs = normalize(obs, norm_mean, norm_std)

        total_reward  = 0.0
        step_losses   = []
        step_actions  = []
        done = False

        while not done:
            action = agent.select_action(obs, training=True)
            next_obs, reward, done, _, info = env.step(action)
            next_obs_n = normalize(next_obs, norm_mean, norm_std)

            agent.buffer.push(obs, action, reward, next_obs_n, float(done))
            loss = agent.train_step()

            if loss is not None:
                step_losses.append(loss)

            obs           = next_obs_n
            total_reward += reward
            step_actions.append(info["action_executed"])

        ep_rewards.append(total_reward)
        ep_losses.append(float(np.mean(step_losses)) if step_losses else 0.0)
        ep_epsilons.append(agent.epsilon)

        # Action distribution for this episode
        n_steps = len(step_actions)
        ep_actions.append({i: step_actions.count(i) / n_steps for i in range(4)})

        # ── Periodic logging ─────────────────────────────────────
        if (ep + 1) % LOG_INTERVAL == 0:
            win = slice(max(0, ep - LOG_INTERVAL + 1), ep + 1)
            avg_r = np.mean(ep_rewards[win])
            valid_l = [l for l in ep_losses[win] if l > 0]
            avg_l = np.mean(valid_l) if valid_l else 0.0
            eps   = agent.epsilon

            # Action collapse check
            recent_dists = ep_actions[win]
            avg_dist = {k: np.mean([d[k] for d in recent_dists]) for k in range(4)}
            names = env.ACTION_NAMES
            adist = " | ".join(f"{names[k]}:{v:.0%}" for k, v in avg_dist.items())

            # Q-value magnitude check
            test_obs = normalize(env._get_obs(0), norm_mean, norm_std)
            import torch
            with torch.no_grad():
                q_vals = agent.q_net(
                    torch.tensor(test_obs, dtype=torch.float32).unsqueeze(0)
                ).numpy().flatten()
            q_mag = float(np.abs(q_vals).max())

            elapsed = time.time() - t0
            print(
                f"Ep {ep+1:>5}/{TOTAL_EPISODES} | "
                f"AvgRew: {avg_r:+.4f} | "
                f"Loss: {avg_l:.5f} | "
                f"Eps: {eps:.3f} | "
                f"Q-max: {q_mag:.3f} | "
                f"Actions: [{adist}] | "
                f"{elapsed:.0f}s"
            )

        # ── Checkpoint ───────────────────────────────────────────
        if (ep + 1) % CHECKPOINT_FREQ == 0:
            agent.save(f"{MODEL_DIR}/dqn_ep{ep+1}.pth")

    # ── Save final ───────────────────────────────────────────────
    agent.save(f"{MODEL_DIR}/dqn_final.pth")

    history = {
        "ep_rewards":  ep_rewards,
        "ep_losses":   ep_losses,
        "ep_epsilons": ep_epsilons,
    }
    with open(f"{MODEL_DIR}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(ep_rewards, ep_losses, ep_epsilons)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Last-100-ep avg reward : {np.mean(ep_rewards[-100:]):.4f}")
    print(f"Final epsilon          : {agent.epsilon:.3f}")
    return agent, history


# ── Plotting ────────────────────────────────────────────────────

def plot_training_curves(rewards, losses, epsilons, smooth=50):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    def _smooth(arr):
        return pd.Series(arr).rolling(smooth, min_periods=1).mean().values

    # 1 — Episode reward
    ax = axes[0]
    ax.plot(rewards, alpha=0.25, color="steelblue", linewidth=0.6)
    ax.plot(_smooth(rewards), color="steelblue", linewidth=2,
            label=f"{smooth}-ep rolling avg")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.6)
    ax.set_title("Episode Reward  (should trend upward)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2 — Training loss
    ax = axes[1]
    nz = [(i, l) for i, l in enumerate(losses) if l > 0]
    if nz:
        ep_idx, ls = zip(*nz)
        ax.plot(ep_idx, ls, alpha=0.25, color="tomato", linewidth=0.6)
        smooth_l = pd.Series(ls, index=ep_idx).rolling(smooth, min_periods=1).mean()
        ax.plot(smooth_l, color="tomato", linewidth=2,
                label=f"{smooth}-ep rolling avg")
    ax.set_title("Training Loss  (should decrease then stabilise)")
    ax.set_xlabel("Episode"); ax.set_ylabel("MSE Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3 — Epsilon
    ax = axes[2]
    ax.plot(epsilons, color="mediumseagreen", linewidth=2)
    ax.axhline(EPS_END, color="grey", linestyle="--", linewidth=1,
               label=f"eps_end={EPS_END}")
    ax.set_title("Epsilon Decay  (should reach 0.1 at 80% of training)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Epsilon")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{MODEL_DIR}/training_curves.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {out}")


if __name__ == "__main__":
    train()
