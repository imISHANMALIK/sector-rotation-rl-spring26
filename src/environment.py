import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class SectorRotationEnv(gym.Env):
    """
    OpenAI Gym-compatible environment for sector rotation using IV features.

    State:   [iv_xlk, iv_xlf, iv_xlv, zscore_xlk, zscore_xlf, zscore_xlv,
              realized_vol_xlk, realized_vol_xlf, realized_vol_xlv] — 9-dim vector
    Actions: Discrete(4) → {0: XLK, 1: XLF, 2: XLV, 3: CASH}
    Reward:  Sortino-shaped daily return
    Override: If ALL sector z-scores > 2.5, force CASH (Vasant Dhar safety layer)
    """

    metadata = {"render_modes": []}
    SECTORS   = ["xlk", "xlf", "xlv"]
    STATE_DIM = 9
    ACTION_NAMES = {0: "XLK", 1: "XLF", 2: "XLV", 3: "CASH"}

    def __init__(
        self,
        features_path: str = "data/processed/iv_features.csv",
        mode: str = "train",           # "train" = 2020-2023, "test" = 2024
        override_threshold: float = 2.5,
        sortino_window: int = 20,
    ):
        super().__init__()

        self.override_threshold = override_threshold
        self.sortino_window     = sortino_window

        # ── Load and split data by mode ────────────────────────
        df = pd.read_csv(features_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if mode == "train":
            self._df = df[df["date"] <= "2023-12-31"].reset_index(drop=True)
        elif mode == "test":
            self._df = df[df["date"] >= "2024-01-01"].reset_index(drop=True)
        else:
            self._df = df  # full dataset

        self._validate_columns()

        # ── Gym spaces ─────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.STATE_DIM,), dtype=np.float32
        )
        # FIX: 4 actions (3 sectors + CASH)
        self.action_space = spaces.Discrete(4)

        # ── Episode state ──────────────────────────────────────
        self._t              = 0
        self._return_history = []   # rolling window for Sortino reward
        self._override_count = 0

    # ── Validation ─────────────────────────────────────────────
    def _validate_columns(self):
        required = (
            [f"iv_{s}"  for s in self.SECTORS] +
            [f"ret_{s}" for s in self.SECTORS] +
            ["rf_daily"]
        )
        missing = [c for c in required if c not in self._df.columns]
        if missing:
            raise ValueError(f"iv_features.csv missing columns: {missing}")

    # ── State construction ─────────────────────────────────────
    def _compute_zscores(self, t: int, window: int = 60) -> np.ndarray:
        start = max(0, t - window)
        zscores = []
        for s in self.SECTORS:
            col = self._df[f"iv_{s}"].iloc[start: t + 1]
            mu, sigma = col.mean(), col.std()
            val = float(self._df[f"iv_{s}"].iloc[t])
            zscores.append(0.0 if not (sigma > 0) else float((val - mu) / sigma))
        return np.array(zscores, dtype=np.float32)

    def _compute_realized_vols(self, t: int, window: int = 20) -> np.ndarray:
        start = max(0, t - window)
        rvols = []
        for s in self.SECTORS:
            rets = self._df[f"ret_{s}"].iloc[start: t + 1]
            rvols.append(float(rets.std()) if len(rets) > 1 else 0.0)
        return np.array(rvols, dtype=np.float32)

    def _get_obs(self, t: int) -> np.ndarray:
        ivs     = np.array([self._df[f"iv_{s}"].iloc[t]  for s in self.SECTORS], dtype=np.float32)
        zscores = self._compute_zscores(t)
        rvols   = self._compute_realized_vols(t)
        return np.concatenate([ivs, zscores, rvols])

    # ── Override check ─────────────────────────────────────────
    def _check_override(self, t: int) -> bool:
        """
        Vasant Dhar safety layer: if ALL sector z-scores exceed the threshold
        simultaneously, the market is in extreme fear and the agent moves to CASH.
        This rule is hard-coded OUTSIDE the gradient pathway — the agent cannot
        learn to ignore it during training.
        """
        zscores = self._compute_zscores(t)
        return bool(np.all(zscores > self.override_threshold))

    # ── Sortino reward shaping ─────────────────────────────────
    def _sortino_reward(self, daily_return: float) -> float:
        """
        Shape the reward using a rolling Sortino ratio.
        Only penalizes DOWNSIDE returns (negative), not all volatility.
        This encourages the agent to avoid losses specifically.
        """
        self._return_history.append(daily_return)
        if len(self._return_history) > self.sortino_window:
            self._return_history = self._return_history[-self.sortino_window:]

        if len(self._return_history) < 5:
            return daily_return  # not enough history yet

        arr      = np.array(self._return_history)
        downside = arr[arr < 0.0]

        if len(downside) == 0:
            return daily_return  # no losses → no penalty

        downside_dev = np.std(downside)
        if downside_dev < 1e-8:
            return daily_return

        sortino = np.mean(arr) / downside_dev
        # Base reward + small Sortino bonus/penalty (scale=0.1 keeps it from dominating)
        return float(daily_return + 0.1 * np.clip(sortino, -2.0, 2.0))

    # ── Gym interface ──────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t              = 0
        self._return_history = []
        self._override_count = 0
        return self._get_obs(self._t), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # FIX 1 — Check override BEFORE executing agent's action
        override_triggered = self._check_override(self._t)
        if override_triggered:
            executed_action = 3  # Force CASH
            self._override_count += 1
        else:
            executed_action = action

        # FIX 2 — Get return for executed action
        if executed_action == 3:  # CASH
            daily_return = float(self._df["rf_daily"].iloc[self._t])
        else:
            sector       = self.SECTORS[executed_action]
            daily_return = float(self._df[f"ret_{sector}"].iloc[self._t])

        if np.isnan(daily_return):
            daily_return = 0.0

        # FIX 3 — Sortino-shaped reward instead of raw return
        reward = self._sortino_reward(daily_return)

        # Advance time
        self._t += 1
        done     = self._t >= len(self._df) - 1
        next_obs = self._get_obs(self._t) if not done else np.zeros(self.STATE_DIM, dtype=np.float32)

        info = {
            "date":               str(self._df["date"].iloc[self._t - 1])[:10],
            "action_requested":   action,
            "action_executed":    executed_action,
            "action_name":        self.ACTION_NAMES[executed_action],
            "override_triggered": override_triggered,
            "daily_return":       daily_return,
            "reward":             reward,
            "override_count":     self._override_count,
        }
        return next_obs, reward, done, False, info

    def render(self):
        pass

    def get_episode_stats(self):
        """Summary stats for logging after each episode."""
        return {"override_count": self._override_count}
