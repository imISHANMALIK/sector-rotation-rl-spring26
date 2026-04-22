import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class SectorRotationEnv(gym.Env):
    """
    OpenAI Gym-compatible environment for sector rotation using IV features.

    State: [iv_xlk, iv_xlf, iv_xlv, zscore_xlk, zscore_xlf, zscore_xlv,
            realized_vol_xlk, realized_vol_xlf, realized_vol_xlv] — 9-dim vector
    Action: Discrete(3) → {0: XLK, 1: XLF, 2: XLV}
    """

    metadata = {"render_modes": []}

    SECTORS = ["xlk", "xlf", "xlv"]
    STATE_DIM = 9

    def __init__(self, features_path: str = "data/processed/iv_features.csv"):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self._df = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        self._validate_columns()

        self._t = 0
        self._current_action = 0

    def _validate_columns(self):
        required = [f"iv_{s}" for s in self.SECTORS] + \
                   [f"ret_{s}" for s in self.SECTORS]
        missing = [c for c in required if c not in self._df.columns]
        if missing:
            raise ValueError(f"iv_features.csv missing columns: {missing}")

    def _compute_zscores(self, t: int, window: int = 60) -> np.ndarray:
        start = max(0, t - window)
        zscores = []
        for s in self.SECTORS:
            col = self._df[f"iv_{s}"].iloc[start : t + 1]
            mu, sigma = col.mean(), col.std()
            val = self._df[f"iv_{s}"].iloc[t]
            zscores.append(0.0 if sigma == 0 else float((val - mu) / sigma))
        return np.array(zscores, dtype=np.float32)

    def _compute_realized_vols(self, t: int, window: int = 20) -> np.ndarray:
        start = max(0, t - window)
        rvols = []
        for s in self.SECTORS:
            rets = self._df[f"ret_{s}"].iloc[start : t + 1]
            rvols.append(float(rets.std()) if len(rets) > 1 else 0.0)
        return np.array(rvols, dtype=np.float32)

    def _get_obs(self, t: int) -> np.ndarray:
        ivs = np.array(
            [self._df[f"iv_{s}"].iloc[t] for s in self.SECTORS], dtype=np.float32
        )
        zscores = self._compute_zscores(t)
        rvols = self._compute_realized_vols(t)
        return np.concatenate([ivs, zscores, rvols])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        obs = self._get_obs(self._t)
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        sector = self.SECTORS[action]
        ret = float(self._df[f"ret_{sector}"].iloc[self._t])

        self._t += 1
        done = self._t >= len(self._df) - 1

        next_obs = self._get_obs(self._t)
        info = {
            "date": self._df["date"].iloc[self._t],
            "sector": sector,
            "ret": ret,
            "t": self._t,
        }

        return next_obs, ret, done, False, info

    def render(self):
        pass
