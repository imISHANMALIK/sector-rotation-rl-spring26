"""FastAPI backend — real-time inference stream for Sector Rotation RL dashboard."""

from __future__ import annotations

import asyncio
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dqn_agent import DQNAgent          # noqa: E402
from environment import SectorRotationEnv  # noqa: E402

DATA_PATH    = os.path.join(ROOT, "data/processed/iv_features.csv")
MODEL_PATH   = os.path.join(ROOT, "checkpoints/best_model.pt")
TRAIN_HIST   = os.path.join(ROOT, "models/training_history.json")
EVAL_RESULTS = os.path.join(ROOT, "models/eval_results.json")

app = FastAPI(title="Sector Rotation RL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_agent() -> DQNAgent:
    agent = DQNAgent(state_dim=9, action_dim=4, hidden=128)
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0
    agent.q_net.eval()
    return agent


@app.get("/api/data")
async def get_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.to_dict("records")


@app.get("/api/training-history")
async def get_training_history():
    with open(TRAIN_HIST) as f:
        return json.load(f)


@app.get("/api/eval-results")
async def get_eval_results():
    with open(EVAL_RESULTS) as f:
        return json.load(f)


@app.get("/api/inference/stream")
async def stream_inference(delay: float = Query(default=0.06, ge=0.0, le=3.0)):
    """SSE endpoint: streams day-by-day greedy inference on the 2024 test set."""

    async def generate():
        agent = _load_agent()

        env = SectorRotationEnv(features_path=DATA_PATH, mode="test")
        df_full = pd.read_csv(DATA_PATH, parse_dates=["date"])
        spy_rets = df_full[df_full["date"] >= "2024-01-01"]["ret_spy"].values.tolist()

        state, _ = env.reset()
        portfolio = 1.0
        spy_port  = 1.0
        port_hist: list[float] = []
        returns:   list[float] = []
        action_counts: dict[str, int] = {"XLK": 0, "XLF": 0, "XLV": 0, "CASH": 0}
        step = 0

        while True:
            with torch.no_grad():
                t  = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                qv = agent.q_net(t).squeeze().numpy().tolist()

            action = int(np.argmax(qv))
            next_state, _reward, done, _, info = env.step(action)

            daily_ret = info["daily_return"]
            act_name  = info["action_name"]
            override  = info["override_triggered"]
            date_str  = info["date"]

            portfolio *= np.exp(daily_ret)
            spy_ret    = spy_rets[step] if step < len(spy_rets) else 0.0
            spy_port  *= np.exp(spy_ret)

            port_hist.append(portfolio)
            returns.append(daily_ret)
            action_counts[act_name] = action_counts.get(act_name, 0) + 1

            if len(returns) >= 5:
                arr   = np.array(returns)
                down  = arr[arr < 0]
                sortino = float((arr.mean() / down.std()) * np.sqrt(252)) if (len(down) > 1 and down.std() > 0) else 0.0
            else:
                sortino = 0.0

            peak   = max(port_hist)
            max_dd = float((portfolio - peak) / peak * 100)

            payload = {
                "step":         step,
                "date":         date_str,
                "action":       info["action_executed"],
                "action_name":  act_name,
                "override":     bool(override),
                "daily_return": round(float(daily_ret * 100), 4),
                "portfolio":    round(float(portfolio), 6),
                "spy":          round(float(spy_port), 6),
                "total_return": round(float((portfolio - 1) * 100), 3),
                "spy_total":    round(float((spy_port - 1) * 100), 3),
                "sortino":      round(sortino, 3),
                "max_drawdown": round(max_dd, 3),
                "q_values":     [round(float(q), 4) for q in qv],
                "state": {
                    "iv_xlk": round(float(state[0]), 4),
                    "iv_xlf": round(float(state[1]), 4),
                    "iv_xlv": round(float(state[2]), 4),
                    "z_xlk":  round(float(state[3]), 4),
                    "z_xlf":  round(float(state[4]), 4),
                    "z_xlv":  round(float(state[5]), 4),
                    "rv_xlk": round(float(state[6]), 4),
                    "rv_xlf": round(float(state[7]), 4),
                    "rv_xlv": round(float(state[8]), 4),
                },
                "action_counts": action_counts.copy(),
                "done":         bool(done),
            }

            yield f"data: {json.dumps(payload)}\n\n"

            if done:
                break

            state = next_state
            step += 1

            if delay > 0:
                await asyncio.sleep(delay)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
