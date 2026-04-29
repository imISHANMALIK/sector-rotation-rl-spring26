# Risk-Aware Reinforcement Learning for Sector Rotation

**Team Lennox | DS-GA 3001 Reinforcement Learning | NYU Center for Data Science | Spring 2026**

Rishit Maheshwari · Ishan Malik · Maanas Lalwani
`{rm7336, im2854, ml10092}@nyu.edu`

---

## Results Summary

| Metric | DQN Agent | SPY Buy-Hold | Improvement |
|--------|-----------|-------------|-------------|
| **Sortino Ratio** | **5.25** | 2.34 | +2.91 (+124%) |
| **Sharpe Ratio** | **3.31** | 1.80 | +1.51 (+84%) |
| **Total Return** | **+51.6%** | +25.3% | +26.3% |
| **Max Drawdown** | **-5.3%** | -8.4% | 37% smaller |
| **Override Triggers** | 2 | N/A | N/A |

**Agent beats SPY in all 4 stress-tested market regimes:**

| Period | Agent | SPY |
|--------|-------|-----|
| COVID Crash (Feb-Apr 2020) | **+27.4%** | -9.2% |
| 2022 Bear Market (Jan-Oct) | **-4.6%** | -17.7% |
| Aug 2024 Flash Crash | **+13.0%** | +0.4% |
| 2024 Full Year | **+51.6%** | +25.3% |

---

## Project Overview

A risk-aware DQN agent that rotates capital among three U.S. sector ETFs —
XLK (Technology), XLF (Financials), XLV (Healthcare) — using Implied Volatility
as a forward-looking fear signal.

### Three Core Design Choices

**1. IV as State Signal**
Options market fear levels (forward-looking) rather than backward-looking
price indicators. IV rises BEFORE crashes — giving the agent early warning.

**2. Sortino Reward Shaping**
Only penalizes downside volatility — teaches the agent to specifically
avoid losses, not just reduce all volatility.

**3. Vasant Dhar Safety Override**
Hard-coded rule outside the gradient pathway: when ALL sector IV z-scores
exceed 2.5 simultaneously → force CASH. The agent knows when it doesn't know.

---

## Quick Start — Full Dashboard Demo

### Prerequisites

```bash
# Python 3.10+
python --version

# Node.js (for the React frontend)
node --version   # If missing: brew install node
```

### Run the Demo

```bash
# Clone and setup
git clone https://github.com/imISHANMALIK/sector-rotation-rl-spring26.git
cd sector-rotation-rl-spring26

# Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Launch everything (backend + frontend)
./demo/start.sh
```

Open **http://localhost:3000** in your browser.

The dashboard starts two servers simultaneously:
- **FastAPI backend** at `http://localhost:8000` — serves the trained model and streams live inference
- **Next.js frontend** at `http://localhost:3000` — the interactive web dashboard

---

## Dashboard Features

### Tab 1: Simulation

- Step through 2024 data day-by-day
- Real-time portfolio value + sector allocation
- Interactive charts + trading logs + clear directives on where to invest
- Manual Hyperparameter Selection + Option to Autonomize Agent

### Tab 2: Market

- Dataset overview (Train/Test splits, coverage, feature/action space)
- Implied Volatility (IV) and 60-day rolling IV Z-scores
- Buy-and-hold cumulative returns vs. SPY benchmark
- Realized vs. Implied Volatility (Vol Risk Premium)
- Daily return distributions across sectors

### Tab 3: Training

- Test set performance summary (Test Return, Sortino Ratio, vs. SPY)
- Final action distribution breakdown (Cash vs. Sectors)
- Live training curves (Reward per episode, Bellman loss, Epsilon decay)
- Detailed view of Agent Architecture & Hyperparameters

### Tab 4: Guide

- IV z-scores for all 3 sectors (2020–24). Project overview and team credits.
- Plain-English explanation of the Reinforcement Learning agent's mechanics
- Educational glossary defining key financial metrics (Total Return, Sortino Ratio, Max Drawdown, Alpha, etc.)

---

## RUN the Full Demo with a Single Command

```bash
bash demo/start.sh
```

---

## Alternative: Streamlit Demo

If you prefer the simpler Streamlit interface:

```bash
streamlit run demo/app.py
# Open http://localhost:8501
```

---

## Full Pipeline (Training from Scratch)

### 1. Generate Data

```bash
python data/scripts/download_wrds.py
python data/scripts/compute_iv.py
```

### 2. Train Agent

```bash
# Full training (2000 episodes, ~22 minutes)
python src/train.py --episodes 2000 --run-name "baseline"

# Watch training in MLflow
mlflow ui --port 5000
# Open http://localhost:5000
```

### 3. Backtest

```bash
python -m src.backtest --checkpoint checkpoints/best_model.pt
```

### 4. Stress Tests

```bash
python -m src.stress_test
```

### 5. Ablation Studies

```bash
python -m src.ablation
```

### 6. Hyperparameter Optimization

```bash
python -m src.optuna_tune --trials 10
```

---

## Repository Structure

```
sector-rotation-rl-spring26/
├── configs/
│   ├── hyperparams.yaml          # Default hyperparameters
│   └── best_hyperparams.yaml     # Optuna-optimized hyperparameters
├── data/
│   ├── processed/
│   │   ├── iv_features.csv       # Master feature dataset (1,238 days, 15 features)
│   │   ├── train.csv             # Training set (2020-2023, 987 days)
│   │   └── test.csv              # Test set (2024, 251 days)
│   ├── raw/                      # ETF prices, returns, proxy IV, Treasury yields
│   └── scripts/
│       ├── download_wrds.py      # Data download pipeline
│       └── compute_iv.py         # Feature engineering (z-scores, realized vol)
├── checkpoints/
│   ├── best_model.pt             # Best trained agent (by validation Sortino)
│   └── final_model.pt            # Final episode checkpoint
├── demo/
│   ├── start.sh                  # ONE COMMAND to launch full dashboard
│   ├── app.py                    # Alternative: Streamlit demo
│   ├── backend/
│   │   ├── main.py               # FastAPI backend — SSE inference stream
│   │   └── requirements.txt      # fastapi, uvicorn
│   └── web/                      # Next.js React frontend
│       ├── app/                  # Next.js pages
│       └── components/
│           ├── Dashboard.tsx     # Main container, state management
│           ├── EquityCurve.tsx   # Cumulative returns chart
│           ├── DrawdownChart.tsx # Drawdown visualization
│           ├── SectorPie.tsx     # Sector allocation pie
│           ├── IVZScoreChart.tsx # IV z-scores + override timeline
│           ├── MarketAnalysis.tsx# Stress test results
│           ├── TrainingHistory.tsx# Training curves
│           └── Guide.tsx         # Built-in explanation panel
├── models/
│   ├── training_history.json     # Training curves data
│   ├── eval_results.json         # Evaluation metrics
│   └── equity_curve_2024.png     # Training equity curve
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_results.ipynb          # Deep results analysis
│   └── backtest_results.csv      # Backtesting results
├── report/
│   ├── report.md                 # Full academic report
│   └── report.pdf                # PDF version
├── src/
│   ├── environment.py            # OpenAI Gym-compatible trading environment
│   ├── dqn_agent.py              # DQN: QNetwork + ReplayBuffer + DQNAgent
│   ├── rewards.py                # Sortino ratio reward shaping
│   ├── evaluate.py               # Performance metrics (Sortino, Sharpe, etc.)
│   ├── train.py                  # Main training loop + MLflow tracking
│   ├── backtest.py               # Backtesting engine
│   ├── risk_override.py          # Vasant Dhar safety layer (standalone)
│   ├── stress_test.py            # Crisis period analysis
│   ├── ablation.py               # Ablation studies
│   └── optuna_tune.py            # Bayesian hyperparameter optimization
├── mlruns/                       # MLflow experiment logs
├── requirements.txt
└── README.md
```

---

## Technical Details

### State Space (9-dimensional)

| Feature | Description | Why |
|---------|-------------|-----|
| iv_xlk, iv_xlf, iv_xlv | Implied Volatility per sector | Forward-looking fear signal |
| zscore_xlk, zscore_xlf, zscore_xlv | 60-day rolling z-scores | For override detection |
| realvol_xlk, realvol_xlf, realvol_xlv | 20-day realized volatility | Backward-looking context |

### Action Space

| Action | Description | Reward |
|--------|-------------|--------|
| 0: XLK | Technology ETF | XLK daily log return |
| 1: XLF | Financials ETF | XLF daily log return |
| 2: XLV | Healthcare ETF | XLV daily log return |
| 3: CASH | Treasury bills | Daily risk-free rate |

### DQN Architecture

```
Input(9) → Linear(128) → ReLU → Linear(128) → ReLU → Linear(4)
```

### Key Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning Rate | 0.001 | Balances speed and stability |
| Discount Factor (γ) | 0.99 | Values near-term returns appropriately |
| Hidden Layer | 128 neurons | Sufficient capacity for 9-dim state |
| Replay Buffer | 10,000 | ~10 episodes of memory |
| Batch Size | 64 | Standard for stable gradient estimates |
| Target Update | Every 100 steps | Stable Bellman targets |
| Episodes | 2,000 | Full convergence |
| Override Threshold | 2.5 std devs | Top 0.6% of fear distribution |

### Data

| Split | Period | Days |
|-------|--------|------|
| Training | 2020-2023 | 987 |
| Testing | 2024 | 251 |
| **Total** | **2020-2024** | **1,238** |

---

## Ablation Study

Proves every component contributes:

| Variant | Sortino | Return | Max DD | Overrides |
|---------|---------|--------|--------|-----------|
| **Full Model** | **3.92** | **+31.0%** | **-4.8%** | 2 |
| No Override | 1.23 | +18.5% | -12.1% | 0 |
| Raw Return Reward | 1.77 | +25.3% | -10.3% | 2 |
| Random Policy | 1.07 | +13.9% | -13.6% | 0 |
| SPY Baseline | 2.34 | +25.3% | -8.4% | 0 |

*Ablation uses 500 episodes for speed. Relative comparisons are valid.*

---

## How the Dashboard Works

```
Browser (Next.js) ←──SSE stream──→ FastAPI Backend ←── trained model
     ↑                                    ↑
Real-time charts                   Loads best_model.pt
Portfolio updates                  Steps through 2024 data
Q-value display                    Pushes each day as JSON
```

**Server-Sent Events (SSE):** The backend keeps a persistent HTTP connection
open and pushes each trading day's data to the browser as it's computed.
At 60ms per day, the full 2024 simulation takes ~15 seconds.

---

## References

1. V. Dhar, "When to Trust Robots with Decisions," HBR, 2016
2. Z. Jiang et al., "Deep RL for Portfolio Management," arXiv:1706.10059, 2017
3. V. Mnih et al., "Human-level Control through Deep RL," Nature, 2015
4. F. Black & M. Scholes, "The Pricing of Options," JPE, 1973
5. F. Sortino & R. van der Meer, "Downside Risk," JPM, 1991
6. J. Schulman et al., "Proximal Policy Optimization," arXiv:1707.06347, 2017
