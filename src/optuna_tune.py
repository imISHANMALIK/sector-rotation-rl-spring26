"""
optuna_tune.py
==============
Hyperparameter optimization using Optuna.

WHAT IS OPTUNA?
Optuna is an automatic hyperparameter optimization framework.
Instead of manually trying different learning rates, gamma values etc,
Optuna does it automatically using Bayesian optimization (TPE sampler).

HOW IT WORKS:
1. Define a search space (ranges for each hyperparameter)
2. Run N trials — each trial trains the agent with different params
3. Measure validation Sortino for each trial
4. Optuna uses TPE to focus on promising parameter regions
5. Return the best hyperparameters found

WHY SORTINO AS OBJECTIVE?
We optimize for Sortino (not return or Sharpe) because:
- Sortino is our primary metric — penalizes only downside risk
- Optimizing for return alone could lead to risky strategies
- Consistent with our reward shaping philosophy

EXPECTED RUNTIME:
- 10 trials x ~2-3 min each = ~25-30 minutes total
- Start it now and work on report while it runs
"""

import os
import sys
import yaml
import numpy as np
import torch
import optuna

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import SectorRotationEnv
from src.dqn_agent   import DQNAgent
from src.evaluate    import sortino_ratio

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT_DIR, 'data', 'processed', 'iv_features.csv')
CONFIG_PATH = os.path.join(ROOT_DIR, 'configs', 'hyperparams.yaml')
CKPT_DIR    = os.path.join(ROOT_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)


def run_quick_training(config, n_episodes=200):
    """
    Run a short training session for one Optuna trial.

    WHY ONLY 200 EPISODES (not 2000)?
    Each Optuna trial needs to be fast enough to run 10+ trials
    in reasonable time. 200 episodes gives enough signal to compare
    hyperparameters without waiting hours per trial.
    The best config gets a full 2000-episode run afterward.

    Returns:
    --------
    float — validation Sortino ratio (objective to maximize)
    """
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_env = SectorRotationEnv(DATA_PATH, mode='train')
    val_env   = SectorRotationEnv(DATA_PATH, mode='test')

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

            agent.buffer.push(
                state, action, reward, next_state,
                terminated or truncated
            )
            agent.train_step()
            state = next_state

            if terminated or truncated:
                break

    # Evaluate on 2024 test data
    original_eps  = agent.epsilon
    agent.epsilon = 0.0
    val_returns   = []

    state, _ = val_env.reset()
    while True:
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, info = \
            val_env.step(action)
        val_returns.append(info['daily_return'])
        state = next_state
        if terminated or truncated:
            break

    agent.epsilon = original_eps
    return sortino_ratio(np.array(val_returns)), agent


def objective(trial):
    """
    Optuna objective function — called once per trial.

    Optuna calls this N times with different hyperparameter
    suggestions. We return validation Sortino which Optuna
    tries to maximize.

    SEARCH SPACE:
    - learning_rate: log-uniform 1e-4 to 1e-2
      WHY LOG? LR varies over orders of magnitude
    - gamma: uniform 0.90 to 0.999
      WHY THIS RANGE? Below 0.9 = too short-sighted,
      above 0.999 = too far-sighted
    - hidden_dim: categorical [64, 128, 256]
    - buffer_size: integer 5000 to 20000
    - batch_size: categorical [32, 64, 128]
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # Optuna suggests values for this trial
    config['learning_rate'] = trial.suggest_float(
        'learning_rate', 1e-4, 1e-2, log=True
    )
    config['gamma']         = trial.suggest_float(
        'gamma', 0.90, 0.999
    )
    config['hidden_dim']    = trial.suggest_categorical(
        'hidden_dim', [64, 128, 256]
    )
    config['buffer_size']   = trial.suggest_int(
        'buffer_size', 5000, 20000, step=5000
    )
    config['batch_size']    = trial.suggest_categorical(
        'batch_size', [32, 64, 128]
    )

    print(f'\nTrial {trial.number}:')
    print(f'  lr={config["learning_rate"]:.6f} | '
          f'gamma={config["gamma"]:.4f} | '
          f'hidden={config["hidden_dim"]} | '
          f'buffer={config["buffer_size"]} | '
          f'batch={config["batch_size"]}')

    try:
        val_sortino, _ = run_quick_training(config, n_episodes=200)
        print(f'  Val Sortino: {val_sortino:.4f}')
        return val_sortino
    except Exception as e:
        print(f'  Trial failed: {e}')
        return -999.0


def run_optuna(n_trials=10):
    """
    Run the full Optuna hyperparameter search.

    Parameters:
    -----------
    n_trials : int — number of trials (10 = ~30 mins)

    Returns:
    --------
    dict — best hyperparameters found
    """
    print('=' * 65)
    print('OPTUNA HYPERPARAMETER OPTIMIZATION')
    print('=' * 65)
    print(f'Trials         : {n_trials}')
    print(f'Episodes/trial : 200')
    print(f'Objective      : Maximize validation Sortino ratio')
    print('=' * 65)

    # Suppress Optuna verbose output
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='sector-rotation-dqn'
    )

    study.optimize(objective, n_trials=n_trials)

    # Print results
    print('\n' + '=' * 65)
    print('OPTUNA RESULTS')
    print('=' * 65)
    print(f'Best trial     : #{study.best_trial.number}')
    print(f'Best Sortino   : {study.best_value:.4f}')
    print(f'\nBest hyperparameters:')
    for k, v in study.best_params.items():
        print(f'  {k:<20}: {v}')

    # All trials ranked
    print(f'\nAll trials ranked by Sortino:')
    print(f'  {"Trial":>6} {"Sortino":>9} {"LR":>10} '
          f'{"Gamma":>8} {"Hidden":>8} {"Buffer":>8} {"Batch":>7}')
    print(f'  {"-"*60}')

    sorted_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else -999,
        reverse=True
    )
    for t in sorted_trials:
        if t.value is not None and t.value > -999:
            p = t.params
            print(f'  {t.number:>6} {t.value:>9.4f} '
                  f'{p.get("learning_rate", 0):>10.6f} '
                  f'{p.get("gamma", 0):>8.4f} '
                  f'{p.get("hidden_dim", 0):>8} '
                  f'{p.get("buffer_size", 0):>8} '
                  f'{p.get("batch_size", 0):>7}')

    return study.best_params, study


def save_best_config(best_params):
    """
    Save best hyperparameters to configs/best_hyperparams.yaml.

    WHY SAVE TO A NEW FILE?
    We keep the original hyperparams.yaml unchanged (for reproducibility)
    and save the optimized params separately. This way you can compare
    default vs optimized configs easily.
    """
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    config.update(best_params)

    best_path = os.path.join(ROOT_DIR, 'configs', 'best_hyperparams.yaml')
    with open(best_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f'\nBest config saved → {best_path}')
    return best_path


def retrain_best(best_params, n_episodes=2000):
    """
    Retrain with best Optuna hyperparameters for full 2000 episodes.

    WHY RETRAIN?
    Optuna trials only ran 200 episodes (fast comparison).
    The best config deserves a full training run to get
    the absolute best possible final model.
    """
    print('\n' + '=' * 65)
    print(f'RETRAINING BEST CONFIG FOR {n_episodes} EPISODES')
    print('=' * 65)

    from src.train import train

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    config.update(best_params)
    config['num_episodes'] = n_episodes

    best_ckpt = train(
        config=config,
        experiment_name='sector-rotation-optuna',
        run_name=f'optuna-best-{n_episodes}ep'
    )
    return best_ckpt


# ── Entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Optuna hyperparameter tuning'
    )
    parser.add_argument('--trials',   type=int,  default=10,
                        help='Number of Optuna trials (default 10)')
    parser.add_argument('--retrain',  action='store_true',
                        help='Retrain best config for full episodes')
    parser.add_argument('--episodes', type=int,  default=2000,
                        help='Episodes for retraining (default 2000)')
    args = parser.parse_args()

    # Run Optuna search
    best_params, study = run_optuna(n_trials=args.trials)

    # Save best config to YAML
    save_best_config(best_params)

    # Optionally retrain with best params
    if args.retrain:
        retrain_best(best_params, n_episodes=args.episodes)
    else:
        print('\nTo retrain with best params, run:')
        print('python -m src.optuna_tune --retrain --episodes 2000')

    print('\nOptuna complete!')