"""
train.py
========
Main training loop for the Sector Rotation DQN agent.

WHAT THIS FILE DOES:
1. Loads the training environment (2020-2023 data)
2. Initializes the DQN agent with hyperparameters from config
3. Runs the training loop (episodes x steps)
4. Logs everything to MLflow for experiment tracking
5. Saves the best model checkpoint

THE RL TRAINING LOOP (from your lectures):
For each episode:
    Reset environment → get initial state
    For each step:
        Agent selects action (epsilon-greedy)
        Environment executes action → returns reward + next state
        Store transition in replay buffer
        Sample mini-batch → compute Bellman target → update Q-network
        If episode done → log metrics → start new episode

WHY EPISODES?
One episode = one complete pass through the training data (987 days).
The agent sees the same historical data many times across episodes,
learning better Q-value estimates each time, like studying the same
material repeatedly but understanding it more deeply each time.
"""

import os
import sys
import yaml
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import SectorRotationEnv
from src.dqn_agent   import DQNAgent
from src.rewards     import compute_reward

# ── Paths ──────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT_DIR, 'data', 'processed', 'iv_features.csv')
CONFIG_PATH = os.path.join(ROOT_DIR, 'configs', 'hyperparams.yaml')
CKPT_DIR    = os.path.join(ROOT_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)


def load_config(path=CONFIG_PATH):
    """
    Load hyperparameters from YAML config file.

    WHY A CONFIG FILE?
    Hard-coding hyperparameters inside Python is bad practice:
    - Hard to track what values you used for each experiment
    - Need to edit source code to change hyperparameters
    - Makes reproducibility difficult
    With a YAML config, all hyperparameters are in one place,
    easy to version control, and MLflow logs the exact config
    used for each run.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_episode(env, agent, training=True):
    """
    Run one complete episode (one pass through the dataset).

    Parameters:
    -----------
    env      : SectorRotationEnv
    agent    : DQNAgent
    training : bool — if True, agent explores and learns
                      if False, agent exploits only (evaluation)

    Returns:
    --------
    dict — episode statistics for logging

    WHY SEPARATE TRAINING AND EVAL MODES?
    During training: epsilon > 0, agent sometimes explores randomly
    During evaluation: epsilon = 0, agent always picks best action
    This ensures fair comparison with the SPY baseline.
    """
    state, _      = env.reset()
    total_reward  = 0.0
    total_return  = 0.0
    losses        = []
    returns_hist  = []
    actions_taken = []
    override_count = 0

    while True:
        # Agent selects action (epsilon-greedy during training)
        action = agent.select_action(state, training=training)

        # Environment executes action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Track stats
        total_reward  += reward
        total_return  += info['daily_return']
        returns_hist.append(info['daily_return'])
        actions_taken.append(info['action_executed'])
        if info['override_triggered']:
            override_count += 1

        # Store transition and learn (only during training)
        if training:
            agent.buffer.push(
                state, action, reward, next_state,
                terminated or truncated
            )
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        state = next_state

        if terminated or truncated:
            break

    # Compute episode-level Sortino ratio
    returns_array = np.array(returns_hist)
    ep_sortino = float(compute_reward(returns_array, window=len(returns_array)))

    # Action distribution for this episode
    action_names  = {0: 'XLK', 1: 'XLF', 2: 'XLV', 3: 'CASH'}
    action_counts = {
        action_names[i]: int(np.sum(np.array(actions_taken) == i))
        for i in range(4)
    }

    return {
        'total_reward':   total_reward,
        'total_return':   total_return,
        'mean_loss':      float(np.mean(losses)) if losses else 0.0,
        'sortino':        ep_sortino,
        'override_count': override_count,
        'action_counts':  action_counts,
        'n_steps':        len(returns_hist),
        'epsilon':        agent.epsilon,
    }


def train(config=None, experiment_name='sector-rotation-dqn', run_name=None):
    """
    Main training function.

    Parameters:
    -----------
    config          : dict — hyperparameters (loaded from YAML if None)
    experiment_name : str  — MLflow experiment name
    run_name        : str  — MLflow run name

    Returns:
    --------
    str — path to best saved model checkpoint
    """
    if config is None:
        config = load_config()

    print('=' * 65)
    print('SECTOR ROTATION RL — Training')
    print('=' * 65)
    print(f'Episodes  : {config["num_episodes"]}')
    print(f'Gamma     : {config["gamma"]}')
    print(f'LR        : {config["learning_rate"]}')
    print(f'Hidden    : {config["hidden_dim"]}')
    print(f'Buffer    : {config["buffer_size"]}')
    print('=' * 65)

    # Set random seeds for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environments
    train_env = SectorRotationEnv(DATA_PATH, mode='train')
    val_env   = SectorRotationEnv(DATA_PATH, mode='test')
    print(f'Train env : {len(train_env._df)} days')
    print(f'Val env   : {len(val_env._df)} days\n')

    # Compute epsilon decay per step
    steps_per_episode = len(train_env._df)
    total_steps       = config['num_episodes'] * steps_per_episode
    eps_decay_steps   = int(total_steps * config.get('eps_decay_frac', 0.8))
    eps_decay         = (1.0 - (1.0 - config['eps_end']) / eps_decay_steps
                         if eps_decay_steps > 0 else 0.995)

    # Create agent
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

    # MLflow setup
    mlflow.set_tracking_uri(os.path.join(ROOT_DIR, 'mlruns'))
    mlflow.set_experiment(experiment_name)

    best_sortino   = -np.inf
    best_ckpt_path = None

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f'MLflow run ID: {run_id}')
        print(f'View results : mlflow ui --port 5000\n')

        # Log all hyperparameters
        mlflow.log_params({
            'state_dim':     config['state_dim'],
            'action_dim':    config['action_dim'],
            'hidden_dim':    config['hidden_dim'],
            'learning_rate': config['learning_rate'],
            'gamma':         config['gamma'],
            'batch_size':    config['batch_size'],
            'buffer_size':   config['buffer_size'],
            'target_update': config['target_update'],
            'eps_start':     config['eps_start'],
            'eps_end':       config['eps_end'],
            'num_episodes':  config['num_episodes'],
            'seed':          seed,
            'reward_type':   config.get('reward_type', 'sortino'),
        })

        num_episodes = config['num_episodes']

        for episode in tqdm(range(1, num_episodes + 1), desc='Training'):

            # Run one training episode
            stats = run_episode(train_env, agent, training=True)

            # Log training metrics to MLflow
            mlflow.log_metrics({
                'train/total_reward':   stats['total_reward'],
                'train/total_return':   stats['total_return'],
                'train/sortino':        stats['sortino'],
                'train/mean_loss':      stats['mean_loss'],
                'train/epsilon':        stats['epsilon'],
                'train/override_count': stats['override_count'],
            }, step=episode)

            # Every 50 episodes: validate on 2024 test data
            # WHY? We want to track generalization, not just training performance
            if episode % 50 == 0:
                val_stats = run_episode(val_env, agent, training=False)

                mlflow.log_metrics({
                    'val/total_return':   val_stats['total_return'],
                    'val/sortino':        val_stats['sortino'],
                    'val/override_count': val_stats['override_count'],
                }, step=episode)

                # Save best model based on validation Sortino
                if val_stats['sortino'] > best_sortino:
                    best_sortino   = val_stats['sortino']
                    best_ckpt_path = os.path.join(CKPT_DIR, 'best_model.pt')
                    agent.save(best_ckpt_path)
                    mlflow.log_metric('best_val_sortino', best_sortino, step=episode)
                    tqdm.write(f'  Ep {episode:4d} | '
                               f'Val Sortino: {val_stats["sortino"]:+.4f} | '
                               f'Return: {val_stats["total_return"]:+.4f} | '
                               f'NEW BEST saved!')
                else:
                    tqdm.write(f'  Ep {episode:4d} | '
                               f'Val Sortino: {val_stats["sortino"]:+.4f} | '
                               f'Return: {val_stats["total_return"]:+.4f} | '
                               f'Eps: {stats["epsilon"]:.4f}')

        # Save final model
        final_ckpt = os.path.join(CKPT_DIR, 'final_model.pt')
        agent.save(final_ckpt)
        mlflow.log_artifact(final_ckpt)
        if best_ckpt_path:
            mlflow.log_artifact(best_ckpt_path)

        # Final validation
        print('\nRunning final evaluation on 2024 test data...')
        final_val = run_episode(val_env, agent, training=False)

        mlflow.log_metrics({
            'final/val_sortino':      final_val['sortino'],
            'final/val_total_return': final_val['total_return'],
            'final/val_overrides':    final_val['override_count'],
        })

        print('\n' + '=' * 65)
        print('TRAINING COMPLETE')
        print('=' * 65)
        print(f'Best val Sortino  : {best_sortino:.4f}')
        print(f'Final val Sortino : {final_val["sortino"]:.4f}')
        print(f'Final val Return  : {final_val["total_return"]:.4f}')
        print(f'Override triggers : {final_val["override_count"]}')
        print(f'Best checkpoint   : {best_ckpt_path}')
        print(f'MLflow run ID     : {run_id}')
        print('=' * 65)
        print('\nNext: python src/backtest.py --checkpoint checkpoints/best_model.pt')

    return best_ckpt_path


# ── Entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Sector Rotation DQN')
    parser.add_argument('--episodes', type=int,   default=None,
                        help='Number of training episodes')
    parser.add_argument('--lr',       type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--gamma',    type=float, default=None,
                        help='Discount factor')
    parser.add_argument('--hidden',   type=int,   default=None,
                        help='Hidden layer size')
    parser.add_argument('--run-name', type=str,   default=None,
                        help='MLflow run name')
    args = parser.parse_args()

    config = load_config()

    # Override config with command-line args if provided
    if args.episodes : config['num_episodes']  = args.episodes
    if args.lr       : config['learning_rate'] = args.lr
    if args.gamma    : config['gamma']          = args.gamma
    if args.hidden   : config['hidden_dim']     = args.hidden

    train(config=config, run_name=args.run_name)