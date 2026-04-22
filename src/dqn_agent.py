import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim=9, action_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32),
            torch.tensor(actions,               dtype=torch.long),
            torch.tensor(rewards,               dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones,                 dtype=torch.float32),
        )

    def __len__(self):
        return len(self._buf)

    @property
    def is_ready(self):
        """Don't start training until buffer has at least 500 transitions."""
        return len(self._buf) >= 500


class DQNAgent:
    def __init__(
        self,
        state_dim: int        = 9,
        action_dim: int       = 4,       # FIX: was 3, now 4 (XLK, XLF, XLV, CASH)
        hidden: int           = 128,
        lr: float             = 1e-3,
        gamma: float          = 0.99,
        epsilon: float        = 1.0,
        epsilon_min: float    = 0.05,
        epsilon_decay: float  = 0.995,
        buffer_capacity: int  = 10_000,
        batch_size: int       = 64,
        target_update_freq: int = 100,
        grad_clip: float      = 1.0,     # FIX: added gradient clipping
    ):
        self.action_dim         = action_dim
        self.gamma              = gamma
        self.epsilon            = epsilon
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip          = grad_clip
        self._steps             = 0

        self.q_net      = QNetwork(state_dim, action_dim, hidden)
        self.target_net = QNetwork(state_dim, action_dim, hidden)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        - training=True:  explore randomly with probability epsilon
        - training=False: always pick the greedy best action (evaluation mode)
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            q = self.q_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def train_step(self):
        if not self.buffer.is_ready:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        # Current Q-values for actions taken
        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target: r + γ * max Q_target(s', a')
        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1).values
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()

        # FIX: gradient clipping prevents exploding gradients → NaN losses
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)

        self.optimizer.step()

        # Decay epsilon and sync target network
        self._steps  += 1
        self.epsilon  = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save({
            'q_net':     self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon':   self.epsilon,
            'steps':     self._steps,
        }, path)
        print(f"Agent saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location='cpu')
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon  = ckpt['epsilon']
        self._steps   = ckpt['steps']
        print(f"Agent loaded ← {path}")


# ── Smoke test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing DQN components...")

    # Test QNetwork
    net   = QNetwork(state_dim=9, action_dim=4, hidden=128)
    dummy = torch.randn(1, 9)
    out   = net(dummy)
    assert out.shape == (1, 4), f"Expected (1,4), got {out.shape}"
    print(f"  QNetwork output shape: {out.shape} ✓")

    # Test ReplayBuffer
    buf = ReplayBuffer(1000)
    for _ in range(600):
        buf.push(
            np.random.randn(9).astype(np.float32),
            np.random.randint(4),
            np.random.randn(),
            np.random.randn(9).astype(np.float32),
            False
        )
    assert buf.is_ready
    states, actions, rewards, ns, dones = buf.sample(64)
    assert states.shape == (64, 9)
    print(f"  ReplayBuffer sample shape: {states.shape} ✓")

    # Test DQNAgent full loop
    agent = DQNAgent(state_dim=9, action_dim=4, hidden=64)
    for i in range(100):
        s  = np.random.randn(9).astype(np.float32)
        a  = agent.select_action(s, training=True)
        r  = np.random.randn()
        ns = np.random.randn(9).astype(np.float32)
        agent.buffer.push(s, a, r, ns, i == 99)
        agent.train_step()

    print(f"  DQNAgent steps: {agent._steps}, epsilon: {agent.epsilon:.4f} ✓")
    print("\nAll DQN tests passed!")
