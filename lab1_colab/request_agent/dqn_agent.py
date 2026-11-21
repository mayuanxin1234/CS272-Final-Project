"""Double DQN Agent with Huber loss, target network, and epsilon-greedy exploration."""

from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from request_agent.network import QNetwork
from request_agent.replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    """Double DQN agent for discrete communication request actions."""

    def __init__(
        self,
        state_dim: int = 71,
        action_dim: int = 3,
        hidden_dims: tuple[int, ...] = (128, 128),
        lr: float = 5e-4,
        gamma: float = 0.95,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        grad_clip: float = 1.0,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.q_eval = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()

        self.optimizer = Adam(self.q_eval.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_eval(state_t)
            return int(q_values.argmax(dim=1).item())

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return raw Q-values for analysis."""
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            return self.q_eval(state_t).squeeze(0).cpu().numpy()

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> float | None:
        """Run one Double DQN update step using Huber loss."""
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(
            batch_size, self.device
        )

        # Current Q-values
        q_current = self.q_eval(states).gather(1, actions)

        # Double DQN target calculation:
        # a* = argmax_a Q_eval(s', a)
        # target = r + gamma * (1 - d) * Q_target(s', a*)
        with torch.no_grad():
            next_eval_actions = self.q_eval(next_states).argmax(dim=1, keepdim=True)
            q_target_next = self.q_target(next_states).gather(1, next_eval_actions)
            q_target_value = rewards + (1.0 - dones) * self.gamma * q_target_next

        # Huber loss (Smooth L1)
        loss = F.smooth_l1_loss(q_current, q_target_value)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), self.grad_clip)
        self.optimizer.step()

        # Polyak soft update of target network
        self.soft_update()

        # Decay exploration
        self.decay_epsilon()

        return float(loss.item())

    def soft_update(self) -> None:
        """Soft update model parameters: theta_target = tau*theta + (1-tau)*theta_target."""
        for target_param, eval_param in zip(
            self.q_target.parameters(), self.q_eval.parameters()
        ):
            target_param.data.copy_(
                self.tau * eval_param.data + (1.0 - self.tau) * target_param.data
            )

    def hard_update(self) -> None:
        """Copy Q_eval weights directly to Q_target."""
        self.q_target.load_state_dict(self.q_eval.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_eval": self.q_eval.state_dict(),
                "q_target": self.q_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            destination,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_eval.load_state_dict(checkpoint["q_eval"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "epsilon" in checkpoint:
            self.epsilon = float(checkpoint["epsilon"])

