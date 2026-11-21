"""Experience replay buffer for the request policy."""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-capacity experience replay buffer."""

    def __init__(self, capacity: int = 10000, seed: int = 0) -> None:
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            Transition(
                state=np.asarray(state, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_state, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.rng.sample(self.buffer, batch_size)

        states = torch.tensor(
            np.array([t.state for t in batch]), dtype=torch.float32, device=device
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long, device=device
        ).unsqueeze(1)
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=device
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=device
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

