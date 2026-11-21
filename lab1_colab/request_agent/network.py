"""MLP Q-Network for the 3-action communication request policy."""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Deep Q-Network mapping 71-D request state to 3 Q-values.

    State:
        z_t (64) + one_hot(prev_req) (3) + one_hot(prev_msg) (4) = 71
    Actions:
        0 = NONE
        1 = ASK_FRONT
        2 = ASK_REAR
    """

    def __init__(
        self,
        state_dim: int = 71,
        action_dim: int = 3,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers: list[nn.Module] = []
        prev_dim = state_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all 3 discrete request actions."""
        return self.net(state)

