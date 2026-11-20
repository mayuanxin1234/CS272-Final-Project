"""Communication-aware dynamics model.

Inputs:
    latent + driving action + request action + reply message

Output:
    predicted next latent state
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DynamicsModel(nn.Module):
    """Predict the next latent state using driving and communication actions."""

    def __init__(
        self,
        latent_dim: int,
        num_actions: int,
        num_request_actions: int = 3,
        num_messages: int = 4,
        hidden: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()

        self.num_actions = num_actions
        self.num_request_actions = num_request_actions
        self.num_messages = num_messages

        input_dim = (
            latent_dim
            + num_actions
            + num_request_actions
            + num_messages
        )

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for width in hidden:
            layers.extend(
                [
                    nn.Linear(previous_dim, width),
                    nn.ReLU(inplace=True),
                ]
            )
            previous_dim = width

        layers.append(
            nn.Linear(previous_dim, latent_dim)
        )

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        latent: torch.Tensor,
        action_onehot: torch.Tensor,
        request_onehot: torch.Tensor,
        message_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the next latent state."""

        x = torch.cat(
            [
                latent,
                action_onehot,
                request_onehot,
                message_onehot,
            ],
            dim=-1,
        )

        return self.net(x)