from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lab1.config import EnvConfig, ModelConfig
from lab1.models.decoder import Decoder
from lab1.models.dynamics import DynamicsModel
from lab1.models.encoder import Encoder
from lab1.models.reward import RewardHead


class WorldModel(nn.Module):
    """World model using driving actions, requests, and replies."""

    def __init__(
        self,
        in_channels: int,
        num_actions: int,
        latent_dim: int = 64,
        channels: tuple[int, ...] = (32, 64, 128),
        dynamics_hidden: tuple[int, ...] = (256, 256),
        image_size: int = 64,
        use_decoder: bool = True,
        use_reward_head: bool = True,
        num_request_actions: int = 3,
        num_messages: int = 4,
    ) -> None:
        super().__init__()

        self.num_actions = num_actions
        self.num_request_actions = num_request_actions
        self.num_messages = num_messages
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            channels=channels,
            image_size=image_size,
        )

        self.dynamics = DynamicsModel(
            latent_dim=latent_dim,
            num_actions=num_actions,
            num_request_actions=num_request_actions,
            num_messages=num_messages,
            hidden=dynamics_hidden,
        )

        self.decoder = (
            Decoder(
                out_channels=in_channels,
                latent_dim=latent_dim,
                channels=channels,
                image_size=image_size,
            )
            if use_decoder
            else None
        )

        self.reward_head = (
            RewardHead(
                latent_dim=latent_dim,
                num_actions=num_actions,
                num_request_actions=3,
                num_messages=4,
                hidden=dynamics_hidden,
            )
            if use_reward_head
            else None
        )

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        env_cfg: EnvConfig,
        num_actions: int,
    ) -> WorldModel:
        return cls(
            in_channels=env_cfg.stack_size,
            num_actions=num_actions,
            latent_dim=model_cfg.latent_dim,
            channels=tuple(model_cfg.encoder_channels),
            dynamics_hidden=tuple(model_cfg.dynamics_hidden),
            image_size=env_cfg.image_size,
            use_decoder=model_cfg.use_decoder,
            use_reward_head=model_cfg.use_reward_head,
            num_request_actions=3,
            num_messages=4,
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("Decoder is disabled.")
        return self.decoder(latent)

    def _action_onehot(self, action: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            action.long(),
            num_classes=self.num_actions,
        ).float()

    def _request_onehot(
        self,
        request_action: torch.Tensor,
    ) -> torch.Tensor:
        return F.one_hot(
            request_action.long(),
            num_classes=self.num_request_actions,
        ).float()

    def _message_onehot(
        self,
        message: torch.Tensor,
    ) -> torch.Tensor:
        return F.one_hot(
            message.long(),
            num_classes=self.num_messages,
        ).float()

    def predict_next(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        request_action: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        return self.dynamics(
            latent,
            self._action_onehot(action),
            self._request_onehot(request_action),
            self._message_onehot(message),
        )

    def predict_reward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        request_action: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        if self.reward_head is None:
            raise RuntimeError("Reward head is disabled.")

        return self.reward_head(
            latent,
            self._action_onehot(action),
            self._request_onehot(request_action),
            self._message_onehot(message),
        )

    @torch.no_grad()
    def imagine(
        self,
        obs0: torch.Tensor,
        actions: torch.Tensor,
        request_actions: torch.Tensor,
        messages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Imagine future states using driving and communication actions."""

        z = self.encode(obs0)
        latents = [z]
        rewards = []

        horizon = actions.shape[1]

        for t in range(horizon):
            action_t = actions[:, t]
            request_t = request_actions[:, t]
            message_t = messages[:, t]

            if self.reward_head is not None:
                rewards.append(
                    self.predict_reward(
                        z,
                        action_t,
                        request_t,
                        message_t,
                    )
                )

            z = self.predict_next(
                z,
                action_t,
                request_t,
                message_t,
            )

            latents.append(z)

        output = {
            "latents": torch.stack(latents, dim=1),
        }

        if rewards:
            output["rewards"] = torch.stack(rewards, dim=1)

        return output

    def compute_losses(
        self,
        batch: dict[str, torch.Tensor],
        recon_weight: float = 1.0,
        reward_weight: float = 1.0,
        dynamics_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute communication-aware world-model losses."""

        obs = batch["obs"]
        next_obs = batch["next_obs"]
        action = batch["action"]
        request_action = batch["request_action"]
        message = batch["message"]

        z = self.encode(obs)

        z_next_pred = self.predict_next(
            z,
            action,
            request_action,
            message,
        )

        z_next_target = self.encode(next_obs).detach()

        dynamics_loss = F.mse_loss(
            z_next_pred,
            z_next_target,
        )

        zero = obs.new_zeros(())

        recon_loss = zero
        if self.decoder is not None:
            reconstruction = self.decoder(z)
            recon_loss = F.mse_loss(
                reconstruction,
                obs,
            )

        reward_loss = zero
        if self.reward_head is not None:
            reward_pred = self.predict_reward(
                z,
                action,
                request_action,
                message,
            )

            reward_loss = F.mse_loss(
                reward_pred,
                batch["reward"],
            )

        total = (
            dynamics_weight * dynamics_loss
            + recon_weight * recon_loss
            + reward_weight * reward_loss
        )

        return {
            "total": total,
            "dynamics": dynamics_loss.detach(),
            "recon": recon_loss.detach(),
            "reward": reward_loss.detach(),
        }