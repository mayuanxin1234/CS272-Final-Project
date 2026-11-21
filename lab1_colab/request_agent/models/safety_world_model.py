from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab1.config import EnvConfig, ModelConfig
from lab1.models.world_model import WorldModel


class CollisionHead(nn.Module):
    """Predicts collision logit conditioned on latent state, driving action, request, and message."""

    def __init__(
        self,
        latent_dim: int = 64,
        num_actions: int = 5,
        num_request_actions: int = 3,
        num_messages: int = 4,
        hidden: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        in_dim = latent_dim + num_actions + num_request_actions + num_messages
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        latent: torch.Tensor,
        action_onehot: torch.Tensor,
        request_onehot: torch.Tensor,
        message_onehot: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([latent, action_onehot, request_onehot, message_onehot], dim=-1)
        return self.net(x).squeeze(-1)


class SafetyWorldModel(WorldModel):
    """World model equipped with an explicit collision prediction head."""

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
        collision_head_hidden: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            num_actions=num_actions,
            latent_dim=latent_dim,
            channels=channels,
            dynamics_hidden=dynamics_hidden,
            image_size=image_size,
            use_decoder=use_decoder,
            use_reward_head=use_reward_head,
            num_request_actions=num_request_actions,
            num_messages=num_messages,
        )

        self.collision_head = CollisionHead(
            latent_dim=latent_dim,
            num_actions=num_actions,
            num_request_actions=num_request_actions,
            num_messages=num_messages,
            hidden=collision_head_hidden,
        )

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        env_cfg: EnvConfig,
        num_actions: int,
    ) -> SafetyWorldModel:
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

    def predict_collision_logit(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        request_action: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        return self.collision_head(
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
        """Imagine future states, scalar rewards, and collision logits."""
        z = self.encode(obs0)
        latents = [z]
        rewards = []
        collision_logits = []

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

            collision_logits.append(
                self.predict_collision_logit(
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
            "collision_logits": torch.stack(collision_logits, dim=1),
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
        collision_weight: float = 1.0,
        pos_weight: torch.Tensor | float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute dynamics, reconstruction, reward, and safety collision losses."""
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

        collision_logits = self.predict_collision_logit(
            z,
            action,
            request_action,
            message,
        )

        if "collision" in batch:
            collision_targets = batch["collision"].float()
        else:
            collision_targets = (batch["reward"] <= -0.5).float()

        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pw_tensor = torch.tensor([pos_weight], device=obs.device, dtype=torch.float32)
            else:
                pw_tensor = pos_weight.to(obs.device)
            collision_loss = F.binary_cross_entropy_with_logits(
                collision_logits,
                collision_targets,
                pos_weight=pw_tensor,
            )
        else:
            collision_loss = F.binary_cross_entropy_with_logits(
                collision_logits,
                collision_targets,
            )

        total = (
            dynamics_weight * dynamics_loss
            + recon_weight * recon_loss
            + reward_weight * reward_loss
            + collision_weight * collision_loss
        )

        return {
            "total": total,
            "dynamics": dynamics_loss.detach(),
            "recon": recon_loss.detach(),
            "reward": reward_loss.detach(),
            "collision": collision_loss.detach(),
        }

