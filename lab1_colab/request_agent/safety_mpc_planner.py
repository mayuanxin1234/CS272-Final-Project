"""Safety-Aware World-Model MPC Planner using calibrated collision probability threshold."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from request_agent.models.safety_world_model import SafetyWorldModel


class SafetyMergeMPCPlanner:
    """World-Model MPC planner with explicit collision probability termination handling."""

    def __init__(
        self,
        model: SafetyWorldModel,
        num_sequences: int = 64,
        horizon: int = 6,
        allowed_actions: tuple[int, ...] = (0, 1, 3, 4),
        seed: int = 0,
        device: torch.device | str = "cpu",
        future_mode: str = "none",  # "none" (0/UNKNOWN) or "persistent"
        p_crit: float = 0.40,
        terminal_penalty: float = -1.0,
    ) -> None:
        self.model = model.to(device).eval()
        self.num_sequences = num_sequences
        self.horizon = horizon
        self.allowed_actions = np.array(allowed_actions, dtype=np.int64)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.future_mode = future_mode
        self.p_crit = float(p_crit)
        self.terminal_penalty = float(terminal_penalty)

        # Ensure model is frozen
        for param in self.model.parameters():
            param.requires_grad = False

    def plan(
        self,
        obs: np.ndarray,
        latent: np.ndarray | None = None,
        request_action: int = 0,
        message: int = 0,
        fixed_candidate_actions: np.ndarray | None = None,
    ) -> tuple[int, float, np.ndarray, dict[str, object]]:
        """Plan driving actions with safety-aware termination and reward masking.

        Parameters
        ----------
        obs:
            Stacked grayscale observation array (4, 64, 64), uint8.
        latent:
            Optional latent vector.
        request_action:
            Communication request action {0=NONE, 1=ASK_FRONT, 2=ASK_REAR}.
        message:
            Revealed semantic reply {0=UNKNOWN, 1=MAINTAIN, 2=ACCELERATE, 3=BRAKE}.
        fixed_candidate_actions:
            Optional fixed (num_sequences, horizon) driving sequences.

        Returns
        -------
        best_first_action:
            Driving action of highest-scoring trajectory.
        best_return:
            Predicted cumulative return.
        candidate_returns:
            Predicted returns for all candidates.
        info:
            Diagnostic dict containing collision probabilities and trajectory predictions.
        """
        if fixed_candidate_actions is not None:
            candidate_actions = fixed_candidate_actions
        else:
            indices = self.rng.integers(
                0, len(self.allowed_actions), size=(self.num_sequences, self.horizon)
            )
            candidate_actions = self.allowed_actions[indices]

        actions_tensor = torch.from_numpy(candidate_actions).to(self.device)
        obs_tensor = (
            torch.from_numpy(obs.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .expand(self.num_sequences, -1, -1, -1)
            .to(self.device)
        )

        requests_tensor = torch.zeros(
            (self.num_sequences, self.horizon), dtype=torch.long, device=self.device
        )
        messages_tensor = torch.zeros(
            (self.num_sequences, self.horizon), dtype=torch.long, device=self.device
        )

        # Condition step 0
        requests_tensor[:, 0] = int(request_action)
        messages_tensor[:, 0] = int(message)

        if self.future_mode == "persistent":
            messages_tensor[:, 1:] = int(message)

        with torch.no_grad():
            imagined = self.model.imagine(
                obs_tensor,
                actions_tensor,
                requests_tensor,
                messages_tensor,
            )

            raw_rewards = imagined["rewards"]  # (num_sequences, horizon)
            logits = imagined["collision_logits"]  # (num_sequences, horizon)
            probs = torch.sigmoid(logits)  # (num_sequences, horizon)

            # Detect collisions: p >= p_crit
            coll_mask = (probs >= self.p_crit)  # (N, H)
            has_coll = coll_mask.any(dim=1)  # (N,)

            # For each sequence, find the first collision timestep
            # argmax on bool returns the first index where True occurs
            first_coll_step = torch.where(
                has_coll,
                coll_mask.float().argmax(dim=1),
                torch.full((self.num_sequences,), self.horizon, dtype=torch.long, device=self.device),
            )

            # Apply termination & penalty
            safe_rewards = raw_rewards.clone()
            step_indices = torch.arange(self.horizon, device=self.device).unsqueeze(0).expand(self.num_sequences, -1)

            # Steps strictly after collision are truncated (0 reward)
            post_crash = step_indices > first_coll_step.unsqueeze(1)
            safe_rewards[post_crash] = 0.0

            # Step at collision receives terminal penalty
            at_crash = step_indices == first_coll_step.unsqueeze(1)
            safe_rewards[at_crash] = self.terminal_penalty

            predicted_returns = safe_rewards.sum(dim=1)
            best_idx = torch.argmax(predicted_returns).item()

        best_first_action = int(candidate_actions[best_idx, 0])
        best_return = float(predicted_returns[best_idx].item())
        candidate_returns = predicted_returns.detach().cpu().numpy()

        diag_info = {
            "best_idx": best_idx,
            "best_action": best_first_action,
            "best_return": best_return,
            "best_has_collision": bool(has_coll[best_idx].item()),
            "best_max_prob": float(probs[best_idx].max().item()),
            "best_trajectory": candidate_actions[best_idx].tolist(),
            "best_probs": probs[best_idx].tolist(),
            "best_raw_rewards": raw_rewards[best_idx].tolist(),
            "best_masked_rewards": safe_rewards[best_idx].tolist(),
            "num_candidates_with_collision": int(has_coll.sum().item()),
        }

        return best_first_action, best_return, candidate_returns, diag_info

    def __call__(
        self,
        obs: np.ndarray,
        latent: np.ndarray | None,
        request_action: int,
        message: int,
    ) -> int:
        action, _, _, _ = self.plan(obs, latent, request_action, message)
        return action

