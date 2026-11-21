"""Adapter connecting World-Model MPC planning with the DQN communication agent.

Adapts the Lab 2 MergeMPC design so that:
1. The DQN agent owns the communication decision (request_action in {0, 1, 2}).
2. The semantic reply (message in {0, 1, 2, 3}) is received before planning.
3. The MPC conditions all candidate driving sequences on (request_action, message)
   for the current timestep (step 0).
4. Future horizon steps (1..H-1) use default non-privileged communication
   (NONE=0, UNKNOWN=0, or persistent intent).
5. The MPC evaluates candidate driving action sequences in [0, 1, 3, 4] and selects
   the first action of the highest-scoring trajectory.
"""

from __future__ import annotations

import numpy as np
import torch

from lab1.models import WorldModel


class MergeMPCPlanner:
    """World-Model MPC planner conditioned on external communication decisions."""

    def __init__(
        self,
        model: WorldModel,
        num_sequences: int = 64,
        horizon: int = 6,
        allowed_actions: tuple[int, ...] = (0, 1, 3, 4),
        seed: int = 0,
        device: torch.device | str = "cpu",
        future_mode: str = "none",  # "none" (0/UNKNOWN) or "persistent" (retains received msg)
    ) -> None:
        self.model = model.to(device).eval()
        self.num_sequences = num_sequences
        self.horizon = horizon
        self.allowed_actions = np.array(allowed_actions, dtype=np.int64)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.future_mode = future_mode

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
    ) -> tuple[int, float, np.ndarray]:
        """Plan driving actions conditioned on current (request_action, message).

        Parameters
        ----------
        obs:
            Stacked grayscale observation array (4, 64, 64), uint8.
        latent:
            Optional precomputed 64-D latent vector (unused directly if imagine encodes obs).
        request_action:
            DQN-selected request action in {0=NONE, 1=ASK_FRONT, 2=ASK_REAR}.
        message:
            Revealed semantic reply in {0=UNKNOWN, 1=MAINTAIN, 2=ACCELERATE, 3=BRAKE}.
        fixed_candidate_actions:
            Optional (num_sequences, horizon) candidate driving sequences to evaluate.
            Useful for controlled counterfactual diagnostics on fixed states.

        Returns
        -------
        best_first_action:
            First driving action of the highest predicted-return trajectory.
        best_return:
            Predicted cumulative return over the horizon.
        candidate_returns:
            Array of predicted returns for all candidate sequences.
        """
        if fixed_candidate_actions is not None:
            candidate_actions = fixed_candidate_actions
        else:
            indices = self.rng.integers(
                0, len(self.allowed_actions), size=(self.num_sequences, self.horizon)
            )
            candidate_actions = self.allowed_actions[indices]

        # Convert candidate actions to tensor
        actions_tensor = torch.from_numpy(candidate_actions).to(self.device)

        # Encode observation once or reuse provided latent
        if latent is not None:
            z0 = (
                torch.from_numpy(latent.astype(np.float32))
                .unsqueeze(0)
                .expand(self.num_sequences, -1)
                .to(self.device)
            )
        else:
            obs_single = (
                torch.from_numpy(obs.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.no_grad():
                z0 = self.model.encode(obs_single).expand(self.num_sequences, -1)

        # Allocate communication tensors (num_sequences, horizon)
        requests_tensor = torch.zeros(
            (self.num_sequences, self.horizon), dtype=torch.long, device=self.device
        )
        messages_tensor = torch.zeros(
            (self.num_sequences, self.horizon), dtype=torch.long, device=self.device
        )

        # Step 0 (current decision): conditioned on DQN decision and revealed message
        requests_tensor[:, 0] = int(request_action)
        messages_tensor[:, 0] = int(message)

        # Steps 1..horizon-1 (future):
        if self.future_mode == "persistent":
            messages_tensor[:, 1:] = int(message)

        with torch.no_grad():
            curr_z = z0
            rewards = []
            for t in range(self.horizon):
                act_t = actions_tensor[:, t]
                req_t = requests_tensor[:, t]
                msg_t = messages_tensor[:, t]
                if self.model.reward_head is not None:
                    rewards.append(self.model.predict_reward(curr_z, act_t, req_t, msg_t))
                curr_z = self.model.predict_next(curr_z, act_t, req_t, msg_t)
            predicted_returns = torch.stack(rewards, dim=1).sum(dim=1)
            best_idx = torch.argmax(predicted_returns).item()

        best_first_action = int(candidate_actions[best_idx, 0])
        best_return = float(predicted_returns[best_idx].item())
        candidate_returns = predicted_returns.detach().cpu().numpy()

        return best_first_action, best_return, candidate_returns

    def __call__(
        self,
        obs: np.ndarray,
        latent: np.ndarray | None,
        request_action: int,
        message: int,
    ) -> int:
        """Callback signature matching PlannerCallback: (obs, latent, req, msg) -> driving_action."""
        action, _, _ = self.plan(obs, latent, request_action, message)
        return action

