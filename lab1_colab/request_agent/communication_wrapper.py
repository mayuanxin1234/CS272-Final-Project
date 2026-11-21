"""Non-invasive communication environment wrapper and temporary driving controller."""

from __future__ import annotations

from typing import Any, Callable
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from lab1.config import Config
from lab1.envs.merge import MergeEnv
from lab1.models import WorldModel
from lab1.utils.model_handoff import load_personal_checkpoint

# Planner Callback signature:
# driving_action = planner(obs, latent, request_action, message)
PlannerCallback = Callable[[np.ndarray, np.ndarray, int, int], int]


class CommunicationMergeWrapper:
    """Wrapper around MergeEnv separating communication from physics advancement.

    Sequence per timestep:
        1. Current observation obs_t -> frozen encoder produces z_t (dim 64)
        2. Construct 71-D state: [z_t (64), one_hot(a_{t-1}^req) (3), one_hot(m_{t-1}) (4)]
        3. Request policy selects request_action in {0=NONE, 1=ASK_FRONT, 2=ASK_REAR}
        4. wrapper.request_message(request_action) returns semantic message without advancing physics
        5. Planner/MPC receives (z_t, request_action, message) and outputs driving_action in {0..4}
        6. wrapper.step(driving_action, request_action) advances physics exactly once
        7. Exactly one communication cost is deducted (performed inside MergeEnv)
    """

    NONE = 0
    ASK_FRONT = 1
    ASK_REAR = 2

    UNKNOWN = 0
    MAINTAIN = 1
    ACCELERATE = 2
    BRAKE = 3

    def __init__(
        self,
        cfg: Config,
        world_model: WorldModel | None = None,
        model_checkpoint_path: str | Path | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.env = MergeEnv(cfg)

        # Initialize or load frozen WorldModel
        if world_model is not None:
            self.model = world_model.to(self.device)
        elif model_checkpoint_path is not None:
            from lab1.data import load_num_actions

            dataset_path = cfg.data.dataset_path
            num_actions = load_num_actions(dataset_path)
            self.model = WorldModel.from_config(cfg.model, cfg.env, num_actions).to(
                self.device
            )
            load_personal_checkpoint(model_checkpoint_path, self.model, cfg)
        else:
            raise ValueError(
                "Either world_model or model_checkpoint_path must be provided."
            )

        # Freeze world model completely
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # State memory
        self.last_request: int = self.NONE
        self.last_message: int = self.UNKNOWN
        self.current_obs: np.ndarray | None = None
        self.current_latent: np.ndarray | None = None
        self.current_req_state: np.ndarray | None = None

    @property
    def action_space_requests(self) -> int:
        return 3

    @property
    def action_space_driving(self) -> int:
        return self.env.action_space.n

    @torch.no_grad()
    def _encode_obs(self, obs: np.ndarray) -> np.ndarray:
        """Encode (4, 64, 64) uint8 observation into 64-D latent using frozen encoder."""
        obs_t = (
            torch.from_numpy(obs.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        z = self.model.encode(obs_t)
        return z.squeeze(0).cpu().numpy()

    def _build_request_state(
        self, latent: np.ndarray, prev_req: int, prev_msg: int
    ) -> np.ndarray:
        """Assemble 71-D request state: [latent (64), req_onehot (3), msg_onehot (4)]."""
        req_onehot = np.zeros(3, dtype=np.float32)
        req_onehot[prev_req] = 1.0

        msg_onehot = np.zeros(4, dtype=np.float32)
        msg_onehot[prev_msg] = 1.0

        return np.concatenate([latent.astype(np.float32), req_onehot, msg_onehot])

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment, clear communication memory, return initial 71-D request state."""
        obs, raw_info = self.env.reset(seed=seed)
        self.current_obs = obs
        self.current_latent = self._encode_obs(obs)

        # Initial memory at start of episode
        self.last_request = self.NONE
        self.last_message = self.UNKNOWN
        self.current_req_state = self._build_request_state(
            self.current_latent, self.last_request, self.last_message
        )

        # Sanitized info dict: do NOT expose hidden intents to the policy
        sanitized_info = {
            "target_lane_index": raw_info.get("target_lane_index"),
            "request_count": 0,
            "total_bits": 0,
        }

        return self.current_req_state.copy(), sanitized_info

    def request_message(self, request_action: int) -> int:
        """Non-mutating read of the requested message.

        Does not advance simulation physics or time.
        """
        return int(self.env._get_message(int(request_action)))

    def step(
        self,
        driving_action: int,
        request_action: int,
        message: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance physics by one step.

        MergeEnv handles communication cost and bit counters exactly once.
        """
        # Advance environment
        next_obs, env_reward, terminated, truncated, raw_info = self.env.step(
            driving_action=int(driving_action), request_action=int(request_action)
        )

        comm_cost = float(raw_info.get("communication_cost", 0.0))
        # Note on reward accounting:
        # MergeEnv.step() has ALREADY subtracted comm_cost: env_reward = raw_driving_reward - comm_cost.
        # Therefore, final_request_reward == env_reward.
        final_request_reward = float(env_reward)
        raw_driving_reward = float(env_reward + comm_cost)

        # Compute next latent with frozen encoder
        next_latent = self._encode_obs(next_obs)

        # Update previous request/message history
        self.last_request = int(request_action)
        self.last_message = int(message)

        # Build next 71-D request state
        next_req_state = self._build_request_state(
            next_latent, self.last_request, self.last_message
        )

        self.current_obs = next_obs
        self.current_latent = next_latent
        self.current_req_state = next_req_state

        # Sanitized info for callers/logging (prevents hidden intent leakage to policy)
        base_unwrapped = self.env.env.unwrapped
        is_merged = bool(base_unwrapped._is_merged()) if hasattr(base_unwrapped, "_is_merged") else False
        is_crashed = bool(base_unwrapped.vehicle.crashed) if hasattr(base_unwrapped, "vehicle") and base_unwrapped.vehicle is not None else bool(raw_info.get("crashed", False))

        sanitized_info = {
            "raw_driving_reward": raw_driving_reward,
            "communication_cost": comm_cost,
            "final_request_reward": final_request_reward,
            "message": int(message),
            "request_action": int(request_action),
            "request_count": int(raw_info.get("request_count", 0)),
            "total_bits": int(raw_info.get("total_bits", 0)),
            "crashed": is_crashed,
            "is_merged": is_merged,
            "front_speed": raw_info.get("front_speed", 0.0),
            "rear_speed": raw_info.get("rear_speed", 0.0),
        }

        return (
            next_req_state.copy(),
            final_request_reward,
            terminated,
            truncated,
            sanitized_info,
        )

    def close(self) -> None:
        self.env.close()


class TemporaryDrivingController:
    """TEMPORARY heuristic driving controller for training/testing the request agent.

    NOTE: This is NOT the final Lab 2 World-Model MPC. It serves as a placeholder
    callback matching the interface:
        driving_action = controller(obs, latent, request_action, message)
    so the request policy can learn in an interactive loop.
    """

    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        obs: np.ndarray,
        latent: np.ndarray,
        request_action: int,
        message: int,
    ) -> int:
        """Select a driving action adapting to neighbor message."""
        # If front car is braking or maintaining close, or rear is accelerating, adapt speed
        if message == CommunicationMergeWrapper.BRAKE and request_action == CommunicationMergeWrapper.ASK_FRONT:
            # Front vehicle braking -> match speed or slow down
            return self.SLOWER
        elif message == CommunicationMergeWrapper.ACCELERATE and request_action == CommunicationMergeWrapper.ASK_FRONT:
            # Front vehicle accelerating -> safe to speed up or merge left
            return self.LANE_LEFT if self.rng.random() < 0.6 else self.FASTER
        elif message == CommunicationMergeWrapper.ACCELERATE and request_action == CommunicationMergeWrapper.ASK_REAR:
            # Rear vehicle accelerating -> speed up to create buffer
            return self.FASTER
        elif message == CommunicationMergeWrapper.BRAKE and request_action == CommunicationMergeWrapper.ASK_REAR:
            # Rear vehicle braking -> safe to merge into target lane
            return self.LANE_LEFT

        # Default heuristic: try merging when possible, otherwise maintain speed
        choices = [self.LANE_LEFT, self.IDLE, self.FASTER]
        return int(self.rng.choice(choices, p=[0.4, 0.4, 0.2]))

