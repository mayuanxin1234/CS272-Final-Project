"""Final Candidate #2 Environment for Communication-Aware World Models.

Scenario Geometry:
- Ego Vehicle: x = 30.0 m, speed = 25.0 m/s (on access ramp)
- Front Target-Lane Vehicle: x = 54.0 m, speed = 22.0 m/s (predictable, non-critical lead)
- Rear Target-Lane Vehicle: x = 18.0 m, speed = 22.0 m/s (critical rear agent with hidden intent)
- Target-Lane Gap: 36.0 m
- Rear Intents:
  - MAINTAIN (1)   -> target speed 22.0 m/s
  - ACCELERATE (2) -> target speed 27.0 m/s
  - BRAKE (3)      -> target speed 17.0 m/s
- Requests: 0=NONE, 1=ASK_FRONT, 2=ASK_REAR
- Messages: 0=UNKNOWN, 1=MAINTAIN, 2=ACCELERATE, 3=BRAKE
"""

from __future__ import annotations

from typing import Any
import numpy as np
import gymnasium as gym
from highway_env import utils
from highway_env.envs.merge_env import MergeGenericEnv

from lab1.config import Config
from lab1.envs.merge import EgoMergeGymEnv, MergeEnv
from lab1.envs.highway import make_env


class FinalCandidate2GymEnv(EgoMergeGymEnv):
    """Final Candidate #2 Gym environment."""

    SCENARIO_CONFIG = {
        "ego_x": 30.0,
        "ego_speed": 25.0,
        "front_x": 54.0,
        "front_speed": 22.0,
        "rear_x": 18.0,
        "rear_speed": 22.0,
    }

    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        super().__init__(config=config, render_mode=render_mode)

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "lanes_count": 2,
                "vehicles_count": 2,
                "duration": 30,
                "neighbour_vehicles_connected_lanes": True,
            }
        )
        return cfg

    def _make_vehicles(self) -> None:
        road = self.road
        lanes = self.config["lanes_count"]
        self.target_lane_index = ("a", "b", lanes - 1)
        ramp_lane = road.network.get_lane(("j", "k", 0))

        s_cfg = self.SCENARIO_CONFIG
        ego = self.action_type.vehicle_class(
            road,
            ramp_lane.position(s_cfg["ego_x"], 0.0),
            speed=s_cfg["ego_speed"],
        )
        road.vehicles.append(ego)
        self.vehicle = ego

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        highway_lane = road.network.get_lane(self.target_lane_index)

        # Front vehicle at x=54.0m, speed=22.0m/s (predictable lead)
        front = vehicle_type(
            road,
            highway_lane.position(s_cfg["front_x"], 0.0),
            speed=s_cfg["front_speed"],
        )

        # Rear vehicle at x=18.0m, speed=22.0m/s (decision-critical bottleneck agent)
        rear = vehicle_type(
            road,
            highway_lane.position(s_cfg["rear_x"], 0.0),
            speed=s_cfg["rear_speed"],
        )

        road.vehicles.append(front)
        road.vehicles.append(rear)
        self.front_vehicle = front
        self.rear_vehicle = rear


if "ego-merge-final-candidate2-v0" not in gym.registry:
    gym.register(
        id="ego-merge-final-candidate2-v0",
        entry_point="request_agent.final_candidate2_env:FinalCandidate2GymEnv",
    )


class FinalCandidate2MergeEnv:
    """Communication environment wrapper for Final Candidate #2.

    Ensures hidden intent is NOT leaked to observation info,
    and isolates communication querying from environment physics stepping.
    """

    NONE = 0
    ASK_FRONT = 1
    ASK_REAR = 2

    UNKNOWN = 0
    MAINTAIN = 1
    ACCELERATE = 2
    BRAKE = 3

    def __init__(self, cfg: Config, request_cost: float = 0.05) -> None:
        self.cfg = cfg
        self.request_cost = float(request_cost)

        env_cfg = cfg.env.copy() if hasattr(cfg.env, "copy") else cfg.env
        env_cfg.id = "ego-merge-final-candidate2-v0"
        env_cfg.lanes_count = 2
        env_cfg.vehicles_count = 2
        env_cfg.scaling = 1.75

        self.env = make_env(env_cfg)

        self.front_vehicle = None
        self.rear_vehicle = None
        self.target_lane_index = None

        # Front is predictable MAINTAIN
        self.front_intent = self.MAINTAIN
        # Rear is balanced hidden intent {MAINTAIN, ACCELERATE, BRAKE}
        self.rear_intent = self.UNKNOWN

        self.front_base_speed = 22.0
        self.rear_base_speed = 22.0

        self.request_count = 0
        self.total_bits = 0

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, raw_info = self.env.reset(seed=seed)

        rng = np.random.default_rng(seed)
        self.front_intent = self.MAINTAIN
        self.rear_intent = int(rng.choice([self.MAINTAIN, self.ACCELERATE, self.BRAKE]))

        base_env = self.env.unwrapped
        self.front_vehicle = base_env.front_vehicle
        self.rear_vehicle = base_env.rear_vehicle
        self.target_lane_index = base_env.target_lane_index

        self.front_base_speed = 22.0
        self.rear_base_speed = 22.0

        if self.front_vehicle is not None:
            self.front_vehicle.speed = self.front_base_speed
            self.front_vehicle.intent = self.front_intent
        if self.rear_vehicle is not None:
            self.rear_vehicle.speed = self.rear_base_speed
            self.rear_vehicle.intent = self.rear_intent

        self.request_count = 0
        self.total_bits = 0

        sanitized_info = {
            "target_lane_index": self.target_lane_index,
            "request_count": 0,
            "total_bits": 0,
            "crashed": False,
            "is_merged": False,
        }
        return obs, sanitized_info

    def get_message(self, request_action: int) -> int:
        """Non-mutating read of requested message."""
        req = int(request_action)
        if req == self.ASK_FRONT:
            return int(self.front_intent)  # Always MAINTAIN (1)
        if req == self.ASK_REAR:
            return int(self.rear_intent)   # Hidden rear intent (1, 2, or 3)
        return self.UNKNOWN

    def _apply_intent(self, vehicle: Any, intent: int, base_speed: float) -> None:
        if vehicle is None:
            return
        if intent == self.MAINTAIN:
            target_speed = base_speed          # 22.0 m/s
        elif intent == self.ACCELERATE:
            target_speed = base_speed + 5.0    # 27.0 m/s
        elif intent == self.BRAKE:
            target_speed = max(0.0, base_speed - 5.0)  # 17.0 m/s
        else:
            return

        if hasattr(vehicle, "MAX_SPEED"):
            target_speed = min(target_speed, float(vehicle.MAX_SPEED))
        if hasattr(vehicle, "target_speed"):
            vehicle.target_speed = target_speed
        else:
            vehicle.speed = target_speed

    def step(
        self,
        driving_action: int,
        request_action: int = NONE,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._apply_intent(self.front_vehicle, self.front_intent, self.front_base_speed)
        self._apply_intent(self.rear_vehicle, self.rear_intent, self.rear_base_speed)

        message = self.get_message(request_action)

        obs, reward, terminated, truncated, raw_info = self.env.step(int(driving_action))

        requested = int(request_action != self.NONE)
        communication_cost = self.request_cost * requested
        final_reward = float(reward) - communication_cost

        if requested:
            self.request_count += 1
            self.total_bits += 4

        base_unwrapped = self.env.unwrapped
        is_merged = bool(base_unwrapped._is_merged()) if hasattr(base_unwrapped, "_is_merged") else False
        is_crashed = bool(base_unwrapped.vehicle.crashed) if hasattr(base_unwrapped, "vehicle") and base_unwrapped.vehicle is not None else bool(raw_info.get("crashed", False))

        info = {
            "message": int(message),
            "request_action": int(request_action),
            "communication_cost": float(communication_cost),
            "raw_driving_reward": float(reward),
            "request_count": int(self.request_count),
            "total_bits": int(self.total_bits),
            "crashed": is_crashed,
            "is_merged": is_merged,
            "target_lane_index": self.target_lane_index,
        }

        return obs, final_reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()

