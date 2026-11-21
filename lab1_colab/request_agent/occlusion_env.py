"""Permanent Isolated Final Occlusion Environment for Connected Autonomous Driving.

Scenario: Candidate A — Cooperative Blind-Zone Sensing in Target-Lane Merge
- Road: HighwayEnv Merge road with 2 highway lanes and an on-ramp.
- Ego Vehicle: Starts at s=5.0 (x=235.0 m), speed = 20.0 m/s on on-ramp ("b", "c", 2).
- Front Connected CAV: Starts at s=35.0 (x=265.0 m) in target lane ("b", "c", 1).
- Rear Connected CAV: Starts at s=225.0 (x=225.0 m) in target lane ("a", "b", 1).
- 3 Balanced Hidden States:
  * 0: CLEAR (33.3%) - normal cruising traffic, lane 1 clear.
  * 1: FRONT_HAZARD (33.3%) - stationary obstacle downstream at x=285.0 m in lane 1.
       Front CAV detects obstacle and brakes hard (target_speed = 8.0 m/s).
  * 2: REAR_HAZARD (33.3%) - high-speed overtaker charging at 30.0 m/s in lane 1 at x=225.0 m.
- Semantic Communication:
  * Requests: 0=NONE, 1=ASK_FRONT, 2=ASK_REAR
  * Messages: 0=UNKNOWN, 1=CLEAR, 2=HAZARD_AHEAD, 3=HAZARD_BEHIND
  * FRONT_HAZARD: ASK_FRONT -> HAZARD_AHEAD (2), ASK_REAR -> CLEAR (1)
  * REAR_HAZARD:  ASK_REAR  -> HAZARD_BEHIND (3), ASK_FRONT -> CLEAR (1)
  * CLEAR:        ASK_FRONT -> CLEAR (1), ASK_REAR -> CLEAR (1)
  * NONE:         UNKNOWN (0)
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np
import gymnasium as gym
from highway_env import utils
from highway_env.envs.merge_env import MergeGenericEnv

from lab1.config import Config
from lab1.envs.merge import EgoMergeGymEnv
from lab1.envs.highway import make_env


class FinalOcclusionGymEnv(EgoMergeGymEnv):
    """Underlying HighwayEnv Gym environment for cooperative occlusion sensing."""

    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        self.hazard_state = 0  # 0: CLEAR, 1: FRONT_HAZARD, 2: REAR_HAZARD
        self.hazard_vehicle = None
        super().__init__(config=config, render_mode=render_mode)

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "lanes_count": 2,
                "vehicles_count": 2,
                "duration": 20,
                "neighbour_vehicles_connected_lanes": True,
            }
        )
        return cfg

    def _make_vehicles(self) -> None:
        road = self.road
        self.target_lane_index = ("b", "c", 1)
        merge_lane = road.network.get_lane(("b", "c", 2))
        target_lane = road.network.get_lane(self.target_lane_index)
        ab_lane = road.network.get_lane(("a", "b", 1))

        # Ego starts on merge lane at s=5.0 (x=235.0 m), speed=20.0 m/s
        self.vehicle = self.action_type.vehicle_class(
            road, merge_lane.position(5.0, 0.0), speed=20.0
        )
        road.vehicles.append(self.vehicle)

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Front Connected CAV at x=265.0 m (s=35.0 on target lane)
        front_spd = 8.0 if self.hazard_state == 1 else 22.0
        self.front_vehicle = vehicle_type(road, target_lane.position(35.0, 0.0), speed=front_spd)
        self.front_vehicle.target_speed = front_spd
        road.vehicles.append(self.front_vehicle)

        # Rear Connected CAV at x=225.0 m on ab_lane
        # In REAR_HAZARD: Rear CAV charges at 30.0 m/s
        rear_spd = 30.0 if self.hazard_state == 2 else 20.0
        self.rear_vehicle = vehicle_type(road, ab_lane.position(225.0, 0.0), speed=rear_spd)
        self.rear_vehicle.target_speed = rear_spd
        road.vehicles.append(self.rear_vehicle)

        # Occluded downstream obstacle at x=285.0 m in FRONT_HAZARD
        if self.hazard_state == 1:
            self.hazard_vehicle = vehicle_type(road, target_lane.position(55.0, 0.0), speed=0.0)
            road.vehicles.append(self.hazard_vehicle)
        else:
            self.hazard_vehicle = None

    def _is_merged(self) -> bool:
        if self.vehicle is None:
            return False
        lane = self.vehicle.lane_index
        # Merged when entered highway lanes 0 or 1 (not ramp lane 2)
        return lane[0] in {"b", "c", "d"} and lane[2] < 2

    def _is_terminated(self) -> bool:
        if self.vehicle is None:
            return False
        if self.vehicle.crashed:
            return True
        # Terminate when ego has safely merged and traversed past the merge/hazard zone (x > 320.0 m)
        return bool(self._is_merged() and self.vehicle.position[0] > 320.0)


if "ego-merge-occlusion-final-v0" not in gym.registry:
    gym.register(
        id="ego-merge-occlusion-final-v0",
        entry_point="request_agent.occlusion_env:FinalOcclusionGymEnv",
    )


class FinalOcclusionMergeEnv:
    """Production wrapper providing the communication channel and isolation."""

    NONE = 0
    ASK_FRONT = 1
    ASK_REAR = 2

    MSG_UNKNOWN = 0
    MSG_CLEAR = 1
    MSG_HAZARD_AHEAD = 2
    MSG_HAZARD_BEHIND = 3

    HAZARD_CLEAR = 0
    HAZARD_FRONT = 1
    HAZARD_REAR = 2

    def __init__(self, cfg: Config | None = None, request_cost: float = 0.05) -> None:
        self.request_cost = float(request_cost)
        if cfg is None:
            cfg = Config.load("configs/cpu_fast.yaml")
        self.cfg = cfg

        env_cfg = cfg.env.copy() if hasattr(cfg.env, "copy") else cfg.env
        env_cfg.id = "ego-merge-occlusion-final-v0"
        env_cfg.lanes_count = 2
        env_cfg.vehicles_count = 2
        env_cfg.scaling = 1.75
        self.env = make_env(env_cfg)

        self.hazard_state = self.HAZARD_CLEAR
        self.request_count = 0
        self.total_bits = 0
        self.front_vehicle = None
        self.rear_vehicle = None
        self.target_lane_index = ("b", "c", 1)

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.hazard_state = seed % 3
        else:
            self.hazard_state = int(np.random.choice([0, 1, 2]))

        self.env.unwrapped.hazard_state = self.hazard_state
        obs, raw_info = self.env.reset(seed=seed)

        base_env = self.env.unwrapped
        self.front_vehicle = base_env.front_vehicle
        self.rear_vehicle = base_env.rear_vehicle
        self.target_lane_index = base_env.target_lane_index

        self.request_count = 0
        self.total_bits = 0

        info = {
            "hazard_state": self.hazard_state,
            "target_lane_index": self.target_lane_index,
            "crashed": False,
            "is_merged": False,
            "request_count": 0,
            "total_bits": 0,
        }
        return obs, info

    def get_message(self, request_action: int) -> int:
        req = int(request_action)
        if req == self.NONE:
            return self.MSG_UNKNOWN

        if req == self.ASK_FRONT:
            if self.hazard_state == self.HAZARD_FRONT:
                return self.MSG_HAZARD_AHEAD
            else:
                return self.MSG_CLEAR

        if req == self.ASK_REAR:
            if self.hazard_state == self.HAZARD_REAR:
                return self.MSG_HAZARD_BEHIND
            else:
                return self.MSG_CLEAR

        return self.MSG_UNKNOWN

    def step(
        self, driving_action: int, request_action: int = 0
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        msg = self.get_message(request_action)
        obs, raw_r, term, trunc, raw_info = self.env.step(int(driving_action))

        comm_cost = self.request_cost if request_action != self.NONE else 0.0
        final_utility = float(raw_r) - comm_cost

        if request_action != self.NONE:
            self.request_count += 1
            self.total_bits += 4

        info = {
            "message": msg,
            "request_action": int(request_action),
            "raw_driving_reward": float(raw_r),
            "communication_cost": float(comm_cost),
            "crashed": bool(self.env.unwrapped.vehicle.crashed),
            "is_merged": bool(self.env.unwrapped._is_merged()),
            "hazard_state": self.hazard_state,
            "request_count": self.request_count,
            "total_bits": self.total_bits,
        }
        return obs, final_utility, term, trunc, info

