from __future__ import annotations

import gymnasium as gym
import numpy as np

from highway_env import utils
from highway_env.envs.merge_env import MergeGenericEnv

from lab1.envs.highway import make_env


class EgoMergeGymEnv(MergeGenericEnv):

    @classmethod
    def default_config(cls):
        config = super().default_config()

        #lanes_count does not include ego lane
        #Vehicles count does not include ego vehicle count
        #Duration is the maximum time for merging/ an episode length
        config.update(
            {
                "lanes_count": 2,
                "vehicles_count": 2,
                "duration": 30,
                "neighbour_vehicles_connected_lanes": True,
            }
        )

        return config

    def _make_vehicles(self):
        road = self.road
        lanes = self.config["lanes_count"]

        # Main highway target lane.
        self.target_lane_index = (
            "a",
            "b",
            lanes - 1,
        )

        # Ego starts on the access ramp.
        ramp_lane = road.network.get_lane(
            ("j", "k", 0)
        )

        ego_vehicle = self.action_type.vehicle_class(
            road,
            ramp_lane.position(30.0, 0.0),
            speed=25.0,
        )

        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicle_type = utils.class_from_path(
            self.config["other_vehicles_type"]
        )

        highway_lane = road.network.get_lane(
            self.target_lane_index
        )

        # Front highway neighbor.
        front_vehicle = vehicle_type(
            road,
            highway_lane.position(75.0, 0.0),
            speed=27.0,
        )

        # Rear highway neighbor.
        rear_vehicle = vehicle_type(
            road,
            highway_lane.position(5.0, 0.0),
            speed=27.0,
        )

        road.vehicles.append(front_vehicle)
        road.vehicles.append(rear_vehicle)

        self.front_vehicle = front_vehicle
        self.rear_vehicle = rear_vehicle

    def _is_merged(self):
        """
        Return True when the ego has entered the highway target lane.
        """

        if self.vehicle is None:
            return False

        lane_index = self.vehicle.lane_index

        return (
            lane_index[2]
            == self.config["lanes_count"] - 1
            and lane_index[0] in {"a", "b", "c"}
        )

    def _reward(self, action):
        """
        Reward ego progress, speed, safe merging, and collision avoidance.
        """

        if self.vehicle.crashed:
            return -1.0

        
        #Ego vehicle is rewarded for maintaining high speed. Clip ensures reward is between 0 and 1
        speed_reward = np.clip(
            self.vehicle.speed / 30.0,
            0.0,
            1.0,
        )

        merge_reward = 1.0 if self._is_merged() else 0.0

        reward = (
            0.45 * speed_reward
            + 0.55 * merge_reward
        )

        #Penalise any un-necessary lane change after merging, 0 = left change, 2 = right change
        if self._is_merged and action in [0, 2]:
            reward -= 0.02

        return float(reward)

    def _is_terminated(self):
        """
        End after collision or after the ego successfully merges
        and travels beyond the merge area.
        """

        if self.vehicle.crashed:
            return True

        return bool(
            self._is_merged()
            and self.vehicle.position[0] > 180.0
        )

    def _is_truncated(self):
        return bool(
            self.time >= self.config.get(
                "duration",
                30,
            )
        )


# Register the custom Gymnasium environment once.
if "ego-merge-v0" not in gym.registry:
    gym.register(
        id="ego-merge-v0",
        entry_point=(
            "lab1.envs.merge:EgoMergeGymEnv"
        ),
    )


class MergeEnv:
    """
    Communication wrapper around EgoMergeGymEnv.

    Request actions:
        0 = no request
        1 = ask front vehicle
        2 = ask rear vehicle

    Messages:
        0 = unknown
        1 = maintain
        2 = accelerate
        3 = brake
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
        cfg,
        request_cost=0.05,
    ):
        self.cfg = cfg
        self.request_cost = float(
            request_cost
        )

        self.env = make_env(cfg.env)

        self.front_vehicle = None
        self.rear_vehicle = None
        self.target_lane_index = None

        self.front_intent = self.UNKNOWN
        self.rear_intent = self.UNKNOWN

        self.front_base_speed = 0.0
        self.rear_base_speed = 0.0

        self.request_count = 0
        self.total_bits = 0

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)

        rng = np.random.default_rng(seed)

        self.front_intent = int(
            rng.choice(
                [
                    self.MAINTAIN,
                    self.ACCELERATE,
                    self.BRAKE,
                ]
            )
        )

        self.rear_intent = int(
            rng.choice(
                [
                    self.MAINTAIN,
                    self.ACCELERATE,
                    self.BRAKE,
                ]
            )
        )

        base_env = self.env.unwrapped

        self.front_vehicle = (
            base_env.front_vehicle
        )

        self.rear_vehicle = (
            base_env.rear_vehicle
        )

        self.target_lane_index = (
            base_env.target_lane_index
        )

        self.front_base_speed = (
            float(self.front_vehicle.speed)
            if self.front_vehicle is not None
            else 0.0
        )

        self.rear_base_speed = (
            float(self.rear_vehicle.speed)
            if self.rear_vehicle is not None
            else 0.0
        )

        if self.front_vehicle is not None:
            self.front_vehicle.intent = (
                self.front_intent
            )

        if self.rear_vehicle is not None:
            self.rear_vehicle.intent = (
                self.rear_intent
            )

        self.request_count = 0
        self.total_bits = 0

        info = dict(info)
        info["front_intent"] = self.front_intent
        info["rear_intent"] = self.rear_intent
        info["target_lane_index"] = (
            self.target_lane_index
        )

        return obs, info

    def step(
        self,
        driving_action,
        request_action=NONE,
    ):
        self._apply_intent(
            self.front_vehicle,
            self.front_intent,
            self.front_base_speed,
        )

        self._apply_intent(
            self.rear_vehicle,
            self.rear_intent,
            self.rear_base_speed,
        )

        message = self._get_message(
            request_action
        )

        obs, reward, terminated, truncated, info = (
            self.env.step(
                int(driving_action)
            )
        )

        requested = int(
            request_action != self.NONE
        )

        communication_cost = (
            self.request_cost * requested
        )

        reward = float(reward) - (
            communication_cost
        )

        if requested:
            self.request_count += 1

            # 2 request bits + 2 reply bits.
            self.total_bits += 2 + 2

        info = dict(info)
        info["message"] = int(message)
        info["request_action"] = int(
            request_action
        )
        info["communication_cost"] = float(
            communication_cost
        )
        info["request_count"] = int(
            self.request_count
        )
        info["total_bits"] = int(
            self.total_bits
        )
        info["front_intent"] = int(
            self.front_intent
        )
        info["rear_intent"] = int(
            self.rear_intent
        )
        info["target_lane_index"] = (
            self.target_lane_index
        )

        if self.front_vehicle is not None:
            info["front_speed"] = float(
                self.front_vehicle.speed
            )

        if self.rear_vehicle is not None:
            info["rear_speed"] = float(
                self.rear_vehicle.speed
            )

        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def _apply_intent(
        self,
        vehicle,
        intent,
        base_speed,
    ):
        if vehicle is None:
            return

        if intent == self.MAINTAIN:
            target_speed = base_speed

        elif intent == self.ACCELERATE:
            target_speed = base_speed + 5.0

        elif intent == self.BRAKE:
            target_speed = max(
                0.0,
                base_speed - 5.0,
            )

        else:
            return

        if hasattr(vehicle, "MAX_SPEED"):
            target_speed = min(
                target_speed,
                float(vehicle.MAX_SPEED),
            )

        if hasattr(vehicle, "target_speed"):
            vehicle.target_speed = (
                target_speed
            )
        else:
            vehicle.speed = target_speed

    def _get_message(self, request_action):
        if request_action == self.ASK_FRONT:
            return self.front_intent

        if request_action == self.ASK_REAR:
            return self.rear_intent

        return self.UNKNOWN

    def close(self):
        self.env.close()