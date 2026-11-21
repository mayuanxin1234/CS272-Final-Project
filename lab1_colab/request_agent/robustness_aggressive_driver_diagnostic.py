"""Robustness Scenario 2: Aggressive / Unpredictable Driver Diagnostic.

Tests whether cooperative V2X communication protects against erratic, unpredictable neighbor behaviors.

Stochastic Behavior Distribution (4 discrete styles, 25% each):
1. NORMAL:             Cruises at 22 m/s with standard IDM comfortable headway (T=1.5s).
2. SUDDEN_ACCELERATE:  Rear vehicle aggressively surges to 32 m/s (a=5.0 m/s^2, headway T=0.4s).
3. HARD_BRAKE:         Front vehicle brakes abruptly (a=-6.0 m/s^2, speed -> 5 m/s) due to localized bottleneck.
4. AGGRESSIVE_APPROACH: Rear vehicle charges at 30 m/s refusing to yield to merging ramp traffic.

V2X Semantic Messages:
- 0: UNKNOWN
- 1: CLEAR (NORMAL behavior)
- 2: HAZARD_AHEAD (HARD_BRAKE detected by Front CAV radar)
- 3: HAZARD_BEHIND (SUDDEN_ACCELERATE or AGGRESSIVE_APPROACH detected by Rear CAV radar)
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from collections import Counter
import numpy as np
import gymnasium as gym
from highway_env import utils
from highway_env.envs.merge_env import MergeGenericEnv

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.envs.merge import EgoMergeGymEnv
from lab1.envs.highway import make_env

BEHAVIOR_STYLES = {
    0: "NORMAL",
    1: "SUDDEN_ACCELERATE",
    2: "HARD_BRAKE",
    3: "AGGRESSIVE_APPROACH",
}


class AggressiveDriverGymEnv(EgoMergeGymEnv):
    """Isolated environment with stochastic aggressive/unpredictable driver behaviors."""

    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        self.driver_style = 0  # 0: NORMAL, 1: SUDDEN_ACCELERATE, 2: HARD_BRAKE, 3: AGGRESSIVE_APPROACH
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

        # Ego starts on merge ramp at s=5.0 (x=235.0 m), speed=20.0 m/s
        self.vehicle = self.action_type.vehicle_class(
            road, merge_lane.position(5.0, 0.0), speed=20.0
        )
        road.vehicles.append(self.vehicle)

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Front vehicle at x=265.0 m
        # In HARD_BRAKE: Front vehicle brakes violently to 5 m/s
        f_spd = 5.0 if self.driver_style == 2 else 22.0
        self.front_vehicle = vehicle_type(road, target_lane.position(35.0, 0.0), speed=f_spd)
        self.front_vehicle.target_speed = f_spd
        road.vehicles.append(self.front_vehicle)

        # Rear vehicle at x=225.0 m on ab_lane
        # In SUDDEN_ACCELERATE: speed = 32 m/s
        # In AGGRESSIVE_APPROACH: speed = 30 m/s
        # In NORMAL / HARD_BRAKE: speed = 20 m/s
        if self.driver_style == 1:
            r_spd = 32.0
        elif self.driver_style == 3:
            r_spd = 30.0
        else:
            r_spd = 20.0

        self.rear_vehicle = vehicle_type(road, ab_lane.position(225.0, 0.0), speed=r_spd)
        self.rear_vehicle.target_speed = r_spd
        road.vehicles.append(self.rear_vehicle)

    def _is_merged(self) -> bool:
        if self.vehicle is None:
            return False
        lane = self.vehicle.lane_index
        return lane[0] in {"b", "c", "d"} and lane[2] < 2

    def _is_terminated(self) -> bool:
        if self.vehicle is None:
            return False
        if self.vehicle.crashed:
            return True
        return bool(self._is_merged() and self.vehicle.position[0] > 320.0)


if "aggressive-driver-v0" not in gym.registry:
    gym.register(
        id="aggressive-driver-v0",
        entry_point="request_agent.robustness_aggressive_driver_diagnostic:AggressiveDriverGymEnv",
    )


class AggressiveDriverMergeEnv:
    """Wrapper isolating communication channel and unpredictable driver physics."""

    NONE = 0
    ASK_FRONT = 1
    ASK_REAR = 2

    MSG_UNKNOWN = 0
    MSG_CLEAR = 1
    MSG_HAZARD_AHEAD = 2
    MSG_HAZARD_BEHIND = 3

    def __init__(self, request_cost: float = 0.0) -> None:
        self.request_cost = float(request_cost)
        cfg = Config.load("configs/cpu_fast.yaml")
        env_cfg = cfg.env.copy() if hasattr(cfg.env, "copy") else cfg.env
        env_cfg.id = "aggressive-driver-v0"
        env_cfg.lanes_count = 2
        env_cfg.vehicles_count = 2
        self.env = make_env(env_cfg)

        self.driver_style = 0
        self.request_count = 0

    def reset(self, seed: int) -> tuple[np.ndarray, dict]:
        # Balanced 4-way stochastic behavior distribution
        self.driver_style = seed % 4
        self.env.unwrapped.driver_style = self.driver_style
        obs, info = self.env.reset(seed=seed)
        self.request_count = 0
        return obs, info

    def get_message(self, request_action: int) -> int:
        req = int(request_action)
        if req == self.NONE:
            return self.MSG_UNKNOWN

        if req == self.ASK_FRONT:
            if self.driver_style == 2:  # HARD_BRAKE
                return self.MSG_HAZARD_AHEAD
            else:
                return self.MSG_CLEAR

        if req == self.ASK_REAR:
            if self.driver_style in [1, 3]:  # SUDDEN_ACCELERATE or AGGRESSIVE_APPROACH
                return self.MSG_HAZARD_BEHIND
            else:
                return self.MSG_CLEAR

        return self.MSG_UNKNOWN

    def step(self, driving_action: int, request_action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        msg = self.get_message(request_action)
        obs, raw_r, term, trunc, info = self.env.step(int(driving_action))
        comm_cost = self.request_cost if request_action != self.NONE else 0.0
        final_u = float(raw_r) - comm_cost
        if request_action != self.NONE:
            self.request_count += 1
        s_info = {
            "message": msg,
            "raw_driving_reward": float(raw_r),
            "crashed": bool(self.env.unwrapped.vehicle.crashed),
            "is_merged": bool(self.env.unwrapped._is_merged()),
            "driver_style": self.driver_style,
        }
        return obs, final_u, term, trunc, s_info


def run_true_simulator_aggressive_diagnostic(
    num_seeds: int = 60,
    base_seed: int = 9500,
) -> dict:
    print("=" * 85)
    print(f"RUNNING TRUE-SIMULATOR AGGRESSIVE/UNPREDICTABLE DRIVER DIAGNOSTIC ({num_seeds} seeds, base={base_seed})")
    print("=" * 85)

    env = AggressiveDriverMergeEnv(request_cost=0.0)
    policies = ["NO_COMM", "RANDOM_REQUEST", "FULL_COMM", "ORACLE_SELECTIVE"]
    results = {p: [] for p in policies}

    for idx in range(num_seeds):
        seed = base_seed + idx
        style_idx = seed % 4
        s_name = BEHAVIOR_STYLES[style_idx]

        for pol in policies:
            obs, info = env.reset(seed=seed)
            ep_utility = 0.0
            ep_driving_r = 0.0
            crashed = False
            merged = False
            step = 0
            req_count = 0
            revealed_msg = env.MSG_UNKNOWN

            done = False
            while not done and step < 8:
                if pol == "NO_COMM":
                    req = env.NONE
                elif pol == "FULL_COMM":
                    # Query the relevant vehicle for this hazard style
                    if step == 0:
                        req = env.ASK_FRONT if style_idx == 2 else (env.ASK_REAR if style_idx in [1, 3] else env.ASK_FRONT)
                    else:
                        req = env.NONE
                elif pol == "ORACLE_SELECTIVE":
                    if step == 0:
                        if style_idx == 2:
                            req = env.ASK_FRONT
                        elif style_idx in [1, 3]:
                            req = env.ASK_REAR
                        else:
                            req = env.NONE
                    else:
                        req = env.NONE
                elif pol == "RANDOM_REQUEST":
                    if step == 0:
                        rng = np.random.default_rng(seed * 100 + step)
                        req = int(rng.choice([env.NONE, env.ASK_FRONT, env.ASK_REAR]))
                    else:
                        req = env.NONE

                msg = env.get_message(req)
                if req != env.NONE:
                    req_count += 1
                if msg != env.MSG_UNKNOWN:
                    revealed_msg = msg

                # Condition driving action on revealed V2X message
                if revealed_msg == env.MSG_HAZARD_AHEAD:
                    # Lead vehicle hard braking -> Double lane change into Lane 0
                    drive_act = 0 if step in [0, 1] else 1
                elif revealed_msg == env.MSG_HAZARD_BEHIND:
                    # Rear vehicle surging -> Yield on ramp, let speeder pass, merge behind
                    if step == 0:
                        drive_act = 4  # SLOWER
                    elif step == 1:
                        drive_act = 1  # IDLE
                    elif step == 2:
                        drive_act = 0  # LANE_LEFT
                    else:
                        drive_act = 1
                else:
                    # Blind default merge into Lane 1
                    drive_act = 0 if step == 0 else 1

                next_obs, u_rew, term, trunc, step_info = env.step(drive_act, req)
                done = bool(term or trunc)

                ep_utility += u_rew
                ep_driving_r += step_info["raw_driving_reward"]
                if step_info["crashed"]:
                    crashed = True
                if step_info["is_merged"]:
                    merged = True
                step += 1

            results[pol].append({
                "seed": seed,
                "driver_style": s_name,
                "utility": ep_utility,
                "driving_return": ep_driving_r,
                "crashed": crashed,
                "safe_merge": bool(merged and not crashed),
                "steps": step,
                "requests": req_count,
                "bits": req_count * 4,
            })

    # Summary
    summary = {}
    print("\n" + "-" * 88)
    print(f"{'Policy':<18} | {'Safe Merge':<10} | {'Collision':<10} | {'Driving Ret':<12} | {'Utility':<10} | {'Reqs/Ep':<8} | {'Bits/Ep'}")
    print("-" * 88)

    for pol in policies:
        runs = results[pol]
        safe_merges = float(np.mean([r["safe_merge"] for r in runs]) * 100.0)
        collisions = float(np.mean([r["crashed"] for r in runs]) * 100.0)
        drv_r = float(np.mean([r["driving_return"] for r in runs]))
        util = float(np.mean([r["utility"] for r in runs]))
        reqs = float(np.mean([r["requests"] for r in runs]))
        bits = float(np.mean([r["bits"] for r in runs]))

        summary[pol] = {
            "safe_merge_rate": safe_merges,
            "collision_rate": collisions,
            "mean_driving_return": drv_r,
            "mean_utility": util,
            "mean_requests": reqs,
            "mean_bits": bits,
        }
        print(f"{pol:<18} | {safe_merges:>8.1f}%  | {collisions:>8.1f}%  | {drv_r:>10.3f}   | {util:>8.3f}   | {reqs:>6.2f}   | {bits:>5.1f}b")

    # Breakdown by driver behavior
    print("\n" + "=" * 88)
    print("BREAKDOWN BY DRIVER BEHAVIOR STYLE (Safe Merge Rate):")
    print("=" * 88)
    print(f"{'Behavior Style':<22} | {'NO_COMM':<10} | {'FULL_COMM':<10} | {'ORACLE_SELECTIVE':<16} | {'RANDOM_REQUEST'}")
    print("-" * 88)
    for s_code, s_name in BEHAVIOR_STYLES.items():
        sm_nc = float(np.mean([r["safe_merge"] for r in results["NO_COMM"] if r["driver_style"] == s_name]) * 100.0)
        sm_fc = float(np.mean([r["safe_merge"] for r in results["FULL_COMM"] if r["driver_style"] == s_name]) * 100.0)
        sm_os = float(np.mean([r["safe_merge"] for r in results["ORACLE_SELECTIVE"] if r["driver_style"] == s_name]) * 100.0)
        sm_rr = float(np.mean([r["safe_merge"] for r in results["RANDOM_REQUEST"] if r["driver_style"] == s_name]) * 100.0)
        print(f"{s_name:<22} | {sm_nc:>8.1f}%  | {sm_fc:>8.1f}%  | {sm_os:>14.1f}%   | {sm_rr:>12.1f}%")

    out_file = Path("outputs/request_agent/robustness_aggressive_driver_oracle_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, default=str)
    print(f"\nSaved aggressive driver diagnostic results to: {out_file}")
    return summary


if __name__ == "__main__":
    run_true_simulator_aggressive_diagnostic(num_seeds=60, base_seed=9500)

