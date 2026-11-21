"""True-Simulator Oracle Diagnostic Test for Cooperative Occlusion Sensing.

Evaluates 4 policies across 60 fixed seeds (20 CLEAR, 20 FRONT_HAZARD, 20 REAR_HAZARD):
- NO_COMM: Blind driving (always NONE=0)
- FULL_COMM: Always requests communication / informed on every hazard
- ORACLE_SELECTIVE: Asks FRONT in Front-Hazard, REAR in Rear-Hazard, NONE in Clear
- RANDOM_REQUEST: Randomly chooses NONE (0), ASK_FRONT (1), or ASK_REAR (2)
"""

from __future__ import annotations

import os
import json
import time
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

# Message semantics:
# 0: UNKNOWN (default when request is NONE)
# 1: CLEAR (no hazard detected in that sector)
# 2: HAZARD_AHEAD (detected by Front CAV: stalled obstacle downstream, ego must double lane-change or yield)
# 3: HAZARD_BEHIND (detected by Rear CAV: fast speeder approaching, ego must yield on ramp)


class CooperativeOcclusionGymEnv(EgoMergeGymEnv):
    """Isolated true-simulator environment with cooperative occlusion sensing."""

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

        # Ego starts on merge lane at s=5.0 (x=235.0), speed=20.0 m/s
        self.vehicle = self.action_type.vehicle_class(
            road, merge_lane.position(5.0, 0.0), speed=20.0
        )
        road.vehicles.append(self.vehicle)

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Front Connected CAV at x=265.0 (s=35.0 on target lane)
        front_spd = 8.0 if self.hazard_state == 1 else 22.0
        self.front_vehicle = vehicle_type(road, target_lane.position(35.0, 0.0), speed=front_spd)
        self.front_vehicle.target_speed = front_spd
        road.vehicles.append(self.front_vehicle)

        # Rear Connected CAV at x=225.0 on ab_lane
        # In REAR_HAZARD: Rear CAV charges at 30 m/s
        rear_spd = 30.0 if self.hazard_state == 2 else 20.0
        self.rear_vehicle = vehicle_type(road, ab_lane.position(225.0, 0.0), speed=rear_spd)
        self.rear_vehicle.target_speed = rear_spd
        road.vehicles.append(self.rear_vehicle)

        # Occluded downstream obstacle at x=285.0 in FRONT_HAZARD
        if self.hazard_state == 1:
            self.hazard_vehicle = vehicle_type(road, target_lane.position(55.0, 0.0), speed=0.0)
            road.vehicles.append(self.hazard_vehicle)
        else:
            self.hazard_vehicle = None

    def _is_merged(self) -> bool:
        if self.vehicle is None:
            return False
        lane = self.vehicle.lane_index
        # Merged when safely entered highway lanes 0 or 1 (off ramp lane 2)
        return lane[0] in {"b", "c", "d"} and lane[2] < 2

    def _is_terminated(self) -> bool:
        if self.vehicle is None:
            return False
        if self.vehicle.crashed:
            return True
        # Terminate when ego has safely merged and traversed past the merge/hazard zone (x > 320.0)
        return bool(self._is_merged() and self.vehicle.position[0] > 320.0)


if "coop-occlusion-v0" not in gym.registry:
    gym.register(
        id="coop-occlusion-v0",
        entry_point="request_agent.true_simulator_occlusion_test:CooperativeOcclusionGymEnv",
    )


class CooperativeOcclusionEnvWrapper:
    """Wrapper isolating communication channel and true simulator physics."""

    NONE = 0
    ASK_FRONT = 1
    ASK_REAR = 2

    MSG_UNKNOWN = 0
    MSG_CLEAR = 1
    MSG_HAZARD_AHEAD = 2
    MSG_HAZARD_BEHIND = 3

    def __init__(self, request_cost: float = 0.05) -> None:
        self.request_cost = float(request_cost)
        cfg = Config.load("configs/cpu_fast.yaml")
        env_cfg = cfg.env.copy() if hasattr(cfg.env, "copy") else cfg.env
        env_cfg.id = "coop-occlusion-v0"
        env_cfg.lanes_count = 2
        env_cfg.vehicles_count = 2
        self.env = make_env(env_cfg)

        self.hazard_state = 0  # 0: CLEAR, 1: FRONT_HAZARD, 2: REAR_HAZARD
        self.request_count = 0
        self.total_bits = 0

    def reset(self, seed: int) -> tuple[np.ndarray, dict]:
        # Balanced 3-way split: 0: CLEAR, 1: FRONT_HAZARD, 2: REAR_HAZARD
        self.hazard_state = seed % 3
        self.env.unwrapped.hazard_state = self.hazard_state
        obs, info = self.env.reset(seed=seed)

        self.request_count = 0
        self.total_bits = 0
        return obs, info

    def get_message(self, request_action: int) -> int:
        req = int(request_action)
        if req == self.NONE:
            return self.MSG_UNKNOWN

        if req == self.ASK_FRONT:
            if self.hazard_state == 1:
                return self.MSG_HAZARD_AHEAD  # Front CAV warns: obstacle ahead in Lane 1!
            else:
                return self.MSG_CLEAR         # Front CAV reports: forward sector clear

        if req == self.ASK_REAR:
            if self.hazard_state == 2:
                return self.MSG_HAZARD_BEHIND # Rear CAV warns: high-speed surge approaching behind!
            else:
                return self.MSG_CLEAR         # Rear CAV reports: rear sector clear

        return self.MSG_UNKNOWN

    def step(self, driving_action: int, request_action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        msg = self.get_message(request_action)
        obs, raw_r, term, trunc, info = self.env.step(int(driving_action))

        comm_cost = self.request_cost if request_action != self.NONE else 0.0
        final_utility = float(raw_r) - comm_cost

        if request_action != self.NONE:
            self.request_count += 1
            self.total_bits += 4

        s_info = {
            "message": msg,
            "request_action": request_action,
            "raw_driving_reward": float(raw_r),
            "communication_cost": comm_cost,
            "crashed": bool(self.env.unwrapped.vehicle.crashed),
            "is_merged": bool(self.env.unwrapped._is_merged()),
            "hazard_state": self.hazard_state,
        }
        return obs, final_utility, term, trunc, s_info


def run_oracle_evaluation(num_seeds: int = 60, base_seed: int = 7000) -> dict:
    print("=" * 88)
    print(f"RUNNING TRUE-SIMULATOR OCCLUSION ORACLE TEST ({num_seeds} seeds, base={base_seed})")
    print("=" * 88)

    env = CooperativeOcclusionEnvWrapper(request_cost=0.05)
    policies = ["NO_COMM", "FULL_COMM", "ORACLE_SELECTIVE", "RANDOM_REQUEST"]
    results = {p: [] for p in policies}

    for idx in range(num_seeds):
        seed = base_seed + idx
        hazard_type = seed % 3
        h_name = {0: "CLEAR", 1: "FRONT_HAZARD", 2: "REAR_HAZARD"}[hazard_type]

        for pol in policies:
            obs, info = env.reset(seed=seed)
            ep_utility = 0.0
            ep_driving_r = 0.0
            crashed = False
            merged = False
            step = 0
            req_count = 0
            revealed_info = env.MSG_UNKNOWN

            done = False
            while not done and step < 8:
                # 1. Determine communication request
                if pol == "NO_COMM":
                    req = env.NONE
                elif pol == "FULL_COMM":
                    # Full comm: asks both or the relevant vehicle at t=0
                    if step == 0:
                        req = env.ASK_FRONT if hazard_type == 1 else (env.ASK_REAR if hazard_type == 2 else env.ASK_FRONT)
                    else:
                        req = env.NONE
                elif pol == "ORACLE_SELECTIVE":
                    # Selective: queries FRONT if Front-Hazard, REAR if Rear-Hazard, NONE if Clear
                    if step == 0:
                        if hazard_type == 1:
                            req = env.ASK_FRONT
                        elif hazard_type == 2:
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
                    revealed_info = msg

                # 2. Driving action conditioned on received information:
                # - MSG_HAZARD_AHEAD: Double Lane Change into Lane 0 (bypassing obstacle in Lane 1)
                # - MSG_HAZARD_BEHIND: Yield on ramp, then merge behind speeder
                # - CLEAR / UNKNOWN: Standard blind merge (LANE_LEFT at t=0, then IDLE in Lane 1)
                if revealed_info == env.MSG_HAZARD_AHEAD:
                    if step == 0:
                        drive_act = 0  # LANE_LEFT
                    elif step == 1:
                        drive_act = 0  # LANE_LEFT (crosses into safe Lane 0)
                    else:
                        drive_act = 1  # IDLE cruising in Lane 0
                elif revealed_info == env.MSG_HAZARD_BEHIND:
                    if step == 0:
                        drive_act = 4  # SLOWER (yield on ramp)
                    elif step == 1:
                        drive_act = 1  # IDLE (speeder zooms past in Lane 1)
                    elif step == 2:
                        drive_act = 0  # LANE_LEFT (merge safely behind speeder)
                    else:
                        drive_act = 1  # IDLE cruising
                else:
                    # Blind default merge: LANE_LEFT at t=0, then IDLE
                    if step == 0:
                        drive_act = 0  # LANE_LEFT
                    else:
                        drive_act = 1  # IDLE (stays in Lane 1)

                next_obs, u_reward, term, trunc, s_info = env.step(drive_act, req)
                ep_utility += u_reward
                ep_driving_r += s_info["raw_driving_reward"]
                done = bool(term or trunc)

                if s_info["crashed"]:
                    crashed = True
                if s_info["is_merged"]:
                    merged = True
                step += 1

            results[pol].append({
                "seed": seed,
                "hazard_state": h_name,
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

    # Intent Breakdown
    print("\n" + "=" * 88)
    print("BREAKDOWN BY HAZARD STATE (Safe Merge Rate):")
    print("=" * 88)
    print(f"{'Hazard State':<18} | {'NO_COMM':<10} | {'FULL_COMM':<10} | {'ORACLE_SELECTIVE':<16} | {'RANDOM_REQUEST'}")
    print("-" * 88)
    for h_name in ["CLEAR", "FRONT_HAZARD", "REAR_HAZARD"]:
        sm_nc = float(np.mean([r["safe_merge"] for r in results["NO_COMM"] if r["hazard_state"] == h_name]) * 100.0)
        sm_fc = float(np.mean([r["safe_merge"] for r in results["FULL_COMM"] if r["hazard_state"] == h_name]) * 100.0)
        sm_os = float(np.mean([r["safe_merge"] for r in results["ORACLE_SELECTIVE"] if r["hazard_state"] == h_name]) * 100.0)
        sm_rr = float(np.mean([r["safe_merge"] for r in results["RANDOM_REQUEST"] if r["hazard_state"] == h_name]) * 100.0)
        print(f"{h_name:<18} | {sm_nc:>8.1f}%  | {sm_fc:>8.1f}%  | {sm_os:>14.1f}%   | {sm_rr:>12.1f}%")

    out_path = Path("outputs/request_agent/occlusion_oracle_diagnostic_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, default=str)
    print(f"\nSaved diagnostic results to {out_path}")
    return summary


def run_leakage_verification() -> bool:
    print("\n" + "=" * 85)
    print("RUNNING PIXEL OBSERVATION LEAKAGE VERIFICATION TEST AT t=0")
    print("=" * 85)

    env = CooperativeOcclusionEnvWrapper()
    diffs = []
    for test_seed in [100, 200, 300, 400, 500]:
        env.hazard_state = 0
        env.env.unwrapped.hazard_state = 0
        obs0, _ = env.env.reset(seed=test_seed)

        env.hazard_state = 1
        env.env.unwrapped.hazard_state = 1
        obs1, _ = env.env.reset(seed=test_seed)

        env.hazard_state = 2
        env.env.unwrapped.hazard_state = 2
        obs2, _ = env.env.reset(seed=test_seed)

        d01 = float(np.max(np.abs(obs0.astype(float) - obs1.astype(float))))
        d02 = float(np.max(np.abs(obs0.astype(float) - obs2.astype(float))))
        diffs.append((d01, d02))
        print(f"Seed {test_seed:3d}: max_pixel_diff(CLEAR, FRONT_HAZARD) = {d01:.1f}, max_pixel_diff(CLEAR, REAR_HAZARD) = {d02:.1f}")

    all_zero = all(d01 == 0.0 and d02 == 0.0 for d01, d02 in diffs)
    print(f"Leakage Test Verdict: {'PASSED (Zero Leakage, Byte-for-byte Identical)' if all_zero else 'FAILED (Information Leaked in Pixels)'}")
    return all_zero


if __name__ == "__main__":
    leakage_passed = run_leakage_verification()
    oracle_summary = run_oracle_evaluation(num_seeds=60, base_seed=7000)

    print("\n" + "=" * 88)
    print("OCCLUSION ORACLE GATE CRITERIA CHECK:")
    print("=" * 88)
    sm_nc = oracle_summary["NO_COMM"]["safe_merge_rate"]
    sm_fc = oracle_summary["FULL_COMM"]["safe_merge_rate"]
    sm_os = oracle_summary["ORACLE_SELECTIVE"]["safe_merge_rate"]
    sm_rr = oracle_summary["RANDOM_REQUEST"]["safe_merge_rate"]

    c1 = (sm_nc < 50.0)
    c2 = (sm_fc > 95.0)
    c3 = (sm_os > 95.0)
    c4 = (sm_nc < sm_rr < sm_fc)
    c5 = leakage_passed

    print(f"1. Blind Driving Safe Merge < 50%?          {sm_nc:.1f}% -> {'PASS' if c1 else 'FAIL'}")
    print(f"2. Full Comm Safe Merge > 95%?              {sm_fc:.1f}% -> {'PASS' if c2 else 'FAIL'}")
    print(f"3. Oracle Selective Safe Merge > 95%?       {sm_os:.1f}% -> {'PASS' if c3 else 'FAIL'}")
    print(f"4. Ordering: NO_COMM < RANDOM < FULL_COMM?  {sm_nc:.1f}% < {sm_rr:.1f}% < {sm_fc:.1f}% -> {'PASS' if c4 else 'FAIL'}")
    print(f"5. Observation Leakage = 0 (Zero Diff)?     {'PASS' if c5 else 'FAIL'}")

    passed = c1 and c2 and c3 and c4 and c5
    print("-" * 88)
    if passed:
        print("FINAL VERDICT: OCCLUSION ORACLE GATE PASSED — READY TO BUILD FINAL PIPELINE")
    else:
        print("FINAL VERDICT: OCCLUSION ORACLE GATE FAILED — DO NOT IMPLEMENT")
    print("=" * 88)
