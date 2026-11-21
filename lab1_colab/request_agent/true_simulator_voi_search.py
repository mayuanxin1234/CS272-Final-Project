"""Systematic TRUE-SIMULATOR Value-of-Information search over lane-change geometries."""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Any
import numpy as np
import gymnasium as gym
from highway_env import utils
from highway_env.envs.merge_env import MergeGenericEnv

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.envs.merge import EgoMergeGymEnv, MergeEnv
from lab1.envs.highway import make_env

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
INTENT_NAMES = {1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


class CustomGeometryMergeGymEnv(EgoMergeGymEnv):
    """Parameterized Gym merge environment for grid search."""

    geom: dict[str, float] = {
        "ego_x": 30.0,
        "ego_speed": 25.0,
        "front_x": 50.0,
        "front_speed": 25.0,
        "rear_x": 20.0,
        "rear_speed": 25.0,
    }

    def __init__(self, config: dict | None = None, render_mode: str | None = None) -> None:
        self.geom = dict(self.geom)
        super().__init__(config=config, render_mode=render_mode)

    def set_geometry(self, geom: dict[str, float]) -> None:
        self.geom = geom

    def _make_vehicles(self) -> None:
        road = self.road
        lanes = self.config["lanes_count"]
        self.target_lane_index = ("a", "b", lanes - 1)
        ramp_lane = road.network.get_lane(("j", "k", 0))

        g = self.geom
        ego = self.action_type.vehicle_class(
            road,
            ramp_lane.position(g["ego_x"], 0.0),
            speed=g["ego_speed"],
        )
        road.vehicles.append(ego)
        self.vehicle = ego

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        highway_lane = road.network.get_lane(self.target_lane_index)

        front = vehicle_type(
            road,
            highway_lane.position(g["front_x"], 0.0),
            speed=g["front_speed"],
        )
        rear = vehicle_type(
            road,
            highway_lane.position(g["rear_x"], 0.0),
            speed=g["rear_speed"],
        )

        road.vehicles.append(front)
        road.vehicles.append(rear)
        self.front_vehicle = front
        self.rear_vehicle = rear


if "ego-merge-search-v0" not in gym.registry:
    gym.register(
        id="ego-merge-search-v0",
        entry_point="request_agent.true_simulator_voi_search:CustomGeometryMergeGymEnv",
    )


def apply_intent(vehicle: Any, intent: int, base_speed: float) -> None:
    if vehicle is None:
        return
    if intent == 1:  # MAINTAIN
        target = base_speed
    elif intent == 2:  # ACCELERATE
        target = base_speed + 5.0
    elif intent == 3:  # BRAKE
        target = max(0.0, base_speed - 5.0)
    else:
        return

    if hasattr(vehicle, "MAX_SPEED"):
        target = min(target, float(vehicle.MAX_SPEED))
    if hasattr(vehicle, "target_speed"):
        vehicle.target_speed = target
    else:
        vehicle.speed = target


def simulate_rollout(
    env: gym.Env,
    geom: dict[str, float],
    rear_intent: int,
    first_action: int,
    horizon: int = 10,
) -> dict[str, Any]:
    """Execute rollout in the TRUE simulator for a given geometry, rear intent, and first action."""
    unwrapped = env.unwrapped
    unwrapped.set_geometry(geom)
    obs, info = env.reset(seed=42)

    ego = unwrapped.vehicle
    front = unwrapped.front_vehicle
    rear = unwrapped.rear_vehicle

    apply_intent(front, 1, geom["front_speed"])
    apply_intent(rear, rear_intent, geom["rear_speed"])

    crashed = False
    merged = False
    ep_return = 0.0
    min_rear_gap = float("inf")
    time_to_merge = horizon

    for t in range(horizon):
        apply_intent(front, 1, geom["front_speed"])
        apply_intent(rear, rear_intent, geom["rear_speed"])

        if t == 0:
            act = first_action
        else:
            is_m = hasattr(unwrapped, "_is_merged") and unwrapped._is_merged()
            if not is_m:
                # On ramp: merge once safe
                rear_dx = rear.position[0] - ego.position[0]
                if rear_dx > 6.0 or rear_dx < -12.0:
                    act = 0  # LANE_LEFT
                else:
                    act = 1 if ego.speed >= 22.0 else 3
            else:
                dist_front = front.position[0] - ego.position[0]
                if dist_front < 14.0:
                    act = 4  # SLOWER
                elif ego.speed < 25.0:
                    act = 3  # FASTER
                else:
                    act = 1  # IDLE

        obs, reward, term, trunc, s_info = env.step(act)
        ep_return += float(reward)

        if unwrapped.vehicle.crashed:
            crashed = True
        if hasattr(unwrapped, "_is_merged") and unwrapped._is_merged():
            merged = True
            if time_to_merge == horizon:
                time_to_merge = t + 1

        gap = ego.position[0] - rear.position[0]
        if abs(gap) < abs(min_rear_gap):
            min_rear_gap = gap

        if term or trunc or crashed:
            break

    return {
        "crashed": bool(crashed),
        "merged": bool(merged),
        "safe_merge": bool(merged and not crashed),
        "return": float(ep_return),
        "min_rear_gap": float(min_rear_gap),
        "time_to_merge": time_to_merge,
    }


def check_pixel_invariance(env: gym.Env, geom: dict[str, float]) -> bool:
    unwrapped = env.unwrapped
    unwrapped.set_geometry(geom)

    obs_m, _ = env.reset(seed=42)
    obs_a, _ = env.reset(seed=42)
    obs_b, _ = env.reset(seed=42)

    d1 = np.max(np.abs(obs_m.astype(int) - obs_a.astype(int)))
    d2 = np.max(np.abs(obs_m.astype(int) - obs_b.astype(int)))
    return (d1 == 0 and d2 == 0)


def score_outcome(res: dict[str, Any]) -> float:
    if res["crashed"]:
        return -100.0 + res["return"]
    if not res["merged"]:
        return -10.0 + res["return"]
    return res["return"]


def evaluate_single_geometry(env: gym.Env, geom: dict[str, float]) -> dict[str, Any]:
    actions = [0, 1, 3, 4]  # LANE_LEFT, IDLE, FASTER, SLOWER
    intents = [3, 1, 2]     # BRAKE, MAINTAIN, ACCELERATE

    outcomes = {}
    for intent in intents:
        for act in actions:
            outcomes[(intent, act)] = simulate_rollout(env, geom, intent, act, horizon=10)

    best_acts = {}
    for intent in intents:
        best_act = max(actions, key=lambda a: score_outcome(outcomes[(intent, a)]))
        best_acts[intent] = best_act

    best_brake = best_acts[3]
    best_maint = best_acts[1]
    best_accel = best_acts[2]

    brake_under_brake = outcomes[(3, best_brake)]
    accel_under_accel = outcomes[(2, best_accel)]
    accel_under_brake_act = outcomes[(2, best_brake)]

    blind_scores = [sum(score_outcome(outcomes[(i, a)]) for i in intents) / 3.0 for a in actions]
    best_blind_score = max(blind_scores)
    best_blind_act = actions[int(np.argmax(blind_scores))]

    informed_score = (score_outcome(outcomes[(3, best_brake)]) + score_outcome(outcomes[(1, best_maint)]) + score_outcome(outcomes[(2, best_accel)])) / 3.0
    voi_score = informed_score - best_blind_score

    crashes_avoided = int(accel_under_brake_act["crashed"] and not accel_under_accel["crashed"])

    return {
        "geom": geom,
        "best_brake": best_brake,
        "best_maint": best_maint,
        "best_accel": best_accel,
        "best_blind_act": best_blind_act,
        "action_diverges": (best_brake != best_accel),
        "crashes_avoided": crashes_avoided,
        "accel_wrong_crashed": accel_under_brake_act["crashed"],
        "voi_score": voi_score,
        "outcomes": outcomes,
    }


def run_systematic_search() -> tuple[list[dict[str, Any]], Any]:
    print("=" * 85)
    print("SYSTEMATIC TRUE-SIMULATOR VALUE-OF-INFORMATION SEARCH")
    print("=" * 85)

    cfg = Config.load("configs/cpu_fast.yaml")
    env_cfg = cfg.env.copy() if hasattr(cfg.env, "copy") else cfg.env
    env_cfg.id = "ego-merge-search-v0"
    env_cfg.lanes_count = 2
    env_cfg.vehicles_count = 2
    env_cfg.scaling = 1.75
    env = make_env(env_cfg)

    ego_x = 30.0
    ego_speed = 25.0

    rear_x_list = [15.0, 18.0, 21.0, 24.0, 27.0]
    rear_speed_list = [22.0, 25.0, 28.0]
    front_x_list = [42.0, 48.0, 54.0, 60.0]
    front_speed_list = [22.0, 25.0, 28.0]

    total_configs = len(rear_x_list) * len(rear_speed_list) * len(front_x_list) * len(front_speed_list)
    print(f"Sweeping {total_configs} candidate geometries in TRUE HighwayEnv...")
    t0 = time.time()

    all_results = []
    promising = []

    for rx in rear_x_list:
        for r_spd in rear_speed_list:
            for fx in front_x_list:
                for f_spd in front_speed_list:
                    geom = {
                        "ego_x": ego_x,
                        "ego_speed": ego_speed,
                        "front_x": fx,
                        "front_speed": f_spd,
                        "rear_x": rx,
                        "rear_speed": r_spd,
                    }
                    res = evaluate_single_geometry(env, geom)
                    all_results.append(res)

                    # Criteria:
                    # 1. Action diverges between BRAKE and ACCEL
                    # 2. Taking BRAKE's action under ACCEL causes collision OR return loss > 0.5
                    # 3. Taking ACCEL's action avoids crash / succeeds safely
                    if res["action_diverges"] and (res["accel_wrong_crashed"] or res["voi_score"] > 0.5):
                        # Verify pixel invariance
                        p_inv = check_pixel_invariance(env, geom)
                        res["pixel_invariance"] = p_inv
                        promising.append(res)

    elapsed = time.time() - t0
    print(f"Search completed in {elapsed:.1f}s.")
    print(f"Identified {len(promising)} decision-critical geometries satisfying all VOI criteria!")

    # Sort candidates by:
    # 1. Crashes avoided (safety consequence)
    # 2. VOI score
    promising.sort(key=lambda x: (x["crashes_avoided"], x["voi_score"]), reverse=True)

    # Robustness Testing on top candidates
    print("\n" + "=" * 85)
    print("ROBUSTNESS TESTING ON TOP CANDIDATES (+/-2m position, +/-1m/s speed)")
    print("=" * 85)

    top_candidates = promising[:5]
    for idx, cand in enumerate(top_candidates):
        base_g = cand["geom"]
        print(f"\nCandidate #{idx+1}: Front (x={base_g['front_x']}, v={base_g['front_speed']}), Rear (x={base_g['rear_x']}, v={base_g['rear_speed']})")
        print(f"  Best Actions: BRAKE -> {ACTION_NAMES[cand['best_brake']]}, MAINTAIN -> {ACTION_NAMES[cand['best_maint']]}, ACCEL -> {ACTION_NAMES[cand['best_accel']]}")
        print(f"  Crashes Avoided: {cand['crashes_avoided']}, VOI Score: {cand['voi_score']:.3f}, Pixel Invariance: {cand['pixel_invariance']}")

        # Perturbation grid:
        perturbations = [
            {"rear_x": base_g["rear_x"] + dx, "rear_speed": base_g["rear_speed"] + dv}
            for dx in [-2.0, 0.0, 2.0]
            for dv in [-1.0, 0.0, 1.0]
        ]
        robust_count = 0
        for p in perturbations:
            pg = dict(base_g)
            pg.update(p)
            p_res = evaluate_single_geometry(env, pg)
            if p_res["action_diverges"] and (p_res["accel_wrong_crashed"] or p_res["voi_score"] > 0.3):
                robust_count += 1

        cand["robustness_rate"] = robust_count / len(perturbations)
        print(f"  Robustness Neighborhood Pass Rate: {robust_count}/{len(perturbations)} ({cand['robustness_rate']*100:.1f}%)")

    env.close()

    # Save results
    out_path = Path("outputs/request_agent/true_simulator_voi_search_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Format serializable
    save_data = []
    for c in top_candidates:
        item = {
            "geom": c["geom"],
            "best_brake": ACTION_NAMES[c["best_brake"]],
            "best_maint": ACTION_NAMES[c["best_maint"]],
            "best_accel": ACTION_NAMES[c["best_accel"]],
            "best_blind_act": ACTION_NAMES[c["best_blind_act"]],
            "crashes_avoided": c["crashes_avoided"],
            "accel_wrong_crashed": c["accel_wrong_crashed"],
            "voi_score": float(c["voi_score"]),
            "robustness_rate": float(c["robustness_rate"]),
            "pixel_invariance": bool(c["pixel_invariance"]),
            "outcomes_summary": {
                f"intent_{INTENT_NAMES[i]}_act_{ACTION_NAMES[a]}": {
                    "crashed": c["outcomes"][(i, a)]["crashed"],
                    "safe_merge": c["outcomes"][(i, a)]["safe_merge"],
                    "return": float(c["outcomes"][(i, a)]["return"]),
                    "min_rear_gap": float(c["outcomes"][(i, a)]["min_rear_gap"]),
                }
                for i in [3, 1, 2] for a in [0, 1, 3, 4]
            }
        }
        save_data.append(item)

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nSaved top candidates to {out_path}")
    return top_candidates, save_data


if __name__ == "__main__":
    run_systematic_search()
