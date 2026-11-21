"""True-Simulator Task Reward Investigation for Candidate #2.

Compares task-reward variants using true-simulator Bayesian tree-search / MPC:
- NO_COMM: planner marginalizes over unknown rear intents {BRAKE, MAINTAIN, ACCEL} (1/3 each).
- ORACLE_REAR: planner plans with true rear intent.
- ORACLE_FRONT: planner receives front intent (MAINTAIN), rear remains unknown.
"""

from __future__ import annotations

import os
import copy
import json
import time
from pathlib import Path
from collections import Counter
from typing import Any, Callable
import numpy as np
import gymnasium as gym

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.envs.highway import make_env
from request_agent.final_candidate2_env import FinalCandidate2GymEnv, FinalCandidate2MergeEnv

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def compute_reward_baseline(env_unwrapped: Any, action: int) -> float:
    """Current EgoMergeGymEnv baseline reward."""
    vehicle = env_unwrapped.vehicle
    if vehicle.crashed:
        return -1.0
    speed_reward = np.clip(vehicle.speed / 30.0, 0.0, 1.0)
    is_merged = bool(env_unwrapped._is_merged()) if hasattr(env_unwrapped, "_is_merged") else False
    merge_reward = 1.0 if is_merged else 0.0
    reward = 0.45 * speed_reward + 0.55 * merge_reward
    if is_merged and action in [0, 2]:
        reward -= 0.02
    return float(reward)


def compute_reward_variant_a(env_unwrapped: Any, action: int, coef_effort: float = 0.06) -> float:
    """Variant A: Flow Speed Alignment (22 m/s) + Comfort/Control Effort Penalty.
    
    R = -1.0 (crash)
      + 0.55 * is_merged
      + 0.45 * flow_alignment (maximized at 22 m/s, highway speed)
      - coef_effort * (action in [3, 4])  # throttle/brake deviation penalty
      - 0.02 * (is_merged and action in [0, 2])
    """
    vehicle = env_unwrapped.vehicle
    if vehicle.crashed:
        return -1.0
    is_merged = bool(env_unwrapped._is_merged()) if hasattr(env_unwrapped, "_is_merged") else False
    merge_reward = 1.0 if is_merged else 0.0
    # Flow alignment around 22 m/s (highway speed)
    flow_reward = np.clip(1.0 - abs(vehicle.speed - 22.0) / 10.0, 0.0, 1.0)
    
    reward = 0.55 * merge_reward + 0.45 * flow_reward
    # Symmetrical comfort / energy penalty for active throttle / braking
    if action in [3, 4]:
        reward -= coef_effort
    if is_merged and action in [0, 2]:
        reward -= 0.02
    return float(reward)


def compute_reward_variant_b(env_unwrapped: Any, action: int, coef_effort: float = 0.08) -> float:
    """Variant B: Merge Completion Incentive + Symmetrical Jerk/Effort Penalty.
    
    R = -1.0 (crash)
      + 0.60 * is_merged
      + 0.40 * speed_progress (v / 25 capped at 1.0)
      - coef_effort * (action in [3, 4])
      - 0.02 * (is_merged and action in [0, 2])
    """
    vehicle = env_unwrapped.vehicle
    if vehicle.crashed:
        return -1.0
    is_merged = bool(env_unwrapped._is_merged()) if hasattr(env_unwrapped, "_is_merged") else False
    merge_reward = 1.0 if is_merged else 0.0
    speed_reward = np.clip(vehicle.speed / 25.0, 0.0, 1.0)
    
    reward = 0.60 * merge_reward + 0.40 * speed_reward
    if action in [3, 4]:
        reward -= coef_effort
    if is_merged and action in [0, 2]:
        reward -= 0.02
    return float(reward)


def compute_reward_variant_c(env_unwrapped: Any, action: int, coef_effort: float = 0.05) -> float:
    """Variant C: Quadratic Speed Tracking around Flow Speed (22 m/s) + Action Penalty.
    
    R = -1.0 (crash)
      + 0.60 * is_merged
      + 0.40 * exp(-0.5 * ((v - 22) / 4.0)^2)
      - coef_effort * (action in [3, 4])
      - 0.02 * (is_merged and action in [0, 2])
    """
    vehicle = env_unwrapped.vehicle
    if vehicle.crashed:
        return -1.0
    is_merged = bool(env_unwrapped._is_merged()) if hasattr(env_unwrapped, "_is_merged") else False
    merge_reward = 1.0 if is_merged else 0.0
    speed_error = (vehicle.speed - 22.0) / 4.0
    speed_reward = float(np.exp(-0.5 * speed_error ** 2))
    
    reward = 0.60 * merge_reward + 0.40 * speed_reward
    if action in [3, 4]:
        reward -= coef_effort
    if is_merged and action in [0, 2]:
        reward -= 0.02
    return float(reward)


def save_env_state(u: Any) -> dict[str, Any]:
    veh_states = []
    for v in u.road.vehicles:
        veh_states.append({
            "pos": np.copy(v.position),
            "spd": float(v.speed),
            "target_spd": float(getattr(v, "target_speed", v.speed)),
            "heading": float(v.heading),
            "lane_index": v.lane_index,
            "crashed": bool(v.crashed),
            "action": copy.deepcopy(v.action) if hasattr(v, "action") and isinstance(v.action, dict) else getattr(v, "action", None),
        })
    return {
        "time": float(u.time),
        "steps": int(getattr(u, "steps", 0)),
        "done": bool(getattr(u, "done", False)),
        "vehs": veh_states,
    }


def restore_env_state(u: Any, state: dict[str, Any]) -> None:
    u.time = state["time"]
    if hasattr(u, "steps"):
        u.steps = state["steps"]
    if hasattr(u, "done"):
        u.done = state["done"]
    for v, s in zip(u.road.vehicles, state["vehs"]):
        v.position = np.copy(s["pos"])
        v.speed = s["spd"]
        if hasattr(v, "target_speed"):
            v.target_speed = s["target_spd"]
        v.heading = s["heading"]
        v.lane_index = s["lane_index"]
        v.crashed = s["crashed"]
        if hasattr(v, "action") and s["action"] is not None:
            v.action = copy.deepcopy(s["action"]) if isinstance(s["action"], dict) else s["action"]


def simulate_plan_step(
    base_env: Any,
    hypothetical_rear_intent: int,
    candidate_action: int,
    reward_fn: Callable[[Any, int], float],
    horizon: int = 5,
) -> float:
    """Simulate a short rollout in a copied environment under hypothetical rear intent."""
    saved_state = save_env_state(base_env)

    # Set hypothetical rear intent target speed
    if hypothetical_rear_intent == 3:  # BRAKE
        rear_target_spd = 17.0
    elif hypothetical_rear_intent == 2:  # ACCEL
        rear_target_spd = 27.0
    else:  # MAINTAIN
        rear_target_spd = 22.0

    cum_reward = 0.0

    for t in range(horizon):
        act = candidate_action if t == 0 else (0 if not base_env._is_merged() else 1)
        
        # Apply intent speed adjustments
        base_env.front_vehicle.speed = 22.0
        if hasattr(base_env.front_vehicle, "target_speed"):
            base_env.front_vehicle.target_speed = 22.0
        base_env.rear_vehicle.target_speed = rear_target_spd

        # Advance environment
        obs, raw_r, term, trunc, info = base_env.step(act)
        r = reward_fn(base_env, act)
        cum_reward += r

        if base_env.vehicle.crashed or term or trunc:
            break

    # Restore exact environment state
    restore_env_state(base_env, saved_state)
    return cum_reward


def run_reward_evaluation(
    reward_fn: Callable[[Any, int], float],
    reward_name: str,
    num_seeds: int = 50,
    base_seed: int = 4000,
) -> dict[str, Any]:
    print("-" * 80)
    print(f"EVALUATING REWARD VARIANT: {reward_name}")
    print("-" * 80)

    cfg = Config.load("configs/cpu_fast.yaml")
    env = FinalCandidate2MergeEnv(cfg, request_cost=0.0)
    allowed_actions = [0, 1, 3, 4]  # LANE_LEFT, IDLE, FASTER, SLOWER
    policies = ["NO_COMM", "ORACLE_REAR", "ORACLE_FRONT"]
    intents_all = [1, 2, 3]  # MAINTAIN, ACCEL, BRAKE

    results = {p: [] for p in policies}

    for idx in range(num_seeds):
        seed = base_seed + idx

        for pol in policies:
            obs, info = env.reset(seed=seed)
            true_rear_intent = env.rear_intent
            true_front_intent = env.front_intent

            ep_return = 0.0
            crashed = False
            merged = False
            step = 0
            done = False
            first_action = None
            action_counts = Counter()

            while not done and step < 15:
                # Plan action
                best_act = None
                best_val = -float("inf")

                for act in allowed_actions:
                    if pol == "ORACLE_REAR":
                        # Perfect knowledge of rear intent
                        val = simulate_plan_step(env.env.unwrapped, true_rear_intent, act, reward_fn, horizon=5)
                    elif pol == "NO_COMM" or pol == "ORACLE_FRONT":
                        # Unknown rear intent: Expected return under uniform prior (1/3 each)
                        val = sum(simulate_plan_step(env.env.unwrapped, i, act, reward_fn, horizon=5) for i in intents_all) / 3.0

                    if val > best_val:
                        best_val = val
                        best_act = act

                if step == 0:
                    first_action = best_act
                action_counts[best_act] += 1

                # Execute in true environment
                next_obs, raw_r, term, trunc, s_info = env.step(best_act, 0)
                r = reward_fn(env.env.unwrapped, best_act)
                ep_return += r
                done = bool(term or trunc)

                if s_info["crashed"]:
                    crashed = True
                if s_info["is_merged"]:
                    merged = True
                step += 1

            results[pol].append({
                "seed": seed,
                "return": float(ep_return),
                "crashed": bool(crashed),
                "merged": bool(merged),
                "safe_merge": bool(merged and not crashed),
                "first_action": first_action,
                "rear_intent": MSG_NAMES[true_rear_intent],
                "action_counts": dict(action_counts),
            })

    env.close()

    # Summarize
    summary = {}
    no_comm_runs = results["NO_COMM"]
    for pol in policies:
        runs = results[pol]
        returns = [r["return"] for r in runs]
        crashes = [float(r["crashed"]) for r in runs]
        safe_merges = [float(r["safe_merge"]) for r in runs]
        paired_diffs = [r["return"] - nc["return"] for r, nc in zip(runs, no_comm_runs)]
        action_diffs = sum(1 for r, nc in zip(runs, no_comm_runs) if r["first_action"] != nc["first_action"])

        summary[pol] = {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "collision_rate": float(np.mean(crashes)),
            "safe_merge_rate": float(np.mean(safe_merges)),
            "paired_improvement": float(np.mean(paired_diffs)),
            "action_divergence_pct": float(action_diffs / num_seeds * 100.0),
        }

    # Breakdown by intent for ORACLE_REAR vs NO_COMM
    intent_analysis = {}
    for iname in ["BRAKE", "MAINTAIN", "ACCELERATE"]:
        sub_nc = [r for r in no_comm_runs if r["rear_intent"] == iname]
        sub_or = [r for r in results["ORACLE_REAR"] if r["rear_intent"] == iname]
        intent_analysis[iname] = {
            "no_comm_first_actions": Counter(r["first_action"] for r in sub_nc),
            "oracle_first_actions": Counter(r["first_action"] for r in sub_or),
            "no_comm_safe_merge": float(np.mean([r["safe_merge"] for r in sub_nc])),
            "oracle_safe_merge": float(np.mean([r["safe_merge"] for r in sub_or])),
            "no_comm_return": float(np.mean([r["return"] for r in sub_nc])),
            "oracle_return": float(np.mean([r["return"] for r in sub_or])),
        }

    return {"summary": summary, "intent_analysis": intent_analysis, "raw": results}


def main():
    print("=" * 85)
    print("TRUE-SIMULATOR TASK REWARD INVESTIGATION (CANDIDATE #2)")
    print("=" * 85)

    variants = [
        ("BASELINE", compute_reward_baseline),
        ("VARIANT_A_FLOW_ALIGNMENT", compute_reward_variant_a),
        ("VARIANT_B_EFFORT_PENALTY", compute_reward_variant_b),
        ("VARIANT_C_GAUSSIAN_FLOW", compute_reward_variant_c),
    ]

    all_evals = {}
    for name, r_fn in variants:
        t0 = time.time()
        res = run_reward_evaluation(r_fn, name, num_seeds=50)
        all_evals[name] = res
        s = res["summary"]
        print(f"\nResults for {name} ({time.time() - t0:.1f}s):")
        print(f"{'Policy':<14} | {'Mean Ret':<10} | {'Collision':<10} | {'Safe Merge':<10} | {'Paired Δ':<10} | {'Action Shift'}")
        print("-" * 75)
        for p in ["NO_COMM", "ORACLE_REAR", "ORACLE_FRONT"]:
            print(f"{p:<14} | {s[p]['mean_return']:<10.3f} | {s[p]['collision_rate']*100:>6.1f}%    | {s[p]['safe_merge_rate']*100:>6.1f}%    | {s[p]['paired_improvement']:+10.3f} | {s[p]['action_divergence_pct']:>5.1f}%")

    out_p = Path("outputs/request_agent/task_reward_investigation_results.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(all_evals, f, indent=2, default=str)
    print(f"\nSaved all investigation results to {out_p}")


if __name__ == "__main__":
    from collections import Counter
    main()

