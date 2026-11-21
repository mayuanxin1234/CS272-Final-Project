"""Learned WorldModel MPC Oracle Experiment for Cooperative Occlusion Sensing.

Evaluates the learned communication-aware WorldModel MPC planner across at least 60 held-out seeds.

Policies compared:
1. NO_COMM: Blind driving (always NONE=0, message=UNKNOWN=0)
2. RANDOM_REQUEST: Randomly requests NONE, ASK_FRONT, or ASK_REAR at t=0
3. FULL_COMM: Always requests communication / receives informed hazard message
4. ORACLE_SELECTIVE: Asks FRONT in Front-Hazard, REAR in Rear-Hazard, NONE in Clear

Core safety evaluation uses request_cost=0.0 to measure pure driving capability,
and separately tracks communication requests and bits.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.models import WorldModel
from lab1.utils.checkpoint import load_checkpoint
from request_agent.occlusion_env import FinalOcclusionMergeEnv
from request_agent.mpc_adapter import MergeMPCPlanner

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "CLEAR", 2: "HAZARD_AHEAD", 3: "HAZARD_BEHIND"}
REQ_NAMES = {0: "NONE", 1: "ASK_FRONT", 2: "ASK_REAR"}


def run_learned_mpc_oracle_experiment(
    model_path: str | Path = "outputs/personal_lab1/occlusion_final_attempt_01/my_lab1_world_model_occlusion_final.pt",
    num_seeds: int = 60,
    base_seed: int = 8000,
    num_sequences: int = 64,
    horizon: int = 6,
) -> dict:
    model_path = Path(model_path)
    assert model_path.exists(), f"WorldModel not found at {model_path}"

    print("=" * 88)
    print(f"RUNNING LEARNED-MPC ORACLE EVALUATION ({num_seeds} seeds, base={base_seed})")
    print(f"WorldModel: {model_path}")
    print(f"MPC Horizon: {horizon}, Sequences: {num_sequences}")
    print("=" * 88)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "ego-merge-occlusion-final-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75

    env = FinalOcclusionMergeEnv(cfg, request_cost=0.0)

    # Load WorldModel
    model = WorldModel.from_config(cfg.model, cfg.env, num_actions=5)
    load_checkpoint(model_path, model, map_location="cpu")
    model.eval()

    planner = MergeMPCPlanner(
        model=model,
        num_sequences=num_sequences,
        horizon=horizon,
        allowed_actions=(0, 1, 3, 4),
        seed=base_seed,
        device="cpu",
        future_mode="persistent",
    )

    policies = ["NO_COMM", "FULL_COMM", "ORACLE_SELECTIVE", "RANDOM_REQUEST"]
    results = {p: [] for p in policies}
    action_changes = 0
    total_decision_points = 0

    for idx in range(num_seeds):
        seed = base_seed + idx
        hazard_type = seed % 3
        h_name = {0: "CLEAR", 1: "FRONT_HAZARD", 2: "REAR_HAZARD"}[hazard_type]

        actions_by_policy = {}

        for pol in policies:
            obs, info = env.reset(seed=seed)
            ep_utility = 0.0
            ep_driving_r = 0.0
            crashed = False
            merged = False
            step = 0
            req_count = 0
            revealed_msg = env.MSG_UNKNOWN
            first_action = None

            done = False
            while not done and step < 8:
                # 1. Communication request
                if pol == "NO_COMM":
                    req = env.NONE
                elif pol == "FULL_COMM":
                    if step == 0:
                        req = env.ASK_FRONT if hazard_type == 1 else (env.ASK_REAR if hazard_type == 2 else env.ASK_FRONT)
                    else:
                        req = env.NONE
                elif pol == "ORACLE_SELECTIVE":
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
                    revealed_msg = msg

                # 2. MPC Planning conditioned on (req, revealed_msg)
                # Seed MPC planner reproducibly per step and seed
                planner.rng = np.random.default_rng(seed * 1000 + step)
                drive_act, pred_ret, _ = planner.plan(
                    obs=obs,
                    request_action=req,
                    message=revealed_msg,
                )
                if step == 0:
                    first_action = drive_act

                next_obs, u_reward, term, trunc, s_info = env.step(drive_act, req)
                ep_utility += u_reward
                ep_driving_r += s_info["raw_driving_reward"]
                done = bool(term or trunc)

                if s_info["crashed"]:
                    crashed = True
                if s_info["is_merged"]:
                    merged = True

                obs = next_obs
                step += 1

            actions_by_policy[pol] = first_action
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
                "first_action": first_action,
            })

        # Track action-change frequency at t=0 between NO_COMM and FULL_COMM
        if actions_by_policy["NO_COMM"] != actions_by_policy["FULL_COMM"]:
            action_changes += 1
        total_decision_points += 1

    action_change_freq = (action_changes / total_decision_points) * 100.0

    # Summary table
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
    breakdown = {}
    for h_name in ["CLEAR", "FRONT_HAZARD", "REAR_HAZARD"]:
        sm_nc = float(np.mean([r["safe_merge"] for r in results["NO_COMM"] if r["hazard_state"] == h_name]) * 100.0)
        sm_fc = float(np.mean([r["safe_merge"] for r in results["FULL_COMM"] if r["hazard_state"] == h_name]) * 100.0)
        sm_os = float(np.mean([r["safe_merge"] for r in results["ORACLE_SELECTIVE"] if r["hazard_state"] == h_name]) * 100.0)
        sm_rr = float(np.mean([r["safe_merge"] for r in results["RANDOM_REQUEST"] if r["hazard_state"] == h_name]) * 100.0)
        breakdown[h_name] = {
            "NO_COMM": sm_nc,
            "FULL_COMM": sm_fc,
            "ORACLE_SELECTIVE": sm_os,
            "RANDOM_REQUEST": sm_rr,
        }
        print(f"{h_name:<18} | {sm_nc:>8.1f}%  | {sm_fc:>8.1f}%  | {sm_os:>14.1f}%   | {sm_rr:>12.1f}%")

    print("\n" + "-" * 88)
    print(f"Message Action-Change Frequency at t=0 (NO_COMM vs FULL_COMM): {action_change_freq:.1f}% ({action_changes}/{total_decision_points})")

    out_path = Path("outputs/personal_lab1/occlusion_final_attempt_01/oracle_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "summary": summary,
                "breakdown": breakdown,
                "action_change_frequency": action_change_freq,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved learned-MPC oracle results to {out_path}")
    return summary, breakdown, action_change_freq


if __name__ == "__main__":
    run_learned_mpc_oracle_experiment()

