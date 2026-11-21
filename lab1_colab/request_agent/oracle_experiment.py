"""Oracle Diagnostic Experiment for the Communication-Efficient World Model Project.

Compares:
1. NO_COMM (NONE + UNKNOWN)
2. ORACLE_FRONT (Free true front intent)
3. ORACLE_REAR (Free true rear intent)
4. ORACLE_BEST (Free upper-bound: best predicted return between front and rear oracle plans)

Evaluates on 50 fixed seeds with identical candidate action sequences per decision step.
NO communication cost penalty is applied to any policy.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from request_agent.communication_wrapper import CommunicationMergeWrapper
from request_agent.mpc_adapter import MergeMPCPlanner

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def run_oracle_experiment(num_seeds: int = 50, max_steps: int = 30) -> dict[str, Any]:
    print("=" * 80)
    print(f"RUNNING ORACLE EXPERIMENT ({num_seeds} FIXED SEEDS)")
    print("Zero communication penalty applied. Testing if true information improves MPC driving.")
    print("=" * 80)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication.npz"
    cfg.data.num_transitions = 5000
    cfg.env.id = "ego-merge-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75
    cfg.training.device = "cpu"

    ckpt_path = Path("outputs/personal_lab1/attempt_01/my_lab1_world_model_v2.pt")
    assert ckpt_path.exists(), f"Missing model: {ckpt_path}"

    env = CommunicationMergeWrapper(
        cfg=cfg,
        model_checkpoint_path=ckpt_path,
        device="cpu",
    )

    mpc = MergeMPCPlanner(
        model=env.model,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=42,
        device="cpu",
        future_mode="none",
    )

    eval_seeds = [4000 + i for i in range(num_seeds)]
    policies = ["NO_COMM", "ORACLE_FRONT", "ORACLE_REAR", "ORACLE_BEST"]

    # Store per-policy, per-seed results: policy -> seed -> stats
    results: dict[str, list[dict[str, Any]]] = {p: [] for p in policies}

    # Tracking intent-specific action switches
    # intent_type -> { "total_decisions": 0, "action_changed": 0 }
    intent_analysis = {
        "front_BRAKE": {"total": 0, "changed": 0},
        "front_ACCELERATE": {"total": 0, "changed": 0},
        "front_MAINTAIN": {"total": 0, "changed": 0},
        "rear_BRAKE": {"total": 0, "changed": 0},
        "rear_ACCELERATE": {"total": 0, "changed": 0},
        "rear_MAINTAIN": {"total": 0, "changed": 0},
    }

    start_time = time.time()

    for idx, seed in enumerate(eval_seeds):
        # We want identical candidate sequences across policies for this seed & step
        step_rng = np.random.default_rng(seed * 777)

        seed_data: dict[str, dict[str, Any]] = {}

        for pol in policies:
            env.reset(seed=seed)
            true_front = env.request_message(1)
            true_rear = env.request_message(2)

            ep_return = 0.0
            crashed = False
            merged = False
            step = 0
            done = False

            # Use local copy of seed_rng state for candidate generation
            pol_rng = np.random.default_rng(seed * 777)

            while not done and step < max_steps:
                obs = env.current_obs
                latent = env.current_latent

                # Candidate pool identical across policies for this step
                cand_indices = pol_rng.integers(0, len(mpc.allowed_actions), size=(mpc.num_sequences, mpc.horizon))
                candidates = mpc.allowed_actions[cand_indices]

                if pol == "NO_COMM":
                    act, pred_ret, _ = mpc.plan(obs, latent, request_action=0, message=0, fixed_candidate_actions=candidates)
                elif pol == "ORACLE_FRONT":
                    act, pred_ret, _ = mpc.plan(obs, latent, request_action=1, message=true_front, fixed_candidate_actions=candidates)
                elif pol == "ORACLE_REAR":
                    act, pred_ret, _ = mpc.plan(obs, latent, request_action=2, message=true_rear, fixed_candidate_actions=candidates)
                elif pol == "ORACLE_BEST":
                    act_f, ret_f, _ = mpc.plan(obs, latent, request_action=1, message=true_front, fixed_candidate_actions=candidates)
                    act_r, ret_r, _ = mpc.plan(obs, latent, request_action=2, message=true_rear, fixed_candidate_actions=candidates)
                    if ret_f >= ret_r:
                        act, pred_ret = act_f, ret_f
                    else:
                        act, pred_ret = act_r, ret_r

                # Step environment: request_action=0 so NO comm penalty is charged
                _, reward, terminated, truncated, info = env.step(driving_action=act, request_action=0, message=0)
                done = bool(terminated or truncated)

                # Raw driving reward
                raw_rew = info["raw_driving_reward"]
                ep_return += raw_rew
                if info["crashed"]:
                    crashed = True
                if info["is_merged"]:
                    merged = True
                step += 1

            seed_data[pol] = {
                "return": ep_return,
                "crashed": crashed,
                "merged": merged,
                "safe_merge": merged and (not crashed),
                "length": step,
                "front_intent": true_front,
                "rear_intent": true_rear,
            }
            results[pol].append(seed_data[pol])

        # Step 0 decision comparison for intent sensitivity analysis
        # Check decision under NO_COMM vs ORACLE on Step 0 of each seed
        env.reset(seed=seed)
        t_front = env.request_message(1)
        t_rear = env.request_message(2)
        base_cands = np.random.default_rng(seed * 777).integers(0, len(mpc.allowed_actions), size=(mpc.num_sequences, mpc.horizon))
        base_cands = mpc.allowed_actions[base_cands]
        a_no, _, _ = mpc.plan(env.current_obs, env.current_latent, 0, 0, fixed_candidate_actions=base_cands)
        a_f, _, _ = mpc.plan(env.current_obs, env.current_latent, 1, t_front, fixed_candidate_actions=base_cands)
        a_r, _, _ = mpc.plan(env.current_obs, env.current_latent, 2, t_rear, fixed_candidate_actions=base_cands)

        # Front intent
        f_key = f"front_{MSG_NAMES[t_front]}"
        intent_analysis[f_key]["total"] += 1
        if a_f != a_no:
            intent_analysis[f_key]["changed"] += 1

        # Rear intent
        r_key = f"rear_{MSG_NAMES[t_rear]}"
        intent_analysis[r_key]["total"] += 1
        if a_r != a_no:
            intent_analysis[r_key]["changed"] += 1

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{num_seeds} episodes ({time.time() - start_time:.1f}s)...")

    env.close()
    elapsed = time.time() - start_time
    print(f"\nExperiment finished in {elapsed:.1f}s ({elapsed/num_seeds:.2f}s per seed).")

    # Aggregate Metrics
    summary = {}
    no_comm_data = results["NO_COMM"]

    for pol in policies:
        data = results[pol]
        returns = [d["return"] for d in data]
        crashes = [float(d["crashed"]) for d in data]
        merges = [float(d["merged"]) for d in data]
        safe_merges = [float(d["safe_merge"]) for d in data]
        lengths = [d["length"] for d in data]

        # Paired differences vs NO_COMM
        return_diffs = [d["return"] - n["return"] for d, n in zip(data, no_comm_data)]
        crashes_avoided = sum(1 for d, n in zip(data, no_comm_data) if n["crashed"] and not d["crashed"])
        crashes_introduced = sum(1 for d, n in zip(data, no_comm_data) if not n["crashed"] and d["crashed"])
        merges_gained = sum(1 for d, n in zip(data, no_comm_data) if not n["merged"] and d["merged"])
        merges_lost = sum(1 for d, n in zip(data, no_comm_data) if n["merged"] and not d["merged"])

        summary[pol] = {
            "mean_raw_return": float(np.mean(returns)),
            "std_raw_return": float(np.std(returns)),
            "collision_rate": float(np.mean(crashes)),
            "merge_success_rate": float(np.mean(merges)),
            "safe_merge_rate": float(np.mean(safe_merges)),
            "mean_length": float(np.mean(lengths)),
            "paired_return_improvement": float(np.mean(return_diffs)),
            "paired_return_diff_std": float(np.std(return_diffs)),
            "crashes_avoided": crashes_avoided,
            "crashes_introduced": crashes_introduced,
            "net_collision_reduction": crashes_avoided - crashes_introduced,
            "merges_gained": merges_gained,
            "merges_lost": merges_lost,
        }

    return {
        "num_seeds": num_seeds,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "intent_analysis": intent_analysis,
    }


if __name__ == "__main__":
    out = run_oracle_experiment(num_seeds=50)
    out_path = Path("outputs/request_agent/oracle_experiment_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

