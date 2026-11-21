"""50-Seed Oracle Evaluation using Safety-Aware World Model and SafetyMergeMPCPlanner."""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from request_agent.tight_merge_env import Candidate4MergeEnv
from request_agent.communication_wrapper import CommunicationMergeWrapper
from request_agent.models.safety_world_model import SafetyWorldModel
from request_agent.safety_mpc_planner import SafetyMergeMPCPlanner

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def run_safety_oracle_experiment(
    num_seeds: int = 50,
    base_seed: int = 4000,
    model_checkpoint_path: Path | str = "outputs/personal_lab1/safety_world_model_attempt_01/my_lab1_safety_world_model.pt",
    val_json_path: Path | str = "outputs/personal_lab1/safety_world_model_attempt_01/validation_results.json",
    out_json_path: Path | str = "outputs/personal_lab1/safety_world_model_attempt_01/safety_oracle_results.json",
    p_crit_override: float | None = None,
) -> dict[str, Any]:
    print("=" * 80)
    print("SAFETY-AWARE ORACLE EXPERIMENT ON CANDIDATE 4 (BALANCED TIGHT SCENARIO)")
    print("=" * 80)

    model_path = Path(model_checkpoint_path)
    assert model_path.exists(), f"Model not found: {model_path}"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Determine p_crit
    if p_crit_override is not None:
        p_crit = float(p_crit_override)
    elif Path(val_json_path).exists():
        with open(val_json_path, "r") as f:
            v_data = json.load(f)
            p_crit = float(v_data.get("calibrated_threshold", 0.40))
    else:
        p_crit = 0.40

    print(f"Using collision probability threshold p_crit = {p_crit:.2f}")

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication_tight.npz"
    cfg.env.id = "ego-merge-tight-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75
    cfg.training.device = "cpu"

    # Instantiate model
    model = SafetyWorldModel.from_config(cfg.model, cfg.env, 5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    env = CommunicationMergeWrapper(cfg=cfg, world_model=model, device="cpu")

    mpc = SafetyMergeMPCPlanner(
        model=model,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=42,
        device="cpu",
        p_crit=p_crit,
        terminal_penalty=-1.0,
    )

    eval_seeds = [base_seed + i for i in range(num_seeds)]
    policies = ["NO_COMM", "ORACLE_FRONT", "ORACLE_REAR", "ORACLE_BEST"]
    seed_records: dict[str, list[dict[str, Any]]] = {p: [] for p in policies}

    action_changes = {
        "front": 0,
        "rear": 0,
        "total_steps": 0,
        "by_intent": {
            "front_MAINTAIN": {"total": 0, "changed": 0},
            "front_ACCELERATE": {"total": 0, "changed": 0},
            "front_BRAKE": {"total": 0, "changed": 0},
            "rear_MAINTAIN": {"total": 0, "changed": 0},
            "rear_ACCELERATE": {"total": 0, "changed": 0},
            "rear_BRAKE": {"total": 0, "changed": 0},
        },
    }

    t0 = time.time()

    for idx, seed in enumerate(eval_seeds):
        for pol in policies:
            env.reset(seed=seed)
            t_front = env.request_message(1)
            t_rear = env.request_message(2)

            ep_ret = 0.0
            crashed = False
            merged = False
            done = False
            step = 0

            cand_rng = np.random.default_rng(seed * 777)

            while not done and step < 20:
                obs = env.current_obs
                latent = env.current_latent
                cands = cand_rng.integers(0, len(mpc.allowed_actions), size=(mpc.num_sequences, mpc.horizon))
                cands = mpc.allowed_actions[cands]

                if pol == "NO_COMM":
                    act, _, _, _ = mpc.plan(obs, latent, 0, 0, fixed_candidate_actions=cands)
                    # Counterfactual check on identical visual state
                    a_f, _, _, _ = mpc.plan(obs, latent, 1, t_front, fixed_candidate_actions=cands)
                    a_r, _, _, _ = mpc.plan(obs, latent, 2, t_rear, fixed_candidate_actions=cands)

                    if a_f != act:
                        action_changes["front"] += 1
                        action_changes["by_intent"][f"front_{MSG_NAMES[t_front]}"]["changed"] += 1
                    action_changes["by_intent"][f"front_{MSG_NAMES[t_front]}"]["total"] += 1

                    if a_r != act:
                        action_changes["rear"] += 1
                        action_changes["by_intent"][f"rear_{MSG_NAMES[t_rear]}"]["changed"] += 1
                    action_changes["by_intent"][f"rear_{MSG_NAMES[t_rear]}"]["total"] += 1

                    action_changes["total_steps"] += 1

                elif pol == "ORACLE_FRONT":
                    act, _, _, _ = mpc.plan(obs, latent, 1, t_front, fixed_candidate_actions=cands)
                elif pol == "ORACLE_REAR":
                    act, _, _, _ = mpc.plan(obs, latent, 2, t_rear, fixed_candidate_actions=cands)
                elif pol == "ORACLE_BEST":
                    act_f, ret_f, _, _ = mpc.plan(obs, latent, 1, t_front, fixed_candidate_actions=cands)
                    act_r, ret_r, _, _ = mpc.plan(obs, latent, 2, t_rear, fixed_candidate_actions=cands)
                    act = act_f if ret_f >= ret_r else act_r

                # Step environment
                _, _, term, trunc, info = env.step(act, 0, 0)
                done = bool(term or trunc)
                ep_ret += info["raw_driving_reward"]
                if info["crashed"]:
                    crashed = True
                if info["is_merged"]:
                    merged = True
                step += 1

            run_dict = {
                "seed": seed,
                "return": float(ep_ret),
                "crashed": bool(crashed),
                "merged": bool(merged),
                "safe_merge": bool(merged and not crashed),
                "front_intent": MSG_NAMES[t_front],
                "rear_intent": MSG_NAMES[t_rear],
            }
            seed_records[pol].append(run_dict)

        if (idx + 1) % 10 == 0:
            print(f"Evaluated {idx + 1}/{num_seeds} seeds ({time.time() - t0:.1f}s)...")

    env.close()
    elapsed = time.time() - t0

    # Summary
    summary = {}
    no_comm_runs = seed_records["NO_COMM"]
    for pol in policies:
        runs = seed_records[pol]
        returns = [r["return"] for r in runs]
        crashes = [float(r["crashed"]) for r in runs]
        merges = [float(r["merged"]) for r in runs]
        safe_merges = [float(r["safe_merge"]) for r in runs]
        paired_diffs = [r["return"] - nc["return"] for r, nc in zip(runs, no_comm_runs)]

        crashes_avoided = sum(1 for r, nc in zip(runs, no_comm_runs) if nc["crashed"] and not r["crashed"])
        crashes_introduced = sum(1 for r, nc in zip(runs, no_comm_runs) if not nc["crashed"] and r["crashed"])
        net_crashes_avoided = crashes_avoided - crashes_introduced

        summary[pol] = {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "collision_rate": float(np.mean(crashes)),
            "merge_rate": float(np.mean(merges)),
            "safe_merge_rate": float(np.mean(safe_merges)),
            "paired_improvement": float(np.mean(paired_diffs)),
            "paired_diff_std": float(np.std(paired_diffs)),
            "crashes_avoided": int(crashes_avoided),
            "crashes_introduced": int(crashes_introduced),
            "net_crashes_avoided": int(net_crashes_avoided),
        }

    tot_steps = max(1, action_changes["total_steps"])
    pct_front_action_chg = (action_changes["front"] / tot_steps) * 100.0
    pct_rear_action_chg = (action_changes["rear"] / tot_steps) * 100.0

    print("\n" + "=" * 90)
    print("50-SEED SAFETY-AWARE ORACLE EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Policy':<14} | {'Mean Ret':<10} | {'Collision':<10} | {'Safe Merge':<10} | {'Paired Δ':<10} | {'Net Avoided'}")
    print("-" * 90)
    for pol in policies:
        s = summary[pol]
        print(f"{pol:<14} | {s['mean_return']:<10.3f} | {s['collision_rate']*100:>6.1f}%    | {s['safe_merge_rate']*100:>6.1f}%    | {s['paired_improvement']:+10.3f} | {s['net_crashes_avoided']:>+11d}")

    # Specific inspection of the 4 failure seeds
    failure_seeds = [4003, 4024, 4032, 4047]
    print("\n" + "=" * 90)
    print("INSPECTION OF HISTORICAL FAILURE SEEDS (4003, 4024, 4032, 4047)")
    print("=" * 90)
    failure_seed_report = {}
    for fs in failure_seeds:
        fs_report = {}
        print(f"Seed {fs}:")
        for pol in policies:
            rec = next(r for r in seed_records[pol] if r["seed"] == fs)
            fs_report[pol] = rec
            print(f"  {pol:<14}: Return={rec['return']:.3f}, Crashed={rec['crashed']}, Safe Merge={rec['safe_merge']}")
        failure_seed_report[fs] = fs_report

    # Decision Gate
    # Does Oracle strictly improve safe merge rate, return, or avoid net collisions without introducing new crashes?
    best_oracle_pol = "ORACLE_FRONT" if summary["ORACLE_FRONT"]["safe_merge_rate"] >= summary["ORACLE_REAR"]["safe_merge_rate"] else "ORACLE_REAR"
    best_oracle_safe_merge = summary[best_oracle_pol]["safe_merge_rate"]
    no_comm_safe_merge = summary["NO_COMM"]["safe_merge_rate"]
    safe_merge_gain = best_oracle_safe_merge - no_comm_safe_merge
    best_oracle_net_avoided = summary[best_oracle_pol]["net_crashes_avoided"]
    best_oracle_paired_diff = summary[best_oracle_pol]["paired_improvement"]

    print("\n" + "=" * 90)
    print("DECISION GATE EVALUATION")
    print("=" * 90)
    print(f"Best Oracle Policy: {best_oracle_pol}")
    print(f"NO_COMM Safe Merge Rate:     {no_comm_safe_merge*100:.1f}%")
    print(f"Best Oracle Safe Merge Rate: {best_oracle_safe_merge*100:.1f}% (Gain: {safe_merge_gain*100:+.1f}%)")
    print(f"Net Collisions Avoided:      {best_oracle_net_avoided:+d}")
    print(f"Mean Paired Improvement:     {best_oracle_paired_diff:+.3f}")

    if safe_merge_gain >= 0.04 or (best_oracle_net_avoided > 0 and summary[best_oracle_pol]["crashes_introduced"] == 0) or (best_oracle_paired_diff > 0.05 and safe_merge_gain >= 0):
        decision = "SAFETY WORLD MODEL SUCCESSFUL — ORACLE NOW HELPS"
        print(f"\nDECISION: >>> {decision} <<<")
    else:
        decision = "SAFETY WORLD MODEL INSUFFICIENT — STOP BEFORE DQN"
        print(f"\nDECISION: >>> {decision} <<<")

    results = {
        "num_seeds": num_seeds,
        "elapsed_seconds": elapsed,
        "p_crit": p_crit,
        "summary": summary,
        "action_changes": {
            "front_pct": pct_front_action_chg,
            "rear_pct": pct_rear_action_chg,
            "details": action_changes,
        },
        "failure_seeds_evaluation": failure_seed_report,
        "decision": decision,
        "seed_records": seed_records,
    }

    out_p = Path(out_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {out_p}")
    return results


if __name__ == "__main__":
    run_safety_oracle_experiment()

