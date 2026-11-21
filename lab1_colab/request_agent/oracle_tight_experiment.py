"""Oracle experiment on Candidate 4 (Tight Merge Scenario) using the retrained world model."""

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
from lab1.models import WorldModel
from request_agent.tight_merge_env import Candidate4MergeEnv
from request_agent.communication_wrapper import CommunicationMergeWrapper
from request_agent.mpc_adapter import MergeMPCPlanner

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def run_oracle_tight_experiment(
    num_seeds: int = 50,
    model_checkpoint_path: Path | str = "outputs/personal_lab1/tight_scenario_attempt_01/my_lab1_world_model_tight.pt",
    out_json_path: Path | str = "outputs/request_agent/oracle_tight_experiment_results.json",
    base_seed: int = 4000,
) -> dict[str, Any]:
    print("=" * 80)
    print("ORACLE EXPERIMENT ON CANDIDATE 4 (TIGHT SCENARIO) WITH RETRAINED WORLD MODEL")
    print(f"Seeds: {num_seeds} (base={base_seed}), Horizon=6, Sequences=64, Cost=0.0")
    print("=" * 80)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication_tight.npz"
    cfg.data.num_transitions = 20000
    cfg.env.id = "ego-merge-tight-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75
    cfg.training.device = "cpu"

    ckpt_path = Path(model_checkpoint_path)
    assert ckpt_path.exists(), f"Model not found: {ckpt_path}"

    env = CommunicationMergeWrapper(cfg=cfg, model_checkpoint_path=ckpt_path, device="cpu")
    mpc = MergeMPCPlanner(
        model=env.model,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=42,
        device="cpu",
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
        # Local seed records for paired comparison
        current_seed_runs = {}

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
                    act, _, _ = mpc.plan(obs, latent, 0, 0, fixed_candidate_actions=cands)
                    # Counterfactual action check on exact same state
                    a_f, _, _ = mpc.plan(obs, latent, 1, t_front, fixed_candidate_actions=cands)
                    a_r, _, _ = mpc.plan(obs, latent, 2, t_rear, fixed_candidate_actions=cands)

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
                    act, _, _ = mpc.plan(obs, latent, 1, t_front, fixed_candidate_actions=cands)
                elif pol == "ORACLE_REAR":
                    act, _, _ = mpc.plan(obs, latent, 2, t_rear, fixed_candidate_actions=cands)
                elif pol == "ORACLE_BEST":
                    act_f, ret_f, _ = mpc.plan(obs, latent, 1, t_front, fixed_candidate_actions=cands)
                    act_r, ret_r, _ = mpc.plan(obs, latent, 2, t_rear, fixed_candidate_actions=cands)
                    act = act_f if ret_f >= ret_r else act_r

                # Advance physics with zero request cost
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
            current_seed_runs[pol] = run_dict

        if (idx + 1) % 10 == 0:
            print(f"Evaluated {idx + 1}/{num_seeds} seeds ({time.time() - t0:.1f}s)...")

    env.close()
    elapsed = time.time() - t0

    # Summarize metrics
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

    # Decision criteria
    best_oracle_safe_merge = max(summary["ORACLE_FRONT"]["safe_merge_rate"], summary["ORACLE_BEST"]["safe_merge_rate"])
    safe_merge_gain = best_oracle_safe_merge - summary["NO_COMM"]["safe_merge_rate"]
    best_oracle_net_avoided = summary["ORACLE_BEST"]["net_crashes_avoided"]

    if safe_merge_gain >= 0.06 or best_oracle_net_avoided > 0 or summary["ORACLE_BEST"]["paired_improvement"] > 0.05:
        verdict = "ORACLE NOW HELPS — READY TO TRAIN REQUEST DQN"
    else:
        verdict = "ORACLE STILL DOES NOT HELP — DO NOT TRAIN DQN"

    results = {
        "num_seeds": num_seeds,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "action_changes": {
            "total_steps": tot_steps,
            "pct_front_action_changed": pct_front_action_chg,
            "pct_rear_action_changed": pct_rear_action_chg,
            "by_intent": action_changes["by_intent"],
        },
        "verdict": verdict,
    }

    out_json = Path(out_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("ORACLE EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    for pol in policies:
        s = summary[pol]
        print(f"{pol:<14}: SafeMerge={s['safe_merge_rate']*100:5.1f}%, Collision={s['collision_rate']*100:5.1f}%, Return={s['mean_return']:6.3f} (paired diff: {s['paired_improvement']:+6.3f}, net crashes avoided: {s['net_crashes_avoided']})")

    print("\nAction Change Frequency on Counterfactual Interrogation:")
    print(f"  Front Intent Changed Action: {pct_front_action_chg:.2f}% of steps")
    print(f"  Rear Intent Changed Action:  {pct_rear_action_chg:.2f}% of steps")

    print("\nIntent-Specific Action Change Rates:")
    for intent_key, counts in action_changes["by_intent"].items():
        tot = counts["total"]
        chg = counts["changed"]
        pct = (chg / tot * 100.0) if tot > 0 else 0.0
        print(f"  {intent_key:<18}: {chg}/{tot} ({pct:.1f}%)")

    print("\n" + "=" * 80)
    print(f"FINAL CLASSIFICATION: {verdict}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_oracle_tight_experiment(num_seeds=50)

