"""50-Seed Oracle Experiment for Design 1 (Rear Bottleneck / Yield-vs-Pass)."""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Any
from collections import defaultdict
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.models import WorldModel
from lab1.utils.model_handoff import load_personal_checkpoint
from request_agent.rear_bottleneck_env import RearBottleneckMergeEnv
from request_agent.communication_wrapper import CommunicationMergeWrapper
from request_agent.mpc_adapter import MergeMPCPlanner

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def run_oracle_experiment(
    num_seeds: int = 50,
    base_seed: int = 4000,
    model_checkpoint_path: Path | str = "outputs/personal_lab1/rear_bottleneck_attempt_01/my_lab1_world_model_rear_bottleneck.pt",
    out_json_path: Path | str = "outputs/personal_lab1/rear_bottleneck_attempt_01/oracle_results.json",
) -> dict[str, Any]:
    print("=" * 85)
    print("DESIGN 1 ORACLE EXPERIMENT (REAR BOTTLENECK / YIELD-VS-PASS)")
    print(f"Seeds: {num_seeds} (base={base_seed}), Horizon=6, Sequences=64, Cost=0.0")
    print("=" * 85)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_rear_bottleneck.npz"
    cfg.env.id = "ego-merge-rear-bottleneck-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75
    cfg.training.device = "cpu"

    ckpt_path = Path(model_checkpoint_path)
    assert ckpt_path.exists(), f"Model checkpoint not found: {ckpt_path}"

    model = WorldModel.from_config(cfg.model, cfg.env, 5)
    load_personal_checkpoint(ckpt_path, model, cfg)
    model.eval()

    env = RearBottleneckMergeEnv(cfg, request_cost=0.0)

    mpc = MergeMPCPlanner(
        model=model,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=42,
        device="cpu",
    )

    eval_seeds = [base_seed + i for i in range(num_seeds)]
    policies = ["NO_COMM", "ORACLE_REAR", "ORACLE_FRONT", "ORACLE_BEST"]
    seed_records: dict[str, list[dict[str, Any]]] = {p: [] for p in policies}

    action_changes = {
        "rear": 0,
        "front": 0,
        "total_steps": 0,
        "by_intent": {
            "BRAKE": {"total": 0, "changed": 0},
            "MAINTAIN": {"total": 0, "changed": 0},
            "ACCELERATE": {"total": 0, "changed": 0},
        },
    }

    concrete_examples = []
    t0 = time.time()

    for idx, seed in enumerate(eval_seeds):
        # We run identical candidate driving action sequences per step for paired comparison
        for pol in policies:
            obs, info = env.reset(seed=seed)
            t_front = env.get_message(env.ASK_FRONT)  # MAINTAIN=1
            t_rear = env.get_message(env.ASK_REAR)    # MAINTAIN=1, ACCEL=2, BRAKE=3

            ep_ret = 0.0
            crashed = False
            merged = False
            done = False
            step = 0

            cand_rng = np.random.default_rng(seed * 777)

            while not done and step < 20:
                cands = cand_rng.integers(0, len(mpc.allowed_actions), size=(mpc.num_sequences, mpc.horizon))
                cands = mpc.allowed_actions[cands]

                if pol == "NO_COMM":
                    act, ret, _ = mpc.plan(obs, None, 0, 0, fixed_candidate_actions=cands)

                    # Check counterfactual action on identical state
                    act_r, ret_r, _ = mpc.plan(obs, None, 2, t_rear, fixed_candidate_actions=cands)
                    act_f, ret_f, _ = mpc.plan(obs, None, 1, t_front, fixed_candidate_actions=cands)

                    if act_r != act:
                        action_changes["rear"] += 1
                        action_changes["by_intent"][MSG_NAMES[t_rear]]["changed"] += 1
                    action_changes["by_intent"][MSG_NAMES[t_rear]]["total"] += 1

                    if act_f != act:
                        action_changes["front"] += 1

                    action_changes["total_steps"] += 1

                    # Save concrete examples from early steps
                    if step == 0 and len(concrete_examples) < 6:
                        concrete_examples.append({
                            "seed": seed,
                            "step": step,
                            "rear_intent": MSG_NAMES[t_rear],
                            "no_comm_action": ACTION_NAMES[act],
                            "no_comm_return": ret,
                            "oracle_rear_action": ACTION_NAMES[act_r],
                            "oracle_rear_return": ret_r,
                            "oracle_front_action": ACTION_NAMES[act_f],
                            "oracle_front_return": ret_f,
                        })

                elif pol == "ORACLE_REAR":
                    act, _, _ = mpc.plan(obs, None, 2, t_rear, fixed_candidate_actions=cands)
                elif pol == "ORACLE_FRONT":
                    act, _, _ = mpc.plan(obs, None, 1, t_front, fixed_candidate_actions=cands)
                elif pol == "ORACLE_BEST":
                    act_r, ret_r, _ = mpc.plan(obs, None, 2, t_rear, fixed_candidate_actions=cands)
                    act_f, ret_f, _ = mpc.plan(obs, None, 1, t_front, fixed_candidate_actions=cands)
                    act = act_r if ret_r >= ret_f else act_f

                obs, reward, term, trunc, s_info = env.step(act, 0)
                done = bool(term or trunc)
                ep_ret += s_info["raw_driving_reward"]
                if s_info["crashed"]:
                    crashed = True
                if s_info["is_merged"]:
                    merged = True
                step += 1

            run_dict = {
                "seed": seed,
                "return": float(ep_ret),
                "crashed": bool(crashed),
                "merged": bool(merged),
                "safe_merge": bool(merged and not crashed),
                "rear_intent": MSG_NAMES[t_rear],
                "front_intent": MSG_NAMES[t_front],
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

    # Breakdown by rear intent
    intent_summary = defaultdict(dict)
    for intent_name in ["BRAKE", "MAINTAIN", "ACCELERATE"]:
        idx_intent = [i for i, r in enumerate(no_comm_runs) if r["rear_intent"] == intent_name]
        for pol in policies:
            sub_runs = [seed_records[pol][i] for i in idx_intent]
            if sub_runs:
                intent_summary[intent_name][pol] = {
                    "count": len(sub_runs),
                    "mean_return": float(np.mean([r["return"] for r in sub_runs])),
                    "collision_rate": float(np.mean([float(r["crashed"]) for r in sub_runs])),
                    "safe_merge_rate": float(np.mean([float(r["safe_merge"]) for r in sub_runs])),
                }

    tot_steps = max(1, action_changes["total_steps"])
    pct_rear_chg = (action_changes["rear"] / tot_steps) * 100.0
    pct_front_chg = (action_changes["front"] / tot_steps) * 100.0

    print("\n" + "=" * 85)
    print("50-SEED ORACLE EVALUATION SUMMARY")
    print("=" * 85)
    print(f"{'Policy':<14} | {'Mean Ret':<10} | {'Collision':<10} | {'Safe Merge':<10} | {'Paired Δ':<10} | {'Net Avoided'}")
    print("-" * 85)
    for pol in policies:
        s = summary[pol]
        print(f"{pol:<14} | {s['mean_return']:<10.3f} | {s['collision_rate']*100:>6.1f}%    | {s['safe_merge_rate']*100:>6.1f}%    | {s['paired_improvement']:+10.3f} | {s['net_crashes_avoided']:>+11d}")

    print("\n" + "=" * 85)
    print("BREAKDOWN BY REAR INTENT")
    print("=" * 85)
    for intent_name in ["BRAKE", "MAINTAIN", "ACCELERATE"]:
        print(f"\n--- Rear Intent: {intent_name} (N={len([r for r in no_comm_runs if r['rear_intent'] == intent_name])}) ---")
        for pol in policies:
            isum = intent_summary[intent_name].get(pol, {})
            if isum:
                print(f"  {pol:<14}: Return={isum['mean_return']:.3f}, Collision={isum['collision_rate']*100:.1f}%, Safe Merge={isum['safe_merge_rate']*100:.1f}%")

    print("\n" + "=" * 85)
    print("ACTION SELECTION SHIFT FROM COMMUNICATION")
    print("=" * 85)
    print(f"Total evaluated steps: {tot_steps}")
    print(f"ASK_REAR changed driving action on:  {action_changes['rear']:>3d} / {tot_steps} ({pct_rear_chg:.1f}%)")
    print(f"ASK_FRONT changed driving action on: {action_changes['front']:>3d} / {tot_steps} ({pct_front_chg:.1f}%)")
    print("\nAction shifts by rear intent:")
    for iname in ["BRAKE", "MAINTAIN", "ACCELERATE"]:
        b_tot = action_changes["by_intent"][iname]["total"]
        b_chg = action_changes["by_intent"][iname]["changed"]
        pct = (b_chg / max(b_tot, 1)) * 100.0
        print(f"  {iname:<12}: {b_chg:>3d} / {b_tot:>3d} ({pct:>5.1f}%)")

    print("\n" + "=" * 85)
    print("CONCRETE EXAMPLES: SAME OBSERVATION + DIFFERENT REAR INTENT -> MPC ACTION")
    print("=" * 85)
    for ex in concrete_examples[:4]:
        print(f"Seed {ex['seed']} (Rear Intent: {ex['rear_intent']}):")
        print(f"  NO_COMM:      Action = {ex['no_comm_action']:<10} (pred return = {ex['no_comm_return']:.3f})")
        print(f"  ORACLE_REAR:  Action = {ex['oracle_rear_action']:<10} (pred return = {ex['oracle_rear_return']:.3f})")
        print(f"  ORACLE_FRONT: Action = {ex['oracle_front_action']:<10} (pred return = {ex['oracle_front_return']:.3f})")

    # Decision Gate
    or_safe_merge = summary["ORACLE_REAR"]["safe_merge_rate"]
    nc_safe_merge = summary["NO_COMM"]["safe_merge_rate"]
    or_net_avoided = summary["ORACLE_REAR"]["net_crashes_avoided"]
    or_paired_diff = summary["ORACLE_REAR"]["paired_improvement"]

    print("\n" + "=" * 85)
    print("DECISION GATE EVALUATION")
    print("=" * 85)
    print(f"NO_COMM Safe Merge Rate:     {nc_safe_merge*100:.1f}%")
    print(f"ORACLE_REAR Safe Merge Rate: {or_safe_merge*100:.1f}% (Gain: {(or_safe_merge - nc_safe_merge)*100:+.1f}%)")
    print(f"Net Collisions Avoided:      {or_net_avoided:+d}")
    print(f"Mean Paired Improvement:     {or_paired_diff:+.3f}")
    print(f"ORACLE_FRONT Paired Imp:     {summary['ORACLE_FRONT']['paired_improvement']:+.3f}")

    if (or_safe_merge > nc_safe_merge or or_net_avoided > 0 or or_paired_diff > 0.05) and (summary["ORACLE_FRONT"]["safe_merge_rate"] <= or_safe_merge):
        decision = "DESIGN 1 ORACLE GATE PASSED — READY FOR DQN"
        print(f"\nDECISION: >>> {decision} <<<")
    else:
        decision = "DESIGN 1 ORACLE GATE FAILED — DO NOT TRAIN DQN"
        print(f"\nDECISION: >>> {decision} <<<")

    results = {
        "num_seeds": num_seeds,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "intent_summary": intent_summary,
        "action_changes": {
            "pct_rear_chg": pct_rear_chg,
            "pct_front_chg": pct_front_chg,
            "details": action_changes,
        },
        "concrete_examples": concrete_examples,
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
    run_oracle_experiment()

