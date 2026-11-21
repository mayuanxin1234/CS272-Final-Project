"""Zero-Shot Robustness Evaluation of Learned WorldModel + DQN on Aggressive Driver.

Evaluates frozen WorldModel and frozen Double DQN policies (Seeds 1, 2, 3) zero-shot
on the aggressive/unpredictable driver merge scenario without any fine-tuning or retraining.

Behaviors:
0: NORMAL
1: SUDDEN_ACCELERATE
2: HARD_BRAKE
3: AGGRESSIVE_APPROACH

Baselines compared:
- NO_COMM
- RANDOM_REQUEST
- ZERO_SHOT_LEARNED_DQN (Seeds 1, 2, 3 and Aggregate)
- FULL_COMM
- ORACLE_SELECTIVE
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
from request_agent.dqn_agent import DoubleDQNAgent
from request_agent.mpc_adapter import MergeMPCPlanner
from request_agent.robustness_aggressive_driver_diagnostic import (
    AggressiveDriverMergeEnv,
    BEHAVIOR_STYLES,
)


def get_latent_vector(wm: WorldModel, obs: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Encode grayscale (64, 64) image to 64-D latent vector."""
    wm.eval()
    with torch.no_grad():
        t_obs = torch.from_numpy(obs.astype(np.float32) / 255.0).unsqueeze(0).to(device)
        z = wm.encoder(t_obs).squeeze(0).cpu().numpy()
    return z


def build_dqn_state(latent: np.ndarray, prev_req: int, prev_msg: int) -> np.ndarray:
    """Build 71-D state vector: [z (64), req_onehot (3), msg_onehot (4)].
    
    CRITICAL: Does NOT include privileged simulator state or driver behavior style.
    """
    req_oh = np.zeros(3, dtype=np.float32)
    req_oh[int(prev_req)] = 1.0
    msg_oh = np.zeros(4, dtype=np.float32)
    msg_oh[int(prev_msg)] = 1.0
    return np.concatenate([latent.astype(np.float32), req_oh, msg_oh], axis=0)


def evaluate_policy_on_aggressive(
    policy_name: str,
    env: AggressiveDriverMergeEnv,
    wm: WorldModel,
    mpc: MergeMPCPlanner,
    dqn_agent: DoubleDQNAgent | None = None,
    num_seeds: int = 60,
    base_seed: int = 9500,
    device: str = "cpu",
) -> tuple[dict, list[dict], dict[int, Counter]]:
    """Evaluate a single policy on the aggressive driver scenario."""
    records = []
    # Style -> Counter(request_action) at t=0
    req_by_style = {s_idx: Counter() for s_idx in BEHAVIOR_STYLES}

    for idx in range(num_seeds):
        seed = base_seed + idx
        style_idx = seed % 4
        s_name = BEHAVIOR_STYLES[style_idx]

        obs, info = env.reset(seed=seed)
        z = get_latent_vector(wm, obs, device=device)
        state = build_dqn_state(z, prev_req=0, prev_msg=0)

        ep_utility = 0.0
        ep_driving_r = 0.0
        crashed = False
        merged = False
        step = 0
        req_count = 0
        revealed_msg = env.MSG_UNKNOWN

        done = False
        while not done and step < 8:
            if policy_name == "NO_COMM":
                req = env.NONE
            elif policy_name == "RANDOM_REQUEST":
                if step == 0:
                    rng = np.random.default_rng(seed * 100 + step)
                    req = int(rng.choice([env.NONE, env.ASK_FRONT, env.ASK_REAR]))
                else:
                    req = env.NONE
            elif policy_name == "FULL_COMM":
                if step == 0:
                    req = env.ASK_FRONT if style_idx == 2 else (env.ASK_REAR if style_idx in [1, 3] else env.ASK_FRONT)
                else:
                    req = env.NONE
            elif policy_name == "ORACLE_SELECTIVE":
                if step == 0:
                    if style_idx == 2:
                        req = env.ASK_FRONT
                    elif style_idx in [1, 3]:
                        req = env.ASK_REAR
                    else:
                        req = env.NONE
                else:
                    req = env.NONE
            elif policy_name.startswith("ZERO_SHOT_DQN"):
                assert dqn_agent is not None
                req = dqn_agent.select_action(state, evaluate=True)
            else:
                raise ValueError(f"Unknown policy {policy_name}")

            if req != env.NONE:
                req_count += 1

            if step == 0:
                req_by_style[style_idx][req] += 1

            msg = env.get_message(req)
            if msg != env.MSG_UNKNOWN:
                revealed_msg = msg

            # Driving controlled by learned WorldModel MPC
            mpc.rng = np.random.default_rng(seed * 1000 + step)
            drive_act, _, _ = mpc.plan(
                obs=obs,
                latent=z,
                request_action=req,
                message=revealed_msg,
            )

            next_obs, u_rew, term, trunc, s_info = env.step(drive_act, req)
            done = bool(term or trunc)

            ep_utility += u_rew
            ep_driving_r += s_info["raw_driving_reward"]
            if s_info["crashed"]:
                crashed = True
            if s_info["is_merged"]:
                merged = True

            next_z = get_latent_vector(wm, next_obs, device=device)
            next_state = build_dqn_state(next_z, prev_req=req, prev_msg=msg)

            state = next_state
            z = next_z
            obs = next_obs
            step += 1

        records.append({
            "seed": seed,
            "style_idx": style_idx,
            "style_name": s_name,
            "safe_merge": bool(merged and not crashed),
            "crashed": crashed,
            "driving_return": ep_driving_r,
            "utility": ep_utility,
            "requests": req_count,
            "bits": req_count * 4,
            "steps": step,
        })

    summary = {
        "safe_merge_rate": float(np.mean([r["safe_merge"] for r in records]) * 100.0),
        "collision_rate": float(np.mean([r["crashed"] for r in records]) * 100.0),
        "mean_driving_return": float(np.mean([r["driving_return"] for r in records])),
        "mean_utility": float(np.mean([r["utility"] for r in records])),
        "mean_requests": float(np.mean([r["requests"] for r in records])),
        "mean_bits": float(np.mean([r["bits"] for r in records])),
    }
    return summary, records, req_by_style


def run_full_aggressive_zero_shot_suite():
    t_start = time.time()
    print("=" * 90)
    print("ZERO-SHOT AGGRESSIVE DRIVER EVALUATION SUITE")
    print("=" * 90)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "aggressive-driver-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75

    wm_path = "outputs/personal_lab1/occlusion_final_attempt_01/my_lab1_world_model_occlusion_final.pt"
    wm = WorldModel.from_config(cfg.model, cfg.env, 5)
    load_checkpoint(wm_path, wm, map_location="cpu")
    wm.eval()
    print(f"Loaded frozen WorldModel from: {wm_path}")

    mpc = MergeMPCPlanner(
        model=wm,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        future_mode="persistent",
    )

    env = AggressiveDriverMergeEnv(request_cost=0.05)

    num_seeds = 60
    base_seed = 9500

    # 1. Evaluate Standard Baselines with WorldModel MPC
    baselines = ["NO_COMM", "RANDOM_REQUEST", "FULL_COMM", "ORACLE_SELECTIVE"]
    baseline_summaries = {}
    baseline_records = {}
    baseline_req_breakdowns = {}

    for b in baselines:
        print(f"\nEvaluating baseline: {b} ...")
        s, r, req_c = evaluate_policy_on_aggressive(
            b, env, wm, mpc, num_seeds=num_seeds, base_seed=base_seed
        )
        baseline_summaries[b] = s
        baseline_records[b] = r
        baseline_req_breakdowns[b] = req_c
        print(f"  -> Safe Merge: {s['safe_merge_rate']:.1f}%, Collision: {s['collision_rate']:.1f}%, Return: {s['mean_driving_return']:.3f}, Reqs: {s['mean_requests']:.2f}")

    # 2. Evaluate Frozen DQN Seeds (lambda=0.05)
    dqn_seeds = [1, 2, 3]
    dqn_summaries = {}
    dqn_records = {}
    dqn_req_breakdowns = {}

    for s_id in dqn_seeds:
        ckpt_path = f"outputs/request_agent/occlusion_final_evaluation/dqn_seed{s_id}_lambda0.05.pt"
        print(f"\nEvaluating frozen DQN Seed {s_id} ({ckpt_path}) ...")
        agent = DoubleDQNAgent(state_dim=71, action_dim=3, device="cpu", seed=s_id)
        agent.load(ckpt_path)
        p_name = f"ZERO_SHOT_DQN_SEED{s_id}"
        s, r, req_c = evaluate_policy_on_aggressive(
            p_name, env, wm, mpc, dqn_agent=agent, num_seeds=num_seeds, base_seed=base_seed
        )
        dqn_summaries[s_id] = s
        dqn_records[s_id] = r
        dqn_req_breakdowns[s_id] = req_c
        print(f"  -> Safe Merge: {s['safe_merge_rate']:.1f}%, Collision: {s['collision_rate']:.1f}%, Return: {s['mean_driving_return']:.3f}, Reqs: {s['mean_requests']:.2f}")

    # Aggregate DQN across seeds 1, 2, 3
    agg_safe = float(np.mean([dqn_summaries[s]["safe_merge_rate"] for s in dqn_seeds]))
    agg_crash = float(np.mean([dqn_summaries[s]["collision_rate"] for s in dqn_seeds]))
    agg_drv = float(np.mean([dqn_summaries[s]["mean_driving_return"] for s in dqn_seeds]))
    agg_util = float(np.mean([dqn_summaries[s]["mean_utility"] for s in dqn_seeds]))
    agg_reqs = float(np.mean([dqn_summaries[s]["mean_requests"] for s in dqn_seeds]))
    agg_bits = float(np.mean([dqn_summaries[s]["mean_bits"] for s in dqn_seeds]))

    agg_dqn_summary = {
        "safe_merge_rate": agg_safe,
        "collision_rate": agg_crash,
        "mean_driving_return": agg_drv,
        "mean_utility": agg_util,
        "mean_requests": agg_reqs,
        "mean_bits": agg_bits,
    }

    # Combined Table
    print("\n" + "=" * 90)
    print("OVERALL PERFORMANCE COMPARISON ON AGGRESSIVE DRIVER (Zero-Shot)")
    print("=" * 90)
    print(f"{'Policy':<24} | {'Safe Merge':<10} | {'Collision':<10} | {'Driving Ret':<12} | {'Utility':<10} | {'Reqs/Ep':<8} | {'Bits/Ep'}")
    print("-" * 90)

    for b in baselines:
        s = baseline_summaries[b]
        print(f"{b:<24} | {s['safe_merge_rate']:>8.1f}%  | {s['collision_rate']:>8.1f}%  | {s['mean_driving_return']:>10.3f}   | {s['mean_utility']:>8.3f}   | {s['mean_requests']:>6.2f}   | {s['mean_bits']:>5.1f}b")

    for s_id in dqn_seeds:
        s = dqn_summaries[s_id]
        print(f"{f'ZERO_SHOT_DQN_SEED_{s_id}':<24} | {s['safe_merge_rate']:>8.1f}%  | {s['collision_rate']:>8.1f}%  | {s['mean_driving_return']:>10.3f}   | {s['mean_utility']:>8.3f}   | {s['mean_requests']:>6.2f}   | {s['mean_bits']:>5.1f}b")

    print(f"{'ZERO_SHOT_DQN_AGGREGATE':<24} | {agg_safe:>8.1f}%  | {agg_crash:>8.1f}%  | {agg_drv:>10.3f}   | {agg_util:>8.3f}   | {agg_reqs:>6.2f}   | {agg_bits:>5.1f}b")
    print("-" * 90)

    # Breakdown by behavior style
    print("\n" + "=" * 90)
    print("PERFORMANCE BREAKDOWN BY BEHAVIOR STYLE (Safe Merge Rate %)")
    print("=" * 90)
    print(f"{'Behavior Style':<22} | {'NO_COMM':<10} | {'RANDOM':<10} | {'FULL_COMM':<10} | {'ORACLE':<10} | {'DQN_S1':<8} | {'DQN_S2':<8} | {'DQN_S3':<8} | {'DQN_AGG'}")
    print("-" * 90)

    style_breakdown = {}
    for s_idx, s_name in BEHAVIOR_STYLES.items():
        sm_nc = float(np.mean([r["safe_merge"] for r in baseline_records["NO_COMM"] if r["style_idx"] == s_idx]) * 100.0)
        sm_rr = float(np.mean([r["safe_merge"] for r in baseline_records["RANDOM_REQUEST"] if r["style_idx"] == s_idx]) * 100.0)
        sm_fc = float(np.mean([r["safe_merge"] for r in baseline_records["FULL_COMM"] if r["style_idx"] == s_idx]) * 100.0)
        sm_os = float(np.mean([r["safe_merge"] for r in baseline_records["ORACLE_SELECTIVE"] if r["style_idx"] == s_idx]) * 100.0)

        sm_s1 = float(np.mean([r["safe_merge"] for r in dqn_records[1] if r["style_idx"] == s_idx]) * 100.0)
        sm_s2 = float(np.mean([r["safe_merge"] for r in dqn_records[2] if r["style_idx"] == s_idx]) * 100.0)
        sm_s3 = float(np.mean([r["safe_merge"] for r in dqn_records[3] if r["style_idx"] == s_idx]) * 100.0)
        sm_agg = (sm_s1 + sm_s2 + sm_s3) / 3.0

        style_breakdown[s_name] = {
            "NO_COMM": sm_nc,
            "RANDOM_REQUEST": sm_rr,
            "FULL_COMM": sm_fc,
            "ORACLE_SELECTIVE": sm_os,
            "DQN_SEED1": sm_s1,
            "DQN_SEED2": sm_s2,
            "DQN_SEED3": sm_s3,
            "DQN_AGGREGATE": sm_agg,
        }
        print(f"{s_name:<22} | {sm_nc:>8.1f}%  | {sm_rr:>8.1f}%  | {sm_fc:>8.1f}%  | {sm_os:>8.1f}%  | {sm_s1:>6.1f}%  | {sm_s2:>6.1f}%  | {sm_s3:>6.1f}%  | {sm_agg:>6.1f}%")

    # DQN Request Choices by Style (% NONE, % ASK_FRONT, % ASK_REAR)
    print("\n" + "=" * 90)
    print("DQN REQUEST ACTION CHOICES BY BEHAVIOR STYLE at t=0 (% NONE / % ASK_FRONT / % ASK_REAR)")
    print("=" * 90)
    print(f"{'Behavior Style':<22} | {'Seed 1 Request Breakdown':<28} | {'Seed 2 Request Breakdown':<28} | {'Seed 3 Request Breakdown'}")
    print("-" * 90)

    req_choice_summary = {}
    for s_idx, s_name in BEHAVIOR_STYLES.items():
        row_str = f"{s_name:<22} | "
        req_choice_summary[s_name] = {}
        for s_id in dqn_seeds:
            c = dqn_req_breakdowns[s_id][s_idx]
            total = sum(c.values())
            pct_n = c.get(0, 0) / max(1, total) * 100.0
            pct_f = c.get(1, 0) / max(1, total) * 100.0
            pct_r = c.get(2, 0) / max(1, total) * 100.0
            req_choice_summary[s_name][f"seed_{s_id}"] = {
                "NONE": pct_n,
                "ASK_FRONT": pct_f,
                "ASK_REAR": pct_r,
            }
            sub_str = f"N:{pct_n:>4.1f}% F:{pct_f:>4.1f}% R:{pct_r:>4.1f}%"
            row_str += f"{sub_str:<28} | "
        print(row_str.rstrip(" |"))

    # Aggregate DQN Request Choices by Style
    print("-" * 90)
    print("Aggregate DQN Request Choices Across 3 Seeds:")
    for s_idx, s_name in BEHAVIOR_STYLES.items():
        all_c = Counter()
        for s_id in dqn_seeds:
            all_c.update(dqn_req_breakdowns[s_id][s_idx])
        tot = sum(all_c.values())
        p_n = all_c.get(0, 0) / max(1, tot) * 100.0
        p_f = all_c.get(1, 0) / max(1, tot) * 100.0
        p_r = all_c.get(2, 0) / max(1, tot) * 100.0
        req_choice_summary[s_name]["aggregate"] = {"NONE": p_n, "ASK_FRONT": p_f, "ASK_REAR": p_r}
        print(f"  {s_name:<20}: % NONE = {p_n:>5.1f}%, % ASK_FRONT = {p_f:>5.1f}%, % ASK_REAR = {p_r:>5.1f}%")

    out_payload = {
        "baseline_summaries": baseline_summaries,
        "dqn_summaries": dqn_summaries,
        "agg_dqn_summary": agg_dqn_summary,
        "style_breakdown": style_breakdown,
        "req_choice_summary": req_choice_summary,
        "eval_meta": {
            "num_seeds": num_seeds,
            "base_seed": base_seed,
            "lambda_cost": 0.05,
            "wm_path": wm_path,
            "elapsed_seconds": time.time() - t_start,
        }
    }

    out_file = Path("outputs/request_agent/robustness_aggressive_zero_shot_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out_payload, f, indent=2, default=str)
    print(f"\nSaved zero-shot evaluation results to: {out_file}")

    # Decision verdict
    print("\n" + "=" * 90)
    print("VERDICT DETERMINATION:")
    print("=" * 90)
    if agg_safe >= 80.0:
        verdict = "ZERO-SHOT AGGRESSIVE ROBUSTNESS PASSED"
    else:
        verdict = "ZERO-SHOT AGGRESSIVE ROBUSTNESS LIMITED — RETRAINING WOULD BE REQUIRED"
    print(verdict)
    print("=" * 90)
    return verdict, out_payload


if __name__ == "__main__":
    run_full_aggressive_zero_shot_suite()

