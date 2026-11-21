"""Comprehensive Diagnostic Script for Oracle-Conditioned MPC Failure Analysis.

Covers:
- Part 1: Identify paired failure seeds and generate side-by-side traces.
- Part 2: Measure predicted vs actual returns at action-divergence timesteps.
- Part 4: One-step world-model accuracy breakdown by communication condition.
- Part 5: Error evaluation on collision-critical subset vs ordinary transitions.
- Part 6: True-environment simulator counterfactual rollouts vs MPC predicted actions.
"""

from __future__ import annotations

import os
import copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.data import load_episodes, TransitionDataset
from lab1.data.dataset import split_episodes
from lab1.models import WorldModel
from lab1.utils.model_handoff import load_personal_checkpoint
from request_agent.tight_merge_env import Candidate4MergeEnv
from request_agent.communication_wrapper import CommunicationMergeWrapper
from request_agent.mpc_adapter import MergeMPCPlanner

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}
REQ_NAMES = {0: "NONE", 1: "ASK_FRONT", 2: "ASK_REAR"}


def part1_and_part2_paired_analysis(
    model_path: Path,
    num_seeds: int = 50,
    base_seed: int = 4000,
):
    print("=" * 80)
    print("PART 1 & PART 2: PAIRED FAILURE SEEDS & PREDICTED VS ACTUAL CONSEQUENCE")
    print("=" * 80)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication_tight.npz"
    cfg.data.num_transitions = 20000
    cfg.env.id = "ego-merge-tight-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75
    cfg.training.device = "cpu"

    env = CommunicationMergeWrapper(cfg=cfg, model_checkpoint_path=model_path, device="cpu")
    mpc = MergeMPCPlanner(
        model=env.model,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=42,
        device="cpu",
    )

    eval_seeds = [base_seed + i for i in range(num_seeds)]
    divergence_records = []
    failure_seeds = []

    for idx, seed in enumerate(eval_seeds):
        # We need traces for NO_COMM, ORACLE_FRONT, ORACLE_REAR
        pol_runs = {}

        for pol, req_type in [("NO_COMM", 0), ("ORACLE_FRONT", 1), ("ORACLE_REAR", 2)]:
            env.reset(seed=seed)
            tf = env.request_message(1)
            tr = env.request_message(2)
            msg_val = 0 if req_type == 0 else (tf if req_type == 1 else tr)

            ep_steps = []
            done = False
            step = 0
            cand_rng = np.random.default_rng(seed * 777)

            while not done and step < 20:
                obs = env.current_obs
                latent = env.current_latent
                cands = cand_rng.integers(0, len(mpc.allowed_actions), size=(mpc.num_sequences, mpc.horizon))
                cands = mpc.allowed_actions[cands]

                act, pred_ret, c_rets = mpc.plan(obs, latent, req_type, msg_val, fixed_candidate_actions=cands)

                base_unwrapped = env.env.env.unwrapped
                ego_pos = copy.deepcopy(base_unwrapped.vehicle.position)
                ego_spd = float(base_unwrapped.vehicle.speed)
                front_pos = copy.deepcopy(base_unwrapped.front_vehicle.position)
                front_spd = float(base_unwrapped.front_vehicle.speed)
                rear_pos = copy.deepcopy(base_unwrapped.rear_vehicle.position)
                rear_spd = float(base_unwrapped.rear_vehicle.speed)

                _, _, term, trunc, info = env.step(act, 0, 0)
                done = bool(term or trunc)

                ep_steps.append({
                    "step": step,
                    "ego_x": float(ego_pos[0]),
                    "ego_y": float(ego_pos[1]),
                    "ego_speed": ego_spd,
                    "front_x": float(front_pos[0]),
                    "rear_x": float(rear_pos[0]),
                    "req_action": req_type,
                    "message": msg_val,
                    "chosen_action": act,
                    "pred_return": float(pred_ret),
                    "actual_reward": float(info["raw_driving_reward"]),
                    "crashed": bool(info["crashed"]),
                    "is_merged": bool(info["is_merged"]),
                    "cands": cands,
                    "obs": obs,
                    "latent": latent,
                })
                step += 1

            pol_runs[pol] = {
                "steps": ep_steps,
                "crashed": bool(ep_steps[-1]["crashed"]),
                "merged": bool(ep_steps[-1]["is_merged"]),
                "return": sum(s["actual_reward"] for s in ep_steps),
                "front_intent": MSG_NAMES[tf],
                "rear_intent": MSG_NAMES[tr],
            }

        # Check for failure conditions: NO_COMM succeeded safely, but ORACLE crashed
        nc_safe = pol_runs["NO_COMM"]["merged"] and not pol_runs["NO_COMM"]["crashed"]
        of_crash = pol_runs["ORACLE_FRONT"]["crashed"]
        or_crash = pol_runs["ORACLE_REAR"]["crashed"]

        if nc_safe and (of_crash or or_crash):
            failure_seeds.append({
                "seed": seed,
                "of_crashed": of_crash,
                "or_crashed": or_crash,
                "runs": pol_runs,
            })

        # Record action divergences at matching states
        nc_steps = pol_runs["NO_COMM"]["steps"]
        of_steps = pol_runs["ORACLE_FRONT"]["steps"]
        min_len = min(len(nc_steps), len(of_steps))
        for t in range(min_len):
            if nc_steps[t]["chosen_action"] != of_steps[t]["chosen_action"]:
                divergence_records.append({
                    "seed": seed,
                    "step": t,
                    "no_comm_action": nc_steps[t]["chosen_action"],
                    "no_comm_pred_ret": nc_steps[t]["pred_return"],
                    "oracle_action": of_steps[t]["chosen_action"],
                    "oracle_pred_ret": of_steps[t]["pred_return"],
                    "no_comm_actual_ep_ret": pol_runs["NO_COMM"]["return"],
                    "oracle_actual_ep_ret": pol_runs["ORACLE_FRONT"]["return"],
                    "no_comm_crashed": pol_runs["NO_COMM"]["crashed"],
                    "oracle_crashed": pol_runs["ORACLE_FRONT"]["crashed"],
                    "front_intent": pol_runs["NO_COMM"]["front_intent"],
                })

    print(f"\nTotal paired failure seeds identified: {len(failure_seeds)}")
    for f in failure_seeds:
        seed = f["seed"]
        of_cr = f["of_crashed"]
        or_cr = f["or_crashed"]
        f_int = f["runs"]["NO_COMM"]["front_intent"]
        r_int = f["runs"]["NO_COMM"]["rear_intent"]
        print(f"\n>>> SEED {seed} | Front Intent: {f_int}, Rear Intent: {r_int}")
        print(f"    NO_COMM: Safe Merge (ret={f['runs']['NO_COMM']['return']:.3f})")
        if of_cr:
            print(f"    ORACLE_FRONT: CRASHED (ret={f['runs']['ORACLE_FRONT']['return']:.3f})")
        if or_cr:
            print(f"    ORACLE_REAR:  CRASHED (ret={f['runs']['ORACLE_REAR']['return']:.3f})")

        # Print side-by-side trace
        print("\n    Side-by-Side Trace (NO_COMM vs ORACLE_FRONT):")
        print(f"    {'Step':<4} | {'NO_COMM (Act/Pred/Rew/Crash)':<32} | {'ORACLE_FRONT (Act/Pred/Rew/Crash)':<32}")
        print("    " + "-" * 70)
        nc_s = f["runs"]["NO_COMM"]["steps"]
        of_s = f["runs"]["ORACLE_FRONT"]["steps"]
        max_l = max(len(nc_s), len(of_s))
        for t in range(max_l):
            nc_str = f"{ACTION_NAMES[nc_s[t]['chosen_action']]:<10} p={nc_s[t]['pred_return']:.2f} r={nc_s[t]['actual_reward']:.2f} cr={nc_s[t]['crashed']}" if t < len(nc_s) else "DONE"
            of_str = f"{ACTION_NAMES[of_s[t]['chosen_action']]:<10} p={of_s[t]['pred_return']:.2f} r={of_s[t]['actual_reward']:.2f} cr={of_s[t]['crashed']}" if t < len(of_s) else "DONE"
            print(f"    {t:<4} | {nc_str:<32} | {of_str:<32}")

    # Analyze Divergences
    print("\n" + "=" * 80)
    print("PART 2: PREDICTED VS ACTUAL CONSEQUENCE AT ACTION DIVERGENCES")
    print("=" * 80)
    print(f"Total divergence steps observed: {len(divergence_records)}")
    overestimated_oracle_count = 0
    for d in divergence_records:
        pred_diff = d["oracle_pred_ret"] - d["no_comm_pred_ret"]
        actual_diff = d["oracle_actual_ep_ret"] - d["no_comm_actual_ep_ret"]
        if pred_diff > 0 and actual_diff < 0:
            overestimated_oracle_count += 1

    print(f"Cases where Oracle predicted HIGHER return but got WORSE actual return: {overestimated_oracle_count}/{len(divergence_records)} ({overestimated_oracle_count / max(1, len(divergence_records)) * 100:.1f}%)")
    for d in divergence_records[:10]:
        print(f"  Seed {d['seed']} Step {d['step']}: NC={ACTION_NAMES[d['no_comm_action']]} (pred={d['no_comm_pred_ret']:.2f}), OF={ACTION_NAMES[d['oracle_action']]} (pred={d['oracle_pred_ret']:.2f}) | Actual: NC_ret={d['no_comm_actual_ep_ret']:.2f}, OF_ret={d['oracle_actual_ep_ret']:.2f}, OF_crashed={d['oracle_crashed']}")


def part4_and_part5_dataset_accuracy(model_path: Path):
    print("\n" + "=" * 80)
    print("PART 4: ONE-STEP WORLD MODEL ACCURACY BY COMMUNICATION CONDITION")
    print("=" * 80)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication_tight.npz"
    episodes = load_episodes(cfg.data.dataset_path)

    # Use exact validation split
    _, val_episodes = split_episodes(episodes, val_fraction=0.1, seed=cfg.seed)
    val_dataset = TransitionDataset(val_episodes)

    device = torch.device("cpu")
    model = WorldModel.from_config(cfg.model, cfg.env, 5).to(device)
    load_personal_checkpoint(model_path, model, cfg)
    model.eval()

    # Breakdown by condition: (req, msg)
    cond_stats = defaultdict(lambda: {"reward_errors": [], "latent_errors": [], "count": 0})

    # Part 5: Collision-critical vs ordinary
    collision_stats = {"reward_errors": [], "count": 0}
    ordinary_stats = {"reward_errors": [], "count": 0}

    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            obs = sample["obs"].unsqueeze(0).to(device)
            next_obs = sample["next_obs"].unsqueeze(0).to(device)
            act = sample["action"].unsqueeze(0).to(device)
            req = sample["request_action"].unsqueeze(0).to(device)
            msg = sample["message"].unsqueeze(0).to(device)
            true_r = sample["reward"].item()
            done = sample["done"].item()

            z = model.encode(obs)
            true_next_z = model.encode(next_obs)

            pred_next_z = model.predict_next(z, act, req, msg)
            pred_r = model.predict_reward(z, act, req, msg).item()

            r_err = abs(pred_r - true_r)
            z_err = torch.norm(pred_next_z - true_next_z, p=2).item()

            req_val = req.item()
            msg_val = msg.item()
            cond_key = (req_val, msg_val)

            cond_stats[cond_key]["reward_errors"].append(r_err)
            cond_stats[cond_key]["latent_errors"].append(z_err)
            cond_stats[cond_key]["count"] += 1

            # Collision-critical transition: true reward <= -0.5 or done with low reward
            if true_r <= -0.5 or (done > 0.5 and true_r < 0.0):
                collision_stats["reward_errors"].append(r_err)
                collision_stats["count"] += 1
            else:
                ordinary_stats["reward_errors"].append(r_err)
                ordinary_stats["count"] += 1

    print(f"{'Condition':<28} | {'Req':<4} | {'Msg':<12} | {'Count':<6} | {'Reward MAE':<12} | {'Latent L2 Error'}")
    print("-" * 80)
    for (req_val, msg_val), s in sorted(cond_stats.items()):
        cond_label = f"{REQ_NAMES[req_val]} + {MSG_NAMES[msg_val]}"
        mae_r = float(np.mean(s["reward_errors"]))
        mae_z = float(np.mean(s["latent_errors"]))
        print(f"{cond_label:<28} | {req_val:<4} | {MSG_NAMES[msg_val]:<12} | {s['count']:<6} | {mae_r:<12.5f} | {mae_z:<12.5f}")

    print("\n" + "=" * 80)
    print("PART 5: COLLISION-CRITICAL VS ORDINARY SUBSET EVALUATION")
    print("=" * 80)
    coll_mae = float(np.mean(collision_stats["reward_errors"])) if collision_stats["count"] > 0 else 0.0
    ord_mae = float(np.mean(ordinary_stats["reward_errors"]))
    print(f"Ordinary Transitions:       Count={ordinary_stats['count']:<5} | Reward MAE={ord_mae:.5f}")
    print(f"Collision-Critical Subset:  Count={collision_stats['count']:<5} | Reward MAE={coll_mae:.5f}")
    if coll_mae > ord_mae:
        ratio = coll_mae / max(1e-6, ord_mae)
        print(f">>> CRITICAL FINDING: Collision-critical reward error is {ratio:.1f}x HIGHER than ordinary error!")


def part6_true_simulator_oracle():
    print("\n" + "=" * 80)
    print("PART 6: TRUE-ENVIRONMENT SIMULATOR ORACLE SANITY CHECK")
    print("=" * 80)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication_tight.npz"
    cfg.env.id = "ego-merge-tight-v0"
    cfg.env.scaling = 1.75
    model_path = Path("outputs/personal_lab1/tight_scenario_attempt_01/my_lab1_world_model_tight.pt")

    env = CommunicationMergeWrapper(cfg=cfg, model_checkpoint_path=model_path, device="cpu")
    mpc = MergeMPCPlanner(model=env.model, num_sequences=64, horizon=6, allowed_actions=(0, 1, 3, 4), seed=42, device="cpu")

    # Pick 5 failure/divergence seeds
    test_seeds = [4003, 4011, 4016, 4022, 4038]

    for seed in test_seeds:
        env.reset(seed=seed)
        tf = env.request_message(1)
        base_unwrapped = env.env.env.unwrapped
        f_int = MSG_NAMES[tf]
        r_int = MSG_NAMES[env.request_message(2)]

        obs = env.current_obs
        latent = env.current_latent
        cand_rng = np.random.default_rng(seed * 777)
        cands = cand_rng.integers(0, len(mpc.allowed_actions), size=(mpc.num_sequences, mpc.horizon))
        cands = mpc.allowed_actions[cands]

        # MPC predicted best actions
        act_no, p_no, _ = mpc.plan(obs, latent, 0, 0, fixed_candidate_actions=cands)
        act_f, p_f, _ = mpc.plan(obs, latent, 1, tf, fixed_candidate_actions=cands)

        # Simulator true rollouts for all 5 driving actions from this exact initial state:
        action_sim_outcomes = {}
        for candidate_act in [0, 1, 2, 3, 4]:
            sim_env = Candidate4MergeEnv(cfg, request_cost=0.0)
            sim_env.reset(seed=seed)
            # Step candidate action
            _, r1, term1, trunc1, info1 = sim_env.step(candidate_act, 0)
            crashed = info1["crashed"]
            # Step default IDLE action for remaining horizon
            ret = r1
            d = term1 or trunc1
            st = 1
            while not d and st < 6:
                _, r_next, t_next, tr_next, inf_next = sim_env.step(1, 0)
                ret += r_next
                d = t_next or tr_next
                if inf_next["crashed"]:
                    crashed = True
                st += 1
            sim_env.close()
            action_sim_outcomes[candidate_act] = {"ret": ret, "crashed": crashed}

        true_best_act = max(action_sim_outcomes.keys(), key=lambda a: (not action_sim_outcomes[a]["crashed"], action_sim_outcomes[a]["ret"]))
        print(f"\nSeed {seed} | Front={f_int}, Rear={r_int}:")
        print(f"  NO_COMM MPC:      Act={ACTION_NAMES[act_no]} (pred_ret={p_no:.2f})")
        print(f"  ORACLE_FRONT MPC: Act={ACTION_NAMES[act_f]} (pred_ret={p_f:.2f})")
        print(f"  True Simulator Best Act: {ACTION_NAMES[true_best_act]} (sim_ret={action_sim_outcomes[true_best_act]['ret']:.2f}, crashed={action_sim_outcomes[true_best_act]['crashed']})")
        for a, out in action_sim_outcomes.items():
            print(f"    Act {ACTION_NAMES[a]:<10}: 6-step sim return={out['ret']:5.2f}, crashed={out['crashed']}")


def main():
    model_path = Path("outputs/personal_lab1/tight_scenario_attempt_01/my_lab1_world_model_tight.pt")
    part1_and_part2_paired_analysis(model_path, num_seeds=50, base_seed=4000)
    part4_and_part5_dataset_accuracy(model_path)
    part6_true_simulator_oracle()


if __name__ == "__main__":
    main()
