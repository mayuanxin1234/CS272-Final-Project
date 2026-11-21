"""Empirical Threshold Analysis and MPC Safety Ablation on Candidate 4."""

from __future__ import annotations

import os
import json
import time
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


def step1_empirical_threshold_analysis(model_path: Path):
    print("=" * 80)
    print("STEP 1: EMPIRICAL COLLISION-REWARD THRESHOLD ANALYSIS")
    print("=" * 80)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication_tight.npz"
    episodes = load_episodes(cfg.data.dataset_path)

    _, val_episodes = split_episodes(episodes, val_fraction=0.1, seed=cfg.seed)
    val_dataset = TransitionDataset(val_episodes)

    device = torch.device("cpu")
    model = WorldModel.from_config(cfg.model, cfg.env, 5).to(device)
    load_personal_checkpoint(model_path, model, cfg)
    model.eval()

    collision_preds = []
    ordinary_preds = []

    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            obs = sample["obs"].unsqueeze(0).to(device)
            act = sample["action"].unsqueeze(0).to(device)
            req = sample["request_action"].unsqueeze(0).to(device)
            msg = sample["message"].unsqueeze(0).to(device)
            true_r = sample["reward"].item()
            done = sample["done"].item()

            z = model.encode(obs)
            pred_r = model.predict_reward(z, act, req, msg).item()

            if true_r <= -0.5 or (done > 0.5 and true_r < 0.0):
                collision_preds.append(pred_r)
            else:
                ordinary_preds.append(pred_r)

    c_arr = np.array(collision_preds)
    o_arr = np.array(ordinary_preds)

    def stats(arr):
        return {
            "min": float(np.min(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    c_stats = stats(c_arr)
    o_stats = stats(o_arr)

    print(f"\nCollision Transitions (N={len(c_arr)}):")
    for k, v in c_stats.items():
        print(f"  {k:<8}: {v:+.5f}")

    print(f"\nOrdinary Transitions (N={len(o_arr)}):")
    for k, v in o_stats.items():
        print(f"  {k:<8}: {v:+.5f}")

    thresholds = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30]
    print("\nCandidate Threshold Evaluation:")
    print("-" * 80)
    print(f"{'Threshold':<10} | {'Collision Recall':<18} | {'Ordinary FPR':<16} | {'Precision':<12} | {'F1 Score'}")
    print("-" * 80)

    best_thresh = None
    best_f1 = -1.0
    thresh_table = []

    for th in thresholds:
        tp = np.sum(c_arr <= th)
        fn = np.sum(c_arr > th)
        fp = np.sum(o_arr <= th)
        tn = np.sum(o_arr > th)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"{th:<10.2f} | {recall*100:6.1f}% ({tp}/{len(c_arr)}) | {fpr*100:6.2f}% ({fp}/{len(o_arr)}) | {precision*100:6.1f}% | {f1:.3f}")
        row = {"threshold": th, "recall": float(recall), "fpr": float(fpr), "precision": float(precision), "f1": float(f1)}
        thresh_table.append(row)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

    print("-" * 80)
    print(f"Empirically selected threshold by F1 / Recall-FPR tradeoff: {best_thresh:.2f}")

    return best_thresh


class SafetyMergeMPCPlanner(MergeMPCPlanner):
    """Extended MPC planner supporting temporal discounting and trajectory termination on collision."""

    def __init__(
        self,
        model: WorldModel,
        num_sequences: int = 64,
        horizon: int = 6,
        allowed_actions: tuple[int, ...] = (0, 1, 3, 4),
        seed: int = 0,
        device: torch.device | str = "cpu",
        gamma: float = 1.0,
        collision_threshold: float | None = None,
    ) -> None:
        super().__init__(
            model=model,
            num_sequences=num_sequences,
            horizon=horizon,
            allowed_actions=allowed_actions,
            seed=seed,
            device=device,
        )
        self.gamma = float(gamma)
        self.collision_threshold = collision_threshold

    def plan(
        self,
        obs: np.ndarray,
        latent: np.ndarray | None = None,
        request_action: int = 0,
        message: int = 0,
        fixed_candidate_actions: np.ndarray | None = None,
    ) -> tuple[int, float, np.ndarray]:
        if fixed_candidate_actions is not None:
            candidate_actions = fixed_candidate_actions
        else:
            indices = self.rng.integers(
                0, len(self.allowed_actions), size=(self.num_sequences, self.horizon)
            )
            candidate_actions = self.allowed_actions[indices]

        actions_tensor = torch.from_numpy(candidate_actions).to(self.device)
        obs_tensor = (
            torch.from_numpy(obs.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .expand(self.num_sequences, -1, -1, -1)
            .to(self.device)
        )

        requests_tensor = torch.zeros(
            (self.num_sequences, self.horizon), dtype=torch.long, device=self.device
        )
        messages_tensor = torch.zeros(
            (self.num_sequences, self.horizon), dtype=torch.long, device=self.device
        )
        requests_tensor[:, 0] = int(request_action)
        messages_tensor[:, 0] = int(message)

        with torch.no_grad():
            imagined = self.model.imagine(
                obs_tensor,
                actions_tensor,
                requests_tensor,
                messages_tensor,
            )
            rewards = imagined["rewards"]  # shape (num_sequences, horizon)

            # Apply termination mask if collision threshold is set
            if self.collision_threshold is not None:
                # Mask out any rewards following a predicted collision
                is_crashed = rewards <= self.collision_threshold
                # Cumulative crash indicator along horizon
                has_crashed_before = torch.cumsum(is_crashed.long(), dim=1)
                # Keep step h where crash happened (has_crashed_before == 1 and is_crashed),
                # zero out subsequent steps (has_crashed_before > 1 or (has_crashed_before == 1 and not is_crashed))
                active_mask = (has_crashed_before == 0) | ((has_crashed_before == 1) & is_crashed)
                effective_rewards = rewards * active_mask.float()
            else:
                effective_rewards = rewards

            # Apply discounting
            if self.gamma < 1.0:
                discount_weights = torch.tensor(
                    [self.gamma ** h for h in range(self.horizon)],
                    dtype=torch.float32,
                    device=self.device,
                )
                predicted_returns = (effective_rewards * discount_weights).sum(dim=1)
            else:
                predicted_returns = effective_rewards.sum(dim=1)

            best_idx = torch.argmax(predicted_returns).item()

        best_first_action = int(candidate_actions[best_idx, 0])
        best_return = float(predicted_returns[best_idx].item())
        candidate_returns = predicted_returns.detach().cpu().numpy()

        return best_first_action, best_return, candidate_returns


def evaluate_variant(
    variant_name: str,
    planner: SafetyMergeMPCPlanner,
    env: CommunicationMergeWrapper,
    seeds: list[int],
) -> dict:
    policies = ["NO_COMM", "ORACLE_FRONT", "ORACLE_REAR", "ORACLE_BEST"]
    seed_records = {p: [] for p in policies}
    action_changes = {"front": 0, "rear": 0, "total_steps": 0}

    for seed in seeds:
        for pol in policies:
            env.reset(seed=seed)
            tf = env.request_message(1)
            tr = env.request_message(2)

            ep_ret = 0.0
            crashed = False
            merged = False
            done = False
            step = 0

            cand_rng = np.random.default_rng(seed * 777)

            while not done and step < 20:
                obs = env.current_obs
                latent = env.current_latent
                cands = cand_rng.integers(0, len(planner.allowed_actions), size=(planner.num_sequences, planner.horizon))
                cands = planner.allowed_actions[cands]

                if pol == "NO_COMM":
                    act, _, _ = planner.plan(obs, latent, 0, 0, fixed_candidate_actions=cands)
                    a_f, _, _ = planner.plan(obs, latent, 1, tf, fixed_candidate_actions=cands)
                    a_r, _, _ = planner.plan(obs, latent, 2, tr, fixed_candidate_actions=cands)
                    if a_f != act:
                        action_changes["front"] += 1
                    if a_r != act:
                        action_changes["rear"] += 1
                    action_changes["total_steps"] += 1
                elif pol == "ORACLE_FRONT":
                    act, _, _ = planner.plan(obs, latent, 1, tf, fixed_candidate_actions=cands)
                elif pol == "ORACLE_REAR":
                    act, _, _ = planner.plan(obs, latent, 2, tr, fixed_candidate_actions=cands)
                elif pol == "ORACLE_BEST":
                    act_f, ret_f, _ = planner.plan(obs, latent, 1, tf, fixed_candidate_actions=cands)
                    act_r, ret_r, _ = planner.plan(obs, latent, 2, tr, fixed_candidate_actions=cands)
                    act = act_f if ret_f >= ret_r else act_r

                _, _, term, trunc, info = env.step(act, 0, 0)
                done = bool(term or trunc)
                ep_ret += info["raw_driving_reward"]
                if info["crashed"]:
                    crashed = True
                if info["is_merged"]:
                    merged = True
                step += 1

            seed_records[pol].append({
                "seed": seed,
                "return": ep_ret,
                "crashed": crashed,
                "merged": merged,
                "safe_merge": merged and (not crashed),
            })

    # Summarize
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

        summary[pol] = {
            "mean_return": float(np.mean(returns)),
            "collision_rate": float(np.mean(crashes)),
            "merge_rate": float(np.mean(merges)),
            "safe_merge_rate": float(np.mean(safe_merges)),
            "paired_improvement": float(np.mean(paired_diffs)),
            "net_crashes_avoided": int(crashes_avoided - crashes_introduced),
        }

    tot_steps = max(1, action_changes["total_steps"])
    pct_f = (action_changes["front"] / tot_steps) * 100.0
    pct_r = (action_changes["rear"] / tot_steps) * 100.0

    return {
        "variant": variant_name,
        "summary": summary,
        "action_changes": {"pct_front": pct_f, "pct_rear": pct_r},
        "seed_records": seed_records,
    }


def run_full_ablation():
    model_path = Path("outputs/personal_lab1/tight_scenario_attempt_01/my_lab1_world_model_tight.pt")
    selected_threshold = step1_empirical_threshold_analysis(model_path)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication_tight.npz"
    cfg.data.num_transitions = 20000
    cfg.env.id = "ego-merge-tight-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75
    cfg.training.device = "cpu"

    env = CommunicationMergeWrapper(cfg=cfg, model_checkpoint_path=model_path, device="cpu")
    seeds = [4000 + i for i in range(50)]

    variants = [
        ("A. BASE_MPC (gamma=1.0, no threshold)", {"gamma": 1.0, "collision_threshold": None}),
        ("B1. DISCOUNT_ONLY (gamma=0.90)", {"gamma": 0.90, "collision_threshold": None}),
        ("B2. DISCOUNT_ONLY (gamma=0.85)", {"gamma": 0.85, "collision_threshold": None}),
        ("C. TERMINATION_ONLY (gamma=1.0, thresh=-0.30)", {"gamma": 1.0, "collision_threshold": selected_threshold}),
        ("D1. DISCOUNT_PLUS_TERM (gamma=0.90, thresh=-0.30)", {"gamma": 0.90, "collision_threshold": selected_threshold}),
        ("D2. DISCOUNT_PLUS_TERM (gamma=0.85, thresh=-0.30)", {"gamma": 0.85, "collision_threshold": selected_threshold}),
    ]

    all_results = {}
    print("\n" + "=" * 80)
    print("STEP 3: ORACLE EVALUATION FOR EVERY MPC VARIANT (50 SEEDS)")
    print("=" * 80)

    for v_name, v_params in variants:
        t0 = time.time()
        planner = SafetyMergeMPCPlanner(
            model=env.model,
            num_sequences=64,
            horizon=6,
            allowed_actions=(0, 1, 3, 4),
            seed=42,
            device="cpu",
            gamma=v_params["gamma"],
            collision_threshold=v_params["collision_threshold"],
        )
        res = evaluate_variant(v_name, planner, env, seeds)
        all_results[v_name] = res
        print(f"\nEvaluated {v_name} in {time.time() - t0:.1f}s:")
        for pol, s in res["summary"].items():
            print(f"  {pol:<13}: SafeMerge={s['safe_merge_rate']*100:5.1f}%, Collision={s['collision_rate']*100:5.1f}%, Return={s['mean_return']:6.3f} (paired diff={s['paired_improvement']:+6.3f}, net avoided={s['net_crashes_avoided']})")
        print(f"  Action changes: Front={res['action_changes']['pct_front']:.1f}%, Rear={res['action_changes']['pct_rear']:.1f}%")

    # Step 4: Paired failure seeds analysis
    print("\n" + "=" * 80)
    print("STEP 4: PAIRED FAILURE SEEDS ANALYSIS (SEEDS 4003, 4024, 4032, 4047)")
    print("=" * 80)
    target_seeds = [4003, 4024, 4032, 4047]
    for v_name, res in all_results.items():
        print(f"\n--- {v_name} ---")
        recs = res["seed_records"]
        for ts in target_seeds:
            nc = next(r for r in recs["NO_COMM"] if r["seed"] == ts)
            of = next(r for r in recs["ORACLE_FRONT"] if r["seed"] == ts)
            ob = next(r for r in recs["ORACLE_BEST"] if r["seed"] == ts)
            print(f"  Seed {ts}: NC_crash={nc['crashed']} (ret={nc['return']:.2f}) | OF_crash={of['crashed']} (ret={of['return']:.2f}) | OB_crash={ob['crashed']} (ret={ob['return']:.2f})")

    # Save to disk
    out_file = Path("outputs/request_agent/mpc_safety_ablation_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    clean_json = {}
    for k, v in all_results.items():
        clean_json[k] = {
            "summary": v["summary"],
            "action_changes": v["action_changes"],
        }
    with open(out_file, "w") as f:
        json.dump(clean_json, f, indent=2)
    print(f"\nSaved ablation results to {out_file}")


if __name__ == "__main__":
    run_full_ablation()
