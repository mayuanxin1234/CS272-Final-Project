"""Full Double DQN Request Policy Training, Multi-Seed Evaluation, and Cost Sweep.

Final Candidate #2 Setup:
- State: [z_t(64), prev_request_onehot(3), prev_message_onehot(4)] = 71-D
- Actions: 0=NONE, 1=ASK_FRONT, 2=ASK_REAR
- Driving: Communication-aware WorldModel MPC
- Reward: driving_reward - lambda * I(request != NONE)
"""

from __future__ import annotations

import os
import sys
import time
import json
import random
import csv
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.models import WorldModel
from lab1.utils.model_handoff import load_personal_checkpoint
from request_agent.final_candidate2_env import FinalCandidate2MergeEnv
from request_agent.mpc_adapter import MergeMPCPlanner
from request_agent.dqn_agent import DoubleDQNAgent
from request_agent.replay_buffer import ReplayBuffer

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
REQ_NAMES = {0: "NONE", 1: "ASK_FRONT", 2: "ASK_REAR"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def get_latent_vector(model: WorldModel, obs: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Encode stacked grayscale observation into 64-D latent vector."""
    obs_t = (
        torch.from_numpy(obs.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        z = model.encoder(obs_t)
        return z.squeeze(0).cpu().numpy()


def build_dqn_state(z: np.ndarray, prev_req: int, prev_msg: int) -> np.ndarray:
    """Construct 71-D state vector: [z(64), prev_req_oh(3), prev_msg_oh(4)]."""
    req_oh = np.zeros(3, dtype=np.float32)
    req_oh[np.clip(prev_req, 0, 2)] = 1.0

    msg_oh = np.zeros(4, dtype=np.float32)
    msg_oh[np.clip(prev_msg, 0, 3)] = 1.0

    return np.concatenate([z, req_oh, msg_oh], axis=0)


def train_single_dqn(
    seed: int,
    lambda_cost: float = 0.05,
    num_episodes: int = 500,
    model_checkpoint_path: str = "outputs/personal_lab1/final_candidate2_attempt_01/my_lab1_world_model_final_candidate2.pt",
    output_dir: str = "outputs/request_agent/final_evaluation",
    log_interval: int = 25,
    device: str = "cpu",
) -> tuple[DoubleDQNAgent, dict[str, Any], Path]:
    print("=" * 80)
    print(f"TRAINING DOUBLE DQN REQUEST POLICY (Seed={seed}, Lambda={lambda_cost}, Episodes={num_episodes})")
    print("=" * 80)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_final_candidate2.npz"
    cfg.env.id = "ego-merge-final-candidate2-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75

    # Load frozen WorldModel
    wm = WorldModel.from_config(cfg.model, cfg.env, 5)
    load_personal_checkpoint(model_checkpoint_path, wm, cfg)
    wm.eval()

    # Create environment with exact lambda request cost
    env = FinalCandidate2MergeEnv(cfg, request_cost=lambda_cost)

    # Initialize MPC planner
    mpc = MergeMPCPlanner(
        model=wm,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=seed * 100,
        device=device,
    )

    # Initialize DQN agent and replay buffer
    agent = DoubleDQNAgent(
        state_dim=71,
        action_dim=3,
        hidden_dims=(128, 128),
        lr=5e-4,
        gamma=0.95,
        tau=0.005,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.992,
        grad_clip=1.0,
        device=device,
        seed=seed,
    )
    buffer = ReplayBuffer(capacity=10000, seed=seed)

    history: dict[str, list[Any]] = {
        "episode": [],
        "return_utility": [],
        "driving_return": [],
        "comm_cost": [],
        "crashed": [],
        "merged": [],
        "safe_merge": [],
        "steps": [],
        "requests": [],
        "bits": [],
        "req_counts": [],
        "epsilon": [],
        "loss": [],
    }

    t0 = time.time()
    total_steps = 0

    for ep in range(1, num_episodes + 1):
        # Training seeds 1..num_episodes are strictly distinct from held-out eval seeds (5000+)
        ep_seed = 10000 * seed + ep
        obs, info = env.reset(seed=ep_seed)

        prev_req = 0
        prev_msg = 0
        z_t = get_latent_vector(wm, obs, device=device)
        state = build_dqn_state(z_t, prev_req, prev_msg)

        ep_utility = 0.0
        ep_driving_ret = 0.0
        ep_comm_cost = 0.0
        ep_losses = []
        ep_req_counts = Counter()
        crashed = False
        merged = False
        done = False
        step = 0

        while not done and step < 20:
            # 1. DQN selects communication request action
            req_act = agent.select_action(state, evaluate=False)
            ep_req_counts[req_act] += 1

            # 2. Query environment channel
            msg = env.get_message(req_act)

            # 3. World-Model MPC chooses driving action conditioned on (req_act, msg)
            driving_act, _, _ = mpc.plan(obs, None, req_act, msg)

            # 4. Advance true environment physics
            next_obs, reward, term, trunc, s_info = env.step(driving_act, req_act)
            done = bool(term or trunc)

            ep_utility += float(reward)
            ep_driving_ret += float(s_info["raw_driving_reward"])
            ep_comm_cost += float(s_info["communication_cost"])

            if s_info["crashed"]:
                crashed = True
            if s_info["is_merged"]:
                merged = True

            # 5. Construct next state
            next_z = get_latent_vector(wm, next_obs, device=device)
            next_state = build_dqn_state(next_z, req_act, msg)

            # 6. Store transition in replay buffer
            buffer.push(state, req_act, float(reward), next_state, done)
            state = next_state
            prev_req = req_act
            prev_msg = msg
            obs = next_obs
            step += 1
            total_steps += 1

            # 7. Update DQN after warmup
            if total_steps > 64:
                loss = agent.update(buffer, batch_size=64)
                if loss is not None:
                    ep_losses.append(loss)

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        history["episode"].append(ep)
        history["return_utility"].append(ep_utility)
        history["driving_return"].append(ep_driving_ret)
        history["comm_cost"].append(ep_comm_cost)
        history["crashed"].append(crashed)
        history["merged"].append(merged)
        history["safe_merge"].append(merged and not crashed)
        history["steps"].append(step)
        history["requests"].append(int(s_info["request_count"]))
        history["bits"].append(int(s_info["total_bits"]))
        history["req_counts"].append(dict(ep_req_counts))
        history["epsilon"].append(agent.epsilon)
        history["loss"].append(avg_loss)

        if ep % log_interval == 0 or ep == num_episodes:
            recent_slice = slice(-log_interval, None)
            mean_util = float(np.mean(history["return_utility"][recent_slice]))
            mean_drv = float(np.mean(history["driving_return"][recent_slice]))
            safe_rate = float(np.mean(history["safe_merge"][recent_slice])) * 100.0
            mean_reqs = float(np.mean(history["requests"][recent_slice]))
            mean_bits = float(np.mean(history["bits"][recent_slice]))
            
            # Action distribution over last log_interval episodes
            total_acts = sum(sum(c.values()) for c in history["req_counts"][recent_slice])
            pct_none = sum(c.get(0, 0) for c in history["req_counts"][recent_slice]) / max(1, total_acts) * 100.0
            pct_front = sum(c.get(1, 0) for c in history["req_counts"][recent_slice]) / max(1, total_acts) * 100.0
            pct_rear = sum(c.get(2, 0) for c in history["req_counts"][recent_slice]) / max(1, total_acts) * 100.0

            print(
                f"Ep {ep:3d}/{num_episodes} ({time.time() - t0:.1f}s) | "
                f"Util: {mean_util:6.3f} | DrvRet: {mean_drv:6.3f} | SafeMerge: {safe_rate:5.1f}% | "
                f"Reqs: {mean_reqs:4.2f} ({mean_bits:4.1f}b) | None/F/R: {pct_none:4.1f}%/{pct_front:4.1f}%/{pct_rear:4.1f}% | "
                f"Eps: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}"
            )

    env.close()

    # Save model checkpoint and training curves
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir_path / f"dqn_seed{seed}_lambda{lambda_cost:.2f}.pt"
    agent.save(ckpt_path)

    hist_path = out_dir_path / f"dqn_seed{seed}_lambda{lambda_cost:.2f}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"Saved DQN checkpoint to {ckpt_path}")
    print(f"Saved training history to {hist_path}")
    return agent, history, ckpt_path


def evaluate_policy_on_heldout_seeds(
    policy_name: str,
    eval_seeds: list[int],
    lambda_cost: float,
    wm_checkpoint_path: str,
    dqn_agent: DoubleDQNAgent | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Evaluate a policy on fixed held-out test seeds."""
    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_final_candidate2.npz"
    cfg.env.id = "ego-merge-final-candidate2-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75

    wm = WorldModel.from_config(cfg.model, cfg.env, 5)
    load_personal_checkpoint(wm_checkpoint_path, wm, cfg)
    wm.eval()

    env = FinalCandidate2MergeEnv(cfg, request_cost=lambda_cost)
    mpc = MergeMPCPlanner(
        model=wm,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=9999,
        device=device,
    )

    records = []

    for seed in eval_seeds:
        obs, info = env.reset(seed=seed)
        t_front = env.get_message(env.ASK_FRONT)
        t_rear = env.get_message(env.ASK_REAR)

        prev_req = 0
        prev_msg = 0
        crashed = False
        merged = False
        done = False
        step = 0
        ep_utility = 0.0
        ep_driving_ret = 0.0
        ep_comm_cost = 0.0
        action_counts = Counter()
        req_counts = Counter()

        while not done and step < 20:
            z_t = get_latent_vector(wm, obs, device=device)

            if policy_name == "NO_COMM":
                req_act = 0  # NONE
                msg = 0      # UNKNOWN
            elif policy_name == "FULL_COMM":
                req_act = 2  # ASK_REAR every step
                msg = env.get_message(2)
            elif policy_name == "RANDOM_REQUEST":
                rng = np.random.default_rng(seed * 1000 + step)
                req_act = int(rng.choice([0, 1, 2]))
                msg = env.get_message(req_act)
            elif policy_name == "LEARNED_DQN":
                state = build_dqn_state(z_t, prev_req, prev_msg)
                req_act = dqn_agent.select_action(state, evaluate=True)
                msg = env.get_message(req_act)
            elif policy_name == "ORACLE_REAR":
                req_act = 0  # Zero communication cost
                msg = t_rear # True rear intent provided for free to MPC
            else:
                raise ValueError(f"Unknown policy {policy_name}")

            req_counts[req_act] += 1

            # World-Model MPC plan
            driving_act, _, _ = mpc.plan(obs, None, req_act, msg)
            action_counts[driving_act] += 1

            # True environment step
            # Note: For ORACLE_REAR, zero communication cost is charged
            cost_req = req_act if policy_name != "ORACLE_REAR" else 0
            next_obs, r, term, trunc, s_info = env.step(driving_act, cost_req)
            done = bool(term or trunc)

            ep_driving_ret += float(s_info["raw_driving_reward"])
            ep_comm_cost += float(s_info["communication_cost"])
            ep_utility += float(r)

            if s_info["crashed"]:
                crashed = True
            if s_info["is_merged"]:
                merged = True

            obs = next_obs
            prev_req = req_act
            prev_msg = msg
            step += 1

        records.append({
            "seed": seed,
            "policy": policy_name,
            "driving_return": ep_driving_ret,
            "utility": ep_utility,
            "comm_cost": ep_comm_cost,
            "crashed": crashed,
            "merged": merged,
            "safe_merge": bool(merged and not crashed),
            "travel_time": step,
            "requests": sum(v for k, v in req_counts.items() if k != 0),
            "bits": sum(v * 4 for k, v in req_counts.items() if k != 0),
            "req_counts": dict(req_counts),
            "action_counts": dict(action_counts),
            "rear_intent": MSG_NAMES[t_rear],
        })

    env.close()
    return records


def compute_policy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    drv_rets = [r["driving_return"] for r in records]
    utilities = [r["utility"] for r in records]
    crashes = [float(r["crashed"]) for r in records]
    merges = [float(r["merged"]) for r in records]
    safe_merges = [float(r["safe_merge"]) for r in records]
    travel_times = [float(r["travel_time"]) for r in records]
    reqs = [float(r["requests"]) for r in records]
    bits = [float(r["bits"]) for r in records]

    total_req_actions = sum(sum(r["req_counts"].values()) for r in records)
    none_count = sum(r["req_counts"].get(0, 0) for r in records)
    front_count = sum(r["req_counts"].get(1, 0) for r in records)
    rear_count = sum(r["req_counts"].get(2, 0) for r in records)

    return {
        "mean_driving_return": float(np.mean(drv_rets)),
        "std_driving_return": float(np.std(drv_rets)),
        "mean_utility": float(np.mean(utilities)),
        "std_utility": float(np.std(utilities)),
        "collision_rate": float(np.mean(crashes)),
        "merge_rate": float(np.mean(merges)),
        "safe_merge_rate": float(np.mean(safe_merges)),
        "mean_travel_time": float(np.mean(travel_times)),
        "std_travel_time": float(np.std(travel_times)),
        "mean_requests": float(np.mean(reqs)),
        "std_requests": float(np.std(reqs)),
        "mean_bits": float(np.mean(bits)),
        "std_bits": float(np.std(bits)),
        "pct_none": float(none_count / max(1, total_req_actions) * 100.0),
        "pct_ask_front": float(front_count / max(1, total_req_actions) * 100.0),
        "pct_ask_rear": float(rear_count / max(1, total_req_actions) * 100.0),
    }


def main():
    print("=" * 85)
    print("PHASE 6: DOUBLE DQN REQUEST POLICY & COMPREHENSIVE FINAL EVALUATION")
    print("=" * 85)

    out_dir = Path("outputs/request_agent/final_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    wm_ckpt = "outputs/personal_lab1/final_candidate2_attempt_01/my_lab1_world_model_final_candidate2.pt"

    # Held-out fixed test seeds (strictly separate from training)
    eval_seeds = [5000 + i for i in range(50)]

    # ---------------------------------------------------------
    # 1. Multi-seed training of DQN (lambda = 0.05, 3 seeds)
    # ---------------------------------------------------------
    dqn_seeds = [1, 2, 3]
    trained_agents = {}
    training_histories = {}

    for s in dqn_seeds:
        agent, hist, ckpt = train_single_dqn(
            seed=s,
            lambda_cost=0.05,
            num_episodes=500,
            model_checkpoint_path=wm_ckpt,
            output_dir=str(out_dir),
            log_interval=50,
        )
        trained_agents[s] = agent
        training_histories[s] = hist

    # ---------------------------------------------------------
    # 2. Communication-cost sweep: lambda in [0.00, 0.01, 0.10]
    # ---------------------------------------------------------
    cost_sweep_lambdas = [0.00, 0.01, 0.10]
    sweep_agents = {}
    sweep_histories = {}

    for l_val in cost_sweep_lambdas:
        agent, hist, ckpt = train_single_dqn(
            seed=1,
            lambda_cost=l_val,
            num_episodes=500,
            model_checkpoint_path=wm_ckpt,
            output_dir=str(out_dir),
            log_interval=50,
        )
        sweep_agents[l_val] = agent
        sweep_histories[l_val] = hist

    # ---------------------------------------------------------
    # 3. Final Baseline Evaluation on Held-Out Test Seeds
    # ---------------------------------------------------------
    print("\n" + "=" * 85)
    print("EVALUATING FINAL BASELINES ON 50 HELD-OUT TEST SEEDS (Seeds 5000-5049)")
    print("=" * 85)

    baseline_policies = ["NO_COMM", "FULL_COMM", "RANDOM_REQUEST", "ORACLE_REAR"]
    eval_results = {}

    for pol in baseline_policies:
        print(f"Evaluating baseline: {pol}...")
        recs = evaluate_policy_on_heldout_seeds(
            policy_name=pol,
            eval_seeds=eval_seeds,
            lambda_cost=0.05,
            wm_checkpoint_path=wm_ckpt,
            dqn_agent=None,
        )
        eval_results[pol] = {"records": recs, "summary": compute_policy_summary(recs)}

    # Evaluate each of the 3 trained DQN seeds (lambda=0.05)
    for s in dqn_seeds:
        pol_name = f"LEARNED_DQN_seed{s}"
        print(f"Evaluating {pol_name}...")
        recs = evaluate_policy_on_heldout_seeds(
            policy_name="LEARNED_DQN",
            eval_seeds=eval_seeds,
            lambda_cost=0.05,
            wm_checkpoint_path=wm_ckpt,
            dqn_agent=trained_agents[s],
        )
        eval_results[pol_name] = {"records": recs, "summary": compute_policy_summary(recs)}

    # Aggregate across the 3 DQN seeds
    dqn_summaries = [eval_results[f"LEARNED_DQN_seed{s}"]["summary"] for s in dqn_seeds]
    agg_dqn_summary = {}
    for k in dqn_summaries[0].keys():
        vals = [s[k] for s in dqn_summaries]
        agg_dqn_summary[k] = float(np.mean(vals))
        agg_dqn_summary[f"{k}_seed_std"] = float(np.std(vals))
    eval_results["LEARNED_DQN_AGGREGATE"] = {"summary": agg_dqn_summary}

    # Evaluate cost-sweep policies under their respective evaluation costs
    sweep_results = {}
    all_sweep_lambdas = [0.00, 0.01, 0.05, 0.10]
    for l_val in all_sweep_lambdas:
        ag = trained_agents[1] if l_val == 0.05 else sweep_agents[l_val]
        recs = evaluate_policy_on_heldout_seeds(
            policy_name="LEARNED_DQN",
            eval_seeds=eval_seeds,
            lambda_cost=l_val,
            wm_checkpoint_path=wm_ckpt,
            dqn_agent=ag,
        )
        sweep_results[f"lambda_{l_val:.2f}"] = {
            "lambda": l_val,
            "records": recs,
            "summary": compute_policy_summary(recs),
        }

    # ---------------------------------------------------------
    # 4. Save JSON and CSV Outputs
    # ---------------------------------------------------------
    # JSON results
    with open(out_dir / "final_evaluation_results.json", "w") as f:
        json.dump(
            {
                "baselines": {k: v["summary"] for k, v in eval_results.items()},
                "cost_sweep": {k: v["summary"] for k, v in sweep_results.items()},
            },
            f,
            indent=2,
            default=str,
        )

    # CSV: Baseline comparison table
    csv_baseline_path = out_dir / "baseline_comparison_table.csv"
    with open(csv_baseline_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Policy",
            "Driving Return (Mean)",
            "Driving Return (Std)",
            "Utility (Mean)",
            "Utility (Std)",
            "Collision Rate (%)",
            "Safe Merge Rate (%)",
            "Travel Time (Steps)",
            "Requests / Episode",
            "Bits / Episode",
            "% NONE",
            "% ASK_FRONT",
            "% ASK_REAR",
        ])
        for pol in ["NO_COMM", "FULL_COMM", "RANDOM_REQUEST", "LEARNED_DQN_seed1", "LEARNED_DQN_seed2", "LEARNED_DQN_seed3", "ORACLE_REAR"]:
            s = eval_results[pol]["summary"]
            writer.writerow([
                pol,
                f"{s['mean_driving_return']:.3f}",
                f"{s['std_driving_return']:.3f}",
                f"{s['mean_utility']:.3f}",
                f"{s['std_utility']:.3f}",
                f"{s['collision_rate']*100:.1f}",
                f"{s['safe_merge_rate']*100:.1f}",
                f"{s['mean_travel_time']:.2f}",
                f"{s['mean_requests']:.2f}",
                f"{s['mean_bits']:.1f}",
                f"{s['pct_none']:.1f}",
                f"{s['pct_ask_front']:.1f}",
                f"{s['pct_ask_rear']:.1f}",
            ])

    # CSV: Cost sweep table
    csv_sweep_path = out_dir / "cost_sweep_table.csv"
    with open(csv_sweep_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Lambda (Request Cost)",
            "Driving Return",
            "Final Utility",
            "Collision Rate (%)",
            "Safe Merge Rate (%)",
            "Requests / Episode",
            "Bits / Episode",
            "% NONE",
            "% ASK_FRONT",
            "% ASK_REAR",
        ])
        for l_val in all_sweep_lambdas:
            s = sweep_results[f"lambda_{l_val:.2f}"]["summary"]
            writer.writerow([
                f"{l_val:.2f}",
                f"{s['mean_driving_return']:.3f}",
                f"{s['mean_utility']:.3f}",
                f"{s['collision_rate']*100:.1f}",
                f"{s['safe_merge_rate']*100:.1f}",
                f"{s['mean_requests']:.2f}",
                f"{s['mean_bits']:.1f}",
                f"{s['pct_none']:.1f}",
                f"{s['pct_ask_front']:.1f}",
                f"{s['pct_ask_rear']:.1f}",
            ])

    print(f"\nSaved CSV baseline table: {csv_baseline_path}")
    print(f"Saved CSV cost sweep table: {csv_sweep_path}")
    print("\nDONE!")


if __name__ == "__main__":
    main()
