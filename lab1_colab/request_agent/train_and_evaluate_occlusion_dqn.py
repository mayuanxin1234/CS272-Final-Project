"""Full Double DQN Request Policy Training, Evaluation, and Cost Sweep for Frozen Occlusion Scenario.

Final Occlusion Setup:
- Scenario: Candidate A (Cooperative Blind-Zone Sensing in Target-Lane Merge)
- Environment: ego-merge-occlusion-final-v0
- Frozen WorldModel: outputs/personal_lab1/occlusion_final_attempt_01/my_lab1_world_model_occlusion_final.pt
- State: [z_t(64), prev_request_onehot(3), prev_message_onehot(4)] = 71-D
- Request Actions: 0=NONE, 1=ASK_FRONT, 2=ASK_REAR
- Messages: 0=UNKNOWN, 1=CLEAR, 2=HAZARD_AHEAD, 3=HAZARD_BEHIND
- Driving Planner: Frozen learned WorldModel MPC (H=6, N=64)
- Request Reward: driving_reward - lambda * I(request != NONE)
"""

from __future__ import annotations

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.models import WorldModel
from lab1.utils.checkpoint import load_checkpoint
from request_agent.occlusion_env import FinalOcclusionMergeEnv
from request_agent.mpc_adapter import MergeMPCPlanner
from request_agent.dqn_agent import DoubleDQNAgent
from request_agent.replay_buffer import ReplayBuffer

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
REQ_NAMES = {0: "NONE", 1: "ASK_FRONT", 2: "ASK_REAR"}
MSG_NAMES = {0: "UNKNOWN", 1: "CLEAR", 2: "HAZARD_AHEAD", 3: "HAZARD_BEHIND"}
HAZARD_NAMES = {0: "CLEAR", 1: "FRONT_HAZARD", 2: "REAR_HAZARD"}


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

    state = np.concatenate([z, req_oh, msg_oh], axis=0)
    assert state.shape == (71,), f"State shape mismatch: {state.shape} != (71,)"
    return state


def verify_dqn_state_isolation(env: FinalOcclusionMergeEnv, model: WorldModel) -> bool:
    """Programmatically verify zero leakage of hidden hazard state into DQN input."""
    print("\n" + "=" * 80)
    print("PROGRAMMATIC VERIFICATION OF DQN STATE ISOLATION")
    print("=" * 80)

    diffs = []
    for test_seed in [101, 202, 303, 404, 505]:
        # Reset under CLEAR (hazard=0)
        env.hazard_state = 0
        env.env.unwrapped.hazard_state = 0
        obs0, _ = env.env.reset(seed=test_seed)
        z0 = get_latent_vector(model, obs0)
        s0 = build_dqn_state(z0, prev_req=0, prev_msg=0)

        # Reset under FRONT_HAZARD (hazard=1)
        env.hazard_state = 1
        env.env.unwrapped.hazard_state = 1
        obs1, _ = env.env.reset(seed=test_seed)
        z1 = get_latent_vector(model, obs1)
        s1 = build_dqn_state(z1, prev_req=0, prev_msg=0)

        # Reset under REAR_HAZARD (hazard=2)
        env.hazard_state = 2
        env.env.unwrapped.hazard_state = 2
        obs2, _ = env.env.reset(seed=test_seed)
        z2 = get_latent_vector(model, obs2)
        s2 = build_dqn_state(z2, prev_req=0, prev_msg=0)

        d01 = float(np.max(np.abs(s0 - s1)))
        d02 = float(np.max(np.abs(s0 - s2)))
        diffs.append((d01, d02))
        print(f"Seed {test_seed}: max_dqn_state_diff(CLEAR, FRONT) = {d01:.6f}, max_dqn_state_diff(CLEAR, REAR) = {d02:.6f}")

    all_zero = all(d01 == 0.0 and d02 == 0.0 for d01, d02 in diffs)
    print(f"Verification Verdict: {'PASSED (Zero Leakage into DQN State)' if all_zero else 'FAILED (Privileged Leakage Detected)'}")
    print("=" * 80 + "\n")
    assert all_zero, "CRITICAL ERROR: Privileged hazard information leaked into DQN state vector!"
    return all_zero


def train_single_dqn(
    seed: int,
    lambda_cost: float = 0.05,
    num_episodes: int = 1000,
    model_checkpoint_path: str = "outputs/personal_lab1/occlusion_final_attempt_01/my_lab1_world_model_occlusion_final.pt",
    output_dir: str = "outputs/request_agent/occlusion_final_evaluation",
    log_interval: int = 100,
    device: str = "cpu",
) -> tuple[DoubleDQNAgent, dict[str, Any], Path]:
    print("=" * 85)
    print(f"TRAINING FINAL DOUBLE DQN REQUEST POLICY")
    print(f"Seed={seed}, Lambda={lambda_cost}, Episodes={num_episodes}")
    print("=" * 85)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "ego-merge-occlusion-final-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75

    # Load frozen WorldModel
    wm = WorldModel.from_config(cfg.model, cfg.env, 5)
    load_checkpoint(model_checkpoint_path, wm, map_location="cpu")
    wm.eval()

    env = FinalOcclusionMergeEnv(cfg, request_cost=lambda_cost)

    mpc = MergeMPCPlanner(
        model=wm,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=seed * 100,
        device=device,
        future_mode="persistent",
    )

    agent = DoubleDQNAgent(
        state_dim=71,
        action_dim=3,
        hidden_dims=(128, 128),
        lr=5e-4,
        gamma=0.95,
        tau=0.005,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        grad_clip=1.0,
        device=device,
        seed=seed,
    )
    buffer = ReplayBuffer(capacity=20000, seed=seed)

    history: dict[str, list[Any]] = {
        "episode": [],
        "return_utility": [],
        "driving_return": [],
        "crashed": [],
        "safe_merge": [],
        "requests": [],
        "bits": [],
        "steps": [],
        "epsilon": [],
        "loss": [],
        "req_counts": [],
    }

    recent_window = 100
    t0 = time.time()

    for ep in range(1, num_episodes + 1):
        # Training seed: distinct from held-out seeds (e.g. seed*10000 + ep)
        ep_seed = seed * 10000 + ep
        obs, info = env.reset(seed=ep_seed)

        # Initial DQN state
        z = get_latent_vector(wm, obs, device=device)
        state = build_dqn_state(z, prev_req=0, prev_msg=0)

        prev_req = 0
        prev_msg = 0
        ep_utility = 0.0
        ep_driving_r = 0.0
        crashed = False
        merged = False
        step = 0
        ep_requests = 0
        ep_req_counter = Counter()
        losses = []

        done = False
        while not done and step < 8:
            # 1. DQN selects request action
            req_act = agent.select_action(state, evaluate=False)
            ep_req_counter[req_act] += 1
            if req_act != env.NONE:
                ep_requests += 1

            # 2. Receive semantic message
            msg = env.get_message(req_act)

            # 3. Driving controlled by frozen WorldModel MPC
            mpc.rng = np.random.default_rng(ep_seed * 100 + step)
            driving_act, _, _ = mpc.plan(
                obs=obs,
                latent=z,
                request_action=req_act,
                message=msg,
            )

            # 4. Step environment
            next_obs, u_rew, term, trunc, step_info = env.step(driving_act, req_act)
            done = bool(term or trunc)

            ep_utility += u_rew
            ep_driving_r += step_info["raw_driving_reward"]
            if step_info["crashed"]:
                crashed = True
            if step_info["is_merged"]:
                merged = True

            # 5. Build next DQN state
            next_z = get_latent_vector(wm, next_obs, device=device)
            next_state = build_dqn_state(next_z, prev_req=req_act, prev_msg=msg)

            # 6. DQN reward: u_rew is (raw_driving_reward - lambda * I(request != NONE))
            # Push transition into replay buffer
            buffer.push(state, req_act, u_rew, next_state, done)

            # 7. Train DQN step
            loss = agent.update(buffer, batch_size=64)
            if loss is not None:
                losses.append(loss)

            state = next_state
            z = next_z
            obs = next_obs
            prev_req = req_act
            prev_msg = msg
            step += 1

        mean_loss = float(np.mean(losses)) if losses else 0.0
        safe_merge = bool(merged and not crashed)

        history["episode"].append(ep)
        history["return_utility"].append(ep_utility)
        history["driving_return"].append(ep_driving_r)
        history["crashed"].append(crashed)
        history["safe_merge"].append(safe_merge)
        history["requests"].append(ep_requests)
        history["bits"].append(ep_requests * 4)
        history["steps"].append(step)
        history["epsilon"].append(agent.epsilon)
        history["loss"].append(mean_loss)
        history["req_counts"].append(dict(ep_req_counter))

        if ep % log_interval == 0 or ep == num_episodes:
            recent_slice = slice(max(0, ep - recent_window), ep)
            r_util = np.mean(history["return_utility"][recent_slice])
            r_drv = np.mean(history["driving_return"][recent_slice])
            r_safe = np.mean(history["safe_merge"][recent_slice]) * 100.0
            r_crash = np.mean(history["crashed"][recent_slice]) * 100.0
            r_reqs = np.mean(history["requests"][recent_slice])
            r_loss = np.mean(history["loss"][recent_slice])

            total_req_actions = sum(
                sum(c.values()) for c in history["req_counts"][recent_slice]
            )
            pct_none = (
                sum(c.get(0, 0) for c in history["req_counts"][recent_slice])
                / max(1, total_req_actions)
                * 100.0
            )
            pct_front = (
                sum(c.get(1, 0) for c in history["req_counts"][recent_slice])
                / max(1, total_req_actions)
                * 100.0
            )
            pct_rear = (
                sum(c.get(2, 0) for c in history["req_counts"][recent_slice])
                / max(1, total_req_actions)
                * 100.0
            )

            elapsed = time.time() - t0
            print(
                f"Ep {ep:4d}/{num_episodes} | "
                f"Util: {r_util:>6.3f} | "
                f"DrvRet: {r_drv:>6.3f} | "
                f"Safe: {r_safe:>5.1f}% | "
                f"Crash: {r_crash:>5.1f}% | "
                f"Reqs/Ep: {r_reqs:>4.2f} | "
                f"Reqs: [{pct_none:>4.1f}% N, {pct_front:>4.1f}% F, {pct_rear:>4.1f}% R] | "
                f"Eps: {agent.epsilon:.3f} | "
                f"Loss: {r_loss:.4f} | "
                f"{elapsed:.1f}s"
            )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"dqn_seed{seed}_lambda{lambda_cost:.2f}.pt"
    agent.save(str(ckpt_path))

    hist_path = out_dir / f"dqn_seed{seed}_lambda{lambda_cost:.2f}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Saved DQN checkpoint to: {ckpt_path}")
    print(f"Saved training history to: {hist_path}")
    return agent, history, ckpt_path


def evaluate_policy_on_held_out(
    policy_type: str,
    dqn_agent: DoubleDQNAgent | None,
    wm: WorldModel,
    num_seeds: int = 60,
    base_seed: int = 9000,
    lambda_cost: float = 0.05,
    device: str = "cpu",
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[int, int]]]:
    """Evaluate a policy across fixed held-out seeds and compute request breakdown by true hidden state."""
    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "ego-merge-occlusion-final-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75

    env = FinalOcclusionMergeEnv(cfg, request_cost=lambda_cost)
    mpc = MergeMPCPlanner(
        model=wm,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=base_seed,
        device=device,
        future_mode="persistent",
    )

    records = []
    # Request confusion tracking: true_hazard -> {req_0: count, req_1: count, req_2: count}
    confusion_tracker = {0: Counter(), 1: Counter(), 2: Counter()}

    for idx in range(num_seeds):
        seed = base_seed + idx
        hazard_type = seed % 3
        h_name = HAZARD_NAMES[hazard_type]

        obs, info = env.reset(seed=seed)
        z = get_latent_vector(wm, obs, device=device)
        state = build_dqn_state(z, prev_req=0, prev_msg=0)

        ep_utility = 0.0
        ep_driving_r = 0.0
        crashed = False
        merged = False
        step = 0
        req_count = 0
        ep_req_acts = []
        revealed_msg = env.MSG_UNKNOWN

        done = False
        while not done and step < 8:
            # Determine request action
            if policy_type == "NO_COMM":
                req = env.NONE
            elif policy_type == "FULL_COMM":
                # Relevant CAV at t=0
                if step == 0:
                    req = env.ASK_FRONT if hazard_type == 1 else (env.ASK_REAR if hazard_type == 2 else env.ASK_FRONT)
                else:
                    req = env.NONE
            elif policy_type == "ORACLE_SELECTIVE":
                if step == 0:
                    if hazard_type == 1:
                        req = env.ASK_FRONT
                    elif hazard_type == 2:
                        req = env.ASK_REAR
                    else:
                        req = env.NONE
                else:
                    req = env.NONE
            elif policy_type == "RANDOM_REQUEST":
                if step == 0:
                    rng = np.random.default_rng(seed * 100 + step)
                    req = int(rng.choice([env.NONE, env.ASK_FRONT, env.ASK_REAR]))
                else:
                    req = env.NONE
            elif policy_type == "LEARNED_DQN":
                assert dqn_agent is not None
                req = dqn_agent.select_action(state, evaluate=True)
            else:
                raise ValueError(f"Unknown policy: {policy_type}")

            ep_req_acts.append(req)
            if req != env.NONE:
                req_count += 1

            # Track request choice at t=0 for confusion analysis
            if step == 0:
                confusion_tracker[hazard_type][req] += 1

            msg = env.get_message(req)
            if msg != env.MSG_UNKNOWN:
                revealed_msg = msg

            # Driving controlled by MPC
            mpc.rng = np.random.default_rng(seed * 1000 + step)
            driving_act, _, _ = mpc.plan(
                obs=obs,
                latent=z,
                request_action=req,
                message=revealed_msg,
            )

            next_obs, u_rew, term, trunc, step_info = env.step(driving_act, req)
            done = bool(term or trunc)

            ep_utility += u_rew
            ep_driving_r += step_info["raw_driving_reward"]
            if step_info["crashed"]:
                crashed = True
            if step_info["is_merged"]:
                merged = True

            next_z = get_latent_vector(wm, next_obs, device=device)
            next_state = build_dqn_state(next_z, prev_req=req, prev_msg=msg)

            state = next_state
            z = next_z
            obs = next_obs
            step += 1

        records.append({
            "seed": seed,
            "hazard_state": h_name,
            "utility": ep_utility,
            "driving_return": ep_driving_r,
            "crashed": crashed,
            "safe_merge": bool(merged and not crashed),
            "travel_time": step,
            "requests": req_count,
            "bits": req_count * 4,
            "request_actions": ep_req_acts,
        })

    # Summary
    safe_merges = float(np.mean([r["safe_merge"] for r in records]) * 100.0)
    collisions = float(np.mean([r["crashed"] for r in records]) * 100.0)
    drv_r = float(np.mean([r["driving_return"] for r in records]))
    util = float(np.mean([r["utility"] for r in records]))
    travel_t = float(np.mean([r["travel_time"] for r in records]))
    reqs = float(np.mean([r["requests"] for r in records]))
    bits = float(np.mean([r["bits"] for r in records]))

    all_req_acts = [act for r in records for act in r["request_actions"]]
    pct_none = float(np.mean([a == 0 for a in all_req_acts]) * 100.0) if all_req_acts else 100.0
    pct_front = float(np.mean([a == 1 for a in all_req_acts]) * 100.0) if all_req_acts else 0.0
    pct_rear = float(np.mean([a == 2 for a in all_req_acts]) * 100.0) if all_req_acts else 0.0

    summary = {
        "policy": policy_type,
        "safe_merge_rate": safe_merges,
        "collision_rate": collisions,
        "mean_driving_return": drv_r,
        "mean_utility": util,
        "mean_travel_time": travel_t,
        "mean_requests": reqs,
        "mean_bits": bits,
        "pct_none": pct_none,
        "pct_front": pct_front,
        "pct_rear": pct_rear,
    }

    # Breakdown by hazard
    hazard_breakdown = {}
    for h_code, h_name in HAZARD_NAMES.items():
        sub = [r for r in records if r["hazard_state"] == h_name]
        hazard_breakdown[h_name] = {
            "safe_merge_rate": float(np.mean([r["safe_merge"] for r in sub]) * 100.0) if sub else 0.0,
            "collision_rate": float(np.mean([r["crashed"] for r in sub]) * 100.0) if sub else 0.0,
            "driving_return": float(np.mean([r["driving_return"] for r in sub])) if sub else 0.0,
            "utility": float(np.mean([r["utility"] for r in sub])) if sub else 0.0,
            "requests": float(np.mean([r["requests"] for r in sub])) if sub else 0.0,
        }
    summary["hazard_breakdown"] = hazard_breakdown

    return summary, records, confusion_tracker


def run_full_evaluation_pipeline() -> dict[str, Any]:
    out_dir = Path("outputs/request_agent/occlusion_final_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    wm_path = "outputs/personal_lab1/occlusion_final_attempt_01/my_lab1_world_model_occlusion_final.pt"
    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "ego-merge-occlusion-final-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75

    wm = WorldModel.from_config(cfg.model, cfg.env, 5)
    load_checkpoint(wm_path, wm, map_location="cpu")
    wm.eval()

    test_env = FinalOcclusionMergeEnv(cfg)
    verify_dqn_state_isolation(test_env, wm)

    # 1. Train 3 seeds with lambda=0.05
    seeds = [1, 2, 3]
    trained_agents = {}
    histories = {}
    dqn_eval_summaries = []
    dqn_eval_records = {}
    confusion_aggregates = {0: Counter(), 1: Counter(), 2: Counter()}

    eval_only = "--eval_only" in sys.argv or "--eval-only" in sys.argv
    print("\n" + "=" * 85)
    if eval_only:
        print("PHASE 1: EVALUATING PRE-TRAINED MULTI-SEED DOUBLE DQN POLICIES (lambda = 0.05)")
    else:
        print("PHASE 1: TRAINING MULTI-SEED DOUBLE DQN POLICIES (lambda = 0.05)")
    print("=" * 85)

    for seed in seeds:
        ckpt = out_dir / f"dqn_seed{seed}_lambda0.05.pt"
        if eval_only and ckpt.exists():
            print(f"Loading pre-trained DQN checkpoint: {ckpt}")
            agent = DoubleDQNAgent(state_dim=71, action_dim=3, device="cpu", seed=seed)
            agent.load(str(ckpt))
            trained_agents[seed] = agent
        else:
            agent, hist, ckpt = train_single_dqn(
                seed=seed,
                lambda_cost=0.05,
                num_episodes=1000,
                model_checkpoint_path=wm_path,
                output_dir=str(out_dir),
                log_interval=100,
            )
            trained_agents[seed] = agent
            histories[seed] = hist

        # Evaluate greedy trained agent on held-out seeds (9000..9059)
        summary, recs, conf = evaluate_policy_on_held_out(
            policy_type="LEARNED_DQN",
            dqn_agent=agent,
            wm=wm,
            num_seeds=60,
            base_seed=9000,
            lambda_cost=0.05,
        )
        summary["seed"] = seed
        dqn_eval_summaries.append(summary)
        dqn_eval_records[f"seed_{seed}"] = recs

        for h in [0, 1, 2]:
            confusion_aggregates[h].update(conf[h])

    # 2. Evaluate Baselines (NO_COMM, FULL_COMM, RANDOM_REQUEST, ORACLE_SELECTIVE)
    print("\n" + "=" * 85)
    print("PHASE 2: EVALUATING BASELINES ON 60 HELD-OUT SEEDS")
    print("=" * 85)

    baseline_summaries = {}
    baseline_records = {}
    for pol in ["NO_COMM", "FULL_COMM", "RANDOM_REQUEST", "ORACLE_SELECTIVE"]:
        s, r, _ = evaluate_policy_on_held_out(
            policy_type=pol,
            dqn_agent=None,
            wm=wm,
            num_seeds=60,
            base_seed=9000,
            lambda_cost=0.05,
        )
        baseline_summaries[pol] = s
        baseline_records[pol] = r

    # Compute aggregate 3-seed DQN metrics
    agg_dqn = {
        "policy": "LEARNED_DQN (3-seed mean)",
        "safe_merge_rate": float(np.mean([s["safe_merge_rate"] for s in dqn_eval_summaries])),
        "safe_merge_std": float(np.std([s["safe_merge_rate"] for s in dqn_eval_summaries])),
        "collision_rate": float(np.mean([s["collision_rate"] for s in dqn_eval_summaries])),
        "collision_std": float(np.std([s["collision_rate"] for s in dqn_eval_summaries])),
        "mean_driving_return": float(np.mean([s["mean_driving_return"] for s in dqn_eval_summaries])),
        "mean_utility": float(np.mean([s["mean_utility"] for s in dqn_eval_summaries])),
        "mean_travel_time": float(np.mean([s["mean_travel_time"] for s in dqn_eval_summaries])),
        "mean_requests": float(np.mean([s["mean_requests"] for s in dqn_eval_summaries])),
        "mean_bits": float(np.mean([s["mean_bits"] for s in dqn_eval_summaries])),
        "pct_none": float(np.mean([s["pct_none"] for s in dqn_eval_summaries])),
        "pct_front": float(np.mean([s["pct_front"] for s in dqn_eval_summaries])),
        "pct_rear": float(np.mean([s["pct_rear"] for s in dqn_eval_summaries])),
    }

    # Aggregate breakdown
    agg_breakdown = {}
    for h_name in ["CLEAR", "FRONT_HAZARD", "REAR_HAZARD"]:
        agg_breakdown[h_name] = {
            "safe_merge_rate": float(np.mean([s["hazard_breakdown"][h_name]["safe_merge_rate"] for s in dqn_eval_summaries])),
            "collision_rate": float(np.mean([s["hazard_breakdown"][h_name]["collision_rate"] for s in dqn_eval_summaries])),
            "driving_return": float(np.mean([s["hazard_breakdown"][h_name]["driving_return"] for s in dqn_eval_summaries])),
            "utility": float(np.mean([s["hazard_breakdown"][h_name]["utility"] for s in dqn_eval_summaries])),
            "requests": float(np.mean([s["hazard_breakdown"][h_name]["requests"] for s in dqn_eval_summaries])),
        }
    agg_dqn["hazard_breakdown"] = agg_breakdown

    # Print baseline comparison table
    print("\n" + "=" * 95)
    print("FINAL HELD-OUT BASELINE COMPARISON TABLE (60 seeds, lambda = 0.05)")
    print("=" * 95)
    headers = ["Policy", "Safe Merge", "Collision", "Driving Ret", "Utility", "Time", "Reqs/Ep", "Bits/Ep"]
    print(f"{headers[0]:<25} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<12} | {headers[4]:<10} | {headers[5]:<6} | {headers[6]:<8} | {headers[7]}")
    print("-" * 95)

    all_pol_summaries = [
        baseline_summaries["NO_COMM"],
        baseline_summaries["RANDOM_REQUEST"],
        agg_dqn,
        baseline_summaries["FULL_COMM"],
        baseline_summaries["ORACLE_SELECTIVE"],
    ]

    for p in all_pol_summaries:
        p_name = p["policy"]
        sm = f"{p['safe_merge_rate']:.1f}%"
        cr = f"{p['collision_rate']:.1f}%"
        dr = f"{p['mean_driving_return']:.3f}"
        ut = f"{p['mean_utility']:.3f}"
        tt = f"{p['mean_travel_time']:.1f}"
        rq = f"{p['mean_requests']:.2f}"
        bt = f"{p['mean_bits']:.1f}b"
        print(f"{p_name:<25} | {sm:>9}  | {cr:>9}  | {dr:>11}  | {ut:>9}  | {tt:>5}  | {rq:>7}  | {bt:>6}")

    # Write CSV
    csv_path = out_dir / "baseline_comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Policy", "Safe Merge Rate (%)", "Collision Rate (%)", "Mean Driving Return", "Mean Utility", "Mean Travel Time", "Mean Requests", "Mean Bits", "% NONE", "% ASK_FRONT", "% ASK_REAR"])
        for p in all_pol_summaries:
            writer.writerow([
                p["policy"],
                f"{p['safe_merge_rate']:.1f}",
                f"{p['collision_rate']:.1f}",
                f"{p['mean_driving_return']:.3f}",
                f"{p['mean_utility']:.3f}",
                f"{p['mean_travel_time']:.1f}",
                f"{p['mean_requests']:.2f}",
                f"{p['mean_bits']:.1f}",
                f"{p['pct_none']:.1f}",
                f"{p['pct_front']:.1f}",
                f"{p['pct_rear']:.1f}",
            ])
    print(f"\nSaved baseline comparison table to: {csv_path}")

    # 3. Critical Behavior Analysis: Request Confusion Table
    print("\n" + "=" * 80)
    print("CRITICAL BEHAVIOR ANALYSIS: REQUEST CHOICES BY TRUE HIDDEN HAZARD STATE")
    print("=" * 80)
    print(f"{'True Hidden State':<18} | {'% NONE':<12} | {'% ASK_FRONT':<14} | {'% ASK_REAR'}")
    print("-" * 80)
    confusion_rows = []
    for h_code, h_name in HAZARD_NAMES.items():
        counts = confusion_aggregates[h_code]
        tot = max(1, sum(counts.values()))
        p_none = (counts.get(0, 0) / tot) * 100.0
        p_front = (counts.get(1, 0) / tot) * 100.0
        p_rear = (counts.get(2, 0) / tot) * 100.0
        confusion_rows.append({"hidden_state": h_name, "pct_none": p_none, "pct_front": p_front, "pct_rear": p_rear, "counts": dict(counts)})
        print(f"{h_name:<18} | {p_none:>10.1f}% | {p_front:>12.1f}% | {p_rear:>10.1f}%")

    conf_csv_path = out_dir / "request_by_hazard_state_table.csv"
    with open(conf_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["True Hidden State", "% NONE", "% ASK_FRONT", "% ASK_REAR", "Count NONE", "Count ASK_FRONT", "Count ASK_REAR"])
        for r in confusion_rows:
            writer.writerow([
                r["hidden_state"],
                f"{r['pct_none']:.1f}",
                f"{r['pct_front']:.1f}",
                f"{r['pct_rear']:.1f}",
                r["counts"].get(0, 0),
                r["counts"].get(1, 0),
                r["counts"].get(2, 0),
            ])
    print(f"Saved request confusion table to: {conf_csv_path}")

    # 4. Cost Sweep: lambda in [0.00, 0.01, 0.05, 0.10]
    print("\n" + "=" * 85)
    print("PHASE 3: COMMUNICATION COST SWEEP (lambda in [0.00, 0.01, 0.05, 0.10])")
    print("=" * 85)
    cost_sweep_results = []
    lambdas = [0.00, 0.01, 0.05, 0.10]

    for lam in lambdas:
        if lam == 0.05:
            # Reuse Seed 1
            ag = trained_agents[1]
        elif eval_only and (out_dir / f"dqn_seed1_lambda{lam:.2f}.pt").exists():
            ckpt_path = out_dir / f"dqn_seed1_lambda{lam:.2f}.pt"
            print(f"\n--- Loading Sweep Checkpoint for Lambda = {lam:.2f} ({ckpt_path}) ---")
            ag = DoubleDQNAgent(state_dim=71, action_dim=3, device="cpu", seed=1)
            ag.load(str(ckpt_path))
        else:
            print(f"\n--- Training Sweep Run for Lambda = {lam:.2f} ---")
            ag, _, _ = train_single_dqn(
                seed=1,
                lambda_cost=lam,
                num_episodes=1000,
                model_checkpoint_path=wm_path,
                output_dir=str(out_dir),
                log_interval=200,
            )

        # Evaluate on held-out seeds
        s, _, _ = evaluate_policy_on_held_out(
            policy_type="LEARNED_DQN",
            dqn_agent=ag,
            wm=wm,
            num_seeds=60,
            base_seed=9000,
            lambda_cost=lam,
        )
        s["lambda"] = lam
        cost_sweep_results.append(s)

    print("\n" + "-" * 85)
    print(f"{'Lambda':<10} | {'Safe Merge':<10} | {'Collision':<10} | {'Driving Ret':<12} | {'Utility':<10} | {'Reqs/Ep':<8} | {'Bits/Ep'}")
    print("-" * 85)
    for c in cost_sweep_results:
        print(f"{c['lambda']:<10.2f} | {c['safe_merge_rate']:>8.1f}%  | {c['collision_rate']:>8.1f}%  | {c['mean_driving_return']:>10.3f}   | {c['mean_utility']:>8.3f}   | {c['mean_requests']:>6.2f}   | {c['mean_bits']:>5.1f}b")

    cost_csv_path = out_dir / "cost_sweep_table.csv"
    with open(cost_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Lambda Cost", "Safe Merge Rate (%)", "Collision Rate (%)", "Mean Driving Return", "Mean Utility", "Mean Requests", "Mean Bits"])
        for c in cost_sweep_results:
            writer.writerow([
                f"{c['lambda']:.2f}",
                f"{c['safe_merge_rate']:.1f}",
                f"{c['collision_rate']:.1f}",
                f"{c['mean_driving_return']:.3f}",
                f"{c['mean_utility']:.3f}",
                f"{c['mean_requests']:.2f}",
                f"{c['mean_bits']:.1f}",
            ])
    print(f"\nSaved cost sweep table to: {cost_csv_path}")

    final_payload = {
        "dqn_seeds": dqn_eval_summaries,
        "dqn_aggregate": agg_dqn,
        "baselines": baseline_summaries,
        "request_confusion": confusion_rows,
        "cost_sweep": cost_sweep_results,
        "held_out_records": {
            "dqn": dqn_eval_records,
            "baselines": baseline_records,
        },
    }
    json_path = out_dir / "final_evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(final_payload, f, indent=2, default=str)
    print(f"Saved complete evaluation results to: {json_path}")

    return final_payload


if __name__ == "__main__":
    run_full_evaluation_pipeline()
