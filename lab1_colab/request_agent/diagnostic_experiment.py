"""100-Episode Diagnostic Training and Baseline Comparison for the Request Agent.

NOTE: This is a diagnostic run using the TemporaryDrivingController.
Lab 2 World-Model MPC lives in Colab and will be integrated later.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from request_agent.communication_wrapper import (
    CommunicationMergeWrapper,
    TemporaryDrivingController,
)
from request_agent.dqn_agent import DoubleDQNAgent
from request_agent.replay_buffer import ReplayBuffer

ACTION_NAMES = {0: "NONE", 1: "ASK_FRONT", 2: "ASK_REAR"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}
DRIVE_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}


def run_diagnostic():
    print("=" * 70)
    print("STARTING DIAGNOSTIC TRAINING (100 EPISODES)")
    print("Notice: Uses TemporaryDrivingController for diagnostic purposes.")
    print("=" * 70)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication.npz"
    cfg.data.num_transitions = 5000
    cfg.env.id = "ego-merge-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75
    cfg.training.device = "cpu"

    checkpoint_model_path = Path("outputs/personal_lab1/attempt_01/my_lab1_world_model_v2.pt")
    assert checkpoint_model_path.exists(), f"Missing required model {checkpoint_model_path}"

    env = CommunicationMergeWrapper(
        cfg=cfg,
        model_checkpoint_path=checkpoint_model_path,
        device="cpu",
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
        epsilon_decay=0.985,  # smooth decay to reach min around ep 70-80
        device="cpu",
        seed=123,
    )

    replay_buffer = ReplayBuffer(capacity=10000, seed=123)
    controller = TemporaryDrivingController(seed=123)

    episodes = 100
    max_steps_per_episode = 30
    batch_size = 32
    warmup_steps = 64

    # Metrics storage
    ep_final_returns = []
    ep_raw_returns = []
    ep_comm_costs = []
    ep_requests = []
    ep_bits = []
    ep_crashes = []
    ep_merges = []
    ep_lengths = []
    ep_epsilons = []
    ep_losses = []
    ep_action_counts = {0: 0, 1: 0, 2: 0}

    total_steps = 0

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=ep * 2026)
        ep_final_ret = 0.0
        ep_raw_ret = 0.0
        ep_cost = 0.0
        crashed = False
        merged = False
        step_count = 0
        losses = []

        done = False
        while not done and step_count < max_steps_per_episode:
            req_act = agent.select_action(state)
            ep_action_counts[req_act] += 1

            msg = env.request_message(req_act)
            drive_act = controller(env.current_obs, env.current_latent, req_act, msg)

            next_state, reward, terminated, truncated, info = env.step(
                drive_act, req_act, msg
            )
            done = bool(terminated or truncated)

            replay_buffer.push(state, req_act, reward, next_state, done)
            state = next_state

            ep_final_ret += reward
            ep_raw_ret += info["raw_driving_reward"]
            ep_cost += info["communication_cost"]
            if info["crashed"]:
                crashed = True
            if info["is_merged"]:
                merged = True

            step_count += 1
            total_steps += 1

            if total_steps > warmup_steps:
                loss = agent.update(replay_buffer, batch_size=batch_size)
                if loss is not None:
                    losses.append(loss)

        avg_loss = float(np.mean(losses)) if losses else 0.0
        ep_final_returns.append(ep_final_ret)
        ep_raw_returns.append(ep_raw_ret)
        ep_comm_costs.append(ep_cost)
        ep_requests.append(info["request_count"])
        ep_bits.append(info["total_bits"])
        ep_crashes.append(float(crashed))
        ep_merges.append(float(merged))
        ep_lengths.append(step_count)
        ep_epsilons.append(agent.epsilon)
        ep_losses.append(avg_loss)

        # Show rolling 20-episode averages at multiples of 20
        if ep % 20 == 0:
            window = 20
            w_final = np.mean(ep_final_returns[-window:])
            w_raw = np.mean(ep_raw_returns[-window:])
            w_cost = np.mean(ep_comm_costs[-window:])
            w_reqs = np.mean(ep_requests[-window:])
            w_bits = np.mean(ep_bits[-window:])
            w_crash = np.mean(ep_crashes[-window:]) * 100
            w_merge = np.mean(ep_merges[-window:]) * 100
            w_len = np.mean(ep_lengths[-window:])
            w_loss = np.mean(ep_losses[-window:])
            print(
                f"[Episodes {ep-19:3d}..{ep:3d}] | Final Ret: {w_final:6.2f} | Raw Ret: {w_raw:6.2f} | "
                f"Cost: {w_cost:5.2f} | Reqs: {w_reqs:4.1f} | Bits: {w_bits:5.1f} | "
                f"Merge%: {w_merge:4.1f}% | Crash%: {w_crash:4.1f}% | Len: {w_len:4.1f} | "
                f"Loss: {w_loss:6.4f} | Eps: {agent.epsilon:.3f}"
            )

    print("\nTraining completed.")
    total_actions = max(1, sum(ep_action_counts.values()))
    print(
        f"Training Action Selection: NONE: {ep_action_counts[0]} ({ep_action_counts[0]/total_actions*100:.1f}%), "
        f"ASK_FRONT: {ep_action_counts[1]} ({ep_action_counts[1]/total_actions*100:.1f}%), "
        f"ASK_REAR: {ep_action_counts[2]} ({ep_action_counts[2]/total_actions*100:.1f}%)"
    )

    # 1. Save trained DQN under new path
    output_dir = Path("outputs/request_agent")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "diagnostic_dqn_100ep.pt"
    agent.save(save_path)
    print(f"Saved diagnostic DQN model to: {save_path.resolve()} ({save_path.stat().st_size} bytes)")

    # 2 & 3. Evaluate on 20 fixed seeds
    eval_seeds = [1000 + i for i in range(20)]
    print(f"\nEvaluating on {len(eval_seeds)} fixed seeds (seeds {eval_seeds[0]}..{eval_seeds[-1]})...")

    def run_eval_policy(policy_fn, label: str):
        eval_raws = []
        eval_finals = []
        eval_reqs = []
        eval_bits = []
        eval_crashes = []
        eval_merges = []
        action_dist = {0: 0, 1: 0, 2: 0}
        first_ep_decisions = []

        for idx, seed in enumerate(eval_seeds):
            state, _ = env.reset(seed=seed)
            ep_final = 0.0
            ep_raw = 0.0
            crashed = False
            merged = False
            done = False
            step = 0

            while not done and step < max_steps_per_episode:
                req_act = policy_fn(state, step)
                action_dist[req_act] += 1
                msg = env.request_message(req_act)
                drive_act = controller(env.current_obs, env.current_latent, req_act, msg)

                next_state, reward, terminated, truncated, info = env.step(
                    drive_act, req_act, msg
                )
                done = bool(terminated or truncated)

                ep_final += reward
                ep_raw += info["raw_driving_reward"]
                if info["crashed"]:
                    crashed = True
                if info["is_merged"]:
                    merged = True

                if idx == 0 and step < 10:
                    first_ep_decisions.append({
                        "step": step,
                        "request": f"{req_act} ({ACTION_NAMES[req_act]})",
                        "message": f"{msg} ({MSG_NAMES[msg]})",
                        "driving": f"{drive_act} ({DRIVE_NAMES[drive_act]})",
                        "raw_reward": info["raw_driving_reward"],
                        "comm_cost": info["communication_cost"],
                        "final_reward": reward,
                    })

                state = next_state
                step += 1

            eval_finals.append(ep_final)
            eval_raws.append(ep_raw)
            eval_reqs.append(info["request_count"])
            eval_bits.append(info["total_bits"])
            eval_crashes.append(float(crashed))
            eval_merges.append(float(merged))

        tot = max(1, sum(action_dist.values()))
        return {
            "label": label,
            "mean_raw": float(np.mean(eval_raws)),
            "std_raw": float(np.std(eval_raws)),
            "mean_final": float(np.mean(eval_finals)),
            "std_final": float(np.std(eval_finals)),
            "collision_rate": float(np.mean(eval_crashes)),
            "merge_success_rate": float(np.mean(eval_merges)),
            "mean_requests": float(np.mean(eval_reqs)),
            "mean_bits": float(np.mean(eval_bits)),
            "action_pct": {k: float(v / tot * 100.0) for k, v in action_dist.items()},
            "first_ep_decisions": first_ep_decisions,
        }

    # Policies
    rng_random = random.Random(999)
    results = [
        run_eval_policy(lambda s, st: agent.select_action(s, evaluate=True), "Double DQN (Greedy)"),
        run_eval_policy(lambda s, st: 0, "Always NONE (No Comm)"),
        run_eval_policy(lambda s, st: 1, "Always ASK_FRONT"),
        run_eval_policy(lambda s, st: 2, "Always ASK_REAR"),
        run_eval_policy(lambda s, st: rng_random.randint(0, 2), "Random Request"),
    ]

    env.close()

    # Save summary results
    summary_path = output_dir / "diagnostic_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    run_diagnostic()

