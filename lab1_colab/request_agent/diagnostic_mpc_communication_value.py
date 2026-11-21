"""Diagnostic experiment testing whether the World-Model MPC actually exploits communication."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from request_agent.communication_wrapper import CommunicationMergeWrapper
from request_agent.mpc_adapter import MergeMPCPlanner

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def run_diagnostic(num_episodes: int = 15, steps_per_episode: int = 4, future_mode: str = "none"):
    print("=" * 75)
    print(f"RUNNING MPC COMMUNICATION VALUE DIAGNOSTIC (future_mode='{future_mode}')")
    print(f"Testing 64 sequences x 6 horizon over {num_episodes * steps_per_episode} real states...")
    print("=" * 75)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication.npz"
    cfg.data.num_transitions = 5000
    cfg.env.id = "ego-merge-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75
    cfg.training.device = "cpu"

    checkpoint_model_path = Path("outputs/personal_lab1/attempt_01/my_lab1_world_model_v2.pt")
    assert checkpoint_model_path.exists()

    env = CommunicationMergeWrapper(
        cfg=cfg,
        model_checkpoint_path=checkpoint_model_path,
        device="cpu",
    )

    mpc = MergeMPCPlanner(
        model=env.model,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=1234,
        device="cpu",
        future_mode=future_mode,
    )

    total_states = 0
    front_changed_count = 0
    rear_changed_count = 0
    any_changed_count = 0

    return_diffs_front = []
    return_diffs_rear = []

    interesting_examples = []

    rng = np.random.default_rng(2026)

    for ep_idx in range(num_episodes):
        seed = 3000 + ep_idx
        env.reset(seed=seed)

        actual_front_intent = env.request_message(1)
        actual_rear_intent = env.request_message(2)

        for step in range(steps_per_episode):
            obs = env.current_obs
            latent = env.current_latent

            # Draw a single fixed pool of 64 candidate sequences for this exact state
            indices = rng.integers(0, len(mpc.allowed_actions), size=(mpc.num_sequences, mpc.horizon))
            fixed_candidates = mpc.allowed_actions[indices]

            # 1. Condition NONE (baseline)
            act_none, ret_none, _ = mpc.plan(
                obs, latent, request_action=0, message=0, fixed_candidate_actions=fixed_candidates
            )

            # 2. Condition ASK_FRONT with actual message
            act_front, ret_front, _ = mpc.plan(
                obs, latent, request_action=1, message=actual_front_intent, fixed_candidate_actions=fixed_candidates
            )

            # 3. Condition ASK_REAR with actual message
            act_rear, ret_rear, _ = mpc.plan(
                obs, latent, request_action=2, message=actual_rear_intent, fixed_candidate_actions=fixed_candidates
            )

            front_diff = ret_front - ret_none
            rear_diff = ret_rear - ret_none
            return_diffs_front.append(front_diff)
            return_diffs_rear.append(rear_diff)

            front_changed = (act_front != act_none)
            rear_changed = (act_rear != act_none)
            any_changed = front_changed or rear_changed

            if front_changed:
                front_changed_count += 1
            if rear_changed:
                rear_changed_count += 1
            if any_changed:
                any_changed_count += 1

            total_states += 1

            if any_changed and len(interesting_examples) < 5:
                interesting_examples.append({
                    "episode": ep_idx + 1,
                    "step": step,
                    "front_intent": f"{actual_front_intent} ({MSG_NAMES[actual_front_intent]})",
                    "rear_intent": f"{actual_rear_intent} ({MSG_NAMES[actual_rear_intent]})",
                    "act_none": f"{act_none} ({ACTION_NAMES[act_none]})",
                    "act_front": f"{act_front} ({ACTION_NAMES[act_front]})",
                    "act_rear": f"{act_rear} ({ACTION_NAMES[act_rear]})",
                    "ret_none": ret_none,
                    "ret_front": ret_front,
                    "ret_rear": ret_rear,
                })

            # Advance environment with the NONE action to step forward to next state
            env.step(driving_action=act_none, request_action=0, message=0)

    env.close()

    pct_front = (front_changed_count / total_states) * 100.0
    pct_rear = (rear_changed_count / total_states) * 100.0
    pct_any = (any_changed_count / total_states) * 100.0

    mean_diff_front = float(np.mean(return_diffs_front))
    mean_diff_rear = float(np.mean(return_diffs_rear))

    print(f"\n--- Diagnostic Results ({future_mode}) ---")
    print(f"Total states tested: {total_states}")
    print(f"ASK_FRONT changed action vs NONE: {front_changed_count}/{total_states} ({pct_front:.1f}%)")
    print(f"ASK_REAR changed action vs NONE : {rear_changed_count}/{total_states} ({pct_rear:.1f}%)")
    print(f"EITHER query changed action vs NONE: {any_changed_count}/{total_states} ({pct_any:.1f}%)")
    print(f"Mean predicted return diff (ASK_FRONT - NONE): {mean_diff_front:+.4f}")
    print(f"Mean predicted return diff (ASK_REAR - NONE) : {mean_diff_rear:+.4f}")

    return {
        "future_mode": future_mode,
        "total_states": total_states,
        "pct_front": pct_front,
        "pct_rear": pct_rear,
        "pct_any": pct_any,
        "mean_diff_front": mean_diff_front,
        "mean_diff_rear": mean_diff_rear,
        "examples": interesting_examples,
    }


if __name__ == "__main__":
    res_none = run_diagnostic(num_episodes=15, steps_per_episode=4, future_mode="none")
    res_pers = run_diagnostic(num_episodes=15, steps_per_episode=4, future_mode="persistent")

