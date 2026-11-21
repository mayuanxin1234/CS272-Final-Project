"""Collect approximately 20,000 transitions for Final Occlusion Dataset.

Dataset Path: datasets/merge_occlusion_final.npz

Ensures:
- Balanced hidden-state coverage (CLEAR, FRONT_HAZARD, REAR_HAZARD)
- Balanced request coverage (NONE, ASK_FRONT, ASK_REAR)
- Rich message coverage (UNKNOWN, CLEAR, HAZARD_AHEAD, HAZARD_BEHIND)
- Rich coverage of decision-critical contingencies:
  * FRONT_HAZARD: blind merge -> collision vs. informed double-lane change -> safe
  * REAR_HAZARD: blind merge -> collision vs. informed ramp-yield -> safe
  * CLEAR: standard merge -> safe
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from request_agent.occlusion_env import FinalOcclusionMergeEnv

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "CLEAR", 2: "HAZARD_AHEAD", 3: "HAZARD_BEHIND"}
REQ_NAMES = {0: "NONE", 1: "ASK_FRONT", 2: "ASK_REAR"}


def collect_occlusion_dataset(
    target_transitions: int = 20000,
    output_path: str | Path = "datasets/merge_occlusion_final.npz",
    base_seed: int = 42,
) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(f"COLLECTING FINAL OCCLUSION DATASET (Target: {target_transitions} transitions)")
    print(f"Destination: {output_path}")
    print("=" * 85)

    env = FinalOcclusionMergeEnv()
    rng = np.random.default_rng(base_seed)

    episodes_data = {}
    total_transitions = 0
    ep_idx = 0

    # Contingency tracking counters
    contingency_counts = {
        "FRONT_HAZARD_blind_collision": 0,
        "FRONT_HAZARD_informed_safe": 0,
        "REAR_HAZARD_blind_collision": 0,
        "REAR_HAZARD_informed_safe": 0,
        "CLEAR_safe_merge": 0,
    }
    request_counter = Counter()
    message_counter = Counter()
    hazard_counter = Counter()
    action_counter = Counter()
    total_crashes = 0
    total_safe_merges = 0

    pbar = tqdm(total=target_transitions, desc="Collecting transitions")

    while total_transitions < target_transitions:
        seed = base_seed + ep_idx * 17
        obs, info = env.reset(seed=seed)
        hazard = env.hazard_state
        hazard_counter[hazard] += 1

        # Determine collection strategy for this episode:
        # 40% informed (expert demonstrations of safe avoidance/merges)
        # 35% blind/critical (demonstrations of collisions and negative rewards)
        # 25% exploratory (random requests, random driving actions)
        strat_roll = rng.random()
        if strat_roll < 0.40:
            strategy = "informed"
        elif strat_roll < 0.75:
            strategy = "blind_critical"
        else:
            strategy = "exploratory"

        ep_obs = [obs]
        ep_actions = []
        ep_reqs = []
        ep_msgs = []
        ep_rewards = []
        ep_dones = []

        step = 0
        done = False
        revealed_msg = env.MSG_UNKNOWN

        while not done:
            # 1. Select request action
            if strategy == "informed":
                if step == 0:
                    if hazard == env.HAZARD_FRONT:
                        req = env.ASK_FRONT
                    elif hazard == env.HAZARD_REAR:
                        req = env.ASK_REAR
                    else:
                        req = env.NONE
                else:
                    req = env.NONE
            elif strategy == "blind_critical":
                # Blind: either NONE or wrong CAV
                if step == 0 and rng.random() < 0.50:
                    # Ask the uninformative CAV
                    req = env.ASK_REAR if hazard == env.HAZARD_FRONT else env.ASK_FRONT
                else:
                    req = env.NONE
            else:  # exploratory
                if step == 0:
                    req = int(rng.choice([env.NONE, env.ASK_FRONT, env.ASK_REAR]))
                else:
                    req = env.NONE if rng.random() < 0.80 else int(rng.choice([env.NONE, env.ASK_FRONT, env.ASK_REAR]))

            msg = env.get_message(req)
            if msg != env.MSG_UNKNOWN:
                revealed_msg = msg

            # 2. Select driving action
            if strategy == "informed":
                if revealed_msg == env.MSG_HAZARD_AHEAD:
                    # Double lane change into Lane 0
                    if step == 0:
                        act = 0  # LANE_LEFT
                    elif step == 1:
                        act = 0  # LANE_LEFT
                    else:
                        act = 1 if rng.random() < 0.85 else int(rng.choice([1, 3]))
                elif revealed_msg == env.MSG_HAZARD_BEHIND:
                    # Yield on ramp, let speeder pass, merge behind
                    if step == 0:
                        act = 4  # SLOWER
                    elif step == 1:
                        act = 1  # IDLE
                    elif step == 2:
                        act = 0  # LANE_LEFT
                    else:
                        act = 1 if rng.random() < 0.85 else int(rng.choice([1, 3]))
                else:
                    # CLEAR / normal merge
                    if step == 0:
                        act = 0  # LANE_LEFT
                    else:
                        act = 1 if rng.random() < 0.85 else int(rng.choice([1, 3]))

            elif strategy == "blind_critical":
                # Standard blind merge directly into hazards
                if step == 0:
                    act = 0  # LANE_LEFT
                else:
                    act = 1  # IDLE

            else:  # exploratory
                # Diverse driving actions
                if not env.env.unwrapped._is_merged():
                    act = int(rng.choice([0, 0, 1, 3, 4]))
                else:
                    act = int(rng.choice([1, 1, 3, 4, 0, 2]))

            next_obs, u_rew, term, trunc, step_info = env.step(act, req)
            done = bool(term or trunc)

            ep_obs.append(next_obs)
            ep_actions.append(act)
            ep_reqs.append(req)
            ep_msgs.append(msg)
            ep_rewards.append(step_info["raw_driving_reward"])
            ep_dones.append(done)

            request_counter[req] += 1
            message_counter[msg] += 1
            action_counter[act] += 1
            step += 1
            total_transitions += 1
            pbar.update(1)

            if total_transitions >= target_transitions:
                break

        # Record contingency stats
        crashed = bool(env.env.unwrapped.vehicle.crashed)
        merged = bool(env.env.unwrapped._is_merged())
        safe_merge = bool(merged and not crashed)

        if crashed:
            total_crashes += 1
        if safe_merge:
            total_safe_merges += 1

        if hazard == env.HAZARD_FRONT:
            if crashed:
                contingency_counts["FRONT_HAZARD_blind_collision"] += 1
            elif safe_merge and revealed_msg == env.MSG_HAZARD_AHEAD:
                contingency_counts["FRONT_HAZARD_informed_safe"] += 1
        elif hazard == env.HAZARD_REAR:
            if crashed:
                contingency_counts["REAR_HAZARD_blind_collision"] += 1
            elif safe_merge and revealed_msg == env.MSG_HAZARD_BEHIND:
                contingency_counts["REAR_HAZARD_informed_safe"] += 1
        else:  # CLEAR
            if safe_merge:
                contingency_counts["CLEAR_safe_merge"] += 1

        episodes_data[f"obs_{ep_idx}"] = np.array(ep_obs, dtype=np.uint8)
        episodes_data[f"actions_{ep_idx}"] = np.array(ep_actions, dtype=np.int64)
        episodes_data[f"request_actions_{ep_idx}"] = np.array(ep_reqs, dtype=np.int64)
        episodes_data[f"messages_{ep_idx}"] = np.array(ep_msgs, dtype=np.int64)
        episodes_data[f"rewards_{ep_idx}"] = np.array(ep_rewards, dtype=np.float32)
        episodes_data[f"dones_{ep_idx}"] = np.array(ep_dones, dtype=bool)

        ep_idx += 1

    pbar.close()

    # Save to npz
    metadata = {
        "num_episodes": ep_idx,
        "num_actions": 5,
        "num_request_actions": 3,
        "num_messages": 4,
    }
    np.savez_compressed(output_path, **metadata, **episodes_data)
    print(f"\nSuccessfully saved {total_transitions} transitions ({ep_idx} episodes) to {output_path}")

    # Summary
    print("\n" + "=" * 85)
    print("FINAL OCCLUSION DATASET DISTRIBUTION SUMMARY")
    print("=" * 85)
    print(f"Total Transitions: {total_transitions}")
    print(f"Total Episodes:    {ep_idx}")
    print(f"Total Safe Merges: {total_safe_merges} ({total_safe_merges / ep_idx * 100:.1f}%)")
    print(f"Total Collisions:  {total_crashes} ({total_crashes / ep_idx * 100:.1f}%)")

    print("\nHidden State Distribution:")
    h_names = {0: "CLEAR", 1: "FRONT_HAZARD", 2: "REAR_HAZARD"}
    for h_code in [0, 1, 2]:
        cnt = hazard_counter[h_code]
        print(f"  {h_names[h_code]:<15}: {cnt:>5d} episodes ({cnt / ep_idx * 100:.1f}%)")

    print("\nRequest Distribution:")
    for r_code in [0, 1, 2]:
        cnt = request_counter[r_code]
        print(f"  {REQ_NAMES[r_code]:<15}: {cnt:>6d} transitions ({cnt / total_transitions * 100:.1f}%)")

    print("\nMessage Distribution:")
    for m_code in [0, 1, 2, 3]:
        cnt = message_counter[m_code]
        print(f"  {MSG_NAMES[m_code]:<15}: {cnt:>6d} transitions ({cnt / total_transitions * 100:.1f}%)")

    print("\nDriving Action Distribution:")
    for a_code in range(5):
        cnt = action_counter[a_code]
        print(f"  {ACTION_NAMES[a_code]:<15}: {cnt:>6d} transitions ({cnt / total_transitions * 100:.1f}%)")

    print("\nDecision-Critical Contingency Counts:")
    for k, v in contingency_counts.items():
        print(f"  {k:<35}: {v:>5d} episodes")
    print("=" * 85)

    stats = {
        "total_transitions": total_transitions,
        "total_episodes": ep_idx,
        "hazard_counts": {h_names[k]: v for k, v in hazard_counter.items()},
        "request_counts": {REQ_NAMES[k]: v for k, v in request_counter.items()},
        "message_counts": {MSG_NAMES[k]: v for k, v in message_counter.items()},
        "action_counts": {ACTION_NAMES[k]: v for k, v in action_counter.items()},
        "contingency_counts": contingency_counts,
    }
    return stats


if __name__ == "__main__":
    collect_occlusion_dataset(target_transitions=20000)

