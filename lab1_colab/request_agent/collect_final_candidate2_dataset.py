"""Collect 15,000 transitions for Candidate #2 into datasets/merge_final_candidate2.npz."""

from __future__ import annotations

import os
import time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from request_agent.final_candidate2_env import FinalCandidate2MergeEnv

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}
REQ_NAMES = {0: "NONE", 1: "ASK_FRONT", 2: "ASK_REAR"}


def get_behavior_action(
    env: FinalCandidate2MergeEnv,
    strategy: str,
    rng: np.random.Generator,
    t: int,
) -> int:
    """Design behavioral actions ensuring balanced coverage of safe merges and critical collisions."""
    if strategy == "random":
        return int(rng.integers(0, 5))

    base_env = env.env.unwrapped
    is_merged = hasattr(base_env, "_is_merged") and base_env._is_merged()
    ego = base_env.vehicle
    rear_intent = env.rear_intent

    # Critical strategy explicitly sampling decision-critical contingencies
    if strategy == "critical":
        if not is_merged:
            if rear_intent == 3:  # BRAKE
                # Under BRAKE: IDLE or LANE_LEFT merges safely
                return 0 if rng.random() < 0.70 else 1
            else:  # MAINTAIN or ACCELERATE
                # 50% take FASTER (safe merge), 50% take IDLE/LANE_LEFT (collision demonstration)
                if rng.random() < 0.50:
                    return 3  # FASTER
                else:
                    return 0 if rng.random() < 0.60 else 1  # LANE_LEFT or IDLE
        else:
            # Once merged:
            return 1 if ego.speed >= 22.0 else 3

    if strategy == "aggressive":
        if not is_merged:
            return 0 if rng.random() < 0.80 else 3
        return 3 if ego.speed < 26.0 else 1

    # strategy == "guided"
    if not is_merged:
        if rear_intent == 3:
            return 0 if rng.random() < 0.75 else 1
        else:
            return 3 if rng.random() < 0.70 else 0
    else:
        front = base_env.front_vehicle
        if front is not None and (front.position[0] - ego.position[0]) < 14.0:
            return 4  # SLOWER
        return 1  # IDLE


def collect_final_dataset(
    target_transitions: int = 15000,
    out_path: str | Path = "datasets/merge_final_candidate2.npz",
    base_seed: int = 42,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "ego-merge-final-candidate2-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75

    env = FinalCandidate2MergeEnv(cfg, request_cost=0.0)
    rng = np.random.default_rng(base_seed)

    episodes: list[dict[str, np.ndarray]] = []
    collected = 0
    ep_idx = 0

    req_counts = Counter()
    msg_counts = Counter()
    act_counts = Counter()
    total_episodes = 0
    collision_episodes = 0
    safe_merge_episodes = 0
    rear_intent_counts = Counter()

    # Decision-critical outcome tracker
    critical_outcomes = {
        "BRAKE_IDLE_OR_LANE_LEFT_SAFE": 0,
        "MAINTAIN_ACCEL_IDLE_OR_LANE_LEFT_CRASH": 0,
        "MAINTAIN_ACCEL_FASTER_SAFE": 0,
    }

    print(f"Collecting {target_transitions} transitions for Candidate #2 into {out_path}...")
    t0 = time.time()
    pbar = tqdm(total=target_transitions, desc="Collecting transitions", unit="step")

    while collected < target_transitions:
        seed = base_seed + ep_idx * 19
        obs, _ = env.reset(seed=seed)

        rear_intent = env.rear_intent
        rear_intent_counts[rear_intent] += 1

        strat_choice = rng.random()
        if strat_choice < 0.45:
            strategy = "critical"
        elif strat_choice < 0.70:
            strategy = "guided"
        elif strat_choice < 0.85:
            strategy = "aggressive"
        else:
            strategy = "random"

        obs_seq: list[np.ndarray] = [np.asarray(obs, dtype=np.uint8)]
        actions: list[int] = []
        req_actions: list[int] = []
        messages: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []

        ep_crashed = False
        ep_merged = False
        done = False
        t_step = 0
        first_act = None

        while not done:
            req_act = int(rng.integers(0, 3))
            msg = env.get_message(req_act)

            drv_act = get_behavior_action(env, strategy, rng, t_step)
            if t_step == 0:
                first_act = drv_act

            next_obs, reward, term, trunc, info = env.step(drv_act, req_act)
            done = bool(term or trunc)

            if info.get("crashed", False):
                ep_crashed = True
            if info.get("is_merged", False):
                ep_merged = True

            obs_seq.append(np.asarray(next_obs, dtype=np.uint8))
            actions.append(drv_act)
            req_actions.append(req_act)
            messages.append(msg)
            rewards.append(float(info["raw_driving_reward"]))
            dones.append(done)

            req_counts[req_act] += 1
            msg_counts[msg] += 1
            act_counts[drv_act] += 1

            collected += 1
            pbar.update(1)
            t_step += 1
            if collected >= target_transitions:
                break

        # Log critical outcome
        if rear_intent == 3 and first_act in [0, 1] and ep_merged and not ep_crashed:
            critical_outcomes["BRAKE_IDLE_OR_LANE_LEFT_SAFE"] += 1
        elif rear_intent in [1, 2] and first_act in [0, 1] and ep_crashed:
            critical_outcomes["MAINTAIN_ACCEL_IDLE_OR_LANE_LEFT_CRASH"] += 1
        elif rear_intent in [1, 2] and first_act == 3 and ep_merged and not ep_crashed:
            critical_outcomes["MAINTAIN_ACCEL_FASTER_SAFE"] += 1

        episodes.append(
            {
                "obs": np.stack(obs_seq).astype(np.uint8),
                "actions": np.asarray(actions, dtype=np.int64),
                "request_actions": np.asarray(req_actions, dtype=np.int64),
                "messages": np.asarray(messages, dtype=np.int64),
                "rewards": np.asarray(rewards, dtype=np.float32),
                "dones": np.asarray(dones, dtype=bool),
            }
        )

        total_episodes += 1
        if ep_crashed:
            collision_episodes += 1
        if ep_merged and not ep_crashed:
            safe_merge_episodes += 1

        ep_idx += 1

    pbar.close()
    env.close()
    elapsed = time.time() - t0

    print(f"\nCollected {collected} transitions in {total_episodes} episodes ({elapsed:.1f}s, {collected/elapsed:.1f} steps/s)")

    # Save npz
    payload: dict[str, object] = {
        "num_episodes": len(episodes),
        "num_actions": 5,
        "num_request_actions": 3,
        "num_messages": 4,
    }
    for i, ep in enumerate(episodes):
        payload[f"obs_{i}"] = ep["obs"]
        payload[f"actions_{i}"] = ep["actions"]
        payload[f"request_actions_{i}"] = ep["request_actions"]
        payload[f"messages_{i}"] = ep["messages"]
        payload[f"rewards_{i}"] = ep["rewards"]
        payload[f"dones_{i}"] = ep["dones"]

    np.savez_compressed(out_path, **payload)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved dataset to {out_path} ({file_size_mb:.2f} MB)")

    # Report distributions
    print("\n" + "=" * 75)
    print("FINAL DATASET SUMMARY & DISTRIBUTIONS")
    print("=" * 75)
    print(f"Total Transitions: {collected}")
    print(f"Total Episodes:    {total_episodes}")
    print(f"Safe Merges:       {safe_merge_episodes} ({safe_merge_episodes/total_episodes*100:.1f}%)")
    print(f"Collisions:        {collision_episodes} ({collision_episodes/total_episodes*100:.1f}%)")

    print("\nDriving Actions:")
    for a in range(5):
        print(f"  {ACTION_NAMES[a]:<12}: {act_counts[a]:>6d} ({act_counts[a]/collected*100:>5.1f}%)")

    print("\nRequest Actions:")
    for r in range(3):
        print(f"  {REQ_NAMES[r]:<12}: {req_counts[r]:>6d} ({req_counts[r]/collected*100:>5.1f}%)")

    print("\nRevealed Messages:")
    for m in range(4):
        print(f"  {MSG_NAMES[m]:<12}: {msg_counts[m]:>6d} ({msg_counts[m]/collected*100:>5.1f}%)")

    print("\nRear Intent Episodes:")
    for intent_val, name in [(1, "MAINTAIN"), (2, "ACCELERATE"), (3, "BRAKE")]:
        cnt = rear_intent_counts[intent_val]
        print(f"  {name:<12}: {cnt:>6d} ({cnt/total_episodes*100:>5.1f}%)")

    print("\n" + "=" * 75)
    print("DECISION-CRITICAL CONTINGENCY OUTCOME COUNTS")
    print("=" * 75)
    print(f"  BRAKE + (IDLE or LANE_LEFT) -> Safe Merge:           {critical_outcomes['BRAKE_IDLE_OR_LANE_LEFT_SAFE']}")
    print(f"  MAINTAIN/ACCEL + (IDLE or LANE_LEFT) -> Collision:   {critical_outcomes['MAINTAIN_ACCEL_IDLE_OR_LANE_LEFT_CRASH']}")
    print(f"  MAINTAIN/ACCEL + FASTER -> Safe Merge:               {critical_outcomes['MAINTAIN_ACCEL_FASTER_SAFE']}")

    return out_path


if __name__ == "__main__":
    collect_final_dataset()

