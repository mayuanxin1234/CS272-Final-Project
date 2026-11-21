"""Collect 12,000 transitions for Design 1 (Rear Bottleneck) into datasets/merge_rear_bottleneck.npz."""

from __future__ import annotations

import os
import time
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from request_agent.rear_bottleneck_env import RearBottleneckMergeEnv

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}
REQ_NAMES = {0: "NONE", 1: "ASK_FRONT", 2: "ASK_REAR"}


def get_mixed_driving_action(
    env: RearBottleneckMergeEnv,
    strategy: str,
    rng: np.random.Generator,
) -> int:
    """Select a driving action according to the designated behavioral strategy."""
    if strategy == "random":
        return int(rng.integers(0, 5))

    base_env = env.env.unwrapped
    ego = base_env.vehicle
    front = base_env.front_vehicle
    rear = base_env.rear_vehicle
    is_merged = hasattr(base_env, "_is_merged") and base_env._is_merged()

    if strategy == "aggressive":
        if not is_merged:
            return 0 if rng.random() < 0.8 else 3  # LANE_LEFT or FASTER
        return 3 if ego.speed < 29.0 else 1  # FASTER or IDLE

    # strategy == "guided"
    if not is_merged:
        # On access ramp: merge left towards highway
        if rng.random() < 0.70:
            return 0  # LANE_LEFT
        return int(rng.choice([1, 3]))

    # Once merged on highway:
    if front is not None:
        dist_front = front.position[0] - ego.position[0]
        if dist_front < 15.0:
            return 4 if rng.random() < 0.75 else 1  # SLOWER
        elif dist_front > 25.0 and ego.speed < 28.0:
            return 3 if rng.random() < 0.75 else 1  # FASTER

    if rng.random() < 0.15:
        return int(rng.integers(0, 5))
    return 1  # IDLE


def collect_dataset(
    target_transitions: int = 12000,
    out_path: str | Path = "datasets/merge_rear_bottleneck.npz",
    base_seed: int = 42,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "ego-merge-rear-bottleneck-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75

    env = RearBottleneckMergeEnv(cfg, request_cost=0.0)
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

    print(f"Collecting {target_transitions} transitions for Design 1 into {out_path}...")
    t0 = time.time()
    pbar = tqdm(total=target_transitions, desc="Collecting transitions", unit="step")

    while collected < target_transitions:
        seed = base_seed + ep_idx * 17
        obs, _ = env.reset(seed=seed)

        rear_intent_counts[env.rear_intent] += 1

        strat_choice = rng.random()
        if strat_choice < 0.40:
            strategy = "guided"
        elif strat_choice < 0.75:
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

        while not done:
            req_act = int(rng.integers(0, 3))
            msg = env.get_message(req_act)

            drv_act = get_mixed_driving_action(env, strategy, rng)

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
            if collected >= target_transitions:
                break

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
    print("\n" + "=" * 70)
    print("DATASET SUMMARY & DISTRIBUTIONS")
    print("=" * 70)
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

    return out_path


if __name__ == "__main__":
    collect_dataset()

