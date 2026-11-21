"""Training loop for the communication request RL agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
import numpy as np

from request_agent.communication_wrapper import (
    CommunicationMergeWrapper,
    PlannerCallback,
    TemporaryDrivingController,
)
from request_agent.dqn_agent import DoubleDQNAgent
from request_agent.replay_buffer import ReplayBuffer


def train_request_agent(
    env: CommunicationMergeWrapper,
    agent: DoubleDQNAgent,
    replay_buffer: ReplayBuffer,
    planner: PlannerCallback,
    episodes: int = 50,
    max_steps_per_episode: int = 30,
    batch_size: int = 32,
    warmup_steps: int = 64,
    checkpoint_dir: str | Path = "outputs/request_agent/checkpoints",
    log_interval: int = 10,
) -> dict[str, list[float]]:
    """Train the request policy while keeping the world model and driving controller pluggable."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "episode_returns": [],
        "communication_costs": [],
        "request_counts": [],
        "total_bits": [],
        "losses": [],
    }

    total_steps = 0
    best_return = -float("inf")

    for ep in range(1, episodes + 1):
        state, _ = env.reset(seed=ep * 100)
        episode_reward = 0.0
        episode_comm_cost = 0.0
        episode_losses: list[float] = []

        done = False
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            # 1. DQN selects request action
            request_action = agent.select_action(state)

            # 2. Query communication channel without advancing physics
            message = env.request_message(request_action)

            # 3. Driving controller (temporary or future MPC) chooses driving action
            driving_action = planner(
                env.current_obs, env.current_latent, request_action, message
            )

            # 4. Advance physics exactly once
            next_state, reward, terminated, truncated, info = env.step(
                driving_action, request_action, message
            )
            done = bool(terminated or truncated)

            # 5. Store transition in replay buffer
            replay_buffer.push(state, request_action, reward, next_state, done)
            state = next_state

            episode_reward += reward
            episode_comm_cost += info["communication_cost"]
            total_steps += 1
            step_count += 1

            # 6. DQN update step after warmup
            if total_steps > warmup_steps:
                loss = agent.update(replay_buffer, batch_size=batch_size)
                if loss is not None:
                    episode_losses.append(loss)

        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        history["episode_returns"].append(episode_reward)
        history["communication_costs"].append(episode_comm_cost)
        history["request_counts"].append(info["request_count"])
        history["total_bits"].append(info["total_bits"])
        history["losses"].append(avg_loss)

        if ep % log_interval == 0 or ep == episodes:
            mean_ret = float(np.mean(history["episode_returns"][-log_interval:]))
            mean_bits = float(np.mean(history["total_bits"][-log_interval:]))
            print(
                f"Episode {ep:3d}/{episodes} | Return: {episode_reward:.3f} (mean_{log_interval}: {mean_ret:.3f}) | "
                f"Bits: {info['total_bits']} | Epsilon: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}"
            )

        if episode_reward > best_return:
            best_return = episode_reward
            agent.save(checkpoint_dir / "request_agent_best.pt")

    agent.save(checkpoint_dir / "request_agent_final.pt")
    with open(checkpoint_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history

