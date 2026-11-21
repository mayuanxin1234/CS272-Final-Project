"""Evaluation script for the trained communication request agent."""

from __future__ import annotations

from typing import Any
import numpy as np

from request_agent.communication_wrapper import (
    CommunicationMergeWrapper,
    PlannerCallback,
)
from request_agent.dqn_agent import DoubleDQNAgent


def evaluate_request_agent(
    env: CommunicationMergeWrapper,
    agent: DoubleDQNAgent,
    planner: PlannerCallback,
    episodes: int = 20,
    max_steps_per_episode: int = 30,
    base_seed: int = 5000,
) -> dict[str, Any]:
    """Evaluate request agent greedily (epsilon=0)."""
    returns = []
    comm_costs = []
    request_counts = []
    total_bits = []
    crashes = []

    action_frequencies = {0: 0, 1: 0, 2: 0}

    for ep in range(episodes):
        state, _ = env.reset(seed=base_seed + ep)
        ep_return = 0.0
        done = False
        step_count = 0

        while not done and step_count < max_steps_per_episode:
            # Greedy action
            request_action = agent.select_action(state, evaluate=True)
            action_frequencies[request_action] += 1

            message = env.request_message(request_action)
            driving_action = planner(
                env.current_obs, env.current_latent, request_action, message
            )

            state, reward, terminated, truncated, info = env.step(
                driving_action, request_action, message
            )
            done = bool(terminated or truncated)
            ep_return += reward
            step_count += 1

        returns.append(ep_return)
        comm_costs.append(info["communication_cost"])
        request_counts.append(info["request_count"])
        total_bits.append(info["total_bits"])
        crashes.append(float(info.get("crashed", False)))

    total_actions = max(1, sum(action_frequencies.values()))
    summary = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_bits": float(np.mean(total_bits)),
        "mean_requests": float(np.mean(request_counts)),
        "crash_rate": float(np.mean(crashes)),
        "action_distribution": {
            k: float(v / total_actions) for k, v in action_frequencies.items()
        },
    }
    return summary

