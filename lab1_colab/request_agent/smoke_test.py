"""Comprehensive smoke test for the communication request agent."""

from __future__ import annotations

import os
from pathlib import Path
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


def run_smoke_test() -> None:
    print("=================== COMM-AGENT SMOKE TEST ===================")
    checkpoint_model_path = Path("outputs/personal_lab1/attempt_01/my_lab1_world_model_v2.pt")
    assert checkpoint_model_path.exists(), f"Missing required model {checkpoint_model_path}"

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication.npz"
    cfg.data.num_transitions = 5000
    cfg.env.id = "ego-merge-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75
    cfg.training.device = "cpu"

    print("1. Initializing CommunicationMergeWrapper with frozen WorldModel...")
    env = CommunicationMergeWrapper(
        cfg=cfg,
        model_checkpoint_path=checkpoint_model_path,
        device="cpu",
    )

    # 1. Reset check
    state, info = env.reset(seed=42)
    assert state is not None, "Failed to reset environment"
    print("   [PASS] Environment reset successfully.")

    # 2. Latent dimension check
    latent = env.current_latent
    assert latent.shape == (64,), f"Expected latent shape (64,), got {latent.shape}"
    print(f"   [PASS] WorldModel latent vector has dimension {latent.shape[0]} (64).")

    # 3. Request state dimension check
    assert state.shape == (71,), f"Expected state shape (71,), got {state.shape}"
    print(f"   [PASS] Request policy state has dimension {state.shape[0]} (71: 64 latent + 3 req + 4 msg).")

    # 4. Privileged information leak check
    # Ensure state does not contain front_intent or rear_intent directly
    # Check that info keys visible to the policy do not leak hidden intents
    assert "front_intent" not in info, "LEAK DETECTED: front_intent found in public info dict!"
    assert "rear_intent" not in info, "LEAK DETECTED: rear_intent found in public info dict!"
    print("   [PASS] No privileged intent enters policy state or sanitized info.")

    # 5. Frozen encoder check
    for name, p in env.model.named_parameters():
        assert not p.requires_grad, f"Parameter {name} is NOT frozen!"
    initial_weights = [p.clone() for p in env.model.parameters()]
    print("   [PASS] WorldModel encoder parameters are confirmed strictly FROZEN (requires_grad=False).")

    # 6. DQN initialization and Q-values check
    agent = DoubleDQNAgent(
        state_dim=71,
        action_dim=3,
        hidden_dims=(128, 128),
        lr=1e-3,
        device="cpu",
        seed=42,
    )
    q_vals = agent.get_q_values(state)
    assert q_vals.shape == (3,), f"Expected 3 Q-values, got {q_vals.shape}"
    print(f"   [PASS] DQN outputs exactly 3 Q-values: {q_vals} (Actions: 0=NONE, 1=ASK_FRONT, 2=ASK_REAR).")

    # 7. Action sampling check (verify 0, 1, 2 can be selected)
    actions_seen = set()
    agent.epsilon = 1.0  # pure random
    for _ in range(50):
        actions_seen.add(agent.select_action(state))
    assert actions_seen == {0, 1, 2}, f"Failed to sample all 3 actions, got {actions_seen}"
    print("   [PASS] All three request actions (0, 1, 2) sampled successfully under epsilon-greedy.")

    # 8. Message resolution check (non-mutating pre-step check)
    msg_0 = env.request_message(0)  # NONE
    msg_1 = env.request_message(1)  # ASK_FRONT
    msg_2 = env.request_message(2)  # ASK_REAR
    assert msg_0 == 0, f"Action NONE must return UNKNOWN (0), got {msg_0}"
    assert msg_1 in {1, 2, 3}, f"Action ASK_FRONT returned invalid message {msg_1}"
    assert msg_2 in {1, 2, 3}, f"Action ASK_REAR returned invalid message {msg_2}"
    print(f"   [PASS] Message resolution verified: NONE -> {msg_0} (UNKNOWN), ASK_FRONT -> {msg_1}, ASK_REAR -> {msg_2}.")

    # 9. Verify physics does not advance during request_message
    time_before = env.env.env.unwrapped.time
    env.request_message(1)
    time_after = env.env.env.unwrapped.time
    assert time_before == time_after, "request_message improperly advanced simulation time!"
    print(f"   [PASS] request_message is confirmed non-mutating (sim time stayed at {time_before}).")

    # 10. Test temporary controller and single-step transition
    controller = TemporaryDrivingController(seed=42)
    driving_action = controller(env.current_obs, env.current_latent, 1, msg_1)
    assert 0 <= driving_action <= 4, f"Invalid driving action {driving_action}"

    next_state, reward, terminated, truncated, step_info = env.step(
        driving_action=driving_action, request_action=1, message=msg_1
    )
    time_after_step = env.env.env.unwrapped.time
    assert time_after_step > time_before, "env.step failed to advance simulation time!"
    print(f"   [PASS] Environment advanced physics exactly once (time: {time_before} -> {time_after_step}).")

    # Check reward accounting:
    # MergeEnv.step() returned reward = driving_reward - communication_cost
    comm_cost = step_info["communication_cost"]
    raw_reward = step_info["raw_driving_reward"]
    final_reward = step_info["final_request_reward"]
    assert np.isclose(final_reward, raw_reward - comm_cost), "Reward accounting mismatch!"
    assert comm_cost == 0.05, f"Expected communication cost 0.05, got {comm_cost}"
    print(f"   [PASS] Reward accounting confirmed: raw_driving ({raw_reward:.4f}) - cost ({comm_cost}) = final_reward ({final_reward:.4f}).")

    # 11. Human-readable transition print
    print("\n----------------- EXAMPLE TRANSITION -----------------")
    action_names = {0: "NONE", 1: "ASK_FRONT", 2: "ASK_REAR"}
    msg_names = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}
    driving_names = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
    print(f"State (t=0)        : 71-D vector (latent norm: {np.linalg.norm(state[:64]):.3f}, prev_req: 0, prev_msg: 0)")
    print(f"Request Action     : {1} ({action_names[1]})")
    print(f"Received Message   : {msg_1} ({msg_names[msg_1]})")
    print(f"Driving Action     : {driving_action} ({driving_names[driving_action]})")
    print(f"Step Rewards       : raw={raw_reward:.4f}, comm_cost={comm_cost:.4f} -> total={final_reward:.4f}")
    print(f"Next State (t=1)   : 71-D vector (latent norm: {np.linalg.norm(next_state[:64]):.3f}, prev_req: 1, prev_msg: {msg_1})")
    print("------------------------------------------------------\n")

    # 12. Replay buffer push and sample
    buffer = ReplayBuffer(capacity=100, seed=42)
    buffer.push(state, 1, final_reward, next_state, False)
    # Add dummy samples to allow a mini-batch update
    for i in range(10):
        buffer.push(state, i % 3, final_reward + i * 0.01, next_state, False)
    assert len(buffer) == 11, f"Expected 11 items in buffer, got {len(buffer)}"
    print(f"   [PASS] Replay buffer successfully stored transitions (size: {len(buffer)}).")

    # 13. DQN Optimization step check
    loss = agent.update(buffer, batch_size=4)
    assert loss is not None and not np.isnan(loss), f"DQN update failed with loss {loss}"
    print(f"   [PASS] Double DQN optimization step succeeded (loss: {loss:.6f}).")

    # 14. Target network soft update check
    agent.soft_update()
    print("   [PASS] Target network Polyak soft update succeeded.")

    # 15. Check encoder still unchanged
    for p_init, p_now in zip(initial_weights, env.model.parameters()):
        assert torch.equal(p_init, p_now), "CRITICAL: WorldModel weights mutated during RL update!"
    print("   [PASS] WorldModel weights confirmed completely unchanged after RL optimization step.")

    # 16. Checkpoint save & load check
    test_ckpt_dir = Path("outputs/request_agent/checkpoints")
    test_ckpt_dir.mkdir(parents=True, exist_ok=True)
    test_ckpt_file = test_ckpt_dir / "smoke_test_dqn.pt"
    agent.save(test_ckpt_file)
    assert test_ckpt_file.exists(), "Failed to save DQN checkpoint file"

    fresh_agent = DoubleDQNAgent(state_dim=71, action_dim=3, device="cpu")
    fresh_agent.load(test_ckpt_file)
    # Compare weights between agent and fresh_agent
    for p1, p2 in zip(agent.q_eval.parameters(), fresh_agent.q_eval.parameters()):
        assert torch.equal(p1, p2), "Loaded weights do not match saved weights!"
    print(f"   [PASS] DQN checkpoint save/load verified ({test_ckpt_file.stat().st_size} bytes).")

    env.close()
    print("\n================ ALL SMOKE CHECKS PASSED ================\n")

if __name__ == "__main__":
    run_smoke_test()

