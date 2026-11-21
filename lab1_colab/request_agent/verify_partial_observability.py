"""Verify partial observability in the merge scenario.

Demonstrates that two identical local observations with different hidden neighbor
intents require different safe driving decisions by the MPC planner.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from request_agent.scenario_sweep_diagnostic import ConfigurableMergeGymEnv
from request_agent.communication_wrapper import CommunicationMergeWrapper
from request_agent.mpc_adapter import MergeMPCPlanner
import gymnasium as gym

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def test_partial_observability(
    scenario_cfg: dict[str, float],
    seed: int = 4001,
):
    print("=" * 80)
    print("PARTIAL OBSERVABILITY VERIFICATION")
    print(f"Scenario: Front x={scenario_cfg['front_x']}, Rear x={scenario_cfg['rear_x']}, Gap={scenario_cfg['front_x'] - scenario_cfg['rear_x']}m")
    print("=" * 80)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "ego-merge-v0"
    ckpt_path = Path("outputs/personal_lab1/attempt_01/my_lab1_world_model_v2.pt")

    ConfigurableMergeGymEnv.SCENARIO_CONFIG = scenario_cfg
    if "ego-merge-v0" in gym.registry:
        del gym.registry["ego-merge-v0"]
    gym.register(id="ego-merge-v0", entry_point=ConfigurableMergeGymEnv)

    env = CommunicationMergeWrapper(cfg=cfg, model_checkpoint_path=ckpt_path, device="cpu")
    mpc = MergeMPCPlanner(
        model=env.model,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=42,
        device="cpu",
    )

    # Reset environment to get initial observation
    req_state, info = env.reset(seed=seed)
    obs = env.current_obs
    latent = env.current_latent

    # Generate fixed candidate action pool for fair counterfactual comparison
    cand_rng = np.random.default_rng(seed * 999)
    cand_indices = cand_rng.integers(0, len(mpc.allowed_actions), size=(mpc.num_sequences, mpc.horizon))
    candidates = mpc.allowed_actions[cand_indices]

    print(f"\nInitial State at t=0 (Identical Visual Observation):")
    base_env = env.env.env.unwrapped
    print(f"  Ego Position: x={base_env.vehicle.position[0]:.1f}m, speed={base_env.vehicle.speed:.1f}m/s")
    print(f"  Front Vehicle: x={base_env.front_vehicle.position[0]:.1f}m, speed={base_env.front_vehicle.speed:.1f}m/s")
    print(f"  Rear Vehicle:  x={base_env.rear_vehicle.position[0]:.1f}m, speed={base_env.rear_vehicle.speed:.1f}m/s")

    # Evaluate MPC driving decision across different counterfactual messages
    # on the EXACT SAME state and candidate trajectories:
    test_cases = [
        ("NO_COMM (Baseline)", 0, 0),
        ("ORACLE_FRONT: MAINTAIN (1)", 1, 1),
        ("ORACLE_FRONT: ACCELERATE (2)", 1, 2),
        ("ORACLE_FRONT: BRAKE (3)", 1, 3),
        ("ORACLE_REAR: MAINTAIN (1)", 2, 1),
        ("ORACLE_REAR: ACCELERATE (2)", 2, 2),
        ("ORACLE_REAR: BRAKE (3)", 2, 3),
    ]

    print("\nCounterfactual Planning on Identical Local Observation at t=0:")
    print("-" * 80)
    print(f"{'Condition':<30} | {'Req':<4} | {'Msg':<12} | {'Chosen Action':<15} | {'Predicted Ret':<12}")
    print("-" * 80)

    results = {}
    for label, req, msg in test_cases:
        act, pred_ret, c_rets = mpc.plan(obs, latent, request_action=req, message=msg, fixed_candidate_actions=candidates)
        act_name = ACTION_NAMES.get(act, str(act))
        msg_name = MSG_NAMES.get(msg, str(msg))
        print(f"{label:<30} | {req:<4} | {msg_name:<12} | {act_name:<15} | {pred_ret:<12.3f}")
        results[label] = {
            "req": req,
            "msg": msg,
            "action": act,
            "action_name": act_name,
            "pred_ret": pred_ret,
        }

    # Now trace during the merge approach to demonstrate how intent alters driving choices
    print("\nStep-by-Step Counterfactual Action Selection along Trajectory:")
    print("-" * 85)
    print(f"{'Step':<5} | {'Ego X':<7} | {'Front X':<8} | {'Rear X':<8} | {'NO_COMM':<11} | {'ORACLE_FRONT':<13} | {'ORACLE_REAR':<13} | {'Diff'}")
    print("-" * 85)

    tf = env.request_message(1)
    tr = env.request_message(2)
    step = 0
    done = False
    step_rng = np.random.default_rng(seed * 777)
    while not done and step < 10:
        c_obs = env.current_obs
        c_lat = env.current_latent
        cands = step_rng.integers(0, len(mpc.allowed_actions), size=(mpc.num_sequences, mpc.horizon))
        cands = mpc.allowed_actions[cands]

        a_no, _, _ = mpc.plan(c_obs, c_lat, 0, 0, fixed_candidate_actions=cands)
        a_f, _, _ = mpc.plan(c_obs, c_lat, 1, tf, fixed_candidate_actions=cands)
        a_r, _, _ = mpc.plan(c_obs, c_lat, 2, tr, fixed_candidate_actions=cands)

        diff = "YES" if (a_f != a_no or a_r != a_no) else "no"
        ego_x = base_env.vehicle.position[0]
        front_x = base_env.front_vehicle.position[0]
        rear_x = base_env.rear_vehicle.position[0]

        print(f"{step:<5} | {ego_x:<7.1f} | {front_x:<8.1f} | {rear_x:<8.1f} | {ACTION_NAMES[a_no]:<11} | {ACTION_NAMES[a_f]:<13} | {ACTION_NAMES[a_r]:<13} | {diff}")

        _, _, term, trunc, _ = env.step(a_no, 0, 0)
        done = bool(term or trunc)
        step += 1

    return results


if __name__ == "__main__":
    test_partial_observability({
        "ego_x": 30.0, "ego_speed": 25.0,
        "front_x": 52.0, "front_speed": 27.0,
        "rear_x": 16.0, "rear_speed": 27.0,
        "accel_delta": 5.0, "brake_delta": 5.0,
    })

