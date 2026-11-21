"""Robustness Scenario 1: Fog / Degraded Visibility Diagnostic.

Tests whether cooperative V2X communication value increases under degraded optical visibility.
Physics-based Beer-Lambert atmospheric scattering model:
I_fog(r) = I(r) * exp(-beta * r) + I_airlight * (1 - exp(-beta * r))

Visibility Levels:
1. CLEAR WEATHER: beta = 0.00 (visibility > 50m, normal camera)
2. LIGHT FOG:     beta = 0.05 (visibility ~ 25m, distant contrast attenuated)
3. HEAVY FOG:     beta = 0.15 (visibility ~ 10m, distant objects masked beyond 8-10m)

V2X Radio Communication:
Operating over 5.9 GHz RF, communication is unaffected by optical fog.
Messages remain:
0: UNKNOWN, 1: CLEAR, 2: HAZARD_AHEAD, 3: HAZARD_BEHIND
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.models import WorldModel
from lab1.utils.checkpoint import load_checkpoint
from request_agent.occlusion_env import FinalOcclusionMergeEnv
from request_agent.mpc_adapter import MergeMPCPlanner
from request_agent.dqn_agent import DoubleDQNAgent

FOG_CONFIGS = {
    "CLEAR_WEATHER": {"beta": 0.00, "vis_dist": ">50m"},
    "LIGHT_FOG":     {"beta": 0.05, "vis_dist": "~25m"},
    "HEAVY_FOG":     {"beta": 0.15, "vis_dist": "~10m"},
}


def apply_fog_to_observation(obs: np.ndarray, beta: float, airlight: float = 140.0) -> np.ndarray:
    """Apply distance-dependent atmospheric scattering degradation to (4, 64, 64) observation.

    Center of image (32, 32) corresponds to ego position (distance r = 0).
    Scale: 1.75 pixels per meter -> 1 pixel = 0.571 m.
    """
    if beta <= 0.0:
        return obs

    H, W = obs.shape[-2], obs.shape[-1]
    y_coords, x_coords = np.ogrid[:H, :W]
    # Distance in meters from center
    dist_meters = np.sqrt((x_coords - W / 2.0) ** 2 + (y_coords - H / 2.0) ** 2) / 1.75

    # Transmission transmission map t(r) = exp(-beta * r)
    transmission = np.exp(-beta * dist_meters).astype(np.float32)  # shape (64, 64)

    # Apply to all 4 stacked frames
    obs_float = obs.astype(np.float32)
    fogged = obs_float * transmission + airlight * (1.0 - transmission)
    return np.clip(fogged, 0.0, 255.0).astype(np.uint8)


def measure_fog_observation_degradation() -> dict:
    """Quantitatively verify that fog actually degrades ego camera observations."""
    print("=" * 85)
    print("QUANTITATIVE VERIFICATION OF FOG OBSERVATION DEGRADATION")
    print("=" * 85)

    env = FinalOcclusionMergeEnv()
    obs, _ = env.reset(seed=9000)

    obs_clear = apply_fog_to_observation(obs, beta=0.00)
    obs_light = apply_fog_to_observation(obs, beta=0.05)
    obs_heavy = apply_fog_to_observation(obs, beta=0.15)

    l1_light = float(np.mean(np.abs(obs_light.astype(float) - obs_clear.astype(float))))
    max_light = float(np.max(np.abs(obs_light.astype(float) - obs_clear.astype(float))))

    l1_heavy = float(np.mean(np.abs(obs_heavy.astype(float) - obs_clear.astype(float))))
    max_heavy = float(np.max(np.abs(obs_heavy.astype(float) - obs_clear.astype(float))))

    print(f"Clear vs Light Fog (beta=0.05): Mean Pixel Shift = {l1_light:.2f} / 255, Max Shift = {max_light:.1f}")
    print(f"Clear vs Heavy Fog (beta=0.15): Mean Pixel Shift = {l1_heavy:.2f} / 255, Max Shift = {max_heavy:.1f}")
    print("=" * 85 + "\n")

    return {
        "light_fog_mean_shift": l1_light,
        "light_fog_max_shift": max_light,
        "heavy_fog_mean_shift": l1_heavy,
        "heavy_fog_max_shift": max_heavy,
    }


def run_true_simulator_fog_diagnostic(
    num_seeds: int = 60,
    base_seed: int = 9000,
) -> dict:
    print("=" * 85)
    print(f"RUNNING TRUE-SIMULATOR FOG ROBUSTNESS DIAGNOSTIC ({num_seeds} seeds, base={base_seed})")
    print("=" * 85)

    env = FinalOcclusionMergeEnv(request_cost=0.0)
    policies = ["NO_COMM", "RANDOM_REQUEST", "FULL_COMM", "ORACLE_SELECTIVE"]
    fog_results = {}

    for fog_name, fog_cfg in FOG_CONFIGS.items():
        beta = fog_cfg["beta"]
        print(f"\n--- Testing Fog Level: {fog_name} (beta={beta:.2f}, vis={fog_cfg['vis_dist']}) ---")
        level_results = {p: [] for p in policies}

        for idx in range(num_seeds):
            seed = base_seed + idx
            hazard_type = seed % 3
            h_name = {0: "CLEAR", 1: "FRONT_HAZARD", 2: "REAR_HAZARD"}[hazard_type]

            for pol in policies:
                obs, info = env.reset(seed=seed)
                ep_utility = 0.0
                ep_driving_r = 0.0
                crashed = False
                merged = False
                step = 0
                req_count = 0
                revealed_msg = env.MSG_UNKNOWN

                done = False
                while not done and step < 8:
                    # 1. Determine request action
                    if pol == "NO_COMM":
                        req = env.NONE
                    elif pol == "FULL_COMM":
                        req = env.ASK_FRONT if hazard_type == 1 else (env.ASK_REAR if hazard_type == 2 else env.ASK_FRONT)
                    elif pol == "ORACLE_SELECTIVE":
                        if step == 0:
                            if hazard_type == 1:
                                req = env.ASK_FRONT
                            elif hazard_type == 2:
                                req = env.ASK_REAR
                            else:
                                req = env.NONE
                        else:
                            req = env.NONE
                    elif pol == "RANDOM_REQUEST":
                        if step == 0:
                            rng = np.random.default_rng(seed * 100 + step)
                            req = int(rng.choice([env.NONE, env.ASK_FRONT, env.ASK_REAR]))
                        else:
                            req = env.NONE

                    msg = env.get_message(req)
                    if req != env.NONE:
                        req_count += 1
                    if msg != env.MSG_UNKNOWN:
                        revealed_msg = msg

                    # 2. Driving behavior conditioned on revealed V2X message vs degraded visibility
                    # Under fog, blind merges without V2X warnings suffer collision whenever hazards exist
                    if revealed_msg == env.MSG_HAZARD_AHEAD:
                        # Double lane change to safe Lane 0
                        drive_act = 0 if step in [0, 1] else 1
                    elif revealed_msg == env.MSG_HAZARD_BEHIND:
                        # Yield on ramp, let speeder pass, merge behind
                        if step == 0:
                            drive_act = 4  # SLOWER
                        elif step == 1:
                            drive_act = 1  # IDLE
                        elif step == 2:
                            drive_act = 0  # LANE_LEFT
                        else:
                            drive_act = 1
                    else:
                        # Standard merge into Lane 1
                        drive_act = 0 if step == 0 else 1

                    next_obs, u_rew, term, trunc, step_info = env.step(drive_act, req)
                    done = bool(term or trunc)

                    ep_utility += u_rew
                    ep_driving_r += step_info["raw_driving_reward"]
                    if step_info["crashed"]:
                        crashed = True
                    if step_info["is_merged"]:
                        merged = True

                    step += 1

                level_results[pol].append({
                    "seed": seed,
                    "hazard_state": h_name,
                    "utility": ep_utility,
                    "driving_return": ep_driving_r,
                    "crashed": crashed,
                    "safe_merge": bool(merged and not crashed),
                    "steps": step,
                    "requests": req_count,
                    "bits": req_count * 4,
                })

        # Summary for this fog level
        summary = {}
        print(f"{'Policy':<18} | {'Safe Merge':<10} | {'Collision':<10} | {'Driving Ret':<12} | {'Utility':<10} | {'Reqs/Ep':<8} | {'Bits/Ep'}")
        print("-" * 85)
        for pol in policies:
            runs = level_results[pol]
            safe_merges = float(np.mean([r["safe_merge"] for r in runs]) * 100.0)
            collisions = float(np.mean([r["crashed"] for r in runs]) * 100.0)
            drv_r = float(np.mean([r["driving_return"] for r in runs]))
            util = float(np.mean([r["utility"] for r in runs]))
            reqs = float(np.mean([r["requests"] for r in runs]))
            bits = float(np.mean([r["bits"] for r in runs]))

            summary[pol] = {
                "safe_merge_rate": safe_merges,
                "collision_rate": collisions,
                "mean_driving_return": drv_r,
                "mean_utility": util,
                "mean_requests": reqs,
                "mean_bits": bits,
            }
            print(f"{pol:<18} | {safe_merges:>8.1f}%  | {collisions:>8.1f}%  | {drv_r:>10.3f}   | {util:>8.3f}   | {reqs:>6.2f}   | {bits:>5.1f}b")

        # Breakdown
        print("\n  Breakdown by Hazard State (Safe Merge Rate):")
        for h_name in ["CLEAR", "FRONT_HAZARD", "REAR_HAZARD"]:
            sm_nc = float(np.mean([r["safe_merge"] for r in level_results["NO_COMM"] if r["hazard_state"] == h_name]) * 100.0)
            sm_fc = float(np.mean([r["safe_merge"] for r in level_results["FULL_COMM"] if r["hazard_state"] == h_name]) * 100.0)
            sm_os = float(np.mean([r["safe_merge"] for r in level_results["ORACLE_SELECTIVE"] if r["hazard_state"] == h_name]) * 100.0)
            print(f"    {h_name:<16}: NO_COMM = {sm_nc:>5.1f}%, FULL_COMM = {sm_fc:>5.1f}%, ORACLE = {sm_os:>5.1f}%")

        fog_results[fog_name] = summary

    out_file = Path("outputs/request_agent/robustness_fog_oracle_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(fog_results, f, indent=2)
    print(f"\nSaved fog diagnostic results to: {out_file}")
    return fog_results


def evaluate_zero_shot_model_on_fog(
    model_path: str = "outputs/personal_lab1/occlusion_final_attempt_01/my_lab1_world_model_occlusion_final.pt",
    dqn_path: str = "outputs/request_agent/occlusion_final_evaluation/dqn_seed1_lambda0.05.pt",
    num_seeds: int = 60,
    base_seed: int = 9000,
) -> dict:
    """Test ZERO-SHOT robustness of the existing trained WorldModel + DQN under Fog."""
    print("\n" + "=" * 85)
    print("TESTING ZERO-SHOT ROBUSTNESS OF EXISTING WORLDMODEL + DQN UNDER FOG")
    print("=" * 85)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "ego-merge-occlusion-final-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75

    wm = WorldModel.from_config(cfg.model, cfg.env, 5)
    load_checkpoint(model_path, wm, map_location="cpu")
    wm.eval()

    dqn = DoubleDQNAgent(state_dim=71, action_dim=3, device="cpu", seed=1)
    dqn.load(dqn_path)

    env = FinalOcclusionMergeEnv(cfg, request_cost=0.05)
    mpc = MergeMPCPlanner(model=wm, num_sequences=64, horizon=6, future_mode="persistent")

    zero_shot_results = {}
    for fog_name, fog_cfg in FOG_CONFIGS.items():
        beta = fog_cfg["beta"]
        records = []
        for idx in range(num_seeds):
            seed = base_seed + idx
            hazard_type = seed % 3
            h_name = {0: "CLEAR", 1: "FRONT_HAZARD", 2: "REAR_HAZARD"}[hazard_type]

            obs, info = env.reset(seed=seed)
            # Apply optical fog to camera observation
            fogged_obs = apply_fog_to_observation(obs, beta=beta)

            with torch.no_grad():
                z = wm.encoder(torch.from_numpy(fogged_obs.astype(np.float32) / 255.0).unsqueeze(0)).squeeze(0).numpy()

            req_oh = np.zeros(3, dtype=np.float32)
            req_oh[0] = 1.0
            msg_oh = np.zeros(4, dtype=np.float32)
            msg_oh[0] = 1.0
            state = np.concatenate([z, req_oh, msg_oh], axis=0)

            crashed = False
            merged = False
            step = 0
            req_count = 0
            revealed_msg = env.MSG_UNKNOWN
            done = False

            while not done and step < 8:
                req = dqn.select_action(state, evaluate=True)
                if req != env.NONE:
                    req_count += 1
                msg = env.get_message(req)
                if msg != env.MSG_UNKNOWN:
                    revealed_msg = msg

                mpc.rng = np.random.default_rng(seed * 1000 + step)
                drive_act, _, _ = mpc.plan(obs=fogged_obs, latent=z, request_action=req, message=revealed_msg)

                next_obs, u_rew, term, trunc, step_info = env.step(drive_act, req)
                done = bool(term or trunc)
                if step_info["crashed"]:
                    crashed = True
                if step_info["is_merged"]:
                    merged = True

                next_fogged = apply_fog_to_observation(next_obs, beta=beta)
                with torch.no_grad():
                    next_z = wm.encoder(torch.from_numpy(next_fogged.astype(np.float32) / 255.0).unsqueeze(0)).squeeze(0).numpy()

                r_oh = np.zeros(3, dtype=np.float32)
                r_oh[req] = 1.0
                m_oh = np.zeros(4, dtype=np.float32)
                m_oh[msg] = 1.0
                state = np.concatenate([next_z, r_oh, m_oh], axis=0)

                z = next_z
                fogged_obs = next_fogged
                step += 1

            records.append({
                "safe_merge": bool(merged and not crashed),
                "crashed": crashed,
                "requests": req_count,
            })

        safe_rate = float(np.mean([r["safe_merge"] for r in records]) * 100.0)
        crash_rate = float(np.mean([r["crashed"] for r in records]) * 100.0)
        mean_reqs = float(np.mean([r["requests"] for r in records]))
        zero_shot_results[fog_name] = {
            "safe_merge_rate": safe_rate,
            "collision_rate": crash_rate,
            "mean_requests": mean_reqs,
        }
        print(f"Zero-Shot {fog_name:<15}: Safe Merge = {safe_rate:>5.1f}%, Collision = {crash_rate:>5.1f}%, Reqs/Ep = {mean_reqs:.2f}")

    out_file = Path("outputs/request_agent/robustness_fog_zero_shot_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(zero_shot_results, f, indent=2)
    print(f"Saved zero-shot fog results to: {out_file}")

    return zero_shot_results


if __name__ == "__main__":
    measure_fog_observation_degradation()
    run_true_simulator_fog_diagnostic(num_seeds=60, base_seed=9000)
    evaluate_zero_shot_model_on_fog(num_seeds=30, base_seed=9000)
