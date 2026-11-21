"""Isolated Scenario-Sweep Diagnostic to discover a communication-dependent merge scenario."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from highway_env import utils
import gymnasium as gym
from lab1.config import Config
from lab1.envs.merge import EgoMergeGymEnv
from request_agent.communication_wrapper import CommunicationMergeWrapper
from request_agent.mpc_adapter import MergeMPCPlanner

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


class ConfigurableMergeGymEnv(EgoMergeGymEnv):
    """Subclass of EgoMergeGymEnv allowing parameterized vehicle starting positions and speeds."""

    SCENARIO_CONFIG = {
        "ego_x": 30.0,
        "ego_speed": 25.0,
        "front_x": 75.0,
        "front_speed": 27.0,
        "rear_x": 5.0,
        "rear_speed": 27.0,
        "accel_delta": 5.0,
        "brake_delta": 5.0,
    }

    def _make_vehicles(self):
        road = self.road
        lanes = self.config["lanes_count"]

        self.target_lane_index = ("a", "b", lanes - 1)
        ramp_lane = road.network.get_lane(("j", "k", 0))

        s_cfg = self.SCENARIO_CONFIG
        ego_vehicle = self.action_type.vehicle_class(
            road,
            ramp_lane.position(s_cfg["ego_x"], 0.0),
            speed=s_cfg["ego_speed"],
        )
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        highway_lane = road.network.get_lane(self.target_lane_index)

        front_vehicle = vehicle_type(
            road,
            highway_lane.position(s_cfg["front_x"], 0.0),
            speed=s_cfg["front_speed"],
        )
        rear_vehicle = vehicle_type(
            road,
            highway_lane.position(s_cfg["rear_x"], 0.0),
            speed=s_cfg["rear_speed"],
        )

        road.vehicles.append(front_vehicle)
        road.vehicles.append(rear_vehicle)

        self.front_vehicle = front_vehicle
        self.rear_vehicle = rear_vehicle


def evaluate_scenario(
    scenario_name: str,
    scenario_cfg: dict[str, float],
    seeds: list[int],
    base_cfg: Config,
    ckpt_path: Path,
) -> dict[str, Any]:
    # Set the scenario config on the gym env class
    ConfigurableMergeGymEnv.SCENARIO_CONFIG = scenario_cfg

    if "ego-merge-v0" in gym.registry:
        del gym.registry["ego-merge-v0"]
    gym.register(id="ego-merge-v0", entry_point=ConfigurableMergeGymEnv)

    env = CommunicationMergeWrapper(cfg=base_cfg, model_checkpoint_path=ckpt_path, device="cpu")
    # Crucial: MPC must use the trained checkpoint loaded into env.model
    mpc = MergeMPCPlanner(
        model=env.model,
        num_sequences=64,
        horizon=6,
        allowed_actions=(0, 1, 3, 4),
        seed=42,
        device="cpu",
    )

    # If scenario modifies speed delta, monkey patch _apply_intent on env.env
    accel_delta = scenario_cfg.get("accel_delta", 5.0)
    brake_delta = scenario_cfg.get("brake_delta", 5.0)

    original_apply_intent = env.env._apply_intent

    def custom_apply_intent(vehicle, intent, base_speed):
        if vehicle is None:
            return
        if intent == env.env.MAINTAIN:
            target_speed = base_speed
        elif intent == env.env.ACCELERATE:
            target_speed = base_speed + accel_delta
        elif intent == env.env.BRAKE:
            target_speed = max(0.0, base_speed - brake_delta)
        else:
            return
        if hasattr(vehicle, "MAX_SPEED"):
            target_speed = min(target_speed, float(vehicle.MAX_SPEED))
        if hasattr(vehicle, "target_speed"):
            vehicle.target_speed = target_speed
        else:
            vehicle.speed = target_speed

    env.env._apply_intent = custom_apply_intent

    policies = ["NO_COMM", "ORACLE_FRONT", "ORACLE_REAR", "ORACLE_BEST"]
    seed_records: dict[str, list[dict[str, Any]]] = {p: [] for p in policies}
    action_changes = {"front": 0, "rear": 0, "total_steps": 0}

    for seed in seeds:
        for pol in policies:
            env.reset(seed=seed)
            t_front = env.request_message(1)
            t_rear = env.request_message(2)

            ep_ret = 0.0
            crashed = False
            merged = False
            done = False
            step = 0

            cand_rng = np.random.default_rng(seed * 777)

            while not done and step < 20:
                obs = env.current_obs
                latent = env.current_latent
                cands = cand_rng.integers(0, len(mpc.allowed_actions), size=(mpc.num_sequences, mpc.horizon))
                cands = mpc.allowed_actions[cands]

                if pol == "NO_COMM":
                    act, _, _ = mpc.plan(obs, latent, 0, 0, fixed_candidate_actions=cands)
                    # Counterfactual check on NO_COMM states: would revealing intent change action?
                    a_f, _, _ = mpc.plan(obs, latent, 1, t_front, fixed_candidate_actions=cands)
                    a_r, _, _ = mpc.plan(obs, latent, 2, t_rear, fixed_candidate_actions=cands)
                    if a_f != act:
                        action_changes["front"] += 1
                    if a_r != act:
                        action_changes["rear"] += 1
                    action_changes["total_steps"] += 1
                elif pol == "ORACLE_FRONT":
                    act, _, _ = mpc.plan(obs, latent, 1, t_front, fixed_candidate_actions=cands)
                elif pol == "ORACLE_REAR":
                    act, _, _ = mpc.plan(obs, latent, 2, t_rear, fixed_candidate_actions=cands)
                elif pol == "ORACLE_BEST":
                    act_f, ret_f, _ = mpc.plan(obs, latent, 1, t_front, fixed_candidate_actions=cands)
                    act_r, ret_r, _ = mpc.plan(obs, latent, 2, t_rear, fixed_candidate_actions=cands)
                    act = act_f if ret_f >= ret_r else act_r

                _, reward, terminated, truncated, info = env.step(act, 0, 0)
                done = bool(terminated or truncated)
                ep_ret += info["raw_driving_reward"]
                if info["crashed"]:
                    crashed = True
                if info["is_merged"]:
                    merged = True
                step += 1

            seed_records[pol].append({
                "return": ep_ret,
                "crashed": crashed,
                "merged": merged,
                "safe_merge": merged and (not crashed),
            })

    env.close()

    # Summarize metrics
    summary = {}
    no_comm = seed_records["NO_COMM"]
    for pol in policies:
        data = seed_records[pol]
        returns = [d["return"] for d in data]
        crashes = [float(d["crashed"]) for d in data]
        merges = [float(d["merged"]) for d in data]
        safe_merges = [float(d["safe_merge"]) for d in data]
        paired_diffs = [d["return"] - n["return"] for d, n in zip(data, no_comm)]

        summary[pol] = {
            "mean_return": float(np.mean(returns)),
            "collision_rate": float(np.mean(crashes)),
            "merge_rate": float(np.mean(merges)),
            "safe_merge_rate": float(np.mean(safe_merges)),
            "paired_improvement": float(np.mean(paired_diffs)),
        }

    tot_steps = max(1, action_changes["total_steps"])
    pct_front_chg = (action_changes["front"] / tot_steps) * 100.0
    pct_rear_chg = (action_changes["rear"] / tot_steps) * 100.0

    return {
        "name": scenario_name,
        "config": scenario_cfg,
        "gap": scenario_cfg["front_x"] - scenario_cfg["rear_x"],
        "front_distance": scenario_cfg["front_x"] - scenario_cfg["ego_x"],
        "rear_distance": scenario_cfg["ego_x"] - scenario_cfg["rear_x"],
        "summary": summary,
        "pct_front_action_changed": pct_front_chg,
        "pct_rear_action_changed": pct_rear_chg,
    }


def main():
    print("=" * 80)
    print("SCENARIO SWEEP DIAGNOSTIC FOR COMMUNICATION-DEPENDENT DRIVING")
    print("=" * 80)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication.npz"
    cfg.data.num_transitions = 5000
    cfg.env.id = "ego-merge-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75
    cfg.training.device = "cpu"

    ckpt_path = Path("outputs/personal_lab1/attempt_01/my_lab1_world_model_v2.pt")
    assert ckpt_path.exists()

    screening_seeds = [4000 + i for i in range(25)]

    candidates = [
        # Baseline (70m gap)
        ("Baseline (70m Gap)", {
            "ego_x": 30.0, "ego_speed": 25.0,
            "front_x": 75.0, "front_speed": 27.0,
            "rear_x": 5.0, "rear_speed": 27.0,
            "accel_delta": 5.0, "brake_delta": 5.0,
        }),
        # Candidate 1: Wide Moderate (45m gap: +30m front, -15m rear)
        ("Candidate 1 (45m Gap)", {
            "ego_x": 30.0, "ego_speed": 25.0,
            "front_x": 60.0, "front_speed": 27.0,
            "rear_x": 15.0, "rear_speed": 27.0,
            "accel_delta": 5.0, "brake_delta": 5.0,
        }),
        # Candidate 2: Front Pinch (38m gap: +20m front, -18m rear)
        ("Candidate 2 (Front Pinch, 38m Gap)", {
            "ego_x": 30.0, "ego_speed": 25.0,
            "front_x": 50.0, "front_speed": 27.0,
            "rear_x": 12.0, "rear_speed": 27.0,
            "accel_delta": 5.0, "brake_delta": 5.0,
        }),
        # Candidate 3: Rear Press (38m gap: +26m front, -12m rear)
        ("Candidate 3 (Rear Press, 38m Gap)", {
            "ego_x": 30.0, "ego_speed": 25.0,
            "front_x": 56.0, "front_speed": 27.0,
            "rear_x": 18.0, "rear_speed": 27.0,
            "accel_delta": 5.0, "brake_delta": 5.0,
        }),
        # Candidate 4: Balanced Tight (36m gap: +22m front, -14m rear)
        ("Candidate 4 (Balanced Tight, 36m Gap)", {
            "ego_x": 30.0, "ego_speed": 25.0,
            "front_x": 52.0, "front_speed": 27.0,
            "rear_x": 16.0, "rear_speed": 27.0,
            "accel_delta": 5.0, "brake_delta": 5.0,
        }),
        # Candidate 5: Dynamic Brake/Accel (36m gap with ±7 m/s delta)
        ("Candidate 5 (Dynamic Delta, 36m Gap)", {
            "ego_x": 30.0, "ego_speed": 25.0,
            "front_x": 52.0, "front_speed": 27.0,
            "rear_x": 16.0, "rear_speed": 27.0,
            "accel_delta": 7.0, "brake_delta": 7.0,
        }),
        # Candidate 6: Dual Pinch (32m gap: +20m front, -12m rear)
        ("Candidate 6 (Dual Pinch, 32m Gap)", {
            "ego_x": 30.0, "ego_speed": 25.0,
            "front_x": 50.0, "front_speed": 27.0,
            "rear_x": 18.0, "rear_speed": 27.0,
            "accel_delta": 5.0, "brake_delta": 5.0,
        }),
    ]

    sweep_results = []
    for name, s_cfg in candidates:
        t0 = time.time()
        print(f"\nEvaluating: {name} (Gap: {s_cfg['front_x'] - s_cfg['rear_x']}m)...")
        res = evaluate_scenario(name, s_cfg, screening_seeds, cfg, ckpt_path)
        print(f"Done in {time.time()-t0:.1f}s.")
        s = res["summary"]
        print(f"  NO_COMM     : SafeMerge={s['NO_COMM']['safe_merge_rate']*100:.1f}%, Collision={s['NO_COMM']['collision_rate']*100:.1f}%, Ret={s['NO_COMM']['mean_return']:.3f}")
        print(f"  ORACLE_FRONT: SafeMerge={s['ORACLE_FRONT']['safe_merge_rate']*100:.1f}%, Collision={s['ORACLE_FRONT']['collision_rate']*100:.1f}%, Ret={s['ORACLE_FRONT']['mean_return']:.3f} (diff: {s['ORACLE_FRONT']['paired_improvement']:+.3f})")
        print(f"  ORACLE_REAR : SafeMerge={s['ORACLE_REAR']['safe_merge_rate']*100:.1f}%, Collision={s['ORACLE_REAR']['collision_rate']*100:.1f}%, Ret={s['ORACLE_REAR']['mean_return']:.3f} (diff: {s['ORACLE_REAR']['paired_improvement']:+.3f})")
        print(f"  ORACLE_BEST : SafeMerge={s['ORACLE_BEST']['safe_merge_rate']*100:.1f}%, Collision={s['ORACLE_BEST']['collision_rate']*100:.1f}%, Ret={s['ORACLE_BEST']['mean_return']:.3f} (diff: {s['ORACLE_BEST']['paired_improvement']:+.3f})")
        print(f"  Action changes: Front={res['pct_front_action_changed']:.1f}%, Rear={res['pct_rear_action_changed']:.1f}%")
        sweep_results.append(res)

    out_file = Path("outputs/request_agent/scenario_sweep_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved all results to {out_file}")


if __name__ == "__main__":
    main()
