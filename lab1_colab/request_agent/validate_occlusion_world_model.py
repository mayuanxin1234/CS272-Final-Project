"""Decision-Critical Validation of Final Occlusion World Model.

Validates that on IDENTICAL initial visual observations (where pixels are byte-for-byte identical),
the learned WorldModel's imagined latent dynamics and reward predictions respond meaningfully
and counterfactually to the semantic messages:
- UNKNOWN (0)
- CLEAR (1)
- HAZARD_AHEAD (2)
- HAZARD_BEHIND (3)

Verifies:
1. Under HAZARD_AHEAD, immediate Lane-1 merge has lower predicted return than double lane change to Lane 0.
2. Under HAZARD_BEHIND, immediate Lane-1 merge has lower predicted return than ramp yielding.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import numpy as np
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.models import WorldModel
from lab1.utils.checkpoint import load_checkpoint
from request_agent.occlusion_env import FinalOcclusionMergeEnv


def validate_message_sensitivity(
    model_path: str | Path = "outputs/personal_lab1/occlusion_final_attempt_01/my_lab1_world_model_occlusion_final.pt",
    test_seed: int = 8000,
) -> dict:
    model_path = Path(model_path)
    assert model_path.exists(), f"Model not found at {model_path}"

    print("=" * 85)
    print("RUNNING DECISION-CRITICAL OCCLUSION WORLD MODEL VALIDATION")
    print(f"Model Checkpoint: {model_path}")
    print("=" * 85)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "ego-merge-occlusion-final-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75

    env = FinalOcclusionMergeEnv(cfg)
    obs, info = env.reset(seed=test_seed)

    # Load trained WorldModel
    model = WorldModel.from_config(cfg.model, cfg.env, num_actions=5)
    load_checkpoint(model_path, model, map_location="cpu")
    model.eval()

    # Define candidate driving trajectories over horizon H=4:
    # Trajectory A: Immediate standard merge into Lane 1: [LANE_LEFT, IDLE, IDLE, IDLE] -> [0, 1, 1, 1]
    # Trajectory B: Double lane change into Lane 0: [LANE_LEFT, LANE_LEFT, IDLE, IDLE] -> [0, 0, 1, 1]
    # Trajectory C: Yield on ramp, then merge: [SLOWER, IDLE, LANE_LEFT, IDLE] -> [4, 1, 0, 1]
    trajectories = {
        "Traj_A_Blind_Merge_Lane1": np.array([0, 1, 1, 1], dtype=np.int64),
        "Traj_B_Double_Change_Lane0": np.array([0, 0, 1, 1], dtype=np.int64),
        "Traj_C_Yield_On_Ramp": np.array([4, 1, 0, 1], dtype=np.int64),
    }

    obs_tensor = torch.from_numpy(obs.astype(np.float32) / 255.0).unsqueeze(0)

    messages_to_test = {
        0: "UNKNOWN",
        1: "CLEAR",
        2: "HAZARD_AHEAD",
        3: "HAZARD_BEHIND",
    }

    predicted_returns = {}
    print(f"\nEvaluating Predicted Returns Across Counterfactual Messages on Identical Observation (Seed {test_seed}):")
    print("-" * 85)
    print(f"{'Message Condition':<20} | {'Traj A (Merge Lane 1)':<22} | {'Traj B (Bypass Lane 0)':<22} | {'Traj C (Yield Ramp)'}")
    print("-" * 85)

    for msg_code, msg_name in messages_to_test.items():
        predicted_returns[msg_name] = {}
        row_str = f"{msg_name:<20} | "
        for traj_name, traj_acts in trajectories.items():
            H = len(traj_acts)
            actions_t = torch.from_numpy(traj_acts).unsqueeze(0)
            requests_t = torch.ones((1, H), dtype=torch.long) if msg_code != 0 else torch.zeros((1, H), dtype=torch.long)
            messages_t = torch.full((1, H), fill_value=msg_code, dtype=torch.long)

            with torch.no_grad():
                imagined = model.imagine(obs_tensor, actions_t, requests_t, messages_t)
                ret = float(imagined["rewards"].sum().item())
                predicted_returns[msg_name][traj_name] = ret
                row_str += f"{ret:>20.4f}   | "

        print(row_str.rstrip(" | "))

    print("-" * 85)

    # Check key directional properties:
    ret_clear_A = predicted_returns["CLEAR"]["Traj_A_Blind_Merge_Lane1"]
    ret_fwd_A = predicted_returns["HAZARD_AHEAD"]["Traj_A_Blind_Merge_Lane1"]
    ret_fwd_B = predicted_returns["HAZARD_AHEAD"]["Traj_B_Double_Change_Lane0"]

    ret_rear_A = predicted_returns["HAZARD_BEHIND"]["Traj_A_Blind_Merge_Lane1"]
    ret_rear_C = predicted_returns["HAZARD_BEHIND"]["Traj_C_Yield_On_Ramp"]

    # In FRONT_HAZARD: Trajectory B should be more attractive than Trajectory A
    front_sensitivity = ret_fwd_B > ret_fwd_A
    # In REAR_HAZARD: Trajectory C should be more attractive than Trajectory A
    rear_sensitivity = ret_rear_C > ret_rear_A

    print("\nMessage-Conditioned Sensitivity Checks:")
    print(f"1. HAZARD_AHEAD: Traj B (Bypass) > Traj A (Lane 1 Merge)? ({ret_fwd_B:.4f} > {ret_fwd_A:.4f}) -> {'PASS' if front_sensitivity else 'FAIL'}")
    print(f"2. HAZARD_BEHIND: Traj C (Yield) > Traj A (Lane 1 Merge)? ({ret_rear_C:.4f} > {ret_rear_A:.4f}) -> {'PASS' if rear_sensitivity else 'FAIL'}")

    out_file = model_path.parent / "decision_critical_sensitivity.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "predicted_returns": predicted_returns,
                "front_sensitivity": bool(front_sensitivity),
                "rear_sensitivity": bool(rear_sensitivity),
            },
            f,
            indent=2,
        )
    print(f"\nSaved sensitivity evaluation to: {out_file}")
    return predicted_returns


if __name__ == "__main__":
    validate_message_sensitivity()

