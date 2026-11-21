"""Validate Candidate 4 tight world model and test message sensitivity."""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.models import WorldModel
from lab1.utils.model_handoff import load_personal_checkpoint
from request_agent.tight_merge_env import Candidate4MergeEnv

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def validate_student_preprocessing(payload: dict, cfg: Config) -> None:
    """Emulate the exact validation check performed by Lab 2."""
    assert "observation_preprocessing" in payload, "Missing observation_preprocessing"
    pre = payload["observation_preprocessing"]
    assert pre["type"] == "GrayscaleObservation", f"Unexpected type: {pre['type']}"
    assert pre["scaling"] == 1.75, f"Scaling must be 1.75, got {pre['scaling']}"
    assert pre["actions"] == ["lane left", "idle", "lane right", "faster", "slower"]
    assert "dataset_sha256" in payload, "Missing dataset_sha256"
    assert "selected_checkpoint_epoch" in payload, "Missing selected_checkpoint_epoch"
    print("Lab 2 observation_preprocessing check: PASSED!")


def test_message_sensitivity(
    model: WorldModel,
    device: torch.device,
) -> dict:
    """Test whether dynamics and reward predictions are sensitive to different semantic messages."""
    model.eval()

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "ego-merge-tight-v0"
    cfg.env.scaling = 1.75
    env = Candidate4MergeEnv(cfg)

    obs, _ = env.reset(seed=4001)
    env.close()

    obs_tensor = torch.from_numpy(obs.astype(np.float32) / 255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model.encode(obs_tensor)

    # Test driving action 0 (LANE_LEFT) and action 3 (FASTER)
    test_actions = [0, 3]
    msg_variants = [
        ("NONE_UNKNOWN", 0, 0),
        ("FRONT_MAINTAIN", 1, 1),
        ("FRONT_ACCELERATE", 1, 2),
        ("FRONT_BRAKE", 1, 3),
        ("REAR_MAINTAIN", 2, 1),
        ("REAR_ACCELERATE", 2, 2),
        ("REAR_BRAKE", 2, 3),
    ]

    print("\n" + "=" * 80)
    print("WORLD MODEL MESSAGE SENSITIVITY TEST (IDENTICAL OBS & ACTION)")
    print("=" * 80)

    sensitivity_report = {}

    for drv_act in test_actions:
        print(f"\nDriving Action: {ACTION_NAMES[drv_act]} ({drv_act})")
        print("-" * 75)
        print(f"{'Condition':<20} | {'Req':<4} | {'Msg':<12} | {'Pred Reward':<12} | {'Latent Diff (L2)'}")
        print("-" * 75)

        base_next_z = None
        base_reward = None

        for name, req, msg in msg_variants:
            act_t = torch.tensor([drv_act], dtype=torch.long, device=device)
            req_t = torch.tensor([req], dtype=torch.long, device=device)
            msg_t = torch.tensor([msg], dtype=torch.long, device=device)

            with torch.no_grad():
                next_z = model.predict_next(z, act_t, req_t, msg_t)
                pred_r = model.predict_reward(z, act_t, req_t, msg_t).item() if model.reward_head is not None else 0.0

            if name == "NONE_UNKNOWN":
                base_next_z = next_z
                base_reward = pred_r
                diff_l2 = 0.0
            else:
                diff_l2 = torch.norm(next_z - base_next_z).item()

            print(f"{name:<20} | {req:<4} | {MSG_NAMES[msg]:<12} | {pred_r:<12.5f} | {diff_l2:<12.5f}")
            sensitivity_report[f"act_{drv_act}_{name}"] = {
                "pred_reward": pred_r,
                "latent_diff_l2": diff_l2,
            }

    return sensitivity_report


def main():
    ckpt_path = Path("outputs/personal_lab1/tight_scenario_attempt_01/my_lab1_world_model_tight.pt")
    assert ckpt_path.exists(), f"Model file not found: {ckpt_path}"

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.env.id = "ego-merge-tight-v0"
    cfg.env.scaling = 1.75
    device = torch.device("cpu")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    validate_student_preprocessing(payload, cfg)

    model = WorldModel.from_config(cfg.model, cfg.env, 5).to(device)
    load_personal_checkpoint(ckpt_path, model, cfg)
    print("load_personal_checkpoint: SUCCESSFUL!")

    test_message_sensitivity(model, device)


if __name__ == "__main__":
    main()

