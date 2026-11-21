"""Validation and Message Sensitivity Analysis for Design 1 World Model."""

from __future__ import annotations

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.data import load_episodes, TransitionDataset
from lab1.data.dataset import split_episodes
from lab1.models import WorldModel
from lab1.utils.model_handoff import load_personal_checkpoint

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def validate_world_model(
    model_path: Path | str = "outputs/personal_lab1/rear_bottleneck_attempt_01/my_lab1_world_model_rear_bottleneck.pt",
    dataset_path: Path | str = "datasets/merge_rear_bottleneck.npz",
    out_json_path: Path | str = "outputs/personal_lab1/rear_bottleneck_attempt_01/validation_results.json",
):
    print("=" * 80)
    print("DESIGN 1 WORLD MODEL VALIDATION & MESSAGE SENSITIVITY")
    print("=" * 80)

    model_path = Path(model_path)
    assert model_path.exists(), f"Model checkpoint not found: {model_path}"

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = str(dataset_path)

    episodes = load_episodes(dataset_path)
    _, val_episodes = split_episodes(episodes, val_fraction=0.1, seed=cfg.seed)
    val_ds = TransitionDataset(val_episodes)

    device = torch.device("cpu")
    model = WorldModel.from_config(cfg.model, cfg.env, 5).to(device)
    load_personal_checkpoint(model_path, model, cfg)
    model.eval()

    print(f"Validation transitions: {len(val_ds)}")

    # 1. Compute validation loss breakdown
    tot_dyn, tot_rec, tot_rew, tot_loss = 0.0, 0.0, 0.0, 0.0
    N = len(val_ds)

    with torch.no_grad():
        for i in range(N):
            sample = val_ds[i]
            batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
            losses = model.compute_losses(batch)
            tot_loss += float(losses["total"].item())
            tot_dyn += float(losses["dynamics"].item())
            tot_rec += float(losses["recon"].item())
            tot_rew += float(losses["reward"].item())

    mean_tot = tot_loss / N
    mean_dyn = tot_dyn / N
    mean_rec = tot_rec / N
    mean_rew = tot_rew / N

    print("\nMean Validation Losses:")
    print(f"  Total:    {mean_tot:.5f}")
    print(f"  Dynamics: {mean_dyn:.5f}")
    print(f"  Recon:    {mean_rec:.5f}")
    print(f"  Reward:   {mean_rew:.5f}")

    # 2. Counterfactual Message Sensitivity on identical visual states
    print("\n" + "=" * 80)
    print("COUNTERFACTUAL MESSAGE SENSITIVITY EVALUATION")
    print("Testing identical initial visual observations across REAR intents")
    print("=" * 80)

    # Pick samples from early episode steps (t=0)
    early_samples = [i for i in range(min(500, N)) if val_ds[i]["obs"] is not None][:5]
    sensitivity_records = []

    for idx, sample_idx in enumerate(early_samples):
        sample = val_ds[sample_idx]
        obs = sample["obs"].unsqueeze(0).to(device)
        z = model.encode(obs)

        print(f"\n--- State Sample #{sample_idx} ---")
        actions_to_test = [0, 1, 3, 4]  # LANE_LEFT, IDLE, FASTER, SLOWER
        req_actions = [2]  # ASK_REAR
        msg_variants = [
            ("REAR_MAINTAIN", 2, 1),
            ("REAR_ACCELERATE", 2, 2),
            ("REAR_BRAKE", 2, 3),
            ("NO_COMM", 0, 0),
        ]

        sample_record = {"sample_idx": sample_idx, "actions": {}}

        for act in actions_to_test:
            act_name = ACTION_NAMES[act]
            print(f"Driving Action: {act_name} ({act}):")
            act_t = torch.tensor([act], device=device, dtype=torch.long)

            preds_by_msg = {}
            latents_by_msg = {}

            for label, req_val, msg_val in msg_variants:
                req_t = torch.tensor([req_val], device=device, dtype=torch.long)
                msg_t = torch.tensor([msg_val], device=device, dtype=torch.long)

                pred_r = model.predict_reward(z, act_t, req_t, msg_t).item()
                z_next = model.predict_next(z, act_t, req_t, msg_t)

                preds_by_msg[label] = float(pred_r)
                latents_by_msg[label] = z_next

                print(f"  {label:<16}: Predicted Reward = {pred_r:+.4f}")

            # Latent distance between REAR_ACCEL and REAR_BRAKE
            latent_diff = torch.norm(latents_by_msg["REAR_ACCELERATE"] - latents_by_msg["REAR_BRAKE"]).item()
            rew_diff = preds_by_msg["REAR_BRAKE"] - preds_by_msg["REAR_ACCELERATE"]
            print(f"  -> ||z_next(BRAKE) - z_next(ACCEL)|| = {latent_diff:.4f}, Rew Diff (BRAKE - ACCEL) = {rew_diff:+.4f}")

            sample_record["actions"][act_name] = {
                "predicted_rewards": preds_by_msg,
                "latent_l2_diff_brake_accel": latent_diff,
                "reward_diff_brake_accel": rew_diff,
            }

        sensitivity_records.append(sample_record)

    results = {
        "mean_validation_loss": {
            "total": mean_tot,
            "dynamics": mean_dyn,
            "recon": mean_rec,
            "reward": mean_rew,
        },
        "message_sensitivity_samples": sensitivity_records,
    }

    out_p = Path(out_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved validation results to {out_p}")
    return results


if __name__ == "__main__":
    validate_world_model()

