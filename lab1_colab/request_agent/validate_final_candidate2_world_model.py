"""Phase 4: Decision-Critical Validation for Final Candidate #2 World Model."""

from __future__ import annotations

import os
import json
import numpy as np
import torch
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.data import load_episodes, TransitionDataset
from lab1.data.dataset import split_episodes
from lab1.models import WorldModel
from lab1.utils.model_handoff import load_personal_checkpoint

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def validate_decision_critical_learning(
    model_path: Path | str = "outputs/personal_lab1/final_candidate2_attempt_01/my_lab1_world_model_final_candidate2.pt",
    dataset_path: Path | str = "datasets/merge_final_candidate2.npz",
    out_json_path: Path | str = "outputs/personal_lab1/final_candidate2_attempt_01/validation_results.json",
):
    print("=" * 80)
    print("PHASE 4: DECISION-CRITICAL WORLD MODEL VALIDATION (CANDIDATE #2)")
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

    N = len(val_ds)
    tot_dyn, tot_rec, tot_rew, tot_loss = 0.0, 0.0, 0.0, 0.0

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

    print(f"\nAverage Validation Losses across {N} transitions:")
    print(f"  Total:    {mean_tot:.5f}")
    print(f"  Dynamics: {mean_dyn:.5f}")
    print(f"  Recon:    {mean_rec:.5f}")
    print(f"  Reward:   {mean_rew:.5f}")

    # Explicit Decision-Critical Action Ordering Test
    print("\n" + "=" * 80)
    print("EXPLICIT ACTION ORDERING TEST AT IDENTICAL INITIAL STATES")
    print("=" * 80)

    actions = [0, 1, 3, 4]  # LANE_LEFT, IDLE, FASTER, SLOWER
    intents = [
        ("BRAKE", 2, 3),
        ("MAINTAIN", 2, 1),
        ("ACCELERATE", 2, 2),
    ]

    # Test across first 5 initial state transitions (t=0)
    samples_to_test = [i for i in range(min(500, N)) if val_ds[i]["obs"] is not None][:5]
    all_sample_orderings = []

    for idx, s_idx in enumerate(samples_to_test):
        sample = val_ds[s_idx]
        obs = sample["obs"].unsqueeze(0).to(device)
        z = model.encode(obs)

        print(f"\n--- State Sample #{s_idx} ---")
        ordering_record = {"sample_idx": s_idx, "by_intent": {}}

        for iname, req_val, msg_val in intents:
            req_t = torch.tensor([req_val], device=device, dtype=torch.long)
            msg_t = torch.tensor([msg_val], device=device, dtype=torch.long)

            preds = {}
            for act in actions:
                act_t = torch.tensor([act], device=device, dtype=torch.long)
                pred_r = model.predict_reward(z, act_t, req_t, msg_t).item()
                preds[ACTION_NAMES[act]] = float(pred_r)

            # Rank actions by predicted reward
            ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)
            ranked_str = " > ".join([f"{a} ({r:+.4f})" for a, r in ranked])
            print(f"  Rear Intent {iname:<11}: {ranked_str}")

            ordering_record["by_intent"][iname] = {
                "predictions": preds,
                "ranking": [a for a, _ in ranked],
            }

        all_sample_orderings.append(ordering_record)

    results = {
        "mean_validation_losses": {
            "total": mean_tot,
            "dynamics": mean_dyn,
            "recon": mean_rec,
            "reward": mean_rew,
        },
        "action_ordering_samples": all_sample_orderings,
    }

    out_p = Path(out_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nValidation output saved to {out_p}")
    return results


if __name__ == "__main__":
    validate_decision_critical_learning()

