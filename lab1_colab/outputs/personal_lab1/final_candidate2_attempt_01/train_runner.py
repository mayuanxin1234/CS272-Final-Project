"""Train standard communication-aware WorldModel on Final Candidate #2 dataset."""

from __future__ import annotations

import os
import sys
import json
import hashlib
from pathlib import Path
import torch

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.data import load_num_actions, make_dataloaders
from lab1.models import WorldModel
from lab1.training import Trainer, resolve_device
from lab1.utils.checkpoint import load_checkpoint
from lab1.utils.model_handoff import save_personal_checkpoint
from lab1.utils.seed import set_seed


def compute_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    base_out = Path("outputs/personal_lab1/final_candidate2_attempt_01")
    ckpt_dir = base_out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    export_path = base_out / "my_lab1_world_model_final_candidate2.pt"

    print("=" * 80)
    print("TRAINING STANDARD COMMUNICATION-AWARE WORLD MODEL ON FINAL CANDIDATE #2")
    print("=" * 80)

    cfg = Config.load("configs/cpu_fast.yaml")
    dataset_path = Path("datasets/merge_final_candidate2.npz")
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"

    cfg.data.dataset_path = str(dataset_path)
    cfg.data.num_transitions = 15000
    cfg.data.batch_size = 64
    cfg.env.id = "ego-merge-final-candidate2-v0"
    cfg.env.lanes_count = 2
    cfg.env.vehicles_count = 2
    cfg.env.scaling = 1.75
    cfg.training.device = "auto"
    cfg.training.epochs = 16
    cfg.training.checkpoint_dir = str(ckpt_dir)
    cfg.training.log_dir = str(base_out / "runs")

    set_seed(cfg.seed)
    device = resolve_device(cfg.training.device)
    print(f"Device: {device}")

    train_loader, val_loader, train_eps, val_eps = make_dataloaders(
        cfg, path=cfg.data.dataset_path
    )
    num_actions = load_num_actions(cfg.data.dataset_path)
    print(f"Actions: {num_actions}, Train transitions: {len(train_loader.dataset)}, Val transitions: {len(val_loader.dataset)}")

    print("\nInitializing standard WorldModel from scratch...")
    model = WorldModel.from_config(cfg.model, cfg.env, num_actions)
    trainer = Trainer(cfg, model, train_loader, val_loader, device)

    print(f"\nStarting 16-epoch training loop on {device}...")
    best_checkpoint_path = trainer.fit()
    print(f"\nTraining loop finished. Best checkpoint: {best_checkpoint_path}")

    # Summary
    best_epoch = None
    min_val_loss = float("inf")
    print("\n" + "=" * 80)
    print("EPOCH-BY-EPOCH TRAINING AND VALIDATION LOSS SUMMARY")
    print("=" * 80)
    for record in trainer.history:
        epoch = record["epoch"]
        tr = record["train"]
        va = record["val"]
        val_total = va["total"]
        if val_total < min_val_loss:
            min_val_loss = val_total
            best_epoch = epoch
        print(f"Epoch {epoch:2d}:")
        print(f"  Train: total={tr['total']:.5f}, dynamics={tr.get('dynamics', 0.0):.5f}, recon={tr.get('recon', 0.0):.5f}, reward={tr.get('reward', 0.0):.5f}")
        print(f"  Val:   total={va['total']:.5f}, dynamics={va.get('dynamics', 0.0):.5f}, recon={va.get('recon', 0.0):.5f}, reward={va.get('reward', 0.0):.5f}")

    print(f"\nSelected Best Epoch: {best_epoch} with Validation Loss: {min_val_loss:.5f}")

    # Load best model for export
    print(f"\nLoading best checkpoint into fresh WorldModel...")
    best_model = WorldModel.from_config(cfg.model, cfg.env, num_actions)
    load_checkpoint(best_checkpoint_path, best_model, map_location="cpu")

    save_personal_checkpoint(
        export_path,
        best_model,
        cfg,
        training_transitions=len(train_loader.dataset),
        epochs=cfg.training.epochs,
        seed=cfg.seed,
    )

    # Inject metadata
    cfg_dict = cfg.to_dict()
    payload = torch.load(export_path, map_location="cpu", weights_only=False)
    payload["observation_preprocessing"] = {
        "type": "GrayscaleObservation",
        "weights": [0.2989, 0.5870, 0.1140],
        "normalization": "uint8 / 255.0",
        "frame_order": "oldest to newest",
        "scaling": 1.75,
        "actions": [
            "lane left",
            "idle",
            "lane right",
            "faster",
            "slower",
        ],
    }
    payload["training_config"] = {
        "seed": cfg.seed,
        "env": {
            "id": "ego-merge-v0",
            "image_size": cfg.env.image_size,
            "stack_size": cfg.env.stack_size,
            "policy_frequency": cfg.env.policy_frequency,
            "simulation_frequency": cfg.env.simulation_frequency,
            "lanes_count": cfg.env.lanes_count,
            "vehicles_count": cfg.env.vehicles_count,
            "duration": cfg.env.duration,
            "offscreen_rendering": cfg.env.offscreen_rendering,
            "scaling": 1.75,
        },
        "data": {
            "num_transitions": len(train_loader.dataset),
            "dataset_path": str(dataset_path),
            "val_fraction": cfg.data.val_fraction,
            "batch_size": cfg.data.batch_size,
            "num_workers": cfg.data.num_workers,
        },
        "model": cfg_dict.get("model", {}),
        "training": cfg_dict.get("training", {}),
    }
    payload["selected_checkpoint_epoch"] = best_epoch
    payload["best_val_loss"] = min_val_loss
    payload["dataset_sha256"] = compute_sha256(dataset_path)

    torch.save(payload, export_path)
    print(f"Successfully exported {export_path} (size: {export_path.stat().st_size / 1024**2:.2f} MB)")


if __name__ == "__main__":
    main()

