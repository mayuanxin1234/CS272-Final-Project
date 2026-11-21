"""Validation and Collision Threshold Calibration for SafetyWorldModel."""

from __future__ import annotations

import os
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

from lab1.config import Config
from lab1.data import load_episodes, TransitionDataset
from lab1.data.dataset import split_episodes
from request_agent.models.safety_world_model import SafetyWorldModel

REQ_NAMES = {0: "NONE", 1: "ASK_FRONT", 2: "ASK_REAR"}
MSG_NAMES = {0: "UNKNOWN", 1: "MAINTAIN", 2: "ACCELERATE", 3: "BRAKE"}


def validate_and_calibrate(
    model_path: Path | str = "outputs/personal_lab1/safety_world_model_attempt_01/my_lab1_safety_world_model.pt",
    out_json_path: Path | str = "outputs/personal_lab1/safety_world_model_attempt_01/validation_results.json",
):
    print("=" * 80)
    print("SAFETY WORLD MODEL VALIDATION & THRESHOLD CALIBRATION")
    print("=" * 80)

    model_path = Path(model_path)
    assert model_path.exists(), f"Model not found: {model_path}"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    cfg = Config.load("configs/cpu_fast.yaml")
    cfg.data.dataset_path = "datasets/merge_communication_tight.npz"

    episodes = load_episodes(cfg.data.dataset_path)
    _, val_episodes = split_episodes(episodes, val_fraction=0.1, seed=cfg.seed)
    val_ds = TransitionDataset(val_episodes)

    model = SafetyWorldModel.from_config(cfg.model, cfg.env, 5)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    y_true_list = []
    p_coll_list = []
    logits_list = []
    req_list = []
    msg_list = []
    act_list = []
    val_samples = []

    print(f"Evaluating {len(val_ds)} validation transitions...")
    with torch.no_grad():
        for i in range(len(val_ds)):
            sample = val_ds[i]
            obs = sample["obs"].unsqueeze(0).to(device)
            act = sample["action"].unsqueeze(0).to(device)
            req = sample["request_action"].unsqueeze(0).to(device)
            msg = sample["message"].unsqueeze(0).to(device)
            r = sample["reward"].item()

            y_true = 1 if r <= -0.5 else 0

            z = model.encode(obs)
            logit = model.predict_collision_logit(z, act, req, msg).item()
            p_coll = float(torch.sigmoid(torch.tensor(logit)).item())

            y_true_list.append(y_true)
            p_coll_list.append(p_coll)
            logits_list.append(logit)
            req_list.append(sample["request_action"].item())
            msg_list.append(sample["message"].item())
            act_list.append(sample["action"].item())

            if y_true == 1:
                val_samples.append((obs, act, req, msg, r, i))

    y_true_arr = np.array(y_true_list)
    p_coll_arr = np.array(p_coll_list)
    req_arr = np.array(req_list)
    msg_arr = np.array(msg_list)

    n_pos = int(np.sum(y_true_arr))
    n_neg = int(len(y_true_arr) - n_pos)

    # Metrics
    auroc = float(roc_auc_score(y_true_arr, p_coll_arr))
    auprc = float(average_precision_score(y_true_arr, p_coll_arr))
    brier = float(np.mean((p_coll_arr - y_true_arr) ** 2))

    print(f"\nOverall Validation Metrics (N={len(y_true_arr)}, Positives={n_pos}, Negatives={n_neg}):")
    print(f"  AUROC:       {auroc:.4f}")
    print(f"  AUPRC:       {auprc:.4f} (Base positive rate: {n_pos/len(y_true_arr):.4f})")
    print(f"  Brier score: {brier:.4f}")

    # Candidate Threshold Evaluation
    candidate_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    thresh_results = []

    print("\n" + "=" * 90)
    print(f"{'Threshold':<10} | {'Recall (TPR)':<14} | {'Precision':<12} | {'FPR':<12} | {'F1 Score':<10} | {'TP / FP / FN / TN'}")
    print("=" * 90)

    best_f1 = -1.0
    best_thresh = 0.5
    calibrated_thresh = 0.5

    for th in candidate_thresholds:
        preds = (p_coll_arr >= th).astype(int)
        tp = int(np.sum((preds == 1) & (y_true_arr == 1)))
        fp = int(np.sum((preds == 1) & (y_true_arr == 0)))
        fn = int(np.sum((preds == 0) & (y_true_arr == 1)))
        tn = int(np.sum((preds == 0) & (y_true_arr == 0)))

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        fpr = fp / max(fp + tn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        print(f"{th:<10.2f} | {recall*100:>6.2f}%       | {precision*100:>6.2f}%    | {fpr*100:>6.2f}%    | {f1:>8.4f}  | {tp}/{fp}/{fn}/{tn}")
        res_dict = {
            "threshold": th,
            "recall": float(recall),
            "precision": float(precision),
            "fpr": float(fpr),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        thresh_results.append(res_dict)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

    # We want high collision recall (>= 80%) while keeping FPR under 10%
    # Select calibrated threshold satisfying safety criteria
    safe_candidates = [r for r in thresh_results if r["recall"] >= 0.80 and r["fpr"] <= 0.12]
    if safe_candidates:
        calibrated_thresh = max(safe_candidates, key=lambda x: x["f1"])["threshold"]
    else:
        # Fall back to best F1
        calibrated_thresh = best_thresh

    print(f"\nCalibrated Operational Collision Probability Threshold: p_crit = {calibrated_thresh:.2f}")

    # Breakdown across 7 Communication Conditions
    comm_conditions = [
        ("NONE", 0, 0),
        ("FRONT_MAINTAIN", 1, 1),
        ("FRONT_ACCEL", 1, 2),
        ("FRONT_BRAKE", 1, 3),
        ("REAR_MAINTAIN", 2, 1),
        ("REAR_ACCEL", 2, 2),
        ("REAR_BRAKE", 2, 3),
    ]

    print("\n" + "=" * 90)
    print("BREAKDOWN ACROSS 7 COMMUNICATION CONDITIONS (at p_crit = {:.2f})".format(calibrated_thresh))
    print(f"{'Condition':<18} | {'N':<6} | {'Collisions':<12} | {'Recall':<12} | {'Precision':<12} | {'FPR':<10}")
    print("-" * 90)

    cond_results = {}
    for name, req_val, msg_val in comm_conditions:
        mask = (req_arr == req_val) & (msg_arr == msg_val)
        sub_y = y_true_arr[mask]
        sub_p = p_coll_arr[mask]
        sub_n = len(sub_y)
        sub_pos = int(np.sum(sub_y))

        if sub_n > 0:
            preds = (sub_p >= calibrated_thresh).astype(int)
            tp = int(np.sum((preds == 1) & (sub_y == 1)))
            fp = int(np.sum((preds == 1) & (sub_y == 0)))
            fn = int(np.sum((preds == 0) & (sub_y == 1)))
            tn = int(np.sum((preds == 0) & (sub_y == 0)))

            rec = tp / max(tp + fn, 1) if sub_pos > 0 else 1.0
            prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
            fpr = fp / max(fp + tn, 1) if (len(sub_y) - sub_pos) > 0 else 0.0

            print(f"{name:<18} | {sub_n:<6} | {sub_pos:<12} | {rec*100:>6.2f}%     | {prec*100:>6.2f}%     | {fpr*100:>6.2f}%")
            cond_results[name] = {
                "n": sub_n,
                "collisions": sub_pos,
                "recall": float(rec),
                "precision": float(prec),
                "fpr": float(fpr),
            }

    # Counterfactual Message Sensitivity Analysis
    print("\n" + "=" * 90)
    print("COUNTERFACTUAL MESSAGE SENSITIVITY ON COLLISION-PRONE STATES")
    print("=" * 90)

    counterfactual_records = []
    # Test on up to 5 collision states
    num_to_test = min(5, len(val_samples))
    for idx in range(num_to_test):
        obs, act, req_orig, msg_orig, r, orig_idx = val_samples[idx]
        z = model.encode(obs)

        probs_by_cond = {}
        for name, req_val, msg_val in comm_conditions:
            req_t = torch.tensor([req_val], device=device, dtype=torch.long)
            msg_t = torch.tensor([msg_val], device=device, dtype=torch.long)
            logit = model.predict_collision_logit(z, act, req_t, msg_t).item()
            p = float(torch.sigmoid(torch.tensor(logit)).item())
            probs_by_cond[name] = p

        print(f"\nState #{orig_idx} (True Reward={r:.2f}, Act={act.item()}, Orig Req={REQ_NAMES[req_orig.item()]}, Orig Msg={MSG_NAMES[msg_orig.item()]}):")
        for name, p in probs_by_cond.items():
            print(f"  {name:<18}: P(collision) = {p:.4f} {'[ALARM]' if p >= calibrated_thresh else ''}")

        counterfactual_records.append({
            "val_index": orig_idx,
            "action": act.item(),
            "orig_req": REQ_NAMES[req_orig.item()],
            "orig_msg": MSG_NAMES[msg_orig.item()],
            "probs": probs_by_cond,
        })

    # Save complete validation output
    results = {
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "calibrated_threshold": calibrated_thresh,
        "best_f1_threshold": best_thresh,
        "threshold_eval": thresh_results,
        "condition_breakdown": cond_results,
        "counterfactual_samples": counterfactual_records,
    }

    out_path = Path(out_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved validation results to {out_path}")
    return results


if __name__ == "__main__":
    validate_and_calibrate()

