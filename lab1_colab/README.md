# Lab 1 Colab Workspace

This directory contains the experimental codebase, datasets, models, and interactive notebooks for the CS272 final project: **Communication-Efficient World Models for Connected Driving**.

For full project documentation, system architecture, verified benchmark tables, cost sweep curves, and robustness analyses, see the canonical [Project README](../README.md).

---

## Workspace Directory Guide

### 1. Interactive Notebooks
- [Lab 1: World-Model Learning](notebooks/1_learn_world_model.ipynb): Trains the autoencoder, recurrent latent transition dynamics, and reward predictors from HighwayEnv experience.
- [Lab 2: Imagination & Planning with MPC](notebooks/2_imagine_and_plan.ipynb): Uses the frozen world model to plan closed-loop driving maneuvers via latent Model Predictive Control (MPC).

### 2. Final Occlusion Experiment & Evaluation
- **Primary Occlusion Environment**: [`request_agent/occlusion_env.py`](request_agent/occlusion_env.py) (`ego-merge-occlusion-final-v0`)
- **Trained WorldModel Checkpoint**: [`outputs/personal_lab1/occlusion_final_attempt_01/my_lab1_world_model_occlusion_final.pt`](outputs/personal_lab1/occlusion_final_attempt_01/my_lab1_world_model_occlusion_final.pt)
- **Trained DQN Policy Checkpoints**: [`outputs/request_agent/occlusion_final_evaluation/`](outputs/request_agent/occlusion_final_evaluation/)
- **Final Benchmark Tables & Logs**:
  - Baseline Comparison: [`baseline_comparison_table.csv`](outputs/request_agent/occlusion_final_evaluation/baseline_comparison_table.csv)
  - Cost Sweep Table: [`cost_sweep_table.csv`](outputs/request_agent/occlusion_final_evaluation/cost_sweep_table.csv)
  - Evaluation Rollout Logs: [`final_evaluation_results.json`](outputs/request_agent/occlusion_final_evaluation/final_evaluation_results.json)

### 3. Supplementary Robustness Suites
- **Degraded Visibility (Optical Fog)**: [`request_agent/robustness_fog_diagnostic.py`](request_agent/robustness_fog_diagnostic.py)
- **Aggressive Drivers**: [`request_agent/evaluate_aggressive_zero_shot.py`](request_agent/evaluate_aggressive_zero_shot.py)

### 4. Research Milestones & History
- Complete chronological diagnostic notes: [`docs/EXPERIMENT_NOTES.md`](docs/EXPERIMENT_NOTES.md)

---

## Quick Reproduction Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run main baseline & cost sweep verification (<30s)
PYTHONPATH=src:. python request_agent/train_and_evaluate_occlusion_dqn.py --eval_only

# Run fog zero-shot robustness
PYTHONPATH=src:. python request_agent/robustness_fog_diagnostic.py

# Run aggressive-driver zero-shot robustness
PYTHONPATH=src:. python request_agent/evaluate_aggressive_zero_shot.py

# Launch interactive notebooks
jupyter lab notebooks/
```
