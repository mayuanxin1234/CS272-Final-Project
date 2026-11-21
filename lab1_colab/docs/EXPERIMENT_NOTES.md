# Research Progression & Experimental Diagnostic Notes

This document provides a concise, chronological record of the experimental milestones, diagnostic failures, and scientific findings that shaped the final project.

---

## 1. Milestone 1 — Original Merge Scenario (70m Gap)
- **Geometry**: Ego $x = 30\text{ m}$, Front $x = 80\text{ m}$, Rear $x = 10\text{ m}$ (initial target lane gap: $70\text{ m}$).
- **Observation**: The baseline `NO_COMM` policy achieved a **$100.0\%$ safe merge rate**.
- **Diagnosis**: The inter-vehicle gap was so wide that the ego vehicle could merge unilaterally into the highway at any time without interacting with or conditioning on neighboring vehicles. The scenario lacked decision-critical partial observability.

---

## 2. Milestone 2 — Candidate 4 (Balanced Tight Geometry)
- **Geometry**: Ego $x = 30\text{ m}, v = 25\text{ m/s}$; Front $x = 52\text{ m}, v = 27\text{ m/s}$; Rear $x = 16\text{ m}, v = 27\text{ m/s}$ (gap: $36\text{ m}$).
- **WorldModel Retraining**: Collected `datasets/merge_communication_tight.npz` (15,000 transitions); trained communication-aware WorldModel (validation loss: $0.0210$).
- **The Oracle Pathology**:
  Evaluating MPC with free oracle communication revealed that oracle information **degraded** closed-loop performance:
  - `NO_COMM`: $8.0\%$ collision rate, $92.0\%$ safe merge rate.
  - `ORACLE_FRONT`: $14.0\%$ collision rate, $86.0\%$ safe merge rate.
  - `ORACLE_REAR`: $12.0\%$ collision rate, $88.0\%$ safe merge rate.
- **Root Cause Analysis**: Timestep-by-timestep failure traces revealed two interacting defects:
  1. *Reward Underestimation*: The learned world model underestimated catastrophic collision penalties on rare terminal transitions.
  2. *Unconstrained MPC Summation*: The continuous MPC planner summed predicted rewards over a 6-step horizon without terminal constraints. When conditioned on an approaching rear vehicle, the planner chose passive hesitation actions (`IDLE` or `SLOWER`), trapping the ego on the ramp until the rear vehicle collided from behind.

---

## 3. Milestone 3 — Safety-Aware World Model & MPC Ablation
- **Architecture Extension**: Added a dedicated `CollisionHead` (76-D $\to$ 128 $\to$ 128 $\to$ 1 logit) predicting $P(\text{collision}_{t+1} \mid z_t, a_t, a_t^{\text{req}}, m_t)$.
- **Classification Performance**: Achieved strong collision detection on held-out transitions (AUROC: $0.993$, precision: $97.7\%$, recall: $96.8\%$).
- **Closed-Loop Ablation Gate**: Incorporating predicted collision risk into MPC objective:
  $$J = \sum_{t=0}^{H-1} \hat{r}_t - w_{\text{safety}} \cdot \sum_{t=0}^{H-1} \hat{P}_{\text{coll}, t}$$
  Testing across threshold weights ($w \in [2.0, 5.0, 10.0]$) failed to improve safe merges. Conservative braking triggered by high collision logits repeatedly stopped the vehicle in the merge lane, directly causing rear-end collisions from high-speed trailing traffic.

---

## 4. Milestone 4 — Design 1 (Rear Bottleneck / Yield-vs-Pass)
- **Geometry**: Front placed safely ahead at $x = 75\text{ m}$, rear bottleneck at $x = 18\text{ m}$, initial speeds $25\text{ m/s}$.
- **Result**: `NO_COMM` achieved $100.0\%$ safe merges under simple forward acceleration. The ego easily out-paced the rear vehicle even when the rear accelerated, confirming that manual geometry tweaking without simulator search was fragile.

---

## 5. Milestone 5 — Systematic True-Simulator Value-of-Information (VoI) Search
- **Methodology**: Suspended world-model training and conducted a grid search over 180 parameterized lane-change configurations directly in the **true HighwayEnv physics simulator**.
- **Discovery of Candidate #2**:
  - Ego: $x = 30\text{ m}, v = 25\text{ m/s}$
  - Front: $x = 54\text{ m}, v = 22\text{ m/s}$
  - Rear: $x = 18\text{ m}, v = 22\text{ m/s}$ (Intents: BRAKE $\to 17\text{ m/s}$, MAINTAIN $\to 22\text{ m/s}$, ACCEL $\to 27\text{ m/s}$)
  - Verification: $100\%$ byte-for-byte pixel invariance at $t=0$, zero intent leakage, and asymmetric local optimal action under true simulator rollouts.

---

## 6. Milestone 6 — True-Simulator Task-Reward Investigation
- **Problem**: Baseline HighwayEnv reward strongly favors forward speed ($0.45 \cdot v/30$), making `FASTER` universally dominant.
- **Investigation**: Evaluated 3 defensible alternative task rewards in the true simulator over 50 seeds:
  - *Variant A*: Flow Alignment ($22\text{ m/s}$) + Effort Penalty.
  - *Variant B*: Merge Completion + Symmetrical Jerk Penalty.
  - *Variant C*: Gaussian Speed Tracking + Action Penalty.
- **Scientific Finding**:
  Across all defensible task rewards:
  - `NO_COMM` achieved $100.0\%$ safe merges with $0.0\%$ collisions.
  - `ORACLE_REAR` produced $0.0\%$ return improvement ($\Delta = +0.000$) and $0.0\%$ action shift.
  - **Conclusion**: The HighwayEnv merge dynamics admit robust conservative driving strategies that guarantee safe merging without requiring neighboring intent. Semantic communication has intrinsically low decision value for the driving task.

---

## 7. Milestone 7 — Final Synthesis: Double DQN Request Policy
- **Objective**: Freeze Candidate #2 and evaluate whether a second RL agent (Double DQN) can learn to minimize communication costs under these dynamics.
- **Outcome**:
  - Across 3 independent training seeds, the DQN learned to suppress unnecessary communication, averaging **$1.59$ requests/episode** ($81.8\%$ reduction vs full communication) while maintaining an identical **$98.0\%$ safe merge rate**.
  - Confirmed that the learned request policy acts as an optimal utility-preserving filter when communication has low task value.

