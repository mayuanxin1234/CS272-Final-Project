# Research Progression & Experimental Diagnostic Notes

This document provides a comprehensive, chronological record of the experimental milestones, diagnostic failures, and scientific insights that culminated in the final project.

---

## 1. Milestone 1 — Original Merge Scenario (70m Gap)
- **Geometry**: Ego $x = 30\text{ m}$, Front $x = 80\text{ m}$, Rear $x = 10\text{ m}$ (target lane gap: $70\text{ m}$).
- **Observation**: The baseline `NO_COMM` policy achieved a **$100.0\%$ safe merge rate**.
- **Diagnosis**: The inter-vehicle gap was so wide that the ego vehicle could merge unconditionally into the highway without interacting with neighboring vehicles. The scenario lacked decision-critical partial observability.

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
  - **Conclusion**: The HighwayEnv merge dynamics admit robust conservative driving strategies that guarantee safe merging without requiring neighboring intent. In open merge geometries, semantic communication has intrinsically low marginal value for driving because a single robust action (`FASTER` or `SLOWER`) can safely resolve the bottleneck regardless of neighbor intent.

---

## 7. Milestone 7 — Candidate #2 Double DQN Evaluation (Utility Filter)
- **Objective**: Freeze Candidate #2 and evaluate whether Double DQN can learn communication cost tradeoffs.
- **Outcome**:
  - Across 3 independent training seeds, the DQN learned to suppress unnecessary communication, averaging **$1.59$ requests/episode** ($81.8\%$ reduction vs full communication) while maintaining an identical **$98.0\%$ safe merge rate**.
  - Confirmed that the learned request policy acts as an optimal utility-preserving filter when communication has low task value.

---

## 8. Milestone 8 — The Information Gap Pivot: Moving to Occlusion
- **Scientific Insight**: In previous merge setups, the neighbor's hidden state was simply its acceleration intent. But highway merging physics often allow an ego vehicle to accelerate past or brake behind an uncooperative car regardless of intent. True decision-critical Value of Information requires a physical **information gap**—specifically, visual occlusions where a connected neighbor has sensory access to critical road hazards that the ego cannot perceive.
- **Hypothesis**: Cooperative blind-zone sensing creates genuine decision-theoretic Value of Information: without communication, the ego is blind to downstream or upstream hazards and will crash; with semantic communication from the situated CAV, the ego can take proactive avoidance maneuvers.

---

## 9. Milestone 9 — Cooperative Occlusion Redesign & True-Simulator Oracle Gate
- **Environment**: Candidate A — Cooperative Blind-Zone Sensing (`ego-merge-occlusion-final-v0`).
  - Balanced 3-way hidden hazard: `CLEAR` (33.3%), `FRONT_HAZARD` (33.3%, stationary obstacle at $x=285\text{ m}$), `REAR_HAZARD` (33.3%, $30\text{ m/s}$ overtaker).
  - 4 semantic messages: `UNKNOWN (0)`, `CLEAR (1)`, `HAZARD_AHEAD (2)`, `HAZARD_BEHIND (3)`.
  - Zero observation leakage: $100\%$ byte-for-byte identical initial frames across states ($\max \Delta = 0.000000$).
- **True-Simulator Oracle Gate Results (60 seeds)**:
  - `NO_COMM`: **$33.3\%$** safe merge rate ($66.7\%$ collision rate).
  - `RANDOM_REQUEST`: **$66.7\%$** safe merge rate ($33.3\%$ collision rate).
  - `FULL_COMM`: **$100.0\%$** safe merge rate ($0.0\%$ collision rate).
  - `ORACLE_SELECTIVE`: **$100.0\%$** safe merge rate ($0.0\%$ collision rate).
- **Verdict**: Gate **PASSED** decisively. Genuine Value of Information established without artificial reward tuning.

---

## 10. Milestone 10 — Learned WorldModel & Learned-MPC Occlusion Gate
- **Dataset Collection**: Collected `datasets/merge_occlusion_final.npz` (20,000 transitions, 5,288 episodes).
- **WorldModel Training**: Trained communication-aware WorldModel (`my_lab1_world_model_occlusion_final.pt`, validation loss: $0.08458$).
- **Learned-MPC Gate**:
  - `NO_COMM`: **$75.0\%$** safe merge rate ($25.0\%$ collision rate).
  - `ORACLE_SELECTIVE`: **$100.0\%$** safe merge rate ($0.0\%$ collision rate, $+51.6\%$ return gain).
- **Verdict**: Learned WorldModel MPC correctly leverages semantic message tokens in latent rollouts. Gate **PASSED**.

---

## 11. Milestone 11 — Final Double DQN Request Policy on Occlusion Scenario
- **DQN Architecture**: 71-D input $[z_t(64), a_{t-1}^{\text{req}}(3), m_{t-1}(4)]$, Huber loss, target network, soft updates.
- **Results Across 3 Seeds ($\lambda=0.05$, 60 Held-Out Seeds)**:
  - Safe Merge Rate: **$100.0\%$** (0.0% collision) across all 3 seeds.
  - Return: **$3.585 \pm 0.002$**, Utility: **$3.504 \pm 0.007$**.
  - Request Frequency: **$1.62$ requests/episode** ($6.5$ bits/ep).
  - Request Distribution: $61.4\%$ `NONE`, $13.7\%$ `ASK_FRONT`, $25.0\%$ `ASK_REAR`.
- **Communication Cost Sweep**:
  - $\lambda=0.00$: $3.27$ reqs/ep ($100\%$ safe).
  - $\lambda=0.01$: $3.17$ reqs/ep ($100\%$ safe).
  - $\lambda=0.05$: $1.83$ reqs/ep ($100\%$ safe).
  - $\lambda=0.10$: $1.08$ reqs/ep ($100\%$ safe).
  - Confirmed: Higher communication cost penalizes unnecessary queries while maintaining zero collisions.

---

## 12. Milestone 12 — Optical Fog / Degraded Visibility Robustness
- **Optical Scattering Model**: Implemented Beer–Lambert atmospheric scattering directly on the camera frames ($I_{\text{fog}}(r) = I(r)e^{-\beta r} + I_{\text{airlight}}(1 - e^{-\beta r})$).
- **Zero-Shot Evaluation**: Tested frozen WorldModel + DQN without retraining across 3 visibility levels:
  - `CLEAR` ($\beta=0.00$, vis $>50\text{ m}$): $100.0\%$ safe merge ($0.0\%$ collision), $1.83$ reqs/ep.
  - `LIGHT FOG` ($\beta=0.05$, vis $\approx 25\text{ m}$): $100.0\%$ safe merge ($0.0\%$ collision), $3.57$ reqs/ep.
  - `HEAVY FOG` ($\beta=0.15$, vis $\approx 10\text{ m}$): $100.0\%$ safe merge ($0.0\%$ collision), $2.13$ reqs/ep.
- **Conclusion**: The learned system exhibits strong zero-shot robustness under degraded vision because V2X radio messages operate out-of-band and condition the world model's latent predictions.

---

## 13. Milestone 13 — Aggressive / Unpredictable Driver Zero-Shot Robustness
- **Behavior Distribution**: 4-mode stochastic driver distribution (`NORMAL`, `SUDDEN_ACCELERATE`, `HARD_BRAKE`, `AGGRESSIVE_APPROACH`).
- **Zero-Shot Evaluation (60 Held-Out Seeds)**:
  - `NO_COMM`: $63.3\%$ safe merge ($36.7\%$ collision rate).
  - `RANDOM_REQUEST`: $88.3\%$ safe merge ($11.7\%$ collision rate).
  - `FULL_COMM`: $98.3\%$ safe merge ($1.7\%$ collision rate).
  - `ORACLE_SELECTIVE`: $98.3\%$ safe merge ($1.7\%$ collision rate).
  - `ZERO_SHOT_DQN (3-Seed Mean)`: **$98.3\%$ safe merge** ($1.7\%$ collision rate), return $= 3.405$, utility $= 3.326$, $1.56$ reqs/ep.
- **Behavior Breakdown**:
  - `NORMAL`: DQN $100.0\%$
  - `SUDDEN_ACCELERATE`: NO_COMM $13.3\% \to$ DQN **$93.3\%$**
  - `HARD_BRAKE`: DQN $100.0\%$
  - `AGGRESSIVE_APPROACH`: NO_COMM $40.0\% \to$ DQN **$100.0\%$**
- **Conclusion**: Zero-shot robustness confirmed. At merge onset ($t=0$), the frozen DQN consistently queries `ASK_REAR` to resolve blind-spot tailgating risk, enabling the WorldModel MPC to adjust merge timing and eliminate over $95\%$ of uncoordinated collisions.
