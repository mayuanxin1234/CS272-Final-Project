# CS272 Final Project Group 1 (Merging task)

# Communication-Efficient World Models for Connected Driving

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![HighwayEnv](https://img.shields.io/badge/Gym-HighwayEnv-green.svg)](https://highway-env.farama.org/)

---

## 1. Research Question & Motivation
**Can an autonomous vehicle learn when and what semantic information to request from connected neighboring vehicles to merge safely under visual occlusion, while minimizing communication overhead?**

Visual line-of-sight sensors (cameras, LiDAR) on autonomous vehicles frequently suffer from severe blind zones caused by roadway curvature, physical obstructions, or neighboring vehicles. While Vehicle-to-Everything (V2X) communication allows connected autonomous vehicles (CAVs) to share information beyond direct line-of-sight, continuous raw sensor sharing or unconditional message broadcasting saturates wireless channels, introduces contention and latency, and consumes substantial computational bandwidth. 

This project studies a hierarchical, communication-efficient architecture where:
1. A **Double DQN request agent** learns to selectively query surrounding vehicles for high-level semantic messages (`NONE`, `ASK_FRONT`, `ASK_REAR`) under an explicit communication penalty $\lambda$.
2. A **communication-aware WorldModel** predicts latent multi-step rollouts conditioned on received semantic messages.
3. A **Model Predictive Control (MPC) driving planner** plans safe lane-merging trajectories in latent space using the world model.

---

## 2. System Architecture

```
[Local Grayscale Observation o_t] ──► [WorldModel CNN Encoder] ──► [Latent z_t (64-D)]
                                                                           │
                                                                           ▼
                                                              [Double DQN Request Policy]
                                                                           │
                                                                           ▼
                                                             [Request: NONE / FRONT / REAR]
                                                                           │
                                                                           ▼
                                                             [Semantic Message m_t (0..3)]
                                                                           │
                                                                           ▼
                                                                  [WorldModel MPC]
                                                                           │
                                                                           ▼
                                                                 [Driving Action a_t]
                                                                           │
                                                                           ▼
                                                                   [HighwayEnv Merge]
```

### Discrete Action & Semantic Spaces
- **Driving Actions ($5$ discrete)**: `0: LANE_LEFT`, `1: IDLE`, `2: LANE_RIGHT`, `3: FASTER`, `4: SLOWER`
- **Request Actions ($3$ discrete)**: `0: NONE`, `1: ASK_FRONT`, `2: ASK_REAR`
- **Semantic Messages ($4$ discrete)**:
  - `0: UNKNOWN` (default prior to inquiry or unqueried vehicle)
  - `1: CLEAR` (no obstacle detected; lane unobstructed)
  - `2: HAZARD_AHEAD` (lead vehicle detects downstream obstacle / emergency braking)
  - `3: HAZARD_BEHIND` (trailing vehicle detects high-speed overtaker charging lane)
- **DQN State Space (71 dimensions):** $s_t = [z_t,\; \mathrm{onehot}(a_{t-1}^{\mathrm{req}}),\; \mathrm{onehot}(m_{t-1})]$
- **Request Reward / Multi-Objective Utility:** $U_t = R_t^{\mathrm{driving}} - \lambda \mathbf{1}[a_t^{\mathrm{req}} \neq \mathrm{NONE}]$

---

## 3. Main Final Experiment: Cooperative Blind-Zone Sensing (Occlusion)

The primary benchmark is **Candidate A: Cooperative Blind-Zone Sensing in Target-Lane Merge** (`ego-merge-occlusion-final-v0` in [`lab1_colab/request_agent/occlusion_env.py`](lab1_colab/request_agent/occlusion_env.py)):
- **Ego Vehicle**: Begins on on-ramp (`"b", "c", 2`) at $s=5.0\text{ m}$ ($x=235.0\text{ m}$), $v=20.0\text{ m/s}$.
- **Front CAV**: Target highway lane (`"b", "c", 1`) at $s=35.0\text{ m}$ ($x=265.0\text{ m}$), $v=22.0\text{ m/s}$.
- **Rear CAV**: Target highway lane (`"a", "b", 1`) at $s=225.0\text{ m}$ ($x=225.0\text{ m}$), $v=22.0\text{ m/s}$.
- **3 Balanced Hidden States ($33.3\%$ each)**:
  1. `CLEAR`: Highway target lane clear; both CAVs report `CLEAR`.
  2. `FRONT_HAZARD`: Stationary obstacle stalled downstream at $x=285.0\text{ m}$ in target lane, hidden from ego by highway geometry. Front CAV observes obstacle and reports `HAZARD_AHEAD`.
  3. `REAR_HAZARD`: High-speed overtaker charging at $30.0\text{ m/s}$ in target lane, hidden from ego. Rear CAV observes charging overtaker and reports `HAZARD_BEHIND`.
- **Zero Initial Observation Leakage:** $\max_{i, j \in \{\mathrm{CLEAR}, \mathrm{FRONT}, \mathrm{REAR}\}} \|o_0^{(i)} - o_0^{(j)}\|_\infty = 0.0$
  Initial camera observations are $100\%$ byte-for-byte identical across all hidden states at $t=0$.

---

## 4. Main Experimental Results

All policies were evaluated across **60 held-out evaluation seeds (Seeds 9000–9059)** strictly excluded from model and DQN training (balanced: $20$ CLEAR, $20$ FRONT_HAZARD, $20$ REAR_HAZARD).

### Baseline Comparison Table (60 Held-Out Seeds, $\lambda = 0.05$)
| Policy | Safe Merge Rate (%) | Collision Rate (%) | Mean Driving Return | Mean Utility | Mean Travel Time (s) | Requests / Ep | Bandwidth (bits/ep) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NO_COMM** | $75.0\%$ | $25.0\%$ | $2.345$ | $2.345$ | $3.3$ | $0.00$ | $0.0$ b |
| **RANDOM_REQUEST** | $90.0\%$ | $10.0\%$ | $2.998$ | $2.969$ | $3.8$ | $0.58$ | $2.3$ b |
| **FULL_COMM** | $100.0\%$ | $0.0\%$ | $3.460$ | $3.410$ | $4.3$ | $1.00$ | $4.0$ b |
| **ORACLE_SELECTIVE** | $100.0\%$ | $0.0\%$ | $3.554$ | $3.521$ | $4.2$ | $0.67$ | $2.7$ b |
| **LEARNED_DQN (3-Seed Mean)** | **$100.0\%$** | **$0.0\%$** | **$3.585 \pm 0.002$** | **$3.504 \pm 0.007$** | **$4.2$** | **$1.62$** | **$6.5$ b** |

### Individual DQN Training Seeds ($\lambda = 0.05$)
- **Seed 1**: $100.0\%$ safe merge, $0.0\%$ collision, return $= 3.586$, utility $= 3.494$, $1.83$ reqs/ep ($7.3$ b).
- **Seed 2**: $100.0\%$ safe merge, $0.0\%$ collision, return $= 3.583$, utility $= 3.510$, $1.47$ reqs/ep ($5.9$ b).
- **Seed 3**: $100.0\%$ safe merge, $0.0\%$ collision, return $= 3.586$, utility $= 3.508$, $1.55$ reqs/ep ($6.2$ b).
- **Overall Request Distribution**: $61.4\%$ `NONE` / $13.7\%$ `ASK_FRONT` / $25.0\%$ `ASK_REAR`.

> [!NOTE]
> **Behavioral Nuance at Decision Initialization**:
> Because initial camera observations $o_0$ are identical across hidden hazard states, the DQN cannot know the true hazard before requesting assistance. At $t=0$, the learned policy consistently issues `ASK_REAR` as a risk-prioritized probe (addressing the fast rear overtaker vulnerability), and then dynamically adapts subsequent requests and driving actions based on the returned semantic message. We do not claim the agent knows the true hazard prior to querying.

---

## 5. Communication-Cost Sweep

We evaluated the sensitivity of the learned Double DQN request policy across varying communication penalty coefficients $\lambda \in [0.00, 0.10]$ on 60 held-out seeds:

| Lambda Cost ($\lambda$) | Safe Merge Rate (%) | Collision Rate (%) | Mean Driving Return | Mean Utility | Requests / Ep | Bandwidth (bits/ep) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$0.00$** | $100.0\%$ | $0.0\%$ | $3.568$ | $3.568$ | $3.27$ | $13.1$ b |
| **$0.01$** | $100.0\%$ | $0.0\%$ | $3.568$ | $3.536$ | $3.17$ | $12.7$ b |
| **$0.05$** | $100.0\%$ | $0.0\%$ | $3.586$ | $3.494$ | $1.83$ | $7.3$ b |
| **$0.10$** | $100.0\%$ | $0.0\%$ | $3.483$ | $3.374$ | $1.08$ | $4.3$ b |

**Key Finding**: Increasing communication penalty reduces request frequency by $67.0\%$ (from $3.27$ to $1.08$ requests/episode) while preserving a $100.0\%$ safe merge rate across the entire range. 

*(Note: Direct bandwidth comparisons between learned DQN and FULL_COMM depend on timing accounting conventions; FULL_COMM here queries the designated channel once per episode at $t=0$, whereas DQN queries reactively across steps).*

---

## 6. Robustness Experiment 1: Degraded Visibility (Optical Fog)

To evaluate whether cooperative V2X communication remains effective when ego vision is compromised, we implemented a physics-based **Beer–Lambert atmospheric scattering model** operating directly on the grayscale camera frames passed to the WorldModel CNN encoder: $I_{\mathrm{fog}}(r) = I(r) \cdot e^{-\beta r} + I_{\mathrm{airlight}} \cdot (1 - e^{-\beta r})$.

We evaluated the **frozen, pre-trained WorldModel and DQN policy zero-shot** (no retraining or fine-tuning):

| Fog Condition | Attenuation ($\beta$) | Visual Range | Safe Merge Rate (%) | Collision Rate (%) | Requests / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CLEAR WEATHER** | $0.00$ | $> 50$ m | **$100.0\%$** | $0.0\%$ | $1.83$ |
| **LIGHT FOG** | $0.05$ | $\approx 25$ m | **$100.0\%$** | $0.0\%$ | $3.57$ |
| **HEAVY FOG** | $0.15$ | $\approx 10$ m | **$100.0\%$** | $0.0\%$ | $2.13$ |

**Finding**: The learned system exhibits strong zero-shot robustness under severe optical degradation ($100\%$ safe merges across all fog levels). While request frequency varied under degraded inputs, safety was fully preserved because V2X radio messages operate out-of-band and condition the world model's latent transition predictions.

---

## 7. Robustness Experiment 2: Aggressive / Unpredictable Drivers

We tested the frozen WorldModel + Double DQN policy **zero-shot** against a 4-mode stochastic driver profile in the HighwayEnv simulator (60 held-out evaluation seeds, $15$ per style):
1. **NORMAL** ($25\%$): Standard polite headway ($T=1.5\text{ s}$), cruising at $22\text{ m/s}$.
2. **SUDDEN_ACCELERATE** ($25\%$): Rear vehicle surges aggressively to $32\text{ m/s}$ ($a=+5.0\text{ m/s}^2$, $T=0.4\text{ s}$).
3. **HARD_BRAKE** ($25\%$): Front vehicle decelerates abruptly to $5\text{ m/s}$ ($a=-6.0\text{ m/s}^2$).
4. **AGGRESSIVE_APPROACH** ($25\%$): Rear vehicle charges at $30\text{ m/s}$ refusing to yield to ramp traffic.

### Zero-Shot Overall Performance Comparison
| Policy | Safe Merge Rate (%) | Collision Rate (%) | Mean Driving Return | Mean Utility | Requests / Ep | Bandwidth (bits/ep) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NO_COMM** | $63.3\%$ | $36.7\%$ | $1.829$ | $1.829$ | $0.00$ | $0.0$ b |
| **RANDOM_REQUEST** | $88.3\%$ | $11.7\%$ | $2.950$ | $2.916$ | $0.68$ | $2.7$ b |
| **FULL_COMM** | $98.3\%$ | $1.7\%$ | $3.267$ | $3.217$ | $1.00$ | $4.0$ b |
| **ORACLE_SELECTIVE** | $98.3\%$ | $1.7\%$ | $3.352$ | $3.315$ | $0.75$ | $3.0$ b |
| **ZERO_SHOT_DQN (3-Seed Mean)** | **$98.3\%$** | **$1.7\%$** | **$3.405$** | **$3.326$** | **$1.56$** | **$6.2$ b** |

### Safe Merge Rate by Behavior Style (%)
| Behavior Style | Episodes | NO_COMM | RANDOM | FULL_COMM | ORACLE | ZERO_SHOT_DQN Aggregate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NORMAL** | 15 | $100.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | **$100.0\%$** |
| **SUDDEN_ACCELERATE** | 15 | $13.3\%$ | $73.3\%$ | $93.3\%$ | $93.3\%$ | **$93.3\%$** |
| **HARD_BRAKE** | 15 | $100.0\%$ | $100.0\%$ | $100.0\%$ | $100.0\%$ | **$100.0\%$** |
| **AGGRESSIVE_APPROACH** | 15 | $40.0\%$ | $80.0\%$ | $100.0\%$ | $100.0\%$ | **$100.0\%$** |

**Zero-Shot Verdict**: `ZERO-SHOT AGGRESSIVE ROBUSTNESS PASSED`. The frozen DQN and WorldModel generalized zero-shot to aggressive drivers, cutting collisions from $36.7\%$ to $1.7\%$ and matching the oracle baseline.

---

## 8. Main Research Conclusion

> In the occluded HighwayEnv merge scenario, selective semantic communication improved safe merging from $75\%$ without communication to $100\%$ with the learned request policy, eliminating observed collisions on the held-out evaluation set. Increasing communication cost reduced request frequency while preserving $100\%$ safe-merge performance. The frozen learned system also maintained strong zero-shot performance under degraded visibility and aggressive-driver behavior.

---

## 9. Limitations & Scope

1. **Simulation Scope**: All experiments were conducted in the 2D kinematic `highway-env` simulation framework; physical vehicle dynamics, road surface friction, and real-world sensor noise were not tested.
2. **Discrete Semantics**: Communication messages were constrained to a 4-token discrete vocabulary. Real-world V2X deployments may use continuous trajectory intents or hybrid messaging protocols.
3. **Synchronous Decision Steps**: The environment assumes synchronized decision rounds without network packet drops or multi-path RF fading.
4. **Safety Claims**: A $100\%$ safe merge rate in simulation does not guarantee real-world safety or imply that the DQN knows neighbor states before querying.

---

## 10. Project Notebooks

The repository includes the two foundational interactive lab notebooks located in [`lab1_colab/notebooks/`](lab1_colab/notebooks/):
- **Lab 1: World-Model Learning ([`1_learn_world_model.ipynb`](lab1_colab/notebooks/1_learn_world_model.ipynb))**: Trains the autoencoder, recurrent latent transition dynamics, and reward predictors on saved HighwayEnv driving experience.
- **Lab 2: Imagination & Planning with MPC ([`2_imagine_and_plan.ipynb`](lab1_colab/notebooks/2_imagine_and_plan.ipynb))**: Freezes the trained world model and performs latent-space Model Predictive Control (MPC) trajectory rollouts to plan closed-loop driving maneuvers.

Both notebooks are pre-configured with package discovery ([`lab1_colab/pyproject.toml`](lab1_colab/pyproject.toml)) and can be run locally or in JupyterLab.

---

## 11. Repository Structure & Final Artifacts

```
.
├── README.md                                   # [CANONICAL] Project documentation
├── lab1_colab/
│   ├── README.md                               # Workspace guide linking to canonical README
│   ├── pyproject.toml                          # Package configuration for notebook imports
│   ├── requirements.txt                        # Pinned dependencies
│   ├── notebooks/
│   │   ├── 1_learn_world_model.ipynb           # [LAB 1] World-model learning notebook
│   │   └── 2_imagine_and_plan.ipynb            # [LAB 2] Imagination & MPC planning notebook
│   ├── datasets/
│   │   ├── merge_occlusion_final.npz           # [FINAL] 20k transitions occlusion dataset
│   │   └── merge_final_candidate2.npz         # [PRIOR] Candidate #2 research dataset
│   ├── outputs/
│   │   ├── personal_lab1/
│   │   │   ├── occlusion_final_attempt_01/
│   │   │   │   ├── my_lab1_world_model_occlusion_final.pt  # [FINAL] Trained WorldModel
│   │   │   │   ├── validation_results.json
│   │   │   │   └── train_runner.py
│   │   │   └── final_candidate2_attempt_01/   # [PRIOR] Candidate #2 WorldModel
│   │   └── request_agent/
│   │       ├── occlusion_final_evaluation/     # [FINAL] 3 DQN seeds & cost sweep
│   │       │   ├── dqn_seed1_lambda0.05.pt
│   │       │   ├── dqn_seed2_lambda0.05.pt
│   │       │   ├── dqn_seed3_lambda0.05.pt
│   │       │   ├── dqn_seed1_lambda0.00.pt
│   │       │   ├── dqn_seed1_lambda0.01.pt
│   │       │   ├── dqn_seed1_lambda0.10.pt
│   │       │   ├── baseline_comparison_table.csv
│   │       │   ├── cost_sweep_table.csv
│   │       │   ├── request_by_hazard_state_table.csv
│   │       │   └── final_evaluation_results.json
│   │       ├── robustness_fog_oracle_results.json
│   │       ├── robustness_fog_zero_shot_results.json
│   │       ├── robustness_aggressive_driver_oracle_results.json
│   │       └── robustness_aggressive_zero_shot_results.json
│   ├── request_agent/
│   │   ├── occlusion_env.py                    # [FINAL] Occlusion Gym environment
│   │   ├── collect_occlusion_dataset.py        # Dataset collection pipeline
│   │   ├── train_and_evaluate_occlusion_dqn.py # DQN training and evaluation runner
│   │   ├── mpc_adapter.py                      # Latent WorldModel MPC planner
│   │   ├── dqn_agent.py                        # Double DQN agent architecture
│   │   ├── network.py                          # QNetwork neural architecture
│   │   ├── replay_buffer.py                    # Prioritized / uniform replay buffer
│   │   ├── robustness_fog_diagnostic.py        # Fog robustness evaluation
│   │   └── evaluate_aggressive_zero_shot.py    # Aggressive driver zero-shot evaluation
│   └── docs/
│       └── EXPERIMENT_NOTES.md                 # Complete research progression & notes
```

---

## 12. Reproducibility Guide

### Environment Setup
```bash
git clone <repo-url>
cd CS272-Final-Project/lab1_colab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Reproduce Main Baseline Comparison & Cost Sweep
```bash
PYTHONPATH=src:. python request_agent/train_and_evaluate_occlusion_dqn.py --eval_only
```

### Reproduce Fog Robustness Zero-Shot Evaluation
```bash
PYTHONPATH=src:. python request_agent/robustness_fog_diagnostic.py
```

### Reproduce Aggressive Driver Zero-Shot Evaluation
```bash
PYTHONPATH=src:. python request_agent/evaluate_aggressive_zero_shot.py
```

### Launch Interactive Notebooks
```bash
jupyter lab notebooks/
```

---

## 13. Contributors

- Devi Kamakshi
- Yuanxin Ma
