# Communication-Efficient World Models for Connected Driving

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![HighwayEnv](https://img.shields.io/badge/Gym-HighwayEnv-green.svg)](https://highway-env.farama.org/)

## 1. Research Question
**Can an autonomous vehicle learn when and what semantic information to request from connected neighboring vehicles while minimizing communication overhead?**

Connected autonomous vehicles (CAVs) have the potential to resolve high-risk driving bottlenecks by sharing intent. However, broadcasting continuous state or requesting communication unconditionally saturates bandwidth and introduces latency. This project studies a hierarchical, communication-efficient architecture where:
1. A **Double DQN request agent** decides at each decision step whether to remain silent or selectively query surrounding vehicles for high-level semantic intent (`NONE`, `ASK_FRONT`, `ASK_REAR`).
2. A **communication-aware WorldModel** predicts latent multi-step rollouts conditioned on received semantic messages.
3. A **Model Predictive Control (MPC) driving planner** searches over candidate driving trajectories using the world model to safely execute highway merging maneuvers.

---

## 2. System Architecture

```
[Local Observation o_t] ──► [CNN Encoder] ──► [Latent z_t] ──────────────┐
                                                    │                    │
                                                    ▼                    ▼
                                      [Double DQN Request Policy]   [WorldModel MPC]
                                                    │                    ▲
                                                    ▼                    │
                                     [Request: NONE/FRONT/REAR]          │
                                                    │                    │
                                                    ▼                    │
                                        [Semantic Message m_t] ──────────┘
                                                    │
                                                    ▼
                                         [Driving Action a_t] ──► [Environment Physics]
```

### Action & Communication Spaces
- **Driving Actions ($5$ discrete)**: `0: LANE_LEFT`, `1: IDLE`, `2: LANE_RIGHT`, `3: FASTER`, `4: SLOWER`
- **Request Actions ($3$ discrete)**: `0: NONE`, `1: ASK_FRONT`, `2: ASK_REAR`
- **Semantic Messages ($4$ discrete)**: `0: UNKNOWN`, `1: MAINTAIN`, `2: ACCELERATE`, `3: BRAKE`
- **DQN State Space ($71\text{-D}$)**: $[z_t(64), \text{one\_hot}(a_{t-1}^{\text{req}}, 3), \text{one\_hot}(m_{t-1}, 4)]$
- **Request Reward / Multi-Objective Utility**:
  $$\mathcal{U}_t = R_{\text{driving}, t} - \lambda \cdot \mathbb{I}(a_t^{\text{req}} \neq \text{NONE})$$

---

## 3. Final Experimental Setup (Candidate #2)

The final evaluation scenario uses **Candidate #2**, a highway on-ramp merge bottleneck under partial observability:
- **Ego Vehicle**: Initial position $x = 30.0\text{ m}$, initial speed $v = 25.0\text{ m/s}$ (on access ramp).
- **Front Vehicle (Target Lane)**: Initial position $x = 54.0\text{ m}$, speed $v = 22.0\text{ m/s}$ (predictable lead, semantic intent `MAINTAIN`).
- **Rear Vehicle (Target Lane)**: Initial position $x = 18.0\text{ m}$, speed $v = 22.0\text{ m/s}$ (decision-critical trailing agent with hidden semantic intent randomly drawn from $\{\text{MAINTAIN}: 22\text{ m/s}, \text{ACCELERATE}: 27\text{ m/s}, \text{BRAKE}: 17\text{ m/s}\}$).
- **Initial Partial Observability**: Initial grayscale observations $o_0$ are $100\%$ byte-for-byte invariant across hidden intents, ensuring intent must be requested via communication or inferred dynamically.

---

## 4. Main Experimental Results

All policies were evaluated across **50 held-out fixed test seeds (Seeds 5000–5049)** that were strictly excluded from model and DQN training.

### Baseline Comparison (50 Held-Out Seeds)

| Policy | Safe Merge Rate (%) | Collision Rate (%) | Driving Return ($\mu \pm \sigma$) | Final Utility ($\mu \pm \sigma$) | Requests / Ep | Bits / Ep | % NONE | % ASK_FRONT | % ASK_REAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`NO_COMM`** | $98.0\%$ | $2.0\%$ | $4.169 \pm 0.310$ | **$4.169 \pm 0.310$** | $0.00$ | $0.0$ | $100.0\%$ | $0.0\%$ | $0.0\%$ |
| **`FULL_COMM`** | $98.0\%$ | $2.0\%$ | $4.228 \pm 0.351$ | $3.793 \pm 0.328$ | $8.70$ | $34.8$ | $0.0\%$ | $0.0\%$ | $100.0\%$ |
| **`RANDOM_REQUEST`** | $98.0\%$ | $2.0\%$ | $4.229 \pm 0.349$ | $3.935 \pm 0.353$ | $5.88$ | $23.5$ | $32.4\%$ | $32.6\%$ | $34.9\%$ |
| **`LEARNED_DQN (Seed 1)`** | $98.0\%$ | $2.0\%$ | $4.169 \pm 0.310$ | **$4.154 \pm 0.300$** | $0.30$ | $1.2$ | $96.5\%$ | $1.2\%$ | $2.4\%$ |
| **`LEARNED_DQN (Seed 2)`** | $98.0\%$ | $2.0\%$ | $4.169 \pm 0.310$ | **$4.148 \pm 0.325$** | $0.42$ | $1.7$ | $95.0\%$ | $2.8\%$ | $2.1\%$ |
| **`LEARNED_DQN (Seed 3)`** | $98.0\%$ | $2.0\%$ | $4.156 \pm 0.293$ | $3.954 \pm 0.284$ | $4.04$ | $16.2$ | $52.4\%$ | $2.8\%$ | $44.8\%$ |
| **`LEARNED_DQN (3-Seed Mean)`** | **$98.0\%$** | **$2.0\%$** | $4.165 \pm 0.006$ | **$4.085 \pm 0.093$** | **$1.59 \pm 1.74$** | **$6.3 \pm 6.9$** | **$81.3\%$** | **$2.3\%$** | **$16.4\%$** |
| **`ORACLE_REAR` (Diagnostic)** | $98.0\%$ | $2.0\%$ | $4.207 \pm 0.343$ | $4.207 \pm 0.343$ | $0.00$ | $0.0$ | $100.0\%$ | $0.0\%$ | $0.0\%$ |

### Key Findings & Nuance
1. **Safety Equivalence**: Collision rate is exactly **$2.0\%$** (1 collision out of 50 seeds) and safe merge rate is **$98.0\%$** across every baseline. Semantic communication **did not improve safety** because the MPC driving controller finds robust trajectories without neighbor intent.
2. **Bandwidth Savings**: Relative to full communication ($8.70$ requests/ep), the learned DQN policy achieves an average of $1.59$ requests/ep across 3 seeds—an **$81.8\%$ reduction in communication** (and up to $96.6\%$ reduction in Seed 1).
3. **Utility Preservation**: Because `FULL_COMM` pays communication costs on every step without improving safety, its utility drops to $3.793$. The learned DQN maintains high utility ($4.085$ aggregate, $4.154$ Seed 1) by learning to suppress unnecessary requests.
4. **Seed Variability**: Seed 1 and Seed 2 converged strongly to silent execution ($96.5\%$ and $95.0\%$ NONE), whereas Seed 3 retained moderate exploration requests ($52.4\%$ NONE), highlighting sensitivity to Q-learning initialization under weak reward signals.

---

## 5. Communication-Cost Sweep ($\lambda \in [0.00, 0.01, 0.05, 0.10]$)

| Request Cost ($\lambda$) | Driving Return | Final Utility | Safe Merge Rate (%) | Collision Rate (%) | Requests / Episode | Transmitted Bits / Ep | % NONE | % ASK_FRONT | % ASK_REAR |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\lambda = 0.00$** | $4.230$ | $4.230$ | $98.0\%$ | $2.0\%$ | $2.34$ | $9.4$ | $72.9\%$ | $13.7\%$ | $13.4\%$ |
| **$\lambda = 0.01$** | $4.205$ | $4.129$ | $98.0\%$ | $2.0\%$ | $7.54$ | $30.2$ | $12.5\%$ | $77.7\%$ | $9.7\%$ |
| **$\lambda = 0.05$** | $4.169$ | $4.154$ | $98.0\%$ | $2.0\%$ | $0.30$ | $1.2$ | $96.5\%$ | $1.2\%$ | $2.4\%$ |
| **$\lambda = 0.10$** | $4.169$ | $4.151$ | $98.0\%$ | $2.0\%$ | $0.18$ | $0.7$ | $97.9\%$ | $2.1\%$ | $0.0\%$ |

### Analysis of Cost Response
- **Suppression Threshold ($\lambda \ge 0.05$)**: Communication is strongly suppressed once $\lambda \ge 0.05$, with request percentages dropping below $0.30$ reqs/ep ($>96\%$ NONE).
- **Non-Monotonicity at $\lambda = 0.01$**: Rather than decreasing monotonically from $\lambda = 0.00$, requests spike to $7.54$ reqs/ep at $\lambda = 0.01$. Because a penalty of $0.01$ is minuscule relative to episode returns ($\sim 4.2$), the cost gradient is weak. The policy drifted toward `ASK_FRONT` ($77.7\%$), querying benign static intent that does not disrupt driving, before sharp suppression takes over at $\lambda \ge 0.05$.

---

## 6. Repository Structure

```
CS272-Final-Project/lab1_colab/
├── configs/
│   └── cpu_fast.yaml                              # Simulation & model configuration
├── datasets/
│   └── merge_final_candidate2.npz                 # Final offline dataset (15,000 transitions)
├── docs/
│   └── EXPERIMENT_NOTES.md                        # Complete research progression & milestone logs
├── outputs/
│   ├── personal_lab1/final_candidate2_attempt_01/
│   │   └── my_lab1_world_model_final_candidate2.pt  # Trained communication-aware WorldModel
│   └── request_agent/final_evaluation/
│       ├── baseline_comparison_table.csv          # Final baseline comparison (CSV)
│       ├── cost_sweep_table.csv                   # Final cost sweep table (CSV)
│       ├── final_evaluation_results.json          # Full evaluation rollout logs & metrics
│       ├── dqn_seed*.pt                           # Trained Double DQN policy checkpoints
│       └── dqn_seed*_history.json                 # 500-episode training curves
├── request_agent/
│   ├── final_candidate2_env.py                    # Candidate #2 Gym environment wrapper
│   ├── mpc_adapter.py                             # WorldModel MPC planner
│   ├── dqn_agent.py                               # Double DQN implementation
│   ├── network.py                                 # QNetwork architecture
│   ├── replay_buffer.py                           # Experience replay buffer
│   ├── train_and_evaluate_dqn.py                  # End-to-end training and evaluation script
│   └── [diagnostic scripts]                       # Diagnostic research tools (VoI, ablation, etc.)
├── src/lab1/                                      # Lab 1 WorldModel source package
├── requirements.txt                               # Pinned / bounded project dependencies
└── README.md                                      # Project documentation
```

---

## 7. Installation & Reproducibility

### Setup Environment
```bash
# Clone the repository
git clone <REPO_URL>
cd CS272-Final-Project/lab1_colab

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Set paths
export PYTHONPATH=src:.
export MPLCONFIGDIR=/tmp/matplotlib
```

### Reproduce Final Baseline Evaluation & Cost Sweep (Using Pretrained Checkpoints)
To reproduce all tables and metrics without retraining:
```bash
PYTHONPATH=src:. python request_agent/train_and_evaluate_dqn.py
```
This loads the frozen WorldModel (`my_lab1_world_model_final_candidate2.pt`) and trains the 3 DQN seeds ($500$ episodes each), executes the cost sweep, and writes `baseline_comparison_table.csv` and `cost_sweep_table.csv`.

---

## 8. Conclusions & Limitations

### Conclusion
The Double DQN request policy successfully learned to optimize multi-objective driving utility under communication penalties. However, because the underlying HighwayEnv merge dynamics admit robust local kinematic solutions (`FASTER` or `SLOWER`), semantic neighbor intent has low marginal decision value for MPC planning. Consequently, the RL policy discovered that **suppressing communication is the utility-maximizing strategy**, eliminating unnecessary network load without degrading driving safety.

### Limitations
1. **Safety Invariance**: Semantic communication did not improve collision or merge rates over uncommunicative baselines in this geometry.
2. **Kinematic Decoupling**: A strong continuous/discrete MPC planner can out-accelerate or yield to trailing traffic without requiring explicit coordination.
3. **Seed Variance**: Weak value-of-information gradients lead to variance in final request frequencies across random training seeds.
4. **Future Directions**: Future research should investigate environments with unavoidable physical occlusions, dense multi-lane intersections, or non-cooperative human drivers where blind autonomous execution is strictly sub-optimal.

