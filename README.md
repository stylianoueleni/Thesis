# Deep Reinforcement Learning for UAV Swarm Path Planning and Obstacle Avoidance

A from-scratch implementation of a **Shared Double DQN** agent that learns cooperative path planning and obstacle avoidance for a swarm of UAVs. The neural network, training loop, replay buffer, and Reynolds-based flocking reward are all written in **NumPy**.

This repository accompanies a diploma thesis at the **National Technical University of Athens (NTUA)**, School of Applied Mathematical and Physical Sciences.

---

## Overview

A shared policy controls every UAV in the swarm: all agents draw actions from a single Double DQN and coordination is induced through **reward shaping** rather than explicit communication. The reward combines a task term (progress toward the goal, obstacle penalties) with the three classic **Reynolds flocking** components - cohesion, separation, and alignment. Swarm sizes from single-UAV up to larger groups are trained and compared under an identical environment.

---

## Key Features

- **Custom Double DQN in NumPy** - He-initialized MLP, Huber loss, Adam, gradient clipping, target network, and experience replay, implemented directly.
- **Reynolds reward shaping** - cohesion, separation, and alignment terms with a configurable desired spacing.
- **Parametric 2D environment** - randomly placed circular obstacles with guaranteed start/goal clearance and fixed evaluation seed for reproducibility.
- **Single-UAV vs. swarm comparison** - baseline and swarm training scripts plus a cross-size comparison pipeline.
- **Reproducible figures** - training curves, evaluation trajectories, and per-metric comparison plots are written to disk on each run.

---

## Repository Structure

| File | Purpose |
|------|---------|
| `uavswarm_env_topology.py` | Swarm environment (`SwarmUAVEnv`): dynamics, LiDAR sensing, observations, reward, termination. |
| `uav_env_topology.py` | Single-UAV environment (`UAVEnv`) sharing the same obstacle topology. |
| `dqn_agent.py` | `DQNAgent` and the NumPy MLP: forward/backward pass, Adam, replay buffer, save/load. |
| `swarm_reward_shaper.py` | `SwarmRewardShaper` — cohesion / separation / alignment reward terms. |
| `train_dqn_swarm.py` | Train the shared Double DQN on the swarm; saves the model and evaluation plots. |
| `train_dqn_single.py` | Train the single-UAV baseline. |
| `train_all_swarm_sizes.py` | Train agents across a range of swarm sizes. |
| `evaluate_dqn.py` | Evaluate a trained agent and generate trajectory/metric figures. |
| `compare_swarm_sizes.py`, `compare_swarm_sizes_fixed.py` | Compare performance across swarm sizes. |
| `thesis.pdf` | Full written thesis (methodology, experiments, and discussion). |

---

## Environment and Problem Setup

| Setting | Value |
|---------|-------|
| Arena | `[-50, 50] × [-50, 50]` |
| Start / Goal | `(-40, -40)` → `(40, 40)`, goal tolerance `3.0 m` |
| Obstacles | 8 static circles, radii `3–8 m` |
| UAV speed / timestep | `3.0 m/s`, `Δt = 0.5 s`, horizon `T = 400` |
| Actions | 5 discrete heading changes: `{-30°, -15°, 0°, +15°, +30°}` |
| Sensing | 5 LiDAR rays at `{0°, ±30°, ±60°}`, range `30 m` |
| Observation | `9 + 2(N-1)` dimensions for swarm size `N` |

**Agent hyperparameters:** Double DQN · two 128-unit hidden layers · He init ·
Huber loss (δ=1) · Adam (lr `1e-3`, L2 `1e-4`) · gradient clip `10` · target
update every 4 steps · replay buffer `1e5` · batch `64` · γ = `0.99`.

**Reward shaping:** Reynolds cohesion / separation / alignment with desired
spacing `6 m` (separation weight `4.0`, alignment weight `0.8`), applied on top of
the task reward.

---

## Installation

Requires **Python 3.9+**. The only dependencies are NumPy and Matplotlib:

```bash
git clone https://github.com/stylianoueleni/Thesis.git
cd Thesis
pip install numpy matplotlib
```

---

## Usage

**Train the swarm agent** (saves `agent_DQN_swarm.npy` and evaluation figures):

```bash
python train_dqn_swarm.py
```

**Train the single-UAV baseline:**

```bash
python train_dqn_single.py
```

**Train across multiple swarm sizes:**

```bash
python train_all_swarm_sizes.py
```

**Evaluate a trained agent / compare swarm sizes:**

```bash
python evaluate_dqn.py
python compare_swarm_sizes_fixed.py
```

Running the scripts writes `.png` figures (training progress, evaluation trajectories, and comparison plots) to the working directory.

---

## Results

Performance is **non-monotonic in swarm size**: coordination is easiest for small groups and becomes markedly harder as the swarm grows.

| Swarm size | Success rate (in-distribution) |
|-----------:|:------------------------------:|
| N = 3 | ~92% |
| N = 6 | ~37% |
| N = 7 | ~29% |

Success peaks at **N = 3** and degrades substantially beyond it - this is a consistent finding of the study, not a training artifact.

**Caveats on the metrics.** Path length, trajectory smoothness, and energy use are shaped *indirectly* through the reward rather than optimized as explicit objectives, and reported swarm-level values depend on the chosen success criterion and on centroid averaging. In-distribution and out-of-distribution results are kept separate. Reported figures should be read with these measurement-methodology limits in mind rather than as guaranteed optima.

---

## Thesis and Acknowledgements

The full thesis is included in this repository as [`thesis.pdf`](thesis.pdf) and contains the complete methodology, experimental setup, and discussion of results.

- **Institution:** NTUA, School of Applied Mathematical and Physical Sciences
- **Topic:** Deep Reinforcement Learning for UAV swarm path planning and obstacle avoidance.
- **Supervision:** Prof. Iakovos Venieris and Dr. Ioannis Bartsiokas.

---

## License

The **code** in this repository is licensed under the MIT License — see the
[`LICENSE`](LICENSE) file for details.

The **thesis document** (`thesis.pdf`) is © 2026 Eleni Stylianou. All rights
reserved unless stated otherwise.
