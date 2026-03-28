# Adaptive Path Planning for AUVs in Dynamic Aquaculture via DDPG-GAN Synergy

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-red)](https://pytorch.org/)

> **Zixin Huang** | Dalian University of Technology 

---

## Overview

This repository contains the implementation of an adaptive path planning framework for Autonomous Underwater Vehicles (AUVs) operating in dynamic aquaculture environments. The framework integrates **Deep Deterministic Policy Gradient (DDPG)** with **Generative Adversarial Networks (GAN)** to enable real-time, collision-aware trajectory optimization under complex hydrodynamic disturbances including ocean currents and unpredictable fish swarm movements.

Traditional path planning algorithms (A\*, RRT\*, APF) exhibit critical limitations when confronted with multiple interacting dynamic factors. This work addresses those limitations through adversarial environmental prediction coupled with continuous policy optimization, achieving substantial improvements across all key performance metrics.

---

## Key Results

| Metric                     | DDPG (Baseline) | **DDPG+GAN (Ours)** |  Improvement  |
| -------------------------- | :-------------: | :-----------------: | :-----------: |
| Mission Success Rate       |      68.3%      |      **91.2%**      |    +22.9%     |
| Path Optimality            |      68.5%      |      **87.3%**      |    +27.4%     |
| Dynamic Obstacle Avoidance |      62.1%      |      **85.1%**      |    +37.0%     |
| Avg. Mission Time (s)      |        —        |      **28.6**       | Best in class |
| Dynamic Collision Rate     |        —        |      **5.0%**       | Best in class |

Compared to traditional baselines at epoch 200:

| Algorithm           | Success Rate | Avg. Time (s) | Static Collision | Dynamic Collision |
| ------------------- | :----------: | :-----------: | :--------------: | :---------------: |
| DDPG+GAN (**Ours**) |  **77.4%**   |   **28.6**    |      15.6%       |     **5.0%**      |
| DQN                 |    63.8%     |     32.7      |       0.0%       |       24.7%       |
| RRT                 |    58.3%     |     41.2      |       0.0%       | — (28.5% timeout) |
| A\*                 |    42.6%     |     35.8      |      37.2%       |       20.2%       |
| APF                 |    31.4%     |     47.5      |      45.6%       |       23.0%       |

---

## Framework Architecture

<img width="456" height="299" alt="image" src="https://github.com/user-attachments/assets/1a65ca37-232d-401e-a1c3-ff9fbadf8c5e" />




### GAN Component

The generator synthesizes high-fidelity environmental parameters — ocean current velocities $(v_{cx}, v_{cy})$ and fish swarm density $\rho$ — from real-time AUV positional data. The discriminator validates these predictions against ground-truth sensor measurements, establishing a closed adversarial feedback loop:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

### DDPG Agent

The actor-critic architecture operates over a 9-dimensional continuous state space and a 3-dimensional continuous action space:

- **State**: $S = [x, y, z, v_x, v_y, v_z, v_{cx}, v_{cy}, \rho]$
- **Action**: $A = [\Delta v_x, \Delta v_y, \Delta v_z]$, each bounded in $[-5.0, 5.0]$ m/s

The critic estimates the action-value function; the actor maximizes it through policy gradient:

$$\nabla_{\theta^\mu} J \approx \mathbb{E}\left[\nabla_a Q(s, a | \theta^Q)\big|_{a=\mu(s)} \cdot \nabla_{\theta^\mu} \mu(s|\theta^\mu)\right]$$

### Hybrid Reward Function

The reward mechanism balances three competing objectives:

$$R_{total} = R_{arrival} + P_{collision} + P_{energy}$$

| Component         | Formula                                                      | Purpose                              |
| ----------------- | ------------------------------------------------------------ | ------------------------------------ |
| Arrival Reward    | $R_{arrival} = -0.1 \cdot d_{target}$                        | Directional guidance toward goal     |
| Collision Penalty | $P_{collision} = -10 \cdot \sum_{obs} \mathbf{1}(\|p - p_{obs}\| < r_{obs})$ | Obstacle avoidance enforcement       |
| Energy Penalty    | $P_{energy} = -0.01 \cdot \|v\|$                             | Energy-efficient velocity regulation |

### Dynamic Environment Modeling

Fish swarm center position evolves sinusoidally around $(50, 50, -10)$:

$$f_c(t) = \begin{pmatrix} x_f + 10\sin(t/10) \\ y_f + 10\cos(t/10) \\ -10 \end{pmatrix}$$

Local fish density decays linearly with AUV–swarm distance:

$$\rho = \max\left(0,\ 1 - \frac{d}{20}\right)$$

Ocean current velocity follows a periodic sinusoidal model:

$$v_{current}(t) = v_{base} + A \cdot \sin\left(\frac{2\pi t}{T}\right)$$

where $v_{base} = [0.2, 0.1]$ m/s, $A = 0.1$ m/s, $T = 60$ s.

---

## Installation

**Prerequisites**

- Python 3.8+
- [AirSim 1.8.1](https://github.com/microsoft/AirSim/releases/tag/v1.8.1)
- Unreal Engine 4.27.2
- Visual Studio 2022
- CUDA 11.3 + cuDNN 8.3.02

**Clone and install dependencies**

```bash
git clone https://github.com/YOUR_USERNAME/auv-ddpg-gan.git
cd auv-ddpg-gan
pip install -r requirements.txt
```

`requirements.txt`:

```
torch==1.12.1
numpy
matplotlib
airsim
msgpack-rpc-python
opencv-python
```

---

## Usage

### Training

```bash
# Train baseline DDPG agent
python train.py --mode ddpg --episodes 1000 --env aquaculture

# Train DDPG+GAN agent
python train.py --mode ddpg_gan --episodes 1000 --env aquaculture
```

### Evaluation

```bash
# Evaluate on static obstacles only
python eval.py --checkpoint checkpoints/ddpg_gan_ep1000.pth --scenario static

# Evaluate on dynamic obstacles (fish swarms + ocean currents)
python eval.py --checkpoint checkpoints/ddpg_gan_ep1000.pth --scenario dynamic
```

### Configuration

Key hyperparameters are in `config/default.yaml`:

```yaml
ddpg:
  actor_lr: 1.0e-4
  critic_lr: 1.0e-3
  gamma: 0.99
  tau: 0.005
  batch_size: 64
  buffer_size: 100000

gan:
  generator_lr: 2.0e-4
  discriminator_lr: 2.0e-4
  latent_dim: 16
  update_freq: 10          # GAN update interval (steps)

env:
  max_velocity: 5.0        # m/s per axis
  fish_swarm_center: [50, 50, -10]
  current_base: [0.2, 0.1] # m/s
  current_amplitude: 0.1
  current_period: 60       # seconds
```

---

## Project Structure

```
auv-ddpg-gan/
├── agents/
│   ├── ddpg.py              # DDPG actor-critic implementation
│   └── ddpg_gan.py          # DDPG+GAN integrated agent
├── envs/
│   ├── aquaculture_env.py   # AirSim-based aquaculture environment
│   ├── fish_swarm.py        # Dynamic fish swarm model
│   └── ocean_current.py     # Sinusoidal current dynamics
├── models/
│   ├── actor.py             # Policy network
│   ├── critic.py            # Value network
│   ├── generator.py         # GAN generator
│   └── discriminator.py     # GAN discriminator
├── utils/
│   ├── replay_buffer.py     # Prioritized experience replay
│   └── metrics.py           # ASR, ACR, APL computation
├── config/
│   └── default.yaml
├── train.py
├── eval.py
├── requirements.txt
└── README.md
```

---

## Simulation Environment

All experiments are conducted in **AirSim 1.8.1 + Unreal Engine 4.27.2** on an NVIDIA GeForce RTX 4060 Laptop GPU with PyTorch 1.12.1 (CUDA 11.3, cuDNN 8.3.02). The simulation provides:

- Photorealistic dynamic ocean current rendering via particle-based fluid dynamics
- Biologically inspired fish swarm behavior with sinusoidal trajectory modeling
- Millimeter-level depth estimation accuracy for obstacle detection
- Real-time AUV state telemetry via AirSim APIs

The AUV is modeled using AirSim's quadcopter platform, adapted for underwater dynamics.

---

## Evaluation Metrics

| Metric                           | Definition                                                   |
| -------------------------------- | ------------------------------------------------------------ |
| **Average Success Rate (ASR)**   | Proportion of trials where the AUV reaches the target without collision or timeout |
| **Average Collision Rate (ACR)** | Proportion of trials ending in collision (static or dynamic) |
| **Average Path Length (APL)**    | Mean Euclidean path length traveled to reach the target      |
| **Path Optimality**              | $\text{Path Optimality}(\%) = (L_a / L_t) \times 100$, where $L_a$ is actual path length and $L_t$ is theoretical optimal |

---

## Ablation Study

| Epoch   | DDPG Success | **DDPG+GAN Success** | DDPG Path Opt. | **DDPG+GAN Path Opt.** | DDPG Dyn. Avoid. | **DDPG+GAN Dyn. Avoid.** |
| ------- | :----------: | :------------------: | :------------: | :--------------------: | :--------------: | :----------------------: |
| 0       |     5.0%     |         8.0%         |     50.0%      |         55.0%          |      30.0%       |          25.0%           |
| 50      |    23.6%     |        30.8%         |     54.6%      |         63.6%          |      27.0%       |          39.8%           |
| 100     |    42.1%     |        53.6%         |     59.3%      |         72.2%          |      24.0%       |          54.5%           |
| 150     |    60.7%     |        76.4%         |     63.9%      |         80.7%          |      21.0%       |          69.3%           |
| **200** |    74.3%     |      **91.2%**       |     68.5%      |       **87.3%**        |      62.1%       |        **85.1%**         |

The GAN component contributes most critically to dynamic obstacle avoidance (+37.0% over DDPG at epoch 200), confirming that predictive environmental modeling — rather than reactive control alone — is the key differentiator in turbulent aquaculture scenarios.

---

## Contributions

1. A novel adaptive motion planning framework integrating DDPG and GAN for dynamic aquaculture AUV navigation, handling both static obstacles (reefs) and dynamic factors (fish swarms, ocean currents).
2. The first implementation of GAN trained via DDPG for underwater navigation, where the GAN dynamically simulates complex environmental conditions and the DDPG agent optimizes real-time path planning through adversarial learning.
3. A multi-dimensional state space formulation incorporating both kinematic parameters (position/velocity) and environmental dynamics (current velocity, fish density).
4. A hybrid reward mechanism with three original components: distance-proportional arrival reward, obstacle radius-based collision penalty, and velocity-norm energy penalty.
5. Extensive validation in simulated aquaculture environments demonstrating 22.7%, 27.4%, and 37.0% gains in success rate, path optimality, and dynamic obstacle avoidance respectively over baseline DDPG.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{huang2025auv,
  title     = {Adaptive Path Planning for AUVs in Dynamic Aquaculture via DDPG-GAN Synergy},
  author    = {Huang, Zixin},
  booktitle = {Proceedings of ...},
  year      = {2025},
  institution = {Dalian University of Technology}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

Experiments were conducted using [AirSim](https://github.com/microsoft/AirSim) (Microsoft) and [Unreal Engine](https://www.unrealengine.com/) (Epic Games). The DDPG implementation builds on the actor-critic framework described in [Lillicrap et al., 2016](https://arxiv.org/abs/1509.02971).
