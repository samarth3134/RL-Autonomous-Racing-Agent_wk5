# RL Autonomous Racing Agent

## Week 4 – Reinforcement Learning for Autonomous Systems

### Overview

This project implements an autonomous racing agent using Reinforcement Learning (RL). The agent is trained to drive around a 2D racing track in the Gymnasium CarRacing-v3 environment using the Proximal Policy Optimization (PPO) algorithm from Stable-Baselines3.

The objective of this project is to understand how an RL agent learns through interaction with its environment by receiving rewards instead of labelled data.

---

## Features

* PPO (Proximal Policy Optimization)
* Gymnasium CarRacing-v3 environment
* Stable-Baselines3 implementation
* CUDA (GPU) training support
* TensorBoard logging
* Model checkpoint saving
* Best model evaluation
* Frame stacking for improved observations

---

## Project Structure

```
RL-Autonomous-Racing-Agent/
│
├── src/
│   ├── env.py
│   ├── train.py
│   └── test.py
│
├── models/
├── submission_logs/
├── submission_models/
├── requirements.txt
└── README.md
```

---

## Files

### env.py

Creates the CarRacing-v3 environment used by the reinforcement learning agent.

---

### train.py

Responsible for training the PPO agent.

Main tasks:

* Creates multiple parallel environments.
* Trains using PPO.
* Saves checkpoints.
* Evaluates the best model.
* Logs TensorBoard statistics.

---

### test.py

Loads the trained model and runs it inside the racing environment to evaluate the learned policy.

---

## Reinforcement Learning Concepts

### State

96×96 image observations from the racing environment.

### Actions

Continuous action space:

* Steering
* Acceleration
* Braking

### Reward

The environment rewards the agent for successfully progressing around the track while discouraging inefficient driving.

### Environment

Gymnasium CarRacing-v3.

### Episode

One complete racing attempt until termination or reset.

---

## Training Configuration

| Parameter             |      Value |
| --------------------- | ---------: |
| Algorithm             |        PPO |
| Policy                | CNN Policy |
| Learning Rate         |     0.0003 |
| Batch Size            |        256 |
| n_steps               |       2048 |
| Gamma                 |       0.99 |
| GAE Lambda            |       0.95 |
| Entropy Coefficient   |      0.001 |
| Parallel Environments |          4 |
| Device                |       CUDA |

---

## Experiments

Several experiments were conducted during development:

* Baseline PPO training
* Reward shaping
* Hyperparameter tuning
* Frame stacking
* Parallel environment training
* GPU acceleration

Although the TensorBoard reward improved significantly during training, the final policy still struggled with aggressive cornering and excessive acceleration, demonstrating the importance of reward design and hyperparameter tuning in reinforcement learning.

---

## What I Learned

This project helped me understand:

* Reinforcement Learning fundamentals
* PPO training process
* Reward design
* Exploration vs exploitation
* Hyperparameter tuning
* TensorBoard monitoring
* GPU-accelerated RL training
* Challenges involved in autonomous decision making

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Train the model:

```bash
python src/train.py
```

Test the trained model:

```bash
python src/test.py
```

---

## Report

The detailed project report explaining the learning process, code implementation, experiments, and observations is included in the submission.

---

## GitHub Repository

https://github.com/samarth3134/RL-Autonomous-Racing-Agent

---

## Author

**Samarth Sharma**
