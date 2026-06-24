# Franka Emika RL — Isaac Lab

RL training environments for the Franka Emika Panda robot arm using [Isaac Lab](https://github.com/isaac-sim/IsaacLab).

## Available Tasks

| Task ID | Description |
|---|---|
| `Franka-Reach-v0` | Reach a random target pose with the end-effector |
| `Franka-Reach-Play-v0` | Evaluation variant (fewer envs, longer episodes) |
| `Franka-PickAndPlace-v0` | Pick a cube and place it at a target location |

## Prerequisites

- Ubuntu 22.04+
- NVIDIA GPU with driver ≥ 535 and CUDA ≥ 12.1
- Python 3.10

## Setup

```bash
chmod +x setup_env.sh
./setup_env.sh
source .venv/bin/activate
```

## Training

```bash
# Reach task (RSL-RL PPO, headless)
python scripts/train.py --task Franka-Reach-v0 --headless

# Reach task with GUI
python scripts/train.py --task Franka-Reach-v0

# Pick and Place
python scripts/train.py --task Franka-PickAndPlace-v0 --headless --num_envs 2048

# Alternative: Stable Baselines3
python scripts/train_sb3.py --task Franka-Reach-v0 --headless
```

## Evaluation

```bash
python scripts/play.py --task Franka-Reach-Play-v0 --checkpoint logs/franka_reach/model_1500.pt
```

## Monitoring

```bash
tensorboard --logdir logs/
```

## Project Structure

```
franka_rl_isaaclab/
├── exts/franka_rl/           # Custom Isaac Lab extension
│   └── franka_rl/
│       └── tasks/
│           └── manipulation/
│               ├── reach/         # Reach task env + agent configs
│               └── pick_and_place/ # Pick-and-place task
├── scripts/
│   ├── train.py              # RSL-RL training
│   ├── train_sb3.py          # Stable Baselines3 training
│   └── play.py               # Evaluate trained policy
├── setup_env.sh              # One-shot environment setup
└── logs/                     # Training outputs (gitignored)
```
