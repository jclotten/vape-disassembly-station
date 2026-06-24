#!/usr/bin/env python3
"""Train using Stable Baselines3 instead of RSL-RL (alternative workflow).

Usage:
    python scripts/train_sb3.py --task Franka-Reach-v0 --headless --num_envs 1024
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Franka RL with Stable Baselines3")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total_timesteps", type=int, default=5_000_000)
parser.add_argument("--log_dir", type=str, default="logs")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import franka_rl


def main():
    env_cfg_cls = gym.spec(args.task).kwargs["env_cfg_entry_point"]
    sb3_cfg = gym.spec(args.task).kwargs.get("sb3_cfg_entry_point", {})

    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args.num_envs

    env = gym.make(args.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    log_dir = os.path.join(args.log_dir, f"{args.task}_sb3")
    os.makedirs(log_dir, exist_ok=True)

    model = PPO(
        sb3_cfg.get("policy", "MlpPolicy"),
        env,
        n_steps=sb3_cfg.get("n_steps", 24),
        batch_size=sb3_cfg.get("batch_size", 1024),
        learning_rate=sb3_cfg.get("learning_rate", 3e-4),
        gamma=sb3_cfg.get("gamma", 0.99),
        gae_lambda=sb3_cfg.get("gae_lambda", 0.95),
        clip_range=sb3_cfg.get("clip_range", 0.2),
        n_epochs=sb3_cfg.get("n_epochs", 8),
        ent_coef=sb3_cfg.get("ent_coef", 0.01),
        vf_coef=sb3_cfg.get("vf_coef", 0.5),
        max_grad_norm=sb3_cfg.get("max_grad_norm", 1.0),
        seed=args.seed,
        verbose=1,
        tensorboard_log=log_dir,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000, save_path=log_dir, name_prefix="franka_sb3"
    )

    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
    model.save(os.path.join(log_dir, "final_model"))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
