#!/usr/bin/env python3
"""Train an RL agent on a Franka Emika task using Isaac Lab + RSL-RL.

Usage:
    python scripts/train.py --task Franka-Reach-v0
    python scripts/train.py --task Franka-Reach-v0 --num_envs 2048 --max_iterations 2000
    python scripts/train.py --task Franka-PickAndPlace-v0 --headless
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Franka RL agent with Isaac Lab")
parser.add_argument("--task", type=str, required=True, help="Task id, e.g. Franka-Reach-v0")
parser.add_argument("--num_envs", type=int, default=None, help="Override number of environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max training iterations")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument("--log_dir", type=str, default="logs", help="Root log directory")
parser.add_argument("--video", action="store_true", default=False, help="Record rgb_array videos during training")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video in env steps")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in env steps")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib
import importlib.metadata as metadata
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import franka_rl  # noqa: F401  — registers gym environments

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _resolve_entry_point(entry_point):
    """Resolve a string entry point like 'module.path:ClassName' to the actual object."""
    if callable(entry_point):
        return entry_point
    if isinstance(entry_point, str) and ":" in entry_point:
        mod_name, attr_name = entry_point.split(":")
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name)
    return entry_point


def main():
    env_cfg_cls = _resolve_entry_point(gym.spec(args.task).kwargs["env_cfg_entry_point"])
    runner_cfg_cls = _resolve_entry_point(gym.spec(args.task).kwargs["rsl_rl_cfg_entry_point"])

    env_cfg = env_cfg_cls()
    agent_cfg = runner_cfg_cls()

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = args.seed

    log_root_path = os.path.join(args.log_dir, "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        video_folder = os.path.join(log_dir, "videos", "train")
        print(f"[INFO] Recording training videos in: {video_folder}")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step % args.video_interval == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    if args.resume:
        runner.load(args.resume)
        print(f"[INFO] Resumed from: {args.resume}")

    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
