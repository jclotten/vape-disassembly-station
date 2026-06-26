#!/usr/bin/env python3
"""Evaluate / visualize a trained Franka RL agent.

Usage:
    python scripts/play.py --task Franka-Reach-Play-v0 --checkpoint logs/rsl_rl/franka_reach/.../model_1500.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained Franka RL policy")
parser.add_argument("--task", type=str, required=True, help="Play task id, e.g. Franka-Reach-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
parser.add_argument("--num_envs", type=int, default=50, help="Number of envs for visualization")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib
import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import franka_rl  # noqa: F401


def _resolve_entry_point(entry_point):
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
    env_cfg.scene.num_envs = args.num_envs

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    print(f"[INFO] Loaded checkpoint: {args.checkpoint}")

    policy = runner.get_inference_policy(device=agent_cfg.device)

    obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        step_result = env.step(actions)
        obs = step_result[0]

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
