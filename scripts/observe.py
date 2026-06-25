#!/usr/bin/env python3
"""Observe an environment without training. Keeps the GUI open for inspection."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Observe environment (no training)")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import franka_rl  # noqa: F401


def main():
    import importlib

    def _resolve(ep):
        if callable(ep):
            return ep
        if isinstance(ep, str) and ":" in ep:
            mod, attr = ep.split(":")
            return getattr(importlib.import_module(mod), attr)
        return ep

    env_cfg_cls = _resolve(gym.spec(args.task).kwargs["env_cfg_entry_point"])
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args.num_envs

    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    print("[INFO] Environment loaded. Use the viewport to inspect the scene.")
    print("[INFO] Close the window or press Ctrl+C to exit.")

    action_dim = env.action_space.shape[-1] if len(env.action_space.shape) > 0 else 0
    action = torch.zeros(args.num_envs, max(action_dim, 1), device="cuda:0")

    while simulation_app.is_running():
        try:
            env.step(action)
        except Exception:
            simulation_app.update()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
