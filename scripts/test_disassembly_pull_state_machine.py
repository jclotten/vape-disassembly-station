#!/usr/bin/env python3
"""Visual physics probe for closing the Tip30 fingers on the lid and pulling up.

This is intentionally not an RL policy. It runs a deterministic state machine:
open gripper, move above the workpiece, move to grasp height, close the two
Tip30 slide edges, then lift the hand while keeping the gripper closed.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run a scripted Franka disassembly pull test.")
parser.add_argument("--task", type=str, default="Franka-Disassembly-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--x_offset", type=float, default=0.0, help="Hand target X offset from lid center in meters.")
parser.add_argument("--y_offset", type=float, default=0.0, help="Hand target Y offset from lid center in meters.")
parser.add_argument("--pregrasp_z_offset", type=float, default=0.18, help="Hand Z above lid center before descent.")
parser.add_argument("--grasp_z_offset", type=float, default=0.105, help="Hand Z above lid center while closing.")
parser.add_argument("--pull_height", type=float, default=0.16, help="Vertical lift after the gripper closes.")
parser.add_argument("--open_width", type=float, default=0.04, help="Panda finger joint target for open gripper.")
parser.add_argument("--close_width", type=float, default=0.0, help="Panda finger joint target for closed gripper.")
parser.add_argument("--approach_time", type=float, default=3.0)
parser.add_argument("--descend_time", type=float, default=2.0)
parser.add_argument("--close_time", type=float, default=1.5)
parser.add_argument("--lift_time", type=float, default=3.0)
parser.add_argument("--hold_time", type=float, default=2.0)
parser.add_argument("--loop", action="store_true", help="Repeat the scripted motion until the window closes.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

import franka_rl  # noqa: F401  # registers gym environments


@dataclass(frozen=True)
class Phase:
    name: str
    duration_s: float
    start_pos_key: str
    end_pos_key: str
    gripper_width: float


def _resolve_entry_point(entry_point):
    if callable(entry_point):
        return entry_point
    if isinstance(entry_point, str) and ":" in entry_point:
        mod_name, attr_name = entry_point.split(":")
        return getattr(importlib.import_module(mod_name), attr_name)
    return entry_point


def _as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def _smoothstep(alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _make_env():
    env_cfg_cls = _resolve_entry_point(gym.spec(args.task).kwargs["env_cfg_entry_point"])
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, 60.0)

    env = gym.make(args.task, cfg=env_cfg)
    env = env.unwrapped
    env.reset()
    env.sim.set_camera_view((1.05, -0.82, 0.82), (0.50, -0.02, 0.52))
    return env


def _set_robot_targets(robot, arm_joint_ids, gripper_joint_ids, arm_target, gripper_width):
    robot.set_joint_position_target_index(target=arm_target, joint_ids=arm_joint_ids)
    finger_target = torch.full(
        (robot.num_instances, len(gripper_joint_ids)),
        gripper_width,
        device=robot.device,
        dtype=arm_target.dtype,
    )
    robot.set_joint_position_target_index(target=finger_target, joint_ids=gripper_joint_ids)


def main():
    env = _make_env()
    scene = env.scene
    robot = scene["robot"]
    workpiece = scene["workpiece"]

    robot_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_cfg.resolve(scene)
    gripper_joint_ids, gripper_joint_names = robot.find_joints("panda_finger_joint.*")

    if robot.is_fixed_base:
        ee_jacobi_body_idx = robot_cfg.body_ids[0] - 1
    else:
        ee_jacobi_body_idx = robot_cfg.body_ids[0]
    jacobi_joint_ids = [joint_id + robot.num_base_dofs for joint_id in robot_cfg.joint_ids]

    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=robot.device)

    sim_dt = env.physics_dt
    current_hand_pose_w = _as_torch(robot.data.body_pose_w)[:, robot_cfg.body_ids[0], :7]
    initial_hand_pos_w = current_hand_pose_w[:, :3].clone()
    initial_hand_quat_w = current_hand_pose_w[:, 3:7].clone()
    initial_lid_pos_w = _as_torch(workpiece.data.root_pos_w)[:, :3].clone()

    xy_offset = torch.tensor((args.x_offset, args.y_offset, 0.0), device=robot.device).repeat(scene.num_envs, 1)
    grasp_pos_w = initial_lid_pos_w + xy_offset
    grasp_pos_w[:, 2] = initial_lid_pos_w[:, 2] + args.grasp_z_offset
    pregrasp_pos_w = grasp_pos_w.clone()
    pregrasp_pos_w[:, 2] = initial_lid_pos_w[:, 2] + args.pregrasp_z_offset
    lift_pos_w = grasp_pos_w.clone()
    lift_pos_w[:, 2] += args.pull_height

    positions = {
        "initial": initial_hand_pos_w,
        "pregrasp": pregrasp_pos_w,
        "grasp": grasp_pos_w,
        "lift": lift_pos_w,
    }
    phases = [
        Phase("approach above lid", args.approach_time, "initial", "pregrasp", args.open_width),
        Phase("descend to grip height", args.descend_time, "pregrasp", "grasp", args.open_width),
        Phase("close Tip30 slide edges", args.close_time, "grasp", "grasp", args.close_width),
        Phase("pull up", args.lift_time, "grasp", "lift", args.close_width),
        Phase("hold lifted pose", args.hold_time, "lift", "lift", args.close_width),
    ]

    print("[INFO] Scripted pull test ready.")
    print(f"[INFO] Arm joints: {robot_cfg.joint_names}")
    print(f"[INFO] Gripper joints: {gripper_joint_names}")
    print(f"[INFO] Lid start position env_0: {initial_lid_pos_w[0].detach().cpu().tolist()}")
    print("[INFO] Close the Isaac Sim window or press Ctrl+C to stop.")

    phase_idx = 0
    phase_step = 0
    total_step = 0
    last_phase_name = None

    while simulation_app.is_running():
        phase = phases[phase_idx]
        phase_steps = max(1, int(phase.duration_s / sim_dt))
        if phase.name != last_phase_name:
            print(f"[STATE] {phase.name}")
            last_phase_name = phase.name

        alpha = _smoothstep(phase_step / phase_steps)
        target_pos_w = (1.0 - alpha) * positions[phase.start_pos_key] + alpha * positions[phase.end_pos_key]
        target_pose_w = torch.cat((target_pos_w, initial_hand_quat_w), dim=-1)

        root_pose_w = _as_torch(robot.data.root_pose_w)
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pose_w[:, 0:3], target_pose_w[:, 3:7]
        )
        ik_command = torch.cat((target_pos_b, target_quat_b), dim=-1)
        ik.set_command(ik_command)

        jacobian = _as_torch(robot.data.body_link_jacobian_w)[:, ee_jacobi_body_idx, :, jacobi_joint_ids]
        ee_pose_w = _as_torch(robot.data.body_pose_w)[:, robot_cfg.body_ids[0], :7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        joint_pos = _as_torch(robot.data.joint_pos)[:, robot_cfg.joint_ids]
        arm_target = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        _set_robot_targets(robot, robot_cfg.joint_ids, gripper_joint_ids, arm_target, phase.gripper_width)
        scene.write_data_to_sim()
        env.sim.step(render=True)
        scene.update(sim_dt)

        if total_step % max(1, int(0.25 / sim_dt)) == 0:
            lid_pos = _as_torch(workpiece.data.root_pos_w)[:, :3]
            lift = lid_pos[:, 2] - initial_lid_pos_w[:, 2]
            finger_pos = _as_torch(robot.data.joint_pos)[:, gripper_joint_ids]
            print(
                f"[TRACE] t={total_step * sim_dt:5.2f}s phase='{phase.name}' "
                f"lid_z={lid_pos[0, 2].item():.4f} lift={lift[0].item():+.4f} "
                f"finger={finger_pos[0].detach().cpu().tolist()}"
            )

        phase_step += 1
        total_step += 1
        if phase_step >= phase_steps:
            phase_step = 0
            phase_idx += 1
            if phase_idx >= len(phases):
                if args.loop:
                    env.reset()
                    scene.update(sim_dt)
                    ik.reset()
                    phase_idx = 0
                    total_step = 0
                    last_phase_name = None
                else:
                    phase_idx = len(phases) - 1
                    phase_step = 0

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
