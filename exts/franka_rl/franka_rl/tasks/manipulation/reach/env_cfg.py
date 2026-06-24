"""Franka Emika Reach environment configuration using Isaac Lab's Manager-Based RL API."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class FrankaReachSceneCfg(InteractiveSceneCfg):
    """Scene with a Franka arm and a ground plane."""

    num_envs = 4096
    env_spacing = 2.5

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


@configclass
class FrankaReachActionsCfg:
    arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action: ActionTermCfg | None = None


@configclass
class FrankaReachObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        target_pos = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "ee_target"}
        )
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def _ee_target_distance(env, asset_cfg: SceneEntityCfg, command_name: str) -> float:
    """Negative L2 distance from end-effector to target."""
    ee_pos = env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids[0], :3]
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    return -(ee_pos - target_pos).norm(dim=-1)


@configclass
class FrankaReachRewardsCfg:
    reaching_target = RewardTermCfg(
        func=_ee_target_distance,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "command_name": "ee_target",
        },
    )
    action_penalty = RewardTermCfg(func=mdp.action_l2, weight=-0.01)
    joint_vel_penalty = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001)


@configclass
class FrankaReachTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class FrankaReachEventsCfg:
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class FrankaReachCommandsCfg:
    ee_target = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.7),
            pos_y=(-0.4, 0.4),
            pos_z=(0.15, 0.6),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class FrankaReachEnvCfg(ManagerBasedRLEnvCfg):
    """Full configuration for the Franka Reach RL environment."""

    scene: FrankaReachSceneCfg = FrankaReachSceneCfg()
    actions: FrankaReachActionsCfg = FrankaReachActionsCfg()
    observations: FrankaReachObservationsCfg = FrankaReachObservationsCfg()
    rewards: FrankaReachRewardsCfg = FrankaReachRewardsCfg()
    terminations: FrankaReachTerminationsCfg = FrankaReachTerminationsCfg()
    events: FrankaReachEventsCfg = FrankaReachEventsCfg()
    commands: FrankaReachCommandsCfg = FrankaReachCommandsCfg()

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0)
    episode_length_s = 5.0
    decimation = 2

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation


@configclass
class FrankaReachEnvCfg_Play(FrankaReachEnvCfg):
    """Play / evaluation variant with fewer parallel envs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 10.0
        self.commands.ee_target.resampling_time_range = (6.0, 6.0)
