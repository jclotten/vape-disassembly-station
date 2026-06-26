"""Franka Emika Pick-and-Place environment configuration."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


@configclass
class FrankaPickAndPlaceSceneCfg(InteractiveSceneCfg):
    num_envs = 2048
    env_spacing = 2.5

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=RigidBodyPropertiesCfg(),
            mass_props=None,
            collision_props=None,
            visual_material=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.04),
        ),
    )


@configclass
class FrankaPickAndPlaceActionsCfg:
    arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action: ActionTermCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger.*": 0.04},
        close_command_expr={"panda_finger.*": 0.0},
    )


@configclass
class FrankaPickAndPlaceObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        cube_pos = ObservationTermCfg(
            func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube")}
        )
        target_pos = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "place_target"}
        )
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


def _grasp_reward(env, robot_cfg: SceneEntityCfg, cube_cfg: SceneEntityCfg) -> float:
    ee_pos = env.scene[robot_cfg.name].data.body_pos_w[:, robot_cfg.body_ids[0], :3]
    cube_pos = env.scene[cube_cfg.name].data.root_pos_w[:, :3]
    return -(ee_pos - cube_pos).norm(dim=-1)


def _place_reward(env, cube_cfg: SceneEntityCfg, command_name: str) -> float:
    cube_pos = env.scene[cube_cfg.name].data.root_pos_w[:, :3]
    target_pos = env.command_manager.get_command(command_name)[:, :3]
    return -(cube_pos - target_pos).norm(dim=-1)


@configclass
class FrankaPickAndPlaceRewardsCfg:
    grasp = RewardTermCfg(
        func=_grasp_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "cube_cfg": SceneEntityCfg("cube"),
        },
    )
    place = RewardTermCfg(
        func=_place_reward,
        weight=2.0,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "command_name": "place_target",
        },
    )
    action_penalty = RewardTermCfg(func=mdp.action_l2, weight=-0.01)


@configclass
class FrankaPickAndPlaceTerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class FrankaPickAndPlaceEventsCfg:
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    reset_cube = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )


@configclass
class FrankaPickAndPlaceCommandsCfg:
    place_target = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(8.0, 8.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.7),
            pos_y=(-0.3, 0.3),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class FrankaPickAndPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: FrankaPickAndPlaceSceneCfg = FrankaPickAndPlaceSceneCfg()
    actions: FrankaPickAndPlaceActionsCfg = FrankaPickAndPlaceActionsCfg()
    observations: FrankaPickAndPlaceObservationsCfg = FrankaPickAndPlaceObservationsCfg()
    rewards: FrankaPickAndPlaceRewardsCfg = FrankaPickAndPlaceRewardsCfg()
    terminations: FrankaPickAndPlaceTerminationsCfg = FrankaPickAndPlaceTerminationsCfg()
    events: FrankaPickAndPlaceEventsCfg = FrankaPickAndPlaceEventsCfg()
    commands: FrankaPickAndPlaceCommandsCfg = FrankaPickAndPlaceCommandsCfg()

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0)
    episode_length_s = 8.0
    decimation = 4

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
