"""Franka disassembly environment: pull bottom cap out of fixture.

The bottom cap is held in place by a spring force applied each step.
The robot must overcome this force to extract the cap.
"""

from __future__ import annotations

import os
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
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
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")

# --- Press-fit friction parameters (the virtual "glue" holding the cap) ---
STATIC_STIFFNESS = 0.0     # N/m — force per meter of displacement while cap hasn't moved much. DISABLED
STATIC_THRESHOLD = 0.002   # m — cap won't start sliding until pushed more than this (2mm)
SLIDING_FRICTION = 0.0     # N — constant drag force while cap is sliding out. DISABLED
SLIDING_DAMPING = 0.0      # N·s/m — extra resistance proportional to speed. DISABLED
RELEASE_DISTANCE = 0.08    # m — cap is fully free once pulled this far (8cm)


# ============================================================
# SCENE — what objects exist in the world
# ============================================================
@configclass
class DisassemblySceneCfg(InteractiveSceneCfg):
    num_envs = 1024       # how many copies of the scene run in parallel
    env_spacing = 2.5     # meters between each copy

    # a big overhead light so you can see things
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    # infinite flat floor
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # brown box acting as a table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0]),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # the Franka Panda robot arm — mounted on top of the table
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(pos=[0.0, 0.0, 0.2]),
    )

    # outer shell + mouthpiece — static, bolted to the world, never moves
    fixture = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Fixture",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.25]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "fixture.usd"),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
            ),
        ),
    )

    # jig that holds the fixture — static, doesn't move (adjust pos later)
    jig = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Jig",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.2]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "vape_jig.usd"),
            scale=(0.01, 0.01, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # bottom cap — the part the robot pulls out. has physics (mass, collision, no gravity)
    workpiece: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.054, 0.25]),
        spawn=UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "bottom_cap.usd"),
            scale=(0.0095, 0.01, 0.0095),    # slightly shrunk in X/Z to fit inside fixture
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,          # floats in place — spring holds it, not gravity
                max_depenetration_velocity=1.0, # limits how fast physics pushes overlapping objects apart
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),  # 50 grams
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
            ),
        ),
    )


# ============================================================
# ACTIONS — what the RL agent can control
# ============================================================
@configclass
class DisassemblyActionsCfg:
    # move each of the 7 arm joints by a position offset
    arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,                  # action=1 moves joint by 0.5 rad
        use_default_offset=True,    # action=0 means "stay at default pose"
    )
    # open or close the gripper (binary: 1=open, 0=close)
    gripper_action: ActionTermCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},   # fingers 4cm apart
        close_command_expr={"panda_finger_.*": 0.0},   # fingers closed
    )


# ============================================================
# HELPER FUNCTIONS — read sensor data from the sim
# ============================================================

# returns [x, y, z] world position of the bottom cap
def _workpiece_pos(env: object) -> torch.Tensor:
    return env.scene["workpiece"].data.root_pos_w[:, :3]


# returns how far (meters) the cap has moved from where it started
def _workpiece_displacement(env: object) -> torch.Tensor:
    current = env.scene["workpiece"].data.root_pos_w[:, :3]
    initial = torch.tensor(
        env.scene["workpiece"].cfg.init_state.pos,
        device=current.device,
    )
    return (current - initial).norm(dim=-1, keepdim=True)


# returns negative distance from gripper to cap (closer = higher value = better)
def _ee_to_workpiece_distance(env: object, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_pos = env.scene[robot_cfg.name].data.body_pos_w[:, robot_cfg.body_ids[0], :3]
    wp_pos = env.scene["workpiece"].data.root_pos_w[:, :3]
    return -(ee_pos - wp_pos).norm(dim=-1)


# returns how far the cap has been pulled out (0 = still assembled)
def _extraction_progress(env: object) -> torch.Tensor:
    current = env.scene["workpiece"].data.root_pos_w[:, :3]
    initial = torch.tensor(
        env.scene["workpiece"].cfg.init_state.pos,
        device=current.device,
    )
    return (current - initial).norm(dim=-1).clamp(min=0.0)


# returns True/False: has the cap been pulled out far enough?
def _workpiece_extracted(env: object, threshold: float = 0.05) -> torch.Tensor:
    current = env.scene["workpiece"].data.root_pos_w[:, :3]
    initial = torch.tensor(
        env.scene["workpiece"].cfg.init_state.pos,
        device=current.device,
    )
    return (current - initial).norm(dim=-1) > threshold


# ============================================================
# OBSERVATIONS — what the RL agent sees each step
# ============================================================
@configclass
class DisassemblyObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)       # current joint angles
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)       # current joint speeds
        workpiece_pos = ObservationTermCfg(func=_workpiece_pos)      # where is the cap?
        workpiece_displacement = ObservationTermCfg(func=_workpiece_displacement)  # how far pulled?
        actions = ObservationTermCfg(func=mdp.last_action)           # what did agent do last step?

        def __post_init__(self):
            self.enable_corruption = False   # no noise added to observations
            self.concatenate_terms = True    # flatten everything into one vector

    policy: PolicyCfg = PolicyCfg()


# ============================================================
# REWARDS — what the RL agent gets points for
# ============================================================
@configclass
class DisassemblyRewardsCfg:
    # +points for moving gripper closer to the cap
    reaching_workpiece = RewardTermCfg(
        func=_ee_to_workpiece_distance,
        weight=0.5,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
    )
    # +points for pulling the cap out further
    extraction_progress = RewardTermCfg(
        func=_extraction_progress,
        weight=5.0,
    )
    # big bonus when cap is fully extracted
    extraction_success = RewardTermCfg(
        func=_workpiece_extracted,
        weight=50.0,
        params={"threshold": 0.05},
    )
    # small penalty for large actions (encourages smooth motion)
    action_penalty = RewardTermCfg(func=mdp.action_l2, weight=-0.01)
    # small penalty for fast joint motion (encourages gentle movement)
    joint_vel_penalty = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001)


# ============================================================
# TERMINATIONS — when does an episode end?
# ============================================================
@configclass
class DisassemblyTerminationsCfg:
    # episode ends after time runs out
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # episode ends early if cap is fully extracted (success!)
    success = TerminationTermCfg(
        func=_workpiece_extracted,
        time_out=False,
        params={"threshold": 0.08},
    )


# ============================================================
# EVENTS — what happens at the start of each episode
# ============================================================
@configclass
class DisassemblyEventsCfg:
    # put robot joints back to default pose (1.0 = exact default, no randomness)
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    # put the cap back to its starting position (no randomness)
    reset_workpiece = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("workpiece"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )


# ============================================================
# CUSTOM ENV — adds the spring force that holds the cap in place
# ============================================================
class DisassemblyEnv(ManagerBasedRLEnv):

    def _pre_physics_step(self, actions):
        # actions = torch.zeros_like(actions)  # TEMP: uncomment to freeze robot
        super()._pre_physics_step(actions)
        self._apply_hold_force()  # apply virtual spring every physics step

    def _apply_hold_force(self):
        """Simulates press-fit friction in 3 phases:
        1. STATIC: cap is locked (tiny wiggles get pushed back hard)
        2. SLIDING: cap moves but feels constant drag + speed-dependent damping
        3. RELEASED: cap is free, no more force
        """
        wp = self.scene["workpiece"]
        current_pos = wp.data.root_pos_w[:, :3]       # where cap is now
        current_vel = wp.data.root_lin_vel_w[:, :3]    # how fast cap is moving
        target_pos = torch.tensor(                      # where cap started (assembled position)
            wp.cfg.init_state.pos, device=current_pos.device
        ).expand_as(current_pos)

        displacement = current_pos - target_pos         # vector from start to current
        dist = displacement.norm(dim=-1, keepdim=True)  # scalar distance from start
        direction = displacement / (dist + 1e-8)        # unit vector pointing away from start

        force = torch.zeros_like(displacement)

        # Phase 1: hasn't moved much → push it back hard (like it's stuck)
        static_mask = (dist < STATIC_THRESHOLD).expand_as(force)
        force = torch.where(static_mask, -STATIC_STIFFNESS * displacement, force)

        # Phase 2: sliding out → constant drag + velocity damping
        sliding_mask = ((dist >= STATIC_THRESHOLD) & (dist < RELEASE_DISTANCE)).expand_as(force)
        friction_force = -SLIDING_FRICTION * direction - SLIDING_DAMPING * current_vel
        force = torch.where(sliding_mask, friction_force, force)

        # Phase 3: pulled out far enough → no force, cap is free

        # actually apply the force to the cap in the physics engine
        wp.set_external_force_and_torque(
            forces=force.unsqueeze(1),
            torques=torch.zeros_like(force).unsqueeze(1),
            body_ids=[0],
        )


# ============================================================
# MAIN CONFIG — ties everything together
# ============================================================
@configclass
class DisassemblyEnvCfg(ManagerBasedRLEnvCfg):

    scene: DisassemblySceneCfg = DisassemblySceneCfg()
    actions: DisassemblyActionsCfg = DisassemblyActionsCfg()
    observations: DisassemblyObservationsCfg = DisassemblyObservationsCfg()
    rewards: DisassemblyRewardsCfg = DisassemblyRewardsCfg()
    terminations: DisassemblyTerminationsCfg = DisassemblyTerminationsCfg()
    events: DisassemblyEventsCfg = DisassemblyEventsCfg()

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0)  # physics runs at 120 Hz
    episode_length_s = 10.0   # each training episode lasts 10 seconds
    decimation = 2            # agent acts every 2 physics steps (60 Hz decisions)

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation


# same config but fewer envs + longer episodes for watching a trained policy
@configclass
class DisassemblyEnvCfg_Play(DisassemblyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 15.0
