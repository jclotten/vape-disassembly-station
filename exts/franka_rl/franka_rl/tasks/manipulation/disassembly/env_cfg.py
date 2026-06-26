"""Franka disassembly environment: pull bottom cap out of fixture.

The bottom cap is held in place by a spring force applied each step.
The robot must overcome this force to extract the cap.
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from isaaclab_physx.physics import PhysxCfg

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
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets")

# --- Press-fit parameters for disc on cylinder ---
PRESS_FIT_HEIGHT = 0.005       # m — disc must be lifted 0.5cm before it's free
LATERAL_STIFFNESS = 500.0      # N/m — strong spring preventing sideways movement during press-fit
VERTICAL_RESISTANCE = 10.0     # N — constant upward resistance while in press-fit zone
VERTICAL_DAMPING = 0.5         # N·s/m — velocity-dependent resistance during extraction
LID_CONTACT_FORCE_THRESHOLD = 0.05
LID_CONTACT_FORCE_NORMALIZATION = 50.0

# --- Front camera (Logitech C920) placement ---
# Adjust these to match the real gooseneck mount position
FRONT_CAM_POS = (1.2, 0.0, 0.5)
FRONT_CAM_ROT = (-0.4755, -0.5277, -0.5148, -0.4801)  # (x,y,z,w) from viewport Euler (84, -265, 5)

# --- Wrist camera (RealSense D405) placement ---
# Offset from panda_hand frame based on cammount CAD (~13cm along mount arm)
# OffsetCfg.rot uses (x, y, z, w) quaternion order!
WRIST_CAM_POS = (-0.11, -0.01, 0.03)
WRIST_CAM_ROT = (-0.6455, -0.6738, 0.2681, 0.2397)

# --- Finger-mounted Tip30 tooling ---
# Tip30.obj is authored in the same CAD units as the vape parts. Scale 0.01 makes it ~2 cm long.
# The OBJ's working point extends mostly along local -Z, so the rotations point it outward
# along the Panda finger's local +Z fingertip direction. They also roll Tip30 so the slide
# faces, not the undersides, point inward toward the opposing finger.
TIP30_FINGER_POS = (0.0, 0.0, 0.046)
TIP30_LEFT_FINGER_ROT = (-0.5, -0.5, -0.5, 0.5)
TIP30_RIGHT_FINGER_ROT = (-0.5, 0.5, -0.5, -0.5)
TIP30_SCALE = (0.01, 0.01, 0.01)

# The dynamic cap is spawned clear of the fixture, then written into this assembled
# pose after PhysX has initialized the static fixture collision.
WORKPIECE_ASSEMBLED_POS = (0.62, -0.02948, 0.5245)
WORKPIECE_STAGING_POS = (
    WORKPIECE_ASSEMBLED_POS[0],
    WORKPIECE_ASSEMBLED_POS[1],
    WORKPIECE_ASSEMBLED_POS[2] + 0.12,
)


def _resnet18_feature_model_zoo_cfg() -> dict:
    """Frozen ResNet18 encoder that returns 512D pooled features instead of classifier logits."""

    def _load_model() -> torch.nn.Module:
        from torchvision import models

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        return model.eval()

    def _inference(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
        model = model.to(images.device)
        image_proc = images.permute(0, 3, 1, 2).float() / 255.0
        image_proc = F.interpolate(image_proc, size=(128, 128), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        image_proc = (image_proc - mean) / std

        with torch.inference_mode():
            return model(image_proc)

    return {
        "resnet18_pool512": {
            "model": _load_model,
            "inference": _inference,
        }
    }


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

    # brown box acting as a table — pos is at center, so Z=0.2 puts bottom on ground
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.2]),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.4),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.58, 0.60),
                metallic=0.8,
                roughness=0.28,
            ),
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(pos=[0.3, 0.0, 0.4]),
    )
    # Tip30 is mounted geometry under the Franka finger links; the contact sensors
    # must be activated on the articulation rigid bodies, not the Tip30 Xforms.
    robot.spawn.activate_contact_sensors = True

    left_tip30 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger/Tip30",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TIP30_FINGER_POS, rot=TIP30_LEFT_FINGER_ROT),
        spawn=UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "tip30.usd"),
            scale=TIP30_SCALE,
            collision_props=None,
            physics_material=RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
        ),
    )

    right_tip30 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger/Tip30",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TIP30_FINGER_POS, rot=TIP30_RIGHT_FINGER_ROT),
        spawn=UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "tip30.usd"),
            scale=TIP30_SCALE,
            collision_props=None,
            physics_material=RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
        ),
    )

    # # outer shell + mouthpiece — static, bolted to the world, never moves
    # fixture = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Fixture",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.62, -0.0775, 0.43674]),
    #     spawn=UsdFileCfg(
    #         usd_path=os.path.join(ASSETS_DIR, "fixture.usd"),
    #         scale=(0.95, 0.95, 1.0),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.48, 0.12)),
    #         collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
    #         physics_material=RigidBodyMaterialCfg(
    #             static_friction=0.8,
    #             dynamic_friction=0.6,
    #         ),
    #     ),
    # )

    wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCamera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.5,
            horizontal_aperture=4.5,
            clipping_range=(0.07, 0.5),
        ),
        offset=CameraCfg.OffsetCfg(pos=WRIST_CAM_POS, rot=WRIST_CAM_ROT, convention="opengl"),
        width=84, height=84, update_period=1.0/30.0, data_types=["rgb", "depth"],
    )
    front_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=3.0,
            horizontal_aperture=4.3,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=FRONT_CAM_POS, rot=FRONT_CAM_ROT, convention="world"),
        width=84, height=84, update_period=1.0/30.0, data_types=["rgb"],
    )

    # jig = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Jig",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.4552]),
    #     spawn=UsdFileCfg(
    #         usd_path=os.path.join(ASSETS_DIR, "vape_jig.usd"),
    #         scale=(0.01, 0.01, 0.01),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    #         collision_props=None,
    #     ),
    # )

    # bottom cap — the part the robot pulls out. Spawn it clear of the fixture;
    # DisassemblyEnv writes it into WORKPIECE_ASSEMBLED_POS after fixture setup.
    workpiece = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workpiece",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.45)),
        spawn=sim_utils.CylinderCfg(
            radius=0.01,   # 2cm diameter
            height=0.10,   # 10cm tall
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.48, 0.12)),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
            ),
        ),
    )

    disc: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Disc",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.51)),
        spawn=sim_utils.CylinderCfg(
            radius=0.01,    # same radius as the tall cylinder
            height=0.005,   # 0.5cm tall disc
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
            ),
        ),
    )

    left_tip30_lid_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Workpiece"],
    )

    right_tip30_lid_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Workpiece"],
    )


# ============================================================
# ACTIONS — what the RL agent can control
# ============================================================
@configclass
class DisassemblyActionsCfg:
    arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action: ActionTermCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


# ============================================================
# HELPER FUNCTIONS — read sensor data from the sim
# ============================================================

def _front_camera_rgb(env: object) -> torch.Tensor:
    """Front camera RGB image, normalized to [0, 1]. Shape: [N, H, W, 3]."""
    return env.scene["front_camera"].data.output["rgb"].float() / 255.0


def _wrist_camera_rgb(env: object) -> torch.Tensor:
    """Wrist camera RGB image, normalized to [0, 1]. Shape: [N, H, W, 3]."""
    return env.scene["wrist_camera"].data.output["rgb"].float() / 255.0


def _wrist_camera_depth(env: object) -> torch.Tensor:
    """Wrist camera depth image in meters. Shape: [N, H, W, 1]."""
    return env.scene["wrist_camera"].data.output["depth"]


# returns [x, y, z] world position of the disc (cap)
def _workpiece_pos(env: object) -> torch.Tensor:
    return env.scene["disc"].data.root_pos_w.torch[:, :3]


def _disc_initial_pos_w(env: object, current: torch.Tensor) -> torch.Tensor:
    return env.scene["disc"].data.default_root_pose.torch[:, :3].to(device=current.device, dtype=current.dtype) + \
        env.scene.env_origins.to(device=current.device, dtype=current.dtype)


# returns how far (meters) the disc has been lifted vertically
def _workpiece_displacement(env: object) -> torch.Tensor:
    current = env.scene["disc"].data.root_pos_w.torch[:, :3]
    initial = _disc_initial_pos_w(env, current)
    return (current[:, 2:3] - initial[:, 2:3]).clamp(min=0.0)


def _tip30_mount_positions_w(env: object, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    finger_pos = robot.data.body_pos_w.torch[:, robot_cfg.body_ids, :3]
    finger_quat = robot.data.body_quat_w.torch[:, robot_cfg.body_ids, :4]
    local_tip30_mount = torch.tensor(TIP30_FINGER_POS, device=finger_pos.device, dtype=finger_pos.dtype)
    local_tip30_mount = local_tip30_mount.expand(finger_pos.shape[0], finger_pos.shape[1], 3)
    return finger_pos + quat_apply(finger_quat.reshape(-1, 4), local_tip30_mount.reshape(-1, 3)).reshape_as(finger_pos)


# returns negative distance from the two mounted Tip30 tools to the cap
def _tip30_to_workpiece_distance(env: object, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    tip30_pos = _tip30_mount_positions_w(env, robot_cfg)
    tip30_midpoint = tip30_pos.mean(dim=1)
    wp_pos = env.scene["disc"].data.root_pos_w.torch[:, :3]
    return -(tip30_midpoint - wp_pos).norm(dim=-1)


# returns how far the cap has been pulled out (0 = still assembled)
def _extraction_progress(env: object) -> torch.Tensor:
    return _workpiece_displacement(env).squeeze(-1).clamp(max=PRESS_FIT_HEIGHT)


# returns True/False: has the cap been pulled out far enough?
def _workpiece_extracted(env: object, threshold: float = 0.05) -> torch.Tensor:
    return _workpiece_displacement(env).squeeze(-1) > threshold


def _sensor_lid_contact_force_norm(env: object, sensor_name: str) -> torch.Tensor:
    force_matrix_data = env.scene[sensor_name].data.force_matrix_w
    if force_matrix_data is None:
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    force_matrix = force_matrix_data.torch
    return force_matrix.norm(dim=-1).flatten(start_dim=1).amax(dim=1)


def _sensor_contacts_lid(env: object, sensor_name: str) -> torch.Tensor:
    return _sensor_lid_contact_force_norm(env, sensor_name) > LID_CONTACT_FORCE_THRESHOLD


def _both_tip30_contacts_lid(env: object) -> torch.Tensor:
    left_contact = _sensor_contacts_lid(env, "left_tip30_lid_contact")
    right_contact = _sensor_contacts_lid(env, "right_tip30_lid_contact")
    return left_contact & right_contact


def _tip30_lid_contact_flags(env: object) -> torch.Tensor:
    left_contact = _sensor_contacts_lid(env, "left_tip30_lid_contact")
    right_contact = _sensor_contacts_lid(env, "right_tip30_lid_contact")
    return torch.stack((left_contact, right_contact), dim=-1).float()


def _tip30_lid_contact_forces(env: object) -> torch.Tensor:
    left_force = _sensor_lid_contact_force_norm(env, "left_tip30_lid_contact")
    right_force = _sensor_lid_contact_force_norm(env, "right_tip30_lid_contact")
    forces = torch.stack((left_force, right_force), dim=-1)
    return forces.clamp(max=LID_CONTACT_FORCE_NORMALIZATION) / LID_CONTACT_FORCE_NORMALIZATION


def _pre_extraction_two_tip_contact_reward(env: object) -> torch.Tensor:
    pre_pull_scale = 1.0 - (_extraction_progress(env) / PRESS_FIT_HEIGHT).clamp(max=1.0)
    return _both_tip30_contacts_lid(env).float() * pre_pull_scale


def _contact_gated_extraction_progress(env: object) -> torch.Tensor:
    return _extraction_progress(env) * _both_tip30_contacts_lid(env).float()


def _contact_gated_workpiece_extracted(env: object, threshold: float = 0.05) -> torch.Tensor:
    return _workpiece_extracted(env, threshold).float() * _both_tip30_contacts_lid(env).float()


# ============================================================
# OBSERVATIONS — what the RL agent sees each step
# ============================================================
@configclass
class DisassemblyObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel)
        workpiece_pos = ObservationTermCfg(func=_workpiece_pos)      # where is the cap?
        workpiece_displacement = ObservationTermCfg(func=_workpiece_displacement)  # how far pulled?
        tip30_lid_contact_flags = ObservationTermCfg(func=_tip30_lid_contact_flags)
        tip30_lid_contact_forces = ObservationTermCfg(func=_tip30_lid_contact_forces)
        wrist_camera_features = ObservationTermCfg(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "model_zoo_cfg": _resnet18_feature_model_zoo_cfg(),
                "model_name": "resnet18_pool512",
            },
        )
        front_camera_features = ObservationTermCfg(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "model_zoo_cfg": _resnet18_feature_model_zoo_cfg(),
                "model_name": "resnet18_pool512",
            },
        )
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
    # +points for moving the mounted Tip30 tools closer to the cap
    reaching_workpiece = RewardTermCfg(
        func=_tip30_to_workpiece_distance,
        weight=0.5,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=["panda_leftfinger", "panda_rightfinger"])},
    )
    # small dense reward for making the physically correct two-sided lid contact
    two_tip_contacts = RewardTermCfg(
        func=_pre_extraction_two_tip_contact_reward,
        weight=1.0,
    )
    # +points for pulling the cap out further
    extraction_progress = RewardTermCfg(
        func=_contact_gated_extraction_progress,
        weight=5.0,
    )
    # big bonus when cap is fully extracted
    extraction_success = RewardTermCfg(
        func=_contact_gated_workpiece_extracted,
        weight=200.0,
        params={"threshold": PRESS_FIT_HEIGHT},
    )
    action_penalty = RewardTermCfg(func=mdp.action_l2, weight=-0.01)
    action_rate_penalty = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02)
    joint_vel_penalty = RewardTermCfg(func=mdp.joint_vel_l2, weight=-0.001)


# ============================================================
# TERMINATIONS — when does an episode end?
# ===========================================================
@configclass
class DisassemblyTerminationsCfg:
    # episode ends after time runs out
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # episode ends early if cap is fully extracted (success!)
    success = TerminationTermCfg(
        func=_workpiece_extracted,
        time_out=False,
        params={"threshold": PRESS_FIT_HEIGHT},
    )


# ============================================================
# EVENTS — what happens at the start of each episode
# ============================================================
@configclass
class DisassemblyEventsCfg:
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    # put the disc back to its starting position (no randomness)
    reset_disc = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("disc"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {},
        },
    )


# ============================================================
# CUSTOM ENV — adds the spring force that holds the cap in place
# ============================================================
class DisassemblyEnv(ManagerBasedRLEnv):

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._disable_stock_finger_collisions()
        # self._place_workpiece_in_fixture_after_setup()
        # self._disable_fixture_workpiece_collision()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # self._place_workpiece_in_fixture_after_setup(env_ids)

    def _place_workpiece_in_fixture_after_setup(self, env_ids=None):
        """Insert the dynamic cap only after the static fixture collision exists."""
        wp = self.scene["workpiece"]

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.int32, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.int32)

        if env_ids.numel() == 0:
            return

        env_ids_long = env_ids.to(dtype=torch.long)
        root_pose = wp.data.default_root_pose.torch[env_ids_long].clone()
        root_vel = torch.zeros_like(wp.data.default_root_vel.torch[env_ids_long])
        assembled_pos = torch.tensor(WORKPIECE_ASSEMBLED_POS, device=root_pose.device, dtype=root_pose.dtype)
        env_origins = self.scene.env_origins[env_ids_long].to(device=root_pose.device, dtype=root_pose.dtype)
        root_pose[:, :3] = env_origins + assembled_pos

        wp.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        wp.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.sim.forward()
        self.scene.update(dt=0.0)

    def _disable_stock_finger_collisions(self):
        """Leave only the Tip30 colliders active on the two moving finger bodies."""
        from pxr import Usd, UsdPhysics

        disabled_count = 0
        for env_path in self.scene.env_prim_paths:
            for finger_name in ("panda_leftfinger", "panda_rightfinger"):
                finger_path = f"{env_path}/Robot/{finger_name}"
                finger_prim = self.scene.stage.GetPrimAtPath(finger_path)
                if not finger_prim.IsValid():
                    continue

                tip30_path = f"{finger_path}/Tip30"
                for prim in Usd.PrimRange(finger_prim):
                    prim_path = str(prim.GetPath())
                    if prim_path.startswith(tip30_path):
                        continue
                    collision_api = UsdPhysics.CollisionAPI(prim)
                    if not collision_api:
                        continue
                    collision_api.GetCollisionEnabledAttr().Set(False)
                    disabled_count += 1

        if disabled_count > 0:
            self.sim.forward()

    def _disable_fixture_workpiece_collision(self):
        """Tell PhysX: fixture and workpiece don't collide with each other."""
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        # Add filtered collision pair on the physics scene
        physics_scene = stage.GetPrimAtPath("/physicsScene")
        if physics_scene.IsValid():
            pair_api = UsdPhysics.FilteredPairsAPI.Apply(physics_scene)
            pair_api.GetFilteredPairsRel().AddTarget("/World/envs/env_0/Fixture")
            pair_api.GetFilteredPairsRel().AddTarget("/World/envs/env_0/Workpiece")

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._apply_hold_force()

    def _apply_hold_force(self):
        """Simulates a press-fit between disc and cylinder.

        While the disc is within PRESS_FIT_HEIGHT (0.5cm) of its start:
          - Strong lateral spring prevents sideways sliding
          - Constant + damping force resists upward pull
        Once lifted past 0.5cm: disc is completely free.
        """
        disc = self.scene["disc"]
        current_pos = disc.data.root_pos_w[:, :3]
        current_vel = disc.data.root_lin_vel_w[:, :3]

        start_pos = disc.data.default_root_pose.torch[:, :3] + self.scene.env_origins
        displacement = current_pos - start_pos
        vertical_lift = displacement[:, 2:3]  # how far above start pos

        in_press_fit = (vertical_lift < PRESS_FIT_HEIGHT) & (vertical_lift >= 0.0)
        in_press_fit_3d = in_press_fit.expand_as(displacement)

        force = torch.zeros_like(displacement)

        # lateral spring: push back any X/Y deviation
        lateral_force = torch.zeros_like(displacement)
        lateral_force[:, 0:2] = -LATERAL_STIFFNESS * displacement[:, 0:2]
        force = torch.where(in_press_fit_3d, lateral_force, force)

        # vertical resistance: oppose upward movement
        vertical_force = torch.zeros_like(displacement)
        vertical_force[:, 2] = -VERTICAL_RESISTANCE - VERTICAL_DAMPING * current_vel[:, 2]
        force[:, 2] = torch.where(in_press_fit.squeeze(), vertical_force[:, 2], force[:, 2])

        disc.set_external_force_and_torque(
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

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 240.0,  # smaller contact timestep for the tight fixture/cap geometry
        physics=PhysxCfg(
            solver_type=1,  # TGS
            solve_articulation_contact_last=True,
            max_position_iteration_count=192,
            max_velocity_iteration_count=8,
            enable_stabilization=True,
            enable_external_forces_every_iteration=True,
            bounce_threshold_velocity=0.01,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**22,
            gpu_collision_stack_size=2**27,
            gpu_max_num_partitions=1,
        ),
    )
    episode_length_s = 10.0   # each training episode lasts 10 seconds
    decimation = 4            # agent still acts at 60 Hz

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.05, -0.82, 0.62)
        self.viewer.lookat = (0.50, -0.02, 0.50)


# same config but fewer envs + longer episodes for watching a trained policy
@configclass
class DisassemblyEnvCfg_Play(DisassemblyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 15.0
