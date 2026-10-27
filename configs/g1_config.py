from pathlib import Path
import mujoco

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.utils.spec_config import CollisionCfg

try:
    from mjlab.utils.os import update_assets
except ImportError:
    def update_assets(
        assets: dict[str, bytes],
        path: str | Path,
        meshdir: str | None = None,
        glob: str = "*",
        recursive: bool = False,
    ):
        for f in Path(path).glob(glob):
            if f.is_file():
                asset_key = f"{meshdir}/{f.name}" if meshdir else f.name
                assets[asset_key] = f.read_bytes()
            elif f.is_dir() and recursive:
                update_assets(assets, f, meshdir, glob, recursive)

G1_XML_PATH = Path(
    "assets/xmls/g1_23dof.xml"
)


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, G1_XML_PATH.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(G1_XML_PATH))
    spec.assets = get_assets(spec.meshdir)
    return spec


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.8),
    joint_pos={
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        ".*_shoulder_pitch_joint": 0.35,
        ".*_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.18,
        "right_shoulder_roll_joint": -0.18,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)


ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_elbow_joint",
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_wrist_roll_joint",
    ),
    stiffness=39.5,
    damping=2.5,
    effort_limit=25.0,
    armature=0.001,
)

ACTUATOR_HIPS = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
    stiffness=102.7,
    damping=6.5,
    effort_limit=88.0,
    armature=0.0026,
)

ACTUATOR_KNEES = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_roll_joint", ".*_knee_joint"),
    stiffness=162.0,
    damping=10.3,
    effort_limit=139.0,
    armature=0.0041,
)

ACTUATOR_ANKLES = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    stiffness=79.0,
    damping=5.0,
    effort_limit=50.0,
    armature=0.002,
)

G1_23DOF_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        ACTUATOR_ARMS,
        ACTUATOR_HIPS,
        ACTUATOR_KNEES,
        ACTUATOR_ANKLES,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_g1_23dof_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=G1_23DOF_ARTICULATION,
    )