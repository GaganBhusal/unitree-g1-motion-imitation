"""G1 Humanoid (23-DOF) Joint Configuration & Reference Reference Guide.

Designed for easy viewing and side-by-side debugging with Genesis
(matching /home/yayy/My/Codeeeeee/RL_Robotics/G1_Humanoid/Genesis/configs/robot_config.py).

Run directly in terminal to print a complete reference table:
    python configs/robot_config.py
"""

import math
from dataclasses import dataclass
import torch



G1_JOINT_NAMES = [
    # --- Left Leg (0-5) ---
    "left_hip_pitch_joint",         #  0
    "left_hip_roll_joint",          #  1
    "left_hip_yaw_joint",           #  2
    "left_knee_joint",              #  3
    "left_ankle_pitch_joint",       #  4
    "left_ankle_roll_joint",        #  5
    # --- Right Leg (6-11) ---
    "right_hip_pitch_joint",        #  6
    "right_hip_roll_joint",         #  7
    "right_hip_yaw_joint",          #  8
    "right_knee_joint",             #  9
    "right_ankle_pitch_joint",      # 10
    "right_ankle_roll_joint",       # 11
    # --- Torso (12) ---
    "waist_yaw_joint",              # 12
    # --- Left Arm (13-17) ---
    "left_shoulder_pitch_joint",    # 13
    "left_shoulder_roll_joint",     # 14
    "left_shoulder_yaw_joint",      # 15
    "left_elbow_joint",             # 16
    "left_wrist_roll_joint",        # 17
    # --- Right Arm (18-22) ---
    "right_shoulder_pitch_joint",   # 18
    "right_shoulder_roll_joint",    # 19
    "right_shoulder_yaw_joint",     # 20
    "right_elbow_joint",            # 21
    "right_wrist_roll_joint",       # 22
]

G1_INITIAL_POSITIONS_RAD = [
    -0.10,   #  0: left_hip_pitch       (-5.73 deg)
     0.00,   #  1: left_hip_roll        ( 0.00 deg)
     0.00,   #  2: left_hip_yaw         ( 0.00 deg)
     0.30,   #  3: left_knee            (17.19 deg)
    -0.20,   #  4: left_ankle_pitch     (-11.46 deg)
     0.00,   #  5: left_ankle_roll      ( 0.00 deg)
    -0.10,   #  6: right_hip_pitch      (-5.73 deg)
     0.00,   #  7: right_hip_roll       ( 0.00 deg)
     0.00,   #  8: right_hip_yaw        ( 0.00 deg)
     0.30,   #  9: right_knee           (17.19 deg)
    -0.20,   # 10: right_ankle_pitch    (-11.46 deg)
     0.00,   # 11: right_ankle_roll     ( 0.00 deg)
     0.00,   # 12: waist_yaw            ( 0.00 deg)
     0.35,   # 13: left_shoulder_pitch  (20.05 deg)
     0.18,   # 14: left_shoulder_roll   (10.31 deg)
     0.00,   # 15: left_shoulder_yaw    ( 0.00 deg)
     0.87,   # 16: left_elbow           (49.85 deg)
     0.00,   # 17: left_wrist_roll      ( 0.00 deg)
     0.35,   # 18: right_shoulder_pitch (20.05 deg)
    -0.18,   # 19: right_shoulder_roll  (-10.31 deg)
     0.00,   # 20: right_shoulder_yaw   ( 0.00 deg)
     0.87,   # 21: right_elbow          (49.85 deg)
     0.00,   # 22: right_wrist_roll     ( 0.00 deg)
]

G1_JOINT_LIMITS_RAD = [
    [-2.5307,   2.8798],   #  0: left_hip_pitch       [-145.0°,  165.0°]
    [-0.5236,   2.9671],   #  1: left_hip_roll        [ -30.0°,  170.0°]
    [-2.7576,   2.7576],   #  2: left_hip_yaw         [-158.0°,  158.0°]
    [-0.0873,   2.8798],   #  3: left_knee            [  -5.0°,  165.0°]
    [-0.8727,   0.5236],   #  4: left_ankle_pitch     [ -50.0°,   30.0°]
    [-0.2618,   0.2618],   #  5: left_ankle_roll      [ -15.0°,   15.0°]
    [-2.5307,   2.8798],   #  6: right_hip_pitch      [-145.0°,  165.0°]
    [-2.9671,   0.5236],   #  7: right_hip_roll       [-170.0°,   30.0°]
    [-2.7576,   2.7576],   #  8: right_hip_yaw        [-158.0°,  158.0°]
    [-0.0873,   2.8798],   #  9: right_knee           [  -5.0°,  165.0°]
    [-0.8727,   0.5236],   # 10: right_ankle_pitch    [ -50.0°,   30.0°]
    [-0.2618,   0.2618],   # 11: right_ankle_roll     [ -15.0°,   15.0°]
    [-2.6180,   2.6180],   # 12: waist_yaw            [-150.0°,  150.0°]
    [-3.0892,   2.6704],   # 13: left_shoulder_pitch  [-177.0°,  153.0°]
    [-1.5882,   2.2515],   # 14: left_shoulder_roll   [ -91.0°,  129.0°]
    [-2.6180,   2.6180],   # 15: left_shoulder_yaw    [-150.0°,  150.0°]
    [-1.0472,   2.0944],   # 16: left_elbow           [ -60.0°,  120.0°]
    [-1.9722,   1.9722],   # 17: left_wrist_roll      [-113.0°,  113.0°]
    [-3.0892,   2.6704],   # 18: right_shoulder_pitch [-177.0°,  153.0°]
    [-2.2515,   1.5882],   # 19: right_shoulder_roll  [-129.0°,   91.0°]
    [-2.6180,   2.6180],   # 20: right_shoulder_yaw   [-150.0°,  150.0°]
    [-1.0472,   2.0944],   # 21: right_elbow          [ -60.0°,  120.0°]
    [-1.9722,   1.9722],   # 22: right_wrist_roll     [-113.0°,  113.0°]
]

GENESIS_INITIAL_POSITIONS_DEG = [
    -10.0,  # 0: left_hip_pitch
    -10.0,  # 1: right_hip_pitch
      0.0,  # 2: waist_yaw
      0.0,  # 3: left_hip_roll
      0.0,  # 4: right_hip_roll
      0.0,  # 5: left_shoulder_pitch
      0.0,  # 6: right_shoulder_pitch
      0.0,  # 7: left_hip_yaw
      0.0,  # 8: right_hip_yaw
     12.0,  # 9: left_shoulder_roll
    -12.0,  # 10: right_shoulder_roll
     15.0,  # 11: left_knee
     15.0,  # 12: right_knee
      0.0,  # 13: left_shoulder_yaw
      0.0,  # 14: right_shoulder_yaw
    -10.0,  # 15: left_ankle_pitch
    -10.0,  # 16: right_ankle_pitch
     90.0,  # 17: left_elbow
     90.0,  # 18: right_elbow
      0.0,  # 19: left_ankle_roll
      0.0,  # 20: right_ankle_roll
      0.0,  # 21: left_wrist_roll
      0.0   # 22: right_wrist_roll
]


class RobotCfg:

    TOTAL_ACTIONS: int = 23
    BASE: str = "pelvis"
    ACTION_SCALE: float = 0.25

    JOINT_NAMES = G1_JOINT_NAMES
    initial_positions_list = G1_INITIAL_POSITIONS_RAD
    joint_limits_list = G1_JOINT_LIMITS_RAD

    # Torch tensors (if torch is installed)
    initial_positions: torch.Tensor = torch.tensor(G1_INITIAL_POSITIONS_RAD, dtype=torch.float32)
    URDF_joint_limits: torch.Tensor = torch.tensor(G1_JOINT_LIMITS_RAD, dtype=torch.float32)

    # Joint Groupings (MuJoCo Order)
    LEFT_LEG: list[int] = [0, 1, 2, 3, 4, 5]
    RIGHT_LEG: list[int] = [6, 7, 8, 9, 10, 11]
    WAIST: list[int] = [12]
    LEFT_ARM: list[int] = [13, 14, 15, 16, 17]
    RIGHT_ARM: list[int] = [18, 19, 20, 21, 22]

    ALL_LEGS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ALL_ARMS: list[int] = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    UPPER_BODY: list[int] = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    LOWER_BODY: list[int] = ALL_LEGS

    HIPS: list[int] = [0, 1, 2, 6, 7, 8]
    KNEES: list[int] = [3, 9]
    ANKLES: list[int] = [4, 5, 10, 11]
    SHOULDERS: list[int] = [13, 14, 15, 18, 19, 20]
    ELBOWS: list[int] = [16, 21]
    WRISTS: list[int] = [17, 22]

    FEET_LINKS: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]

    # All 24 Body Links (Matching MuJoCo and Genesis)
    BODY_LINK_NAMES = [
        "pelvis",                         #  0 (root link)
        "left_hip_pitch_link",            #  1
        "left_hip_roll_link",             #  2
        "left_hip_yaw_link",              #  3
        "left_knee_link",                 #  4
        "left_ankle_pitch_link",          #  5
        "left_ankle_roll_link",           #  6
        "right_hip_pitch_link",           #  7
        "right_hip_roll_link",            #  8
        "right_hip_yaw_link",             #  9
        "right_knee_link",                # 10
        "right_ankle_pitch_link",         # 11
        "right_ankle_roll_link",          # 12
        "torso_link",                     # 13
        "left_shoulder_pitch_link",       # 14
        "left_shoulder_roll_link",        # 15
        "left_shoulder_yaw_link",         # 16
        "left_elbow_link",                # 17
        "left_wrist_roll_rubber_hand",    # 18
        "right_shoulder_pitch_link",      # 19
        "right_shoulder_roll_link",       # 20
        "right_shoulder_yaw_link",        # 21
        "right_elbow_link",               # 22
        "right_wrist_roll_rubber_hand"    # 23
    ]

    GENESIS_TO_MUJOCO = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]

    MUJOCO_TO_GENESIS = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]


def print_g1_joint_table():
    cfg = RobotCfg()

    print("=" * 110)
    print(" " * 32 + "G1 Humanoid (23-DOF) Joint Configuration")
    print(" " * 27 + "MuJoCo (Mjlab) vs. Genesis Side-by-Side Debug Table")
    print("=" * 110)
    print(f"{'Idx':>3} | {'MuJoCo Joint Name':<28} | {'Gen Idx':>7} | {'Init (deg)':>10} | {'Init (rad)':>10} | {'Limits (deg)':<22} | {'Limits (rad)':<16}")
    print("-" * 110)

    for i in range(cfg.TOTAL_ACTIONS):
        name = cfg.JOINT_NAMES[i]
        gen_idx = cfg.MUJOCO_TO_GENESIS[i]
        init_rad = cfg.initial_positions_list[i]
        init_deg = math.degrees(init_rad)
        min_rad, max_rad = cfg.joint_limits_list[i]
        min_deg, max_deg = math.degrees(min_rad), math.degrees(max_rad)

        limits_deg_str = f"[{min_deg:6.1f}°, {max_deg:6.1f}°]"
        limits_rad_str = f"[{min_rad:6.2f}, {max_rad:6.2f}]"

        print(f"{i:3d} | {name:<28} | {gen_idx:7d} | {init_deg:9.2f}° | {init_rad:10.3f} | {limits_deg_str:<22} | {limits_rad_str:<16}")

    print("=" * 110)
    print("\nJoint Groupings (MuJoCo Indices):")
    print(f"  LEFT_LEG   (6): {cfg.LEFT_LEG}")
    print(f"  RIGHT_LEG  (6): {cfg.RIGHT_LEG}")
    print(f"  WAIST      (1): {cfg.WAIST}")
    print(f"  LEFT_ARM   (5): {cfg.LEFT_ARM}")
    print(f"  RIGHT_ARM  (5): {cfg.RIGHT_ARM}")
    print(f"  HIPS       (6): {cfg.HIPS}")
    print(f"  KNEES      (2): {cfg.KNEES}")
    print(f"  ANKLES     (4): {cfg.ANKLES}")
    print(f"  SHOULDERS  (6): {cfg.SHOULDERS}")
    print(f"  ELBOWS     (2): {cfg.ELBOWS}")
    print(f"  WRISTS     (2): {cfg.WRISTS}")
    print("\nBody Links (24 Links):")
    for idx, link in enumerate(cfg.BODY_LINK_NAMES):
        print(f"  {idx:2d}: {link}")
    print("=" * 110)


if __name__ == "__main__":
    print_g1_joint_table()
