from dataclasses import dataclass
import torch

@dataclass
class EnvConfigs:

    
    # Simulation Configs
    sim_physics_dt: float = 0.002
    sim_iterations: int = 10
    sim_ls_iterations: int = 20
    sim_decimation: int = 10
    sim_nconmax: int = 100
    sim_njmax: int = 500

    sim_num_envs: int = 4096
    sim_decimation: int = 10
    sim_episode_length_s: float = 20.0

    # Observation_scales
    base_lin_vel: float = 2.0
    base_ang_vel: float = 0.25
    base_z_pos: float = 1.0
    projected_gravity: float = 1.0
    euler_angles: float = 1.0
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    last_action: float = 1.0
    motion_phase: float = 1.0
    motion_current_joint_position: float = 1.0
    current_motion_joint_velocity: float = 0.05
    motion_joint_position_error: float = 1.0
    motion_future_joint_position: float = 1.0
    future_states_horizon: int = 10
    history_length: int = 5


    # Sigmas
    sigma_joint_pos: float = 0.18
    sigma_body_pos: float = 0.20
    sigma_body_ori: float = 0.30
    sigma_lin_vel: float = 0.5
    sigma_ang_vel: float = 1.0
    sigma_root_pos: float = 0.5
    sigma_root_ori: float = 0.5
    sigma_transition: float = 0.5

    # Motion Tracking Link & DOF Weights (Genesis-style targeted weighting)
    # Body link indices: pelvis(0), left_knee(4), left_ankle(6), right_knee(10), right_ankle(12), torso(13), left_wrist(18), right_wrist(23)
    tracking_body_indices: tuple[int, ...] = (0, 4, 6, 10, 12, 13, 18, 23)
    # Body Position Tracking Weights: high weights on feet (2.0) and knees (1.5)
    tracking_body_pos_weights: tuple[float, ...] = (1.0, 1.5, 2.0, 1.5, 2.0, 1.0, 1.0, 1.0)
    # Body Orientation Tracking Weights: ankle orientation set to 0.0 (like Genesis) to avoid fighting flat foot ground penalty
    tracking_body_ori_weights: tuple[float, ...] = (1.0, 0.5, 0.0, 0.5, 0.0, 2.0, 1.0, 1.0)
    # Joint DOF tracking weights (23 DOF): prioritize hips and knees over arms
    tracking_dof_weights: tuple[float, ...] = (
        1.5, 1.2, 1.2, 2.0, 1.5, 1.2,  # Left leg: hip_pitch, roll, yaw, knee, ankle_pitch, roll
        1.5, 1.2, 1.2, 2.0, 1.5, 1.2,  # Right leg: hip_pitch, roll, yaw, knee, ankle_pitch, roll
        1.0,                            # Waist: waist_yaw
        0.8, 0.8, 0.8, 0.8, 0.8,        # Left arm: shoulder_pitch, roll, yaw, elbow, wrist_roll
        0.8, 0.8, 0.8, 0.8, 0.8,        # Right arm: shoulder_pitch, roll, yaw, elbow, wrist_roll
    )

    # Rewards
    orientation_upright_reward: float = 1.0
    linear_vel_z_reward: float = 0.75
    base_height_reward: float = 1.5
    target_height: float = 0.75
    stand_still_reward: float = 1.0
    
    alive_reward: float = 0.2
    motion_joint_position_tracking_reward: float = 1.5
    motion_body_position_tracking_reward: float = 1.5
    motion_body_orientation_tracking_reward: float = 1.0
    motion_linear_velocity_tracking_reward: float = 1.0
    motion_angular_velocity_tracking_reward: float = 0.8
    motion_root_position_tracking_reward: float = 0.5
    motion_root_orientation_tracking_reward: float = 0.5
    lead_in_lead_out_joint_pos_reward: float = 1.5

    # Regularization Penalties (Negative)
    joint_torque_reward: float = -1.0e-5
    joint_vel_reward: float = -1.0e-4
    joint_acc_reward: float = -2.5e-6
    action_rate_reward: float = -0.005
    action_acc_reward: float = -0.005
    joint_pos_limits_reward: float = -1.0
    collision_reward: float = -1.0
    foot_sliding_during_contact_penalty: float = -0.1
    foot_orientation_penalty: float = -0.5

    # Velocity Locomotion Command Ranges (Twist)
    lin_vel_x_range: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z_range: tuple[float, float] = (-0.6, 0.6)
    rel_standing_envs: float = 0.2
    vel_resampling_time_range: tuple[float, float] = (3.0, 6.0)

    # Motion Tracking
    settle_time_range: tuple[float, float] = (2.0, 4.0)
    rsi_probability: float = 0.5
    
    # Velocity Tracking Reward Weights (Exponential)
    track_vx_reward: float = 2.5
    track_vy_reward: float = 1.5
    track_wz_reward: float = 1.5

    # Locomotion & Gait Rewards (Exponential form)
    feet_air_time_reward: float = 1.0
    hip_deviation_reward: float = 0.5
    upper_body_pose_reward: float = 0.5
    contact_reward: float = 1.0
    feet_swing_height_reward: float = 1.0

    # Locomotion Exponential Sigmas
    sigma_vx: float = 0.25
    sigma_vy: float = 0.25
    sigma_wz: float = 0.5
    sigma_hip_deviation: float = 0.15
    sigma_upper_body_pose: float = 0.2
    sigma_feet_swing_height: float = 0.05
    sigma_upright: float = 0.2
    sigma_height: float = 0.1
    sigma_vel_z: float = 0.2
    sigma_stand: float = 0.3

    # Locomotion & Gait Parameters
    feet_air_time_threshold_min: float = 0.05
    feet_air_time_threshold_max: float = 0.5
    hip_yaw_indices: tuple[int, ...] = (2, 8)
    upper_body_indices: tuple[int, ...] = (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
    gait_period: float = 0.8
    gait_offset: float = 0.5
    stance_ratio: float = 0.55
    target_foot_height: float = 0.08
    default_ankle_height: float = 0.03

    termination_weight: float = -200.0

    # actions
    action_scale: float = 0.25

    # Default Joints (G1 23-DOF standing posture from g1_config.py HOME_KEYFRAME)
    default_joints: torch.Tensor = torch.tensor([
        -0.1,  0.0,   0.0, 0.3, -0.2, 0.0,   # Left leg:  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        -0.1,  0.0,   0.0, 0.3, -0.2, 0.0,   # Right leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
         0.0,                                # Waist:     waist_yaw
         0.35, 0.18,  0.0, 0.87, 0.0,        # Left arm:  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
         0.35, -0.18, 0.0, 0.87, 0.0,        # Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll
    ], dtype=torch.float32)
