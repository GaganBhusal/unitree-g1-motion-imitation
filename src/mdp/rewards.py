import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_mul, quat_inv, quat_error_magnitude


def track_vx_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "motion_command",
    sigma_vx: float = 0.25,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vx = asset.data.root_link_lin_vel_b[:, 0]
    reward = torch.exp(-torch.square(vx) / (sigma_vx ** 2))
    cmd = env.command_manager.get_term(command_name)
    return reward * (1.0 - cmd.trigger.squeeze(-1))

def track_vy_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "motion_command",
    sigma_vy: float = 0.25,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vy = asset.data.root_link_lin_vel_b[:, 1]
    reward = torch.exp(-torch.square(vy) / (sigma_vy ** 2))
    cmd = env.command_manager.get_term(command_name)
    return reward * (1.0 - cmd.trigger.squeeze(-1))

def track_wz_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "motion_command",
    sigma_wz: float = 0.25,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    wz = asset.data.root_link_ang_vel_b[:, 2]
    reward = torch.exp(-torch.square(wz) / (sigma_wz ** 2))
    cmd = env.command_manager.get_term(command_name)
    return reward * (1.0 - cmd.trigger.squeeze(-1))

def orientation_upright_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "motion_command",
    sigma_upright: float = 0.2,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward = torch.exp(-error / (sigma_upright ** 2))
    trigger = cmd.trigger.squeeze(-1)
    return reward * (1.0 - trigger)
# def height_reward(
#     env,
#     target_height,
#     asset_cfg,
#     command_name="motion_command"
# ):
#     asset = env.scene[asset_cfg.name]
#     cmd = env.command_manager.get_term(command_name)
#     base_height = asset.data.root_link_pos_w[:, 2]
#     reward = torch.square(base_height - target_height)
#     trigger = cmd.trigger.squeeze(-1)
#     return reward * (1.0 - trigger)

# def linear_vel_z_reward(
#     env,
#     asset_cfg,
#     command_name="motion_command",
# ):
#     asset = env.scene[asset_cfg.name]
#     cmd = env.command_manager.get_term(command_name)
#     base_lin_vel_body = asset.data.root_link_lin_vel_b
#     reward = torch.square(base_lin_vel_body[:, 2])
#     trigger = cmd.trigger.squeeze(-1)
#     return reward * (1.0 - trigger)

def roll_pitch_velocity_penalty(
    env,
    asset_cfg,
    command_name="motion_command"
):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    base_ang_vel = asset.data.root_link_ang_vel_b
    reward = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
    trigger = cmd.trigger.squeeze(-1)
    return reward * (1.0 - trigger)

# def stand_still_reward(
#     env,
#     asset_cfg,
#     command_name="motion_command",
# ):
#     asset = env.scene[asset_cfg.name]
#     cmd = env.command_manager.get_term(command_name)
#     current_dof_pos = asset.data.joint_pos
#     reward = torch.sum(torch.square(current_dof_pos - asset.data.default_joint_pos), dim=1)
#     trigger = cmd.trigger.squeeze(-1)
#     return reward * (1.0 - trigger)

def stand_still_reward(
    env,
    asset_cfg,
    command_name="motion_command",
    sigma_stand: float = 0.3,
):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    error = torch.mean(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    reward = torch.exp(-error / (sigma_stand ** 2))
    trigger = cmd.trigger.squeeze(-1)
    return reward * (1.0 - trigger)

def height_reward(
    env,
    target_height,
    asset_cfg,
    command_name="motion_command",
    sigma_height: float = 0.1,
):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    base_height = asset.data.root_link_pos_w[:, 2]
    error = torch.square(base_height - target_height)
    reward = torch.exp(-error / (sigma_height ** 2))
    trigger = cmd.trigger.squeeze(-1)
    return reward * (1.0 - trigger)

def linear_vel_z_reward(
    env,
    asset_cfg,
    command_name="motion_command",
    sigma_vel_z: float = 0.2,
):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    error = torch.square(asset.data.root_link_lin_vel_b[:, 2])
    reward = torch.exp(-error / (sigma_vel_z ** 2))
    trigger = cmd.trigger.squeeze(-1)
    return reward * (1.0 - trigger)

# --------------------------Rewards of Motion Tracking--------------------------

def motion_linear_velocity_tracking_reward(
    env, 
    asset_cfg,
    command_name="motion_command",
    sigma_lin_vel_tracking=1.0
):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)

    motion_lin_vel_w = cmd.motion_lib.get_motion_data("body_lin_vel_w", cmd.motion_frame)[:, 0, :]
    motion_root_quat_w = cmd.motion_lib.get_motion_data("body_quat_w", cmd.motion_frame)[:, 0, :]

    # motion_lin_vel_w = motion_lib.get_motion_data("body_lin_vel_w", env.episode_length_buf)[:, 0, :]
    # motion_root_quat_w = motion_lib.get_motion_data("body_quat_w", env.episode_length_buf)[:, 0, :]

    motion_lin_vel_b = quat_apply_inverse(motion_root_quat_w, motion_lin_vel_w)
    robot_lin_vel_b = asset.data.root_link_lin_vel_b

    reward = torch.exp(-torch.sum(torch.square(motion_lin_vel_b - robot_lin_vel_b), dim=1)/ (sigma_lin_vel_tracking ** 2))
    
    # mask = motion_lib.get_motion_data("data_masking", env.episode_length_buf)
    # return torch.where((mask > -0.5) & (mask < 0.5), reward, torch.ones_like(reward))

    trigger = cmd.trigger.squeeze(-1)
    return reward * trigger

def motion_angular_velocity_tracking_reward(
    env,
    asset_cfg,
    command_name="motion_command",
    sigma_ang_vel_tracking=1.0
):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)

    motion_ang_vel_w = cmd.motion_lib.get_motion_data("body_ang_vel_w", cmd.motion_frame)[:, 0, :]
    motion_root_quat_w = cmd.motion_lib.get_motion_data("body_quat_w", cmd.motion_frame)[:, 0, :]
    motion_ang_vel_b = quat_apply_inverse(motion_root_quat_w, motion_ang_vel_w)
    robot_ang_vel_b = asset.data.root_link_ang_vel_b

    reward = torch.exp(-torch.sum(torch.square(motion_ang_vel_b - robot_ang_vel_b), dim=1)/ (sigma_ang_vel_tracking ** 2))

    # mask = motion_lib.get_motion_data("data_masking", env.episode_length_buf)
    # return torch.where((mask > -0.5) & (mask < 0.5), reward, torch.ones_like(reward))

    trigger = cmd.trigger.squeeze(-1)
    return reward * trigger

def motion_joint_position_tracking_reward(
    env,
    asset_cfg,
    command_name="motion_command",
    sigma_joint_pos_tracking=1.0,
    dof_weights=None,
):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)

    motion_joint_pos = cmd.motion_lib.get_motion_data("joint_pos", cmd.motion_frame)
    joint_pos = asset.data.joint_pos

    diff_sq = torch.square(motion_joint_pos - joint_pos)
    if dof_weights is not None:
        if not isinstance(dof_weights, torch.Tensor):
            dof_weights = torch.tensor(dof_weights, device=diff_sq.device, dtype=diff_sq.dtype)
        weighted_diff = torch.sum(diff_sq * dof_weights, dim=-1) / torch.sum(dof_weights)
    else:
        weighted_diff = torch.mean(diff_sq, dim=-1)

    reward = torch.exp(-weighted_diff / (sigma_joint_pos_tracking ** 2))

    trigger = cmd.trigger.squeeze(-1)
    return reward * trigger

def motion_body_position_tracking_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name="motion_command",
    sigma_body_pos_tracking: float = 1.0,
    body_indices=None,
    body_weights=None,
):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    
    motion_body_pos_w = cmd.motion_lib.get_motion_data("body_pos_w", cmd.motion_frame)
    motion_body_quat_w = cmd.motion_lib.get_motion_data("body_quat_w", cmd.motion_frame)
    motion_root_pos_w = cmd.motion_lib.get_motion_data("body_pos_w", cmd.motion_frame)[:, 0:1, :]
    motion_root_quat_w = cmd.motion_lib.get_motion_data("body_quat_w", cmd.motion_frame)[:, 0:1, :]

    robot_body_pos_w = asset.data.body_link_pos_w
    robot_root_pos_w = asset.data.root_link_pos_w.unsqueeze(1)
    robot_root_quat_w = asset.data.root_link_pose_w[:, 3:7].unsqueeze(1)

    motion_body_pos_b = quat_apply_inverse(
        motion_root_quat_w.expand(-1, motion_body_pos_w.shape[1], -1), 
        motion_body_pos_w - motion_root_pos_w
    )
    robot_body_pos_b = quat_apply_inverse(
        robot_root_quat_w.expand(-1, robot_body_pos_w.shape[1], -1), 
        robot_body_pos_w - robot_root_pos_w
    )

    per_body_diff = torch.sum(torch.square(motion_body_pos_b - robot_body_pos_b), dim=-1)
    if body_indices is not None:
        selected_diff = per_body_diff[:, body_indices]
        if body_weights is not None:
            w = torch.as_tensor(body_weights, device=selected_diff.device, dtype=selected_diff.dtype)
            pos_error = torch.sum(selected_diff * w, dim=-1) / torch.sum(w)
        else:
            pos_error = selected_diff.mean(dim=-1)
    else:
        pos_error = per_body_diff.mean(dim=-1)
    
    reward = torch.exp(-pos_error / (sigma_body_pos_tracking ** 2))

    trigger = cmd.trigger.squeeze(-1)
    return reward * trigger

def motion_body_orientation_tracking_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name="motion_command",
    sigma_body_ori_tracking: float = 1.0,
    body_indices=None,
    body_weights=None,
):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)

    motion_body_quat_w = cmd.motion_lib.get_motion_data("body_quat_w", cmd.motion_frame)
    motion_root_quat_w = cmd.motion_lib.get_motion_data("body_quat_w", cmd.motion_frame)[:, 0:1, :]

    robot_body_quat_w = asset.data.body_link_quat_w
    robot_root_quat_w = asset.data.root_link_pose_w[:, 3:7].unsqueeze(1)

    motion_body_quat_b = quat_mul(
        quat_inv(
            motion_root_quat_w.expand(-1, motion_body_quat_w.shape[1], -1)
        ), 
        motion_body_quat_w
    )
    robot_body_quat_b = quat_mul(
        quat_inv(
            robot_root_quat_w.expand(-1, robot_body_quat_w.shape[1], -1)
        ), 
        robot_body_quat_w
    )

    per_body_ori = quat_error_magnitude(motion_body_quat_b, robot_body_quat_b) ** 2
    if body_indices is not None:
        selected_ori = per_body_ori[:, body_indices]
        if body_weights is not None:
            w = torch.as_tensor(body_weights, device=selected_ori.device, dtype=selected_ori.dtype)
            ori_error = torch.sum(selected_ori * w, dim=-1) / torch.sum(w)
        else:
            ori_error = selected_ori.mean(dim=-1)
    else:
        ori_error = per_body_ori.mean(dim=-1)
    
    reward = torch.exp(-ori_error / (sigma_body_ori_tracking ** 2))

    trigger = cmd.trigger.squeeze(-1)
    return reward * trigger


def motion_root_position_tracking_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name="motion_command",
    sigma_root_pos_tracking: float = 0.5,
):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)

    motion_root_pos_w = cmd.motion_lib.get_motion_data("body_pos_w", cmd.motion_frame)[:, 0, :]

    robot_root_pos = asset.data.root_link_pos_w.clone()
    robot_root_pos[:, :2] -= env.scene.env_origins[:, :2]

    pos_error = torch.sum(torch.square(robot_root_pos - motion_root_pos_w), dim=-1)
    reward = torch.exp(-pos_error / (sigma_root_pos_tracking ** 2))

    # mask = motion_lib.get_motion_data("data_masking", env.episode_length_buf)
    # return torch.where((mask > -0.5) & (mask < 0.5), reward, torch.ones_like(reward))
    trigger = cmd.trigger.squeeze(-1)
    return reward * trigger


def motion_root_orientation_tracking_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name="motion_command",
    sigma_root_ori_tracking: float = 0.5,
):
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    motion_root_quat_w = cmd.motion_lib.get_motion_data("body_quat_w", cmd.motion_frame)[:, 0, :]
    robot_root_quat_w = asset.data.root_link_pose_w[:, 3:7]

    ori_error = quat_error_magnitude(motion_root_quat_w, robot_root_quat_w) ** 2
    reward = torch.exp(-ori_error / (sigma_root_ori_tracking ** 2))

    # mask = motion_lib.get_motion_data("data_masking", env.episode_length_buf)
    # return torch.where((mask > -0.5) & (mask < 0.5), reward, torch.ones_like(reward))
    trigger = cmd.trigger.squeeze(-1)
    return reward * trigger


def collision_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str = "self_collision",
    force_threshold: float = 10.0,
):
    sensor = env.scene[sensor_name]
    data = sensor.data
    force_mag = torch.norm(data.force_history, dim=-1)
    collide = (force_mag > force_threshold).any(dim=(-1))
    return collide.sum(dim=-1).float()


def foot_sliding_during_contact(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str = "feet_ground_contact",
    force_threshold: float = 5.0,
):
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[sensor_name]

    foot_contact_forces = torch.norm(contact_sensor.data.force, dim=-1)
    in_contact = (foot_contact_forces > force_threshold).float()

    foot_lin_vel_xy = asset.data.body_link_lin_vel_w[:, [6, 12], :2]
    foot_speed_xy_sq = torch.sum(torch.square(foot_lin_vel_xy), dim=-1)

    slip_penalty = foot_speed_xy_sq * in_contact

    return torch.sum(slip_penalty, dim=-1)


def foot_orientation_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "motion_command",
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_name]
    feet_quat = asset.data.body_link_quat_w[:, [6, 12]]
    
    w = feet_quat[..., 0]
    x = feet_quat[..., 1]
    y = feet_quat[..., 2]
    z = feet_quat[..., 3]
    
    zx = 2.0 * (x * z + w * y)
    zy = 2.0 * (y * z - w * x)
    
    tilt = torch.square(zx) + torch.square(zy)
    in_contact = (sensor.data.found > 0).float()
    return torch.sum(tilt * in_contact, dim=-1)