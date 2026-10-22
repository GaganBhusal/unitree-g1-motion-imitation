import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from src.utils import MotionLib

def base_z_pos(
    env: ManagerBasedRlEnv, 
    asset_cfg: SceneEntityCfg,
):
    return env.scene[asset_cfg.name].data.root_link_pos_w[:, -1:]

# def motion_phase(
#     env: ManagerBasedRlEnv, 
#     asset_cfg: SceneEntityCfg,
#     motion_lib  : MotionLib
# ):  
#     # return motion_lib.get_motion_data("motion_phase", env.episode_length_buf % motion_lib.num_frames).unsqueeze(-1)
#     phase = ((env.episode_length_buf % motion_lib.num_frames).float().unsqueeze(-1)) / motion_lib.num_frames
#     return phase

# def motion_current_joint_position_relative(
#     env: ManagerBasedRlEnv, 
#     asset_cfg: SceneEntityCfg,
#     motion_lib: MotionLib
# ):
#     asset = env.scene[asset_cfg.name]
#     motion_dof_pos = motion_lib.get_motion_data("joint_pos", env.episode_length_buf)
#     dof_pos_init = asset.data.default_joint_pos
#     return motion_dof_pos - dof_pos_init

# def motion_current_joint_velocity(
#     env: ManagerBasedRlEnv, 
#     asset_cfg: SceneEntityCfg,
#     motion_lib: MotionLib
# ):
#     asset = env.scene[asset_cfg.name]
#     motion_dof_vel = motion_lib.get_motion_data("joint_vel", env.episode_length_buf)
#     return motion_dof_vel

def sine_phase(
    env: ManagerBasedRlEnv, 
    command_name: str = "motion_command",
    gait_period: float = 0.8,
) -> torch.Tensor:
    phase = (env.episode_length_buf.float() * env.step_dt) % gait_period / gait_period
    sin_p = torch.sin(2 * torch.pi * phase).unsqueeze(-1)
    motion_cmd = env.command_manager.get_term(command_name)
    return sin_p * (1.0 - motion_cmd.trigger)


def cosine_phase(
    env: ManagerBasedRlEnv, 
    command_name: str = "motion_command",
    gait_period: float = 0.8,
) -> torch.Tensor:
    phase = (env.episode_length_buf.float() * env.step_dt) % gait_period / gait_period
    cos_p = torch.cos(2 * torch.pi * phase).unsqueeze(-1)
    motion_cmd = env.command_manager.get_term(command_name)
    return cos_p * (1.0 - motion_cmd.trigger)


# def motion_future_joint_position(
#     env: ManagerBasedRlEnv, 
#     asset_cfg: SceneEntityCfg,
#     motion_lib: MotionLib,
#     horizon: int = 10
# ):

#     asset = env.scene[asset_cfg.name]
#     future_steps = (env.episode_length_buf + horizon) % motion_lib.num_frames
#     motion_dof_pos = motion_lib.get_motion_data("joint_pos", future_steps)
#     dof_pos_init = asset.data.default_joint_pos
#     return motion_dof_pos - dof_pos_init

# def motion_joint_position_error(
#     env: ManagerBasedRlEnv, 
#     asset_cfg: SceneEntityCfg, 
#     motion_lib: MotionLib
# ) -> torch.Tensor:
#     asset = env.scene[asset_cfg.name]
#     motion_dof_pos = motion_lib.get_motion_data("joint_pos", env.episode_length_buf)
#     current_dof_pos = asset.data.joint_pos
#     return motion_dof_pos - current_dof_pos

def motion_phase(
    env: ManagerBasedRlEnv, 
    command_name: str = "motion_command"
) -> torch.Tensor:  
    cmd = env.command_manager.get_term(command_name)
    return cmd.phase * cmd.trigger

def motion_current_joint_position_relative(
    env: ManagerBasedRlEnv, 
    asset_cfg: SceneEntityCfg,
    command_name: str = "motion_command"
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    return (cmd.get_ref_joint_pos() - asset.data.default_joint_pos) * cmd.trigger

def motion_current_joint_velocity(
    env: ManagerBasedRlEnv, 
    command_name: str = "motion_command"
) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    return cmd.get_ref_joint_vel() * cmd.trigger

def motion_joint_position_error(
    env: ManagerBasedRlEnv, 
    asset_cfg: SceneEntityCfg, 
    command_name: str = "motion_command"
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    return (cmd.get_ref_joint_pos() - asset.data.joint_pos) * cmd.trigger

def motion_future_joint_position(
    env: ManagerBasedRlEnv, 
    asset_cfg: SceneEntityCfg,
    command_name: str = "motion_command",
    horizon: int = 10
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    future_pos = cmd.get_ref_future_joint_pos(horizon=horizon)
    return (future_pos - asset.data.default_joint_pos) * cmd.trigger

def mode_one_hot(
    env: ManagerBasedRlEnv, 
    command_name: str = "motion_command"
) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    num_modes = 1 + cmd.num_motions
    one_hot = torch.zeros(env.num_envs, num_modes, device=env.device)
    one_hot.scatter_(1, cmd.motion_id, 1.0)
    return one_hot


def velocity_commands(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    motion_command_name: str | None = "motion_command",
) -> torch.Tensor:
    
    vel_cmd = env.command_manager.get_command(command_name)
    motion_cmd = env.command_manager.get_term(motion_command_name)
    vel_cmd = vel_cmd * (1.0 - motion_cmd.trigger)
    return vel_cmd