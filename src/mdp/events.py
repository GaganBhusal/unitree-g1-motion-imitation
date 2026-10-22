import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg
from src.utils import MotionLib


def random_motion_state_initialization(
    env: ManagerBasedRlEnv, 
    env_ids: torch.Tensor, 
    asset_cfg: SceneEntityCfg, 
    motion_lib: MotionLib,
    prob_start_from_zero: float = 0.2
):

    asset = env.scene[asset_cfg.name]
    num_resets = len(env_ids)

    mask = motion_lib.motion_data["data_masking"]

    valid_indices = torch.where((mask > -0.5) & (mask < 0.5))[0]
    start_frame = valid_indices[0].item()
    end_frame = valid_indices[-1].item()


    random_steps = torch.randint(
        start_frame, end_frame + 1, (num_resets,), device=env.device
    )

    start_from_zero = torch.rand(num_resets, device=env.device) < prob_start_from_zero
    random_steps[start_from_zero] = 0

    env.episode_length_buf[env_ids] = random_steps

    ref_root_pos = motion_lib.get_motion_data("body_pos_w", random_steps)[:, 0, :].clone()
    ref_root_quat = motion_lib.get_motion_data("body_quat_w", random_steps)[:, 0, :].clone()
    ref_root_lin_vel = motion_lib.get_motion_data("body_lin_vel_w", random_steps)[:, 0, :].clone()
    ref_root_ang_vel = motion_lib.get_motion_data("body_ang_vel_w", random_steps)[:, 0, :].clone()

    ref_joint_pos = motion_lib.get_motion_data("joint_pos", random_steps).clone()
    ref_joint_vel = motion_lib.get_motion_data("joint_vel", random_steps).clone()

    if start_from_zero.any():
        zero_envs_mask = start_from_zero
        default_root = asset.data.default_root_state[env_ids[zero_envs_mask]]
        ref_root_pos[zero_envs_mask] = default_root[:, :3]
        ref_root_quat[zero_envs_mask] = default_root[:, 3:7]
        ref_joint_vel[zero_envs_mask] = 0.0
        ref_root_lin_vel[zero_envs_mask] = 0.0
        ref_root_ang_vel[zero_envs_mask] = 0.0

    ref_root_pos[:, :2] += env.scene.env_origins[env_ids, :2]

    root_state = torch.cat([
        ref_root_pos,
        ref_root_quat,
        ref_root_lin_vel,
        ref_root_ang_vel
    ], dim=-1)

    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    asset.write_joint_state_to_sim(ref_joint_pos, ref_joint_vel, env_ids=env_ids)
