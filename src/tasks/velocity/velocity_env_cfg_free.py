import src.mdp as mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.managers import (
    ObservationTermCfg, ObservationGroupCfg, SceneEntityCfg, 
    RewardTermCfg, EventTermCfg,
    TerminationTermCfg, ActionTermCfg
)
from src.utils import MotionLib
from configs.env_configs import EnvConfigs
from configs.g1_config import get_g1_23dof_robot_cfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
import torch



def make_motion_env_cfg(motion_path: str | list[str], device = torch.device("cuda")) -> ManagerBasedRlEnvCfg:

    robot_cfg = SceneEntityCfg(name="robot")
    configs = EnvConfigs()
    motion_lib = MotionLib(
        device=device, 
        motion_path=motion_path,
        default_pos=configs.default_joints
    )    
    
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=10,
    )


    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=10,
        track_air_time=True,
    )

    actor_terms = {
        "base_lin_vel" : ObservationTermCfg(
            func=mdp.base_lin_vel,
            params={
                "asset_cfg": robot_cfg
            },
            scale = configs.base_lin_vel
        ),

        "base_ang_vel" : ObservationTermCfg(
            func=mdp.base_ang_vel,
            params={
                "asset_cfg": robot_cfg
            },
            scale = configs.base_ang_vel
        ),

        "base_z_pos": ObservationTermCfg(
            func=mdp.base_z_pos,
            params={
                "asset_cfg": robot_cfg
            },
            scale = configs.base_z_pos
        ),

        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity, 
            params={
                "asset_cfg": robot_cfg
            },
            scale=configs.projected_gravity
        ),

        "dof_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": robot_cfg
            },
            scale=configs.dof_pos
        ),

        "dof_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": robot_cfg
            },
            scale=configs.dof_vel
        ),

        "last_action": ObservationTermCfg(
            func=mdp.last_action,
            params={
                "action_name": "joint_pos"
            },
            scale=configs.last_action
        ),

        # "velocity_commands": ObservationTermCfg(
        #     func=mdp.velocity_commands,
        #     params={
        #         "command_name": "twist",
        #         "motion_command_name": "motion_command",
        #     },
        #     scale=1.0,
        # ),

        # "sin_phase": ObservationTermCfg(
        #     func=mdp.sine_phase,
        #     params={
        #         "command_name": "motion_command",
        #         "gait_period": configs.gait_period,
        #     },
        #     scale=1.0,
        # ),

        # "cos_phase": ObservationTermCfg(
        #     func=mdp.cosine_phase,
        #     params={
        #         "command_name": "motion_command",
        #         "gait_period": configs.gait_period,
        #     },
        #     scale=1.0,
        # ),

        "motion_trigger": ObservationTermCfg(
            func=mdp.generated_commands,
            params={
                "command_name": "motion_command"
            },
            scale=1.0,
        ),

        "motion_phase": ObservationTermCfg(
            func=mdp.motion_phase,
            scale=configs.motion_phase
        ),

        "motion_current_joint_position": ObservationTermCfg(
            func=mdp.motion_current_joint_position_relative,
            params={
                "asset_cfg": robot_cfg,
            },
            scale=configs.motion_current_joint_position
        ),

        "current_motion_joint_velocity": ObservationTermCfg(
            func=mdp.motion_current_joint_velocity,
            scale=configs.current_motion_joint_velocity
        ),

        "motion_joint_position_error": ObservationTermCfg(
            func=mdp.motion_joint_position_error,
            params={
                "asset_cfg": robot_cfg,
            },
            scale=configs.motion_joint_position_error
        ),

        "motion_future_joint_position": ObservationTermCfg(
            func=mdp.motion_future_joint_position,
            params={
                "asset_cfg": robot_cfg,
                "horizon": configs.future_states_horizon
            },
            scale=configs.motion_future_joint_position
        ),
    }


    observations = {
        "actor" : ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=configs.history_length
        ),

        "critic" : ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=configs.history_length
        )
    }

    commands = {
        # "twist": UniformVelocityCommandCfg(
        #     entity_name="robot",
        #     resampling_time_range=configs.vel_resampling_time_range,
        #     rel_standing_envs=configs.rel_standing_envs,
        #     ranges=UniformVelocityCommandCfg.Ranges(
        #         lin_vel_x=configs.lin_vel_x_range,
        #         lin_vel_y=configs.lin_vel_y_range,
        #         ang_vel_z=configs.ang_vel_z_range,
        #     ),
        # ),
        "motion_command": mdp.MyCommandCfg(
            motion_lib=motion_lib,
            settle_time_range=configs.settle_time_range,
            prob_rsi=configs.rsi_probability,
        ),
    }

    rewards = {

        # "track_vx_reward": RewardTermCfg(
        #     func=mdp.track_vx_reward,
        #     params={
        #         "asset_cfg": robot_cfg,
        #         "command_name": "motion_command",
        #         "sigma_vx": configs.sigma_vx,
        #     },
        #     weight=configs.track_vx_reward,
        # ),

        # "track_vy_reward": RewardTermCfg(
        #     func=mdp.track_vy_reward,
        #     params={
        #         "asset_cfg": robot_cfg,
        #         "command_name": "motion_command",
        #         "sigma_vy": configs.sigma_vy,
        #     },
        #     weight=configs.track_vy_reward,
        # ),

        "track_wz_reward": RewardTermCfg(
            func=mdp.track_wz_reward,
            params={
                "asset_cfg": robot_cfg,
                "command_name": "motion_command",
                "sigma_wz": configs.sigma_wz,
            },
            weight=configs.track_wz_reward,
        ),

        "orientation_upright_reward": RewardTermCfg(
            func=mdp.orientation_upright_reward,
            params={
                "asset_cfg": robot_cfg,
                "sigma_upright": configs.sigma_upright,
            },
            weight=configs.orientation_upright_reward
        ),

        "linear_vel_z_reward": RewardTermCfg(
            func=mdp.linear_vel_z_reward,
            params={
                "asset_cfg": robot_cfg,
                "sigma_vel_z": configs.sigma_vel_z,
            },
            weight=configs.linear_vel_z_reward
        ),

        "base_height_reward": RewardTermCfg(
            func=mdp.height_reward,
            params={
                "asset_cfg": robot_cfg,
                "target_height": configs.target_height,
                "sigma_height": configs.sigma_height,
            },
            weight=configs.base_height_reward
        ),

        "stand_still_reward": RewardTermCfg(
            func=mdp.stand_still_reward,
            params={
                "asset_cfg": robot_cfg,
                "command_name": "motion_command",
                "sigma_stand": configs.sigma_stand,
            },
            weight=configs.stand_still_reward
        ),

        "collision_reward": RewardTermCfg(
            func=mdp.collision_reward,
            params={
                "sensor_name": "self_collision",
            },
            weight=configs.collision_reward
        ),

        # "feet_air_time": RewardTermCfg(
        #     func=mdp.reward_feet_air_time,
        #     params={
        #         "sensor_name": "feet_ground_contact",
        #         "command_name": "motion_command",
        #         "twist_command_name": "twist",
        #         "threshold_min": configs.feet_air_time_threshold_min,
        #         "threshold_max": configs.feet_air_time_threshold_max,
        #     },
        #     weight=configs.feet_air_time_reward,
        # ),

        # "hip_deviation": RewardTermCfg(
        #     func=mdp.reward_hip_deviation,
        #     params={
        #         "asset_cfg": robot_cfg,
        #         "command_name": "motion_command",
        #         "hip_yaw_indices": configs.hip_yaw_indices,
        #         "sigma_hip_deviation": configs.sigma_hip_deviation,
        #     },
        #     weight=configs.hip_deviation_reward,
        # ),

        # "upper_body_pose": RewardTermCfg(
        #     func=mdp.reward_upper_body_pose,
        #     params={
        #         "asset_cfg": robot_cfg,
        #         "command_name": "motion_command",
        #         "upper_body_indices": configs.upper_body_indices,
        #         "sigma_upper_body_pose": configs.sigma_upper_body_pose,
        #     },
        #     weight=configs.upper_body_pose_reward,
        # ),

        # "contact": RewardTermCfg(
        #     func=mdp.reward_contact,
        #     params={
        #         "sensor_name": "feet_ground_contact",
        #         "command_name": "motion_command",
        #         "twist_command_name": "twist",
        #         "gait_period": configs.gait_period,
        #         "gait_offset": configs.gait_offset,
        #         "stance_ratio": configs.stance_ratio,
        #     },
        #     weight=configs.contact_reward,
        # ),

        # "feet_swing_height": RewardTermCfg(
        #     func=mdp.reward_feet_swing_height,
        #     params={
        #         "asset_cfg": robot_cfg,
        #         "sensor_name": "feet_ground_contact",
        #         "command_name": "motion_command",
        #         "twist_command_name": "twist",
        #         "target_foot_height": configs.target_foot_height,
        #         "default_ankle_height": configs.default_ankle_height,
        #         "gait_period": configs.gait_period,
        #         "gait_offset": configs.gait_offset,
        #         "stance_ratio": configs.stance_ratio,
        #         "sigma_feet_swing_height": configs.sigma_feet_swing_height,
        #     },
        #     weight=configs.feet_swing_height_reward,
        # ),

        "alive_reward": RewardTermCfg(
            func=mdp.is_alive,
            weight=configs.alive_reward
        ),

        "joint_torque_reward": RewardTermCfg(
            func=mdp.joint_torques_l2,
            params={
                "asset_cfg": robot_cfg,
            },
            weight=configs.joint_torque_reward
        ),

        "joint_vel_reward": RewardTermCfg(
            func=mdp.joint_vel_l2,
            params={
                "asset_cfg": robot_cfg,
            },
            weight=configs.joint_vel_reward
        ),

        "joint_acc_reward": RewardTermCfg(
            func=mdp.joint_acc_l2,
            params={
                "asset_cfg": robot_cfg,
            },
            weight=configs.joint_acc_reward
        ),

        "action_rate_reward": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=configs.action_rate_reward
        ),

        "action_acc_reward": RewardTermCfg(
            func=mdp.action_acc_l2,
            weight=configs.action_acc_reward    
        ),

        "joint_pos_limits_reward": RewardTermCfg(
            func=mdp.joint_pos_limits,
            params={
                "asset_cfg": robot_cfg,
            },
            weight=configs.joint_pos_limits_reward
        ),

        "motion_linear_velocity_tracking_reward": RewardTermCfg(
            func=mdp.motion_linear_velocity_tracking_reward,
            params={
                "asset_cfg": robot_cfg,
                "sigma_lin_vel_tracking": configs.sigma_lin_vel
            },
            weight=configs.motion_linear_velocity_tracking_reward
        ),  

        "motion_angular_velocity_tracking_reward": RewardTermCfg(
            func=mdp.motion_angular_velocity_tracking_reward,
            params={
                "asset_cfg": robot_cfg,
                "sigma_ang_vel_tracking": configs.sigma_ang_vel
            },
            weight=configs.motion_angular_velocity_tracking_reward
        ),

        "motion_joint_position_tracking_reward": RewardTermCfg(
            func=mdp.motion_joint_position_tracking_reward,
            params={
                "asset_cfg": robot_cfg,
                "sigma_joint_pos_tracking": configs.sigma_joint_pos,
                "dof_weights": configs.tracking_dof_weights,
            },
            weight=configs.motion_joint_position_tracking_reward
        ),

        "motion_body_position_tracking_reward": RewardTermCfg(
            func=mdp.motion_body_position_tracking_reward,
            params={
                "asset_cfg": robot_cfg,
                "sigma_body_pos_tracking": configs.sigma_body_pos,
                "body_indices": configs.tracking_body_indices,
                "body_weights": configs.tracking_body_pos_weights,
            },
            weight=configs.motion_body_position_tracking_reward
        ),

        "motion_body_orientation_tracking_reward": RewardTermCfg(
            func=mdp.motion_body_orientation_tracking_reward,
            params={
                "asset_cfg": robot_cfg,
                "sigma_body_ori_tracking": configs.sigma_body_ori,
                "body_indices": configs.tracking_body_indices,
                "body_weights": configs.tracking_body_ori_weights,
            },
            weight=configs.motion_body_orientation_tracking_reward
        ),

        "motion_root_position_tracking_reward": RewardTermCfg(
            func=mdp.motion_root_position_tracking_reward,
            params={
                "asset_cfg": robot_cfg,
                "sigma_root_pos_tracking": configs.sigma_root_pos
            },
            weight=configs.motion_root_position_tracking_reward
        ),

        "motion_root_orientation_tracking_reward": RewardTermCfg(
            func=mdp.motion_root_orientation_tracking_reward,
            params={
                "asset_cfg": robot_cfg,
                "sigma_root_ori_tracking": configs.sigma_root_ori
            },
            weight=configs.motion_root_orientation_tracking_reward
        ),

        "foot_sliding_during_contact": RewardTermCfg(
            func=mdp.foot_sliding_during_contact,
            params={
                "asset_cfg": robot_cfg,
                "sensor_name": "feet_ground_contact",
            },
            weight=configs.foot_sliding_during_contact_penalty,
        ),

        "foot_orientation_penalty": RewardTermCfg(
            func=mdp.foot_orientation_penalty,
            params={
                "asset_cfg": robot_cfg,
                "sensor_name": "feet_ground_contact",
                "command_name": "motion_command",
            },
            weight=configs.foot_orientation_penalty,
        ),

        "is_terminated": RewardTermCfg(  
            func=mdp.is_terminated,  
            weight=configs.termination_weight,
        )
    }


    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            scale=configs.action_scale,
            actuator_names=(".*",),
            use_default_offset=True,
        )
    }

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "root_height_below_minimum": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={
                "asset_cfg": robot_cfg,
                "minimum_height" : 0.2
            }   
        )
    }

    events = {
        "reset_scene": EventTermCfg(
            func=mdp.reset_scene_to_default,
            mode="reset",
        ),
    }
    
    sim=SimulationCfg(
      nconmax=configs.sim_nconmax,
      njmax=configs.sim_njmax,
      mujoco=MujocoCfg(
        timestep=configs.sim_physics_dt,
        iterations=configs.sim_iterations,
        ls_iterations=configs.sim_ls_iterations,
      ),
    )

    managerbased_rl_env_cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            num_envs=configs.sim_num_envs,
            env_spacing=2.5,
            extent=2.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": get_g1_23dof_robot_cfg()},
            sensors=(self_collision_cfg, feet_ground_cfg),
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        sim=sim,
        decimation=configs.sim_decimation,
        episode_length_s=configs.sim_episode_length_s,
    )

    return managerbased_rl_env_cfg
    
