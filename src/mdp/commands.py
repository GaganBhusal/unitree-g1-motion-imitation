from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Any
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from src.utils.motion_lib import MotionLib


@dataclass(kw_only=True)
class MyCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1.0, 2.0)
    debug_vis: bool = False
    entity_name: str = "robot"

    motion_lib: MotionLib
    settle_time_range: tuple[float, float] = (1.2, 3.0)
    prob_rsi: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> MyCommand:
        return MyCommand(self, env)


class MyCommand(CommandTerm):
    """Multi-motion command term supporting N arbitrary motions.
    
    Commands:
      0: Standing still
      1: Motion 1 (e.g. Spiderman)
      2: Motion 2 (e.g. Dance)
      ...
      N: Motion N
    """

    cfg: MyCommandCfg

    def __init__(self, cfg: MyCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot = env.scene[cfg.entity_name]
        self.motion_lib = cfg.motion_lib
        self.num_motions = self.motion_lib.num_motions
        self.step_dt = env.step_dt

        # Link motion_lib back to this command term for automatic routing
        self.motion_lib.active_cmd = self

        # Motion ID: 0 = Standing, 1 = Motion 1, 2 = Motion 2, ..., N = Motion N
        self.motion_id = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self.motion_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.default_pos = self.motion_lib.default_pos.to(self.device).view(1, -1)
        self.auto_loop: bool = True

    @property
    def active_motion_idx(self) -> torch.Tensor:
        """0-based index into motion_lib stacked tensors for active environments.
        Returns shape (num_envs,).
        """
        return (self.motion_id.squeeze(-1) - 1).clamp(min=0)

    @property
    def trigger(self) -> torch.Tensor:
        """Binary trigger flag: 0.0 when standing, 1.0 when executing any motion.
        Used by rewards for seamless switching. Shape: (num_envs, 1).
        """
        return (self.motion_id > 0).float()

    @property
    def command(self) -> torch.Tensor:
        """Property read by mdp.generated_commands.
        Returns float tensor of shape (num_envs, 1) containing:
          0.0 for Standing, 1.0 for Motion 1, 2.0 for Motion 2, etc.
        """
        return self.motion_id.float()

    @property
    def num_frames(self) -> torch.Tensor:
        """Number of frames for the active motion of each environment. Shape: (num_envs,)."""
        return self.motion_lib.num_frames[self.active_motion_idx]

    @property
    def phase(self) -> torch.Tensor:
        cur_nf = self.motion_lib.num_frames[self.active_motion_idx]
        raw_phase = (self.motion_frame.float() / cur_nf.float()).unsqueeze(-1)
        return torch.where(self.trigger > 0.5, raw_phase, torch.zeros_like(raw_phase))

    def get_ref_joint_pos(self) -> torch.Tensor:
        motion_joint_pos = self.motion_lib.get_motion_data("joint_pos", self.motion_frame, self.active_motion_idx)
        return torch.where(self.trigger > 0.5, motion_joint_pos, self.default_pos)

    def get_ref_joint_vel(self) -> torch.Tensor:
        motion_joint_vel = self.motion_lib.get_motion_data("joint_vel", self.motion_frame, self.active_motion_idx)
        return torch.where(self.trigger > 0.5, motion_joint_vel, torch.zeros_like(motion_joint_vel))

    def get_ref_future_joint_pos(self, horizon: int = 10) -> torch.Tensor:
        future_frames = self.motion_frame + horizon
        cur_nf = self.motion_lib.num_frames[self.active_motion_idx]
        is_past_end = (future_frames >= cur_nf)

        safe_future_frames = future_frames % cur_nf
        motion_data_future = self.motion_lib.get_motion_data(
            "joint_pos", safe_future_frames, self.active_motion_idx
        )  # (num_envs, 23)

        default_pos = self.default_pos.expand(self.num_envs, -1)
        future_pos = torch.where(is_past_end.unsqueeze(-1), default_pos, motion_data_future)
        return torch.where(self.trigger > 0.5, future_pos, default_pos)

        default_pos = self.default_pos.view(1, 1, -1)
        future_trajectory = torch.where(is_past_end, default_pos, motion_data_future)
        future_trajectory = torch.where(self.trigger.view(-1, 1, 1) > 0.5, future_trajectory, default_pos)
        return future_trajectory.reshape(self.num_envs, -1)

    def get_ref_body_pos_w(self) -> torch.Tensor:
        motion_body_pos = self.motion_lib.get_motion_data("body_pos_w", self.motion_frame, self.active_motion_idx)
        return motion_body_pos + self._env.scene.env_origins.unsqueeze(1)

    def get_ref_body_quat_w(self) -> torch.Tensor:
        return self.motion_lib.get_motion_data("body_quat_w", self.motion_frame, self.active_motion_idx)

    def get_ref_body_lin_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_motion_data("body_lin_vel_w", self.motion_frame, self.active_motion_idx)

    def get_ref_body_ang_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_motion_data("body_ang_vel_w", self.motion_frame, self.active_motion_idx)

    def trigger_motion(self, motion_id: int = 1) -> None:
        """Trigger motion by ID (1 to N). If already in motion, ignore (cannot untrigger mid-motion)."""
        idle_mask = (self.motion_id[:, 0] == 0)
        if idle_mask.any():
            m_id = max(1, min(motion_id, self.num_motions))
            self.motion_id[idle_mask] = m_id
            self.motion_frame[idle_mask] = 0
            m_idx = m_id - 1
            duration = float(self.motion_lib.num_frames_list[m_idx]) * self.step_dt
            self.time_left[idle_mask] = duration
            print(f">>> [Command] Motion {m_id} ({self.motion_lib.motion_names[m_idx]}) Triggered!")

    def reset_to_standing(self) -> None:
        """Reset to standing posture immediately (Command 0)."""
        self.motion_id[:] = 0
        self.motion_frame[:] = 0
        self.time_left[:] = 1e9
        print(">>> [Command] Reset to Standing (Command 0)!")

    def create_gui(
        self,
        name: str,
        server,
        get_env_idx,
        on_change=None,
        request_action=None,
    ) -> None:
        """Create interactive Viser GUI controls for all loaded motions."""
        with server.gui.add_folder(name.capitalize()):
            for m_idx, m_name in enumerate(self.motion_lib.motion_names):
                m_id = m_idx + 1
                btn = server.gui.add_button(f"Trigger {m_id}: {m_name.capitalize()}")

                def make_click_fn(target_id):
                    def _(_):
                        self.trigger_motion(target_id)
                        if on_change is not None:
                            on_change()
                    return _

                btn.on_click(make_click_fn(m_id))

            btn_stand = server.gui.add_button("Stand Still (Cmd 0)")

            @btn_stand.on_click
            def _(_):
                self.reset_to_standing()
                if on_change is not None:
                    on_change()

            auto_loop_cb = server.gui.add_checkbox("Auto Loop", initial_value=self.auto_loop)

            @auto_loop_cb.on_update
            def _(_):
                self.auto_loop = bool(auto_loop_cb.value)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Called automatically when an environment resets or self.time_left reaches 0.0."""
        is_reset = (self.command_counter[env_ids] == 0)
        reset_ids = env_ids[is_reset]
        step_ids = env_ids[~is_reset]

        if len(reset_ids) > 0:
            rsi_mask = torch.rand(len(reset_ids), device=self.device) < self.cfg.prob_rsi
            rsi_envs = reset_ids[rsi_mask]
            non_rsi_envs = reset_ids[~rsi_mask]

            # RSI: start at a random motion and a random frame within that motion
            if len(rsi_envs) > 0:
                self.motion_id[rsi_envs] = torch.randint(
                    1, self.num_motions + 1, (len(rsi_envs), 1), device=self.device
                )
                m_idx = self.active_motion_idx[rsi_envs]
                m_num_frames = self.motion_lib.num_frames[m_idx]
                self.motion_frame[rsi_envs] = (
                    torch.rand(len(rsi_envs), device=self.device) * (m_num_frames - 1).float()
                ).long()
                steps_left = m_num_frames - self.motion_frame[rsi_envs]
                self.time_left[rsi_envs] = steps_left.float() * self.step_dt

                # --- Write reference root state & joint state to simulation (True RSI) ---
                frames = self.motion_frame[rsi_envs]
                ref_root_pos = self.motion_lib.get_motion_data("body_pos_w", frames, m_idx)[:, 0, :].clone()
                ref_root_pos[:, :2] += self._env.scene.env_origins[rsi_envs, :2]
                ref_root_quat = self.motion_lib.get_motion_data("body_quat_w", frames, m_idx)[:, 0, :].clone()
                ref_root_lin_vel = self.motion_lib.get_motion_data("body_lin_vel_w", frames, m_idx)[:, 0, :].clone()
                ref_root_ang_vel = self.motion_lib.get_motion_data("body_ang_vel_w", frames, m_idx)[:, 0, :].clone()
                ref_joint_pos = self.motion_lib.get_motion_data("joint_pos", frames, m_idx).clone()
                ref_joint_vel = self.motion_lib.get_motion_data("joint_vel", frames, m_idx).clone()

                root_state = torch.cat([ref_root_pos, ref_root_quat, ref_root_lin_vel, ref_root_ang_vel], dim=-1)
                self.robot.write_root_state_to_sim(root_state, env_ids=rsi_envs)
                self.robot.write_joint_state_to_sim(ref_joint_pos, ref_joint_vel, env_ids=rsi_envs)

            # Non-RSI: start in standing / locomotion posture (Command 0)
            if len(non_rsi_envs) > 0:
                self.motion_id[non_rsi_envs] = 0
                self.motion_frame[non_rsi_envs] = 0
                r = torch.empty(len(non_rsi_envs), device=self.device)
                self.time_left[non_rsi_envs] = r.uniform_(*self.cfg.settle_time_range)

        if len(step_ids) > 0:
            was_in_motion = (self.motion_id[step_ids, 0] > 0)
            motion_ended_envs = step_ids[was_in_motion]
            settle_ended_envs = step_ids[~was_in_motion]

            # Motion ended: transition to standing settle phase
            # Motion ended: transition to locomotion / standing phase
            if len(motion_ended_envs) > 0:
                self.motion_id[motion_ended_envs] = 0
                self.motion_frame[motion_ended_envs] = 0
                settle_steps = torch.randint(25, 60, (len(motion_ended_envs),), device=self.device)
                self.time_left[motion_ended_envs] = settle_steps.float() * self.step_dt
                r = torch.empty(len(motion_ended_envs), device=self.device)
                self.time_left[motion_ended_envs] = r.uniform_(*self.cfg.settle_time_range)

            # Settle ended: if auto_loop, pick next random motion; otherwise stay standing
            if len(settle_ended_envs) > 0:
                if self.auto_loop:
                    self.motion_id[settle_ended_envs] = torch.randint(
                        1, self.num_motions + 1, (len(settle_ended_envs), 1), device=self.device
                    )
                    m_idx = self.active_motion_idx[settle_ended_envs]
                    m_num_frames = self.motion_lib.num_frames[m_idx]
                    self.motion_frame[settle_ended_envs] = 0
                    self.time_left[settle_ended_envs] = m_num_frames.float() * self.step_dt
                else:
                    self.motion_id[settle_ended_envs] = 0
                    self.motion_frame[settle_ended_envs] = 0
                    self.time_left[settle_ended_envs] = 1e9

    def _update_command(self) -> None:
        """Called EVERY single physics step (dt = env.step_dt)."""
        active_envs = (self.motion_id[:, 0] > 0)

        if active_envs.any():
            self.motion_frame[active_envs] += 1
            m_idx = self.active_motion_idx[active_envs]
            m_max = self.motion_lib.num_frames[m_idx] - 1
            self.motion_frame[active_envs] = torch.min(self.motion_frame[active_envs], m_max)

        idle_envs = ~active_envs
        if idle_envs.any():
            self.motion_frame[idle_envs] = 0

    def _update_metrics(self) -> None:
        pass