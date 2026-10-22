from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Any

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from src.utils.motion_lib import MotionLib

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    import viser


@dataclass(kw_only=True)
class MotionTriggerCommandCfg(CommandTermCfg):

    motion_lib: MotionLib
    settle_time_range: tuple[float, float] = (1.0, 2.5)
    
    prob_rsi: float = 0.0
    
    resampling_time_range: tuple[float, float] = (1.0, 2.5)

    def build(self, env: ManagerBasedRlEnv) -> MotionTriggerCommand:
        return MotionTriggerCommand(self, env)


class MotionTriggerCommand(CommandTerm):

    cfg: MotionTriggerCommandCfg

    def __init__(self, cfg: MotionTriggerCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.motion_lib: MotionLib = cfg.motion_lib
        self.num_frames: int = self.motion_lib.num_frames
        self.step_dt: float = env.step_dt
        self.motion_duration: float = self.num_frames * self.step_dt

        self.trigger = torch.zeros(self.num_envs, 1, device=self.device)
        self.motion_frame = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.default_pos = self.motion_lib.default_pos.to(self.device).view(1, -1)

        self._manual_gui_mode = False
        self._manual_gui_trigger = False
        self._gui_env_idx: int = 0


    @property
    def command(self) -> torch.Tensor:
        """Returns the binary command trigger (shape: [num_envs, 1])."""
        return self.trigger

    @property
    def phase(self) -> torch.Tensor:
        """Returns the motion phase in [0.0, 1.0] when active, or 0.0 when settling."""
        norm_phase = (self.motion_frame.float() / max(self.num_frames, 1)).unsqueeze(-1)
        return torch.where(self.trigger > 0.5, norm_phase, torch.zeros_like(self.trigger))

    def get_ref_joint_pos(self) -> torch.Tensor:
        mocap_pos = self.motion_lib.get_motion_data("joint_pos", self.motion_frame)
        return torch.where(self.trigger > 0.5, mocap_pos, self.default_pos)

    def get_ref_joint_vel(self) -> torch.Tensor:
        mocap_vel = self.motion_lib.get_motion_data("joint_vel", self.motion_frame)
        return torch.where(self.trigger > 0.5, mocap_vel, torch.zeros_like(mocap_vel))

    def get_ref_future_joint_pos(self, horizon: int = 10) -> torch.Tensor:
        future_frame = torch.clamp(self.motion_frame + horizon, max=self.num_frames - 1)
        mocap_future_pos = self.motion_lib.get_motion_data("joint_pos", future_frame)
        return torch.where(self.trigger > 0.5, mocap_future_pos, self.default_pos)

    def get_ref_body_pos_w(self) -> torch.Tensor:
        mocap_body_pos = self.motion_lib.get_motion_data("body_pos_w", self.motion_frame)
        return mocap_body_pos + self._env.scene.env_origins.unsqueeze(1)

    def get_ref_body_quat_w(self) -> torch.Tensor:
        return self.motion_lib.get_motion_data("body_quat_w", self.motion_frame)

    def get_ref_body_lin_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_motion_data("body_lin_vel_w", self.motion_frame)

    def get_ref_body_ang_vel_w(self) -> torch.Tensor:
        return self.motion_lib.get_motion_data("body_ang_vel_w", self.motion_frame)

    
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Handles state transitions when time_left expires or upon environment reset."""
        if len(env_ids) == 0:
            return

        is_reset = (self.command_counter[env_ids] == 0)
        reset_ids = env_ids[is_reset]
        step_ids = env_ids[~is_reset]

        # Case A: Environment Reset -> Start in Settle/Stand (or optional RSI)
        if len(reset_ids) > 0:
            if self.cfg.prob_rsi > 0.0:
                rsi_mask = torch.rand(len(reset_ids), device=self.device) < self.cfg.prob_rsi
                # RSI envs start directly mid-motion
                if rsi_mask.any():
                    rsi_env_ids = reset_ids[rsi_mask]
                    self.trigger[rsi_env_ids] = 1.0
                    self.motion_frame[rsi_env_ids] = torch.randint(
                        0, self.num_frames - 1, (len(rsi_env_ids),), device=self.device
                    )
                    rem_steps = (self.num_frames - self.motion_frame[rsi_env_ids]).float()
                    self.time_left[rsi_env_ids] = rem_steps * self.step_dt

                # Non-RSI reset envs start settling/standing
                settle_env_ids = reset_ids[~rsi_mask]
                if len(settle_env_ids) > 0:
                    self.trigger[settle_env_ids] = 0.0
                    self.motion_frame[settle_env_ids] = 0
                    self.time_left[settle_env_ids] = self.time_left[settle_env_ids].uniform_(
                        *self.cfg.settle_time_range
                    )
            else:
                self.trigger[reset_ids] = 0.0
                self.motion_frame[reset_ids] = 0
                self.time_left[reset_ids] = self.time_left[reset_ids].uniform_(
                    *self.cfg.settle_time_range
                )

    def _update_command(self) -> None:
        """Called every simulation step to advance active motion frames or handle GUI overrides."""
        # Handle manual GUI control (in Viser viewer)
        if self._manual_gui_mode:
            env_i = self._gui_env_idx
            if self._manual_gui_trigger:
                self.trigger[env_i] = 1.0
                if self.motion_frame[env_i] < self.num_frames - 1:
                    self.motion_frame[env_i] += 1
                else:
                    self.motion_frame[env_i] = 0  # Loop in manual GUI viewer
            else:
                self.trigger[env_i] = 0.0
                self.motion_frame[env_i] = 0
            return

        # Normal Training / Rollout Progression:
        active_mask = (self.trigger[:, 0] > 0.5)
        if active_mask.any():
            self.motion_frame[active_mask] += 1
            self.motion_frame[active_mask] = torch.clamp(
                self.motion_frame[active_mask], max=self.num_frames - 1
            )

        idle_mask = ~active_mask
        if idle_mask.any():
            self.motion_frame[idle_mask] = 0

    def _update_metrics(self) -> None:
        """Update metrics for logging."""
        pass

    # -------------------------------------------------------------------------
    # Viser Interactive GUI Integration
    # -------------------------------------------------------------------------

    def create_gui(
        self,
        name: str,
        server: viser.ViserServer,
        get_env_idx: Callable[[], int],
        on_change: Callable[[], None] | None = None,
        request_action: Callable[[str, Any], None] | None = None,
    ) -> None:
        """Creates interactive controls in Viser web viewer."""
        self._gui_env_idx = get_env_idx()

        mode_cb = server.gui.add_checkbox(
            "Manual Override", initial_value=False
        )
        trigger_btn = server.gui.add_button("Trigger Motion (Hold/Click)")

        @mode_cb.on_update
        def _(event):
            self._manual_gui_mode = event.target.value
            self._gui_env_idx = get_env_idx()

        @trigger_btn.on_click
        def _(event):
            self._manual_gui_mode = True
            self._manual_gui_trigger = not self._manual_gui_trigger
            self._gui_env_idx = get_env_idx()
            trigger_btn.name = "Stop Motion" if self._manual_gui_trigger else "Trigger Motion"
            if on_change:
                on_change()
