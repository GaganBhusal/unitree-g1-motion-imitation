import os
import sys
from pathlib import Path

# Add project root to sys.path so running directly works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np 
from tensordict import TensorDict
import torch
from configs.env_configs import EnvConfigs


class MotionLib:

    def __init__(
        self,
        device: str | torch.device,
        motion_path: str | Path | list[str | Path],
        default_pos: torch.Tensor | list[float],
        transition_frames: int = 50,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.default_pos = torch.as_tensor(default_pos, dtype=torch.float32, device=self.device).view(1, -1)
        self.transition_frames = transition_frames

        if isinstance(motion_path, (str, Path)):
            self.motion_paths = [str(motion_path)]
        else:
            self.motion_paths = [str(p) for p in motion_path]

        self.num_motions = len(self.motion_paths)
        self.motion_names = [Path(p).stem for p in self.motion_paths]

        self.active_cmd = None

        self.load_motion_data()

    def load_motion_data(self) -> None:
        raw_motions = []
        num_frames_list = []

        for p in self.motion_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Motion file not found: {p}")
            npz_file = np.load(p, allow_pickle=True)
            m_dict = {
                "body_pos_w": torch.from_numpy(npz_file["body_pos_w"]).to(self.device, dtype=torch.float32),
                "body_quat_w": torch.from_numpy(npz_file["body_quat_w"]).to(self.device, dtype=torch.float32),
                "joint_pos": torch.from_numpy(npz_file["joint_pos"]).to(self.device, dtype=torch.float32),
                "joint_vel": torch.from_numpy(npz_file["joint_vel"]).to(self.device, dtype=torch.float32),
                "body_lin_vel_w": torch.from_numpy(npz_file["body_lin_vel_w"]).to(self.device, dtype=torch.float32),
                "body_ang_vel_w": torch.from_numpy(npz_file["body_ang_vel_w"]).to(self.device, dtype=torch.float32),
            }
            n_frames = m_dict["joint_pos"].shape[0]
            num_frames_list.append(n_frames)
            raw_motions.append(m_dict)

        self.num_frames_list = num_frames_list
        self.num_frames = torch.tensor(num_frames_list, dtype=torch.long, device=self.device)
        self.max_frames = max(num_frames_list)

        keys = ("body_pos_w", "body_quat_w", "joint_pos", "joint_vel", "body_lin_vel_w", "body_ang_vel_w")
        self.stacked_data = TensorDict({}, batch_size=[])

        for k in keys:
            feat_shape = raw_motions[0][k].shape[1:]
            stacked = torch.zeros(
                (self.num_motions, self.max_frames, *feat_shape),
                dtype=torch.float32,
                device=self.device,
            )
            for m_idx, m_dict in enumerate(raw_motions):
                L = num_frames_list[m_idx]
                stacked[m_idx, :L] = m_dict[k]
                if L < self.max_frames:
                    stacked[m_idx, L:] = m_dict[k][-1:]
            self.stacked_data[k] = stacked

        self.motion_data = {k: self.stacked_data[k][0, :self.num_frames_list[0]] for k in keys}
        self._show_info()

    def get_num_frames(self, motion_idx: int | torch.Tensor = 0) -> int | torch.Tensor:
        if isinstance(motion_idx, torch.Tensor):
            return self.num_frames[motion_idx]
        return self.num_frames_list[motion_idx]

    def _resolve_motion_idx(self, motion_idx: int | torch.Tensor | None, batch_size: int | None = None) -> torch.Tensor:
        if motion_idx is not None:
            if isinstance(motion_idx, int):
                sz = batch_size if batch_size is not None else 1
                return torch.full((sz,), motion_idx, dtype=torch.long, device=self.device)
            return motion_idx.long().to(self.device)
        if self.active_cmd is not None and hasattr(self.active_cmd, "active_motion_idx"):
            return self.active_cmd.active_motion_idx
        sz = batch_size if batch_size is not None else 1
        return torch.zeros(sz, dtype=torch.long, device=self.device)

    def get_motion_data(
        self,
        key: str,
        frames: int | torch.Tensor,
        motion_idx: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        data = self.stacked_data[key]

        if isinstance(frames, torch.Tensor):
            m_idx = self._resolve_motion_idx(motion_idx, batch_size=frames.shape[0])
            if m_idx.shape[0] != frames.shape[0]:
                m_idx = m_idx.expand(frames.shape[0])
            safe_frames = frames % self.num_frames[m_idx]
            return data[m_idx, safe_frames]
        else:
            m_idx = 0 if motion_idx is None else (motion_idx if isinstance(motion_idx, int) else int(motion_idx.item()))
            safe_frame = frames % self.num_frames_list[m_idx]
            return data[m_idx, safe_frame]

    def get_future_motion_data(
        self,
        key: str,
        future_frames: torch.Tensor,
        motion_idx: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        data = self.stacked_data[key]
        m_idx = self._resolve_motion_idx(motion_idx, batch_size=future_frames.shape[0])
        if m_idx.dim() == 1:
            m_idx = m_idx.unsqueeze(1)
        safe_frames = torch.clamp(future_frames, 0, self.max_frames - 1)
        return data[m_idx, safe_frames]



if __name__ == "__main__":
    cfg = EnvConfigs()
    lib = MotionLib(
        device="cpu",
        motion_path=["motion_file/spiderman.npz", "motion_file/dance.npz"],
        default_pos=cfg.default_joints,
    )
    # Test batched query
    m_idx = torch.tensor([0, 1, 0, 1])
    frames = torch.tensor([10, 200, 50, 500])
    j_pos = lib.get_motion_data("joint_pos", frames, m_idx)
    print("Test get_motion_data result shape:", j_pos.shape)
    assert j_pos.shape == (4, 23)
    print("MultiMotionLib self-test PASSED!")

