import os
import sys
import glob
import argparse
from pathlib import Path
from dataclasses import asdict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tasks.velocity.velocity_env_cfg_free import make_motion_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from configs.rl_config import get_g1_tracking_ppo_runner_cfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def find_latest_checkpoint(log_dir: str) -> str:
    ckpts = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")
    ckpts.sort(key=lambda x: int(Path(x).stem.split("_")[1]))
    return ckpts[-1]


def play():
    parser = argparse.ArgumentParser(description="Test G1 humanoid multi-motion command imitation.")
    parser.add_argument("-e", "--exp_name", type=str, default="g1_spiderman_free", help="Experiment name (log dir)")
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint iteration number (default: latest)")
    parser.add_argument("--motion_files", nargs="+", default=None, help="List of motion files (1: first, 2: second, etc.)")
    parser.add_argument("--motion_file", type=str, default="motion_file/spiderman.npz", help="Single motion file path")
    parser.add_argument("--viewer", choices=["native", "viser"], default="viser", help="Viewer backend")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log_dir = f"logs/{args.exp_name}"

    if args.ckpt is not None:
        ckpt_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    else:
        ckpt_path = find_latest_checkpoint(log_dir)
    print(f"[INFO] Checkpoint: {ckpt_path}")

    motion_paths = args.motion_files if args.motion_files is not None else [args.motion_file]
    print(f"[INFO] Motions loaded ({len(motion_paths)}): {motion_paths}")

    env_cfg = make_motion_env_cfg(motion_path=motion_paths, device=device)
    env_cfg.scene.num_envs = 1

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env)

    runner_cfg = get_g1_tracking_ppo_runner_cfg(experiment_name=args.exp_name, max_iterations=1)
    runner = MjlabOnPolicyRunner(vec_env, asdict(runner_cfg), log_dir=log_dir, device=device)
    runner.load(str(ckpt_path), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    cmd = env.command_manager.get_term("motion_command")
    cmd.cfg.prob_rsi = 0.0
    cmd.auto_loop = False
    cmd.reset_to_standing()

    if args.viewer == "native":
        def on_key_press(key: int):
            from mjlab.viewer.native.keys import KEY_0, KEY_KP_0, KEY_A, KEY_B

            if key in (KEY_A, ord("a"), ord("A")):
                cmd.trigger_motion(1)
            elif key in (KEY_B, ord("b"), ord("B")):
                cmd.trigger_motion(2)
            elif key in (KEY_0, KEY_KP_0, ord("0")):
                cmd.reset_to_standing()

        NativeMujocoViewer(vec_env, policy, key_callback=on_key_press).run()
    else:
        print(f"\n  Open http://localhost:8080 in your browser")
        ViserPlayViewer(vec_env, policy).run()

    vec_env.close()


if __name__ == "__main__":
    play()
