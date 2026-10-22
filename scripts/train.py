from src.tasks.velocity.velocity_env_cfg_free import make_motion_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from configs.rl_config import get_g1_tracking_ppo_runner_cfg
from dataclasses import asdict
import argparse
import os

MOTION_PATH = r'/home/yayy/My/Codeeeeee/RL_Robotics/G1_Humanoid/Genesis/motion/spiderman.npz' 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="g1_mimic")
    parser.add_argument("-d", "--domain", action="store_true", help="Whether to enable domain randomization")
    parser.add_argument("-r", "--resume", action="store_true", help="Whether to resume training from the last checkpoint")
    parser.add_argument("--ckpt", type=int, default=None, help="Checkpoint number from where to resume training.")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel envs")
    parser.add_argument("--motion_files", nargs="+", default=None, help="List of motion files (1: first, 2: second, etc.)")
    parser.add_argument("--motion_file", type=str, default="motion_file/spiderman.npz", help="Single motion file path")
    args = parser.parse_args()

    exp_name = args.exp_name
    num_envs = args.num_envs
    max_iterations = 15000

    log_dir = f"logs/{exp_name}"

    motion_paths = args.motion_files if args.motion_files is not None else [args.motion_file]
    print(f"[INFO] Training with {len(motion_paths)} motion(s): {motion_paths}")

    env_cfg = make_motion_env_cfg(
        motion_path=motion_paths
    )

    env_cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(env)

    runner_cfg = get_g1_tracking_ppo_runner_cfg(
        max_iterations = max_iterations,
        experiment_name=exp_name
    )
    runner = MjlabOnPolicyRunner(env, asdict(runner_cfg), log_dir=log_dir, device="cuda:0")


    if args.resume:
        if args.ckpt:
            resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
            print(f"Resuming training from checkpoint: {resume_path}")
            runner.load(resume_path)
        else:
            print(f"Try Again!!\nResume path provided but not checkpoint number")
    else:
        print(f"Starting training from scratch. No resume path provided.")

   
    runner.learn(num_learning_iterations=runner_cfg.max_iterations)


if __name__ == "__main__":
    main()