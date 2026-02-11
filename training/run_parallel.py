"""
Single-command parallel PPO training for Pokemon Auto Chess.

Usage:
    python training/run_parallel.py --num-envs 16 --timesteps 1000000

Launches N training servers, runs PPO training across all of them, and
shuts everything down cleanly when training completes or is interrupted.
"""

import argparse
import subprocess
import sys
import os

# Project root is one level up from training/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))

from launch_servers import launch_servers


def main():
    parser = argparse.ArgumentParser(description="Parallel PPO Training")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--base-port", type=int, default=9100)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args, extra_args = parser.parse_known_args()

    # Launch servers
    servers, cleanup = launch_servers(
        num_envs=args.num_envs,
        base_port=args.base_port,
    )

    print(f"\n{'='*50}")
    print(f"All {args.num_envs} servers ready. Starting training.")
    print(f"{'='*50}\n")

    # Run training as a subprocess so cleanup always runs
    train_cmd = [
        sys.executable, "training/train_ppo.py",
        "--num-envs", str(args.num_envs),
        "--base-port", str(args.base_port),
        "--timesteps", str(args.timesteps),
        *extra_args,
    ]

    try:
        train_proc = subprocess.run(train_cmd, cwd=PROJECT_ROOT)
        if train_proc.returncode != 0:
            print(f"\nTraining exited with code {train_proc.returncode}")
        else:
            print(f"\nTraining completed successfully.")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user.")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
