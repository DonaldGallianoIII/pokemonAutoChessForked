"""
PPO Training Script for Pokemon Auto Chess

Trains a MaskablePPO agent using sb3-contrib with native action masking.
The environment exposes 92 discrete actions and 612 observation features,
aligned 1:1 with the agent-io browser extension API.

Prerequisites:
  1. Start the TypeScript training server:
       npx ts-node app/training/index.ts
  2. Install Python dependencies:
       pip install -r training/requirements.txt
  3. Run this script:
       python training/train_ppo.py

The agent learns to:
  - Buy pokemon from the shop (economy management)
  - Build team compositions (synergy optimization)
  - Level up at the right time (tempo decisions)
  - Sell underperforming units (resource recycling)
  - Pick item/pokemon propositions (carousel & PVE rewards)
  - Combine items, move units, lock/unlock shop

Architecture:
  MaskablePPO with 256×256 MLP. Native action masking via env.action_masks().
  612 input features → 256 → 256 → 92 actions.
"""

import argparse
import os
import time
from typing import Optional

import numpy as np
import requests
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from launch_servers import SERVER_CMD, PROJECT_ROOT
from pac_env import PokemonAutoChessEnv
from eval_metrics import FightSnapshot, GameMetrics, format_flags


class TrainingMetricsCallback(BaseCallback):
    """
    Logs custom training metrics to TensorBoard.

    Three tiers of metrics:
    1. training/*   — core episode stats (reward, rank, stage, length)
    2. reward/*     — per-signal reward breakdown (from rewardBreakdown in info)
    3. behavior/*   — behavioral aggregates (gold patterns, star quality, synergies)
    4. ratio/*      — balance ratios (shaped vs placement, positive vs negative)

    Game length is critical for detecting degenerate self-play equilibria:
    - Sudden drop → agents learned to suicide early
    - Sudden spike → agents are stalling / all-padding
    """

    LOG_EVERY = 10  # log every N episodes

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_ranks: list[int] = []
        self.episode_stages: list[int] = []
        self.episode_gold: list[float] = []
        self.episode_board_size: list[int] = []
        self.episode_synergy_count: list[int] = []
        self.episode_items_held: list[int] = []
        self._total_episodes = 0

        # Per-signal reward breakdowns (last N episodes)
        self._reward_breakdowns: list[dict[str, float]] = []
        # Behavioral aggregates from GameMetrics (last N episodes)
        self._behavior_aggregates: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue

            self.episode_rewards.append(info["episode"]["r"])
            self.episode_lengths.append(info["episode"]["l"])
            self._total_episodes += 1

            if "rank" in info:
                self.episode_ranks.append(info["rank"])
            if "stage" in info:
                self.episode_stages.append(info["stage"])
            if "gold" in info:
                self.episode_gold.append(info["gold"])
            if "boardSize" in info:
                self.episode_board_size.append(info["boardSize"])
            if "synergyCount" in info:
                self.episode_synergy_count.append(info["synergyCount"])
            if "itemsHeld" in info:
                self.episode_items_held.append(info["itemsHeld"])

            # Collect reward breakdown
            bd = info.get("rewardBreakdown")
            if bd:
                self._reward_breakdowns.append(dict(bd))

        # Log every N episodes
        if len(self.episode_rewards) < self.LOG_EVERY:
            return True

        n = self.LOG_EVERY

        # ── Core training stats ──────────────────────────────
        self.logger.record("training/mean_reward", np.mean(self.episode_rewards[-n:]))
        self.logger.record("training/mean_game_length", np.mean(self.episode_lengths[-n:]))
        if self.episode_ranks:
            self.logger.record("training/mean_rank", np.mean(self.episode_ranks[-n:]))
        if self.episode_stages:
            self.logger.record("training/mean_final_stage", np.mean(self.episode_stages[-n:]))
        if self.episode_gold:
            self.logger.record("training/mean_gold", np.mean(self.episode_gold[-n:]))
        if self.episode_board_size:
            self.logger.record("training/mean_board_size", np.mean(self.episode_board_size[-n:]))
        if self.episode_synergy_count:
            self.logger.record("training/synergy_count", np.mean(self.episode_synergy_count[-n:]))
        if self.episode_items_held:
            self.logger.record("training/items_held", np.mean(self.episode_items_held[-n:]))
        self.logger.record("training/total_episodes", self._total_episodes)

        # ── Per-signal reward breakdown ──────────────────────
        if self._reward_breakdowns:
            recent = self._reward_breakdowns[-n:]
            all_keys = set(k for bd in recent for k in bd)
            total_positive = 0.0
            total_negative = 0.0
            total_shaped = 0.0
            placement_sum = 0.0

            for key in sorted(all_keys):
                vals = [bd.get(key, 0.0) for bd in recent]
                mean_val = float(np.mean(vals))
                self.logger.record(f"reward/{key}", mean_val)

                # Accumulate for ratio computation
                if mean_val > 0:
                    total_positive += mean_val
                else:
                    total_negative += mean_val

                if key == "placement":
                    placement_sum = mean_val
                else:
                    total_shaped += mean_val

            # ── Balance ratios ───────────────────────────────
            if abs(placement_sum) > 0.01:
                self.logger.record("ratio/shaped_vs_placement", total_shaped / abs(placement_sum))
            if abs(total_negative) > 0.01:
                self.logger.record("ratio/positive_vs_negative", total_positive / abs(total_negative))

        return True


def wait_for_server(url: str, timeout: int = 60):
    """Wait for the training server to become available."""
    print(f"Waiting for training server at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                print("Training server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise TimeoutError(
        f"Training server at {url} did not become available within {timeout}s"
    )


def benchmark_env(url: str):
    """Run a benchmark game and report speed."""
    print("\nRunning benchmark (full game with random actions)...")
    try:
        resp = requests.post(f"{url}/benchmark", timeout=30).json()
    except Exception as e:
        print(f"  WARNING: Benchmark failed ({type(e).__name__}: {e}), skipping.")
        return None
    print(f"  Steps:          {resp['steps']}")
    print(f"  Elapsed:        {resp['elapsedMs']}ms")
    print(f"  Steps/sec:      {resp['stepsPerSecond']:.0f}")
    print(f"  Final rank:     {resp['finalRank']}")
    print(f"  Final stage:    {resp['finalStage']}")
    print(f"  Final reward:   {resp['finalReward']:.2f}")
    print()
    return resp


def make_env(server_url: str):
    """Create and wrap the training environment."""
    env = PokemonAutoChessEnv(server_url=server_url)
    env = Monitor(env)
    return env


def train(
    server_url: str = "http://localhost:9100",
    total_timesteps: int = 500_000,
    learning_rate: float = 2e-4,
    n_steps: int = 2048,
    batch_size: int = 512,
    n_epochs: int = 5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.05,
    save_dir: str = "training/checkpoints",
    log_dir: str = "training/logs",
    resume_from: Optional[str] = None,
    phase: str = "A",
    num_envs: int = 1,
    base_port: int = 9100,
):
    """
    Train a MaskablePPO agent on Pokemon Auto Chess.

    Args:
        server_url: URL of the TypeScript training server (used when num_envs=1)
        total_timesteps: Total training timesteps
        learning_rate: PPO learning rate
        n_steps: Steps per rollout buffer collection
        batch_size: Minibatch size for PPO updates
        n_epochs: Number of optimization epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        clip_range: PPO clipping parameter
        ent_coef: Entropy coefficient (encourages exploration)
        save_dir: Directory for model checkpoints
        log_dir: Directory for TensorBoard logs
        resume_from: Path to checkpoint to resume from
        phase: Training phase label for checkpoint naming.
               "A" = dummy bots (curriculum), "B" = self-play.
               When transitioning A→B, pass --resume with Phase A checkpoint.
        num_envs: Number of parallel environments (1 = single server, >1 = SubprocVecEnv)
        base_port: Base port for training servers (used when num_envs > 1)
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Checkpoint prefix encodes phase + action space for rollback clarity.
    checkpoint_prefix = f"phase{phase}_92act"

    if num_envs > 1:
        from stable_baselines3.common.vec_env import SubprocVecEnv

        print(f"Using {num_envs} parallel environments on ports {base_port}-{base_port + num_envs - 1}")

        # Wait for all servers to be healthy
        for i in range(num_envs):
            port = base_port + i
            wait_for_server(f"http://localhost:{port}")

        # Benchmark on the first server only
        benchmark_env(f"http://localhost:{base_port}")

        # SubprocVecEnv requires factory functions (closures), not env instances
        def make_parallel_env(port):
            def _init():
                env = PokemonAutoChessEnv(
                    server_url=f"http://localhost:{port}",
                    server_cmd=SERVER_CMD,
                    server_cwd=PROJECT_ROOT,
                )
                env = Monitor(env)
                return env
            return _init

        env = SubprocVecEnv(
            [make_parallel_env(base_port + i) for i in range(num_envs)]
        )
    else:
        print(f"Using single environment on port {base_port}")

        # Wait for server
        wait_for_server(server_url)

        # Run benchmark
        benchmark_env(server_url)

        # Create environment
        env = make_env(server_url)

    # Create or load model
    # SB3 saves checkpoints as .zip; MaskablePPO.load() auto-appends .zip,
    # but we must also check for it here so the guard doesn't fall through.
    if resume_from and (os.path.exists(resume_from) or os.path.exists(resume_from + ".zip")):
        print(f"Resuming from checkpoint: {resume_from}")
        model = MaskablePPO.load(resume_from, env=env)
        # Override saved hyperparameters with CLI values
        # (SB3 .load() restores the checkpoint's saved LR, ignoring CLI args)
        # Must set both: learning_rate (the stored value) AND lr_schedule
        # (the callable SB3 actually uses each update step).
        model.learning_rate = learning_rate
        model.lr_schedule = lambda _: learning_rate
        print(f"  LR overridden to: {learning_rate}")
    else:
        if resume_from:
            print(f"WARNING: --resume path not found: {resume_from} (also tried {resume_from}.zip)")
        print("Creating new MaskablePPO model...")
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log=log_dir,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            ),
        )

    # Callbacks
    # Save every 50k steps with a descriptive prefix for staged training.
    # When transitioning Phase A → B, you can roll back to a Phase A checkpoint
    # if self-play destabilizes.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, 50_000 // num_envs),
        save_path=save_dir,
        name_prefix=checkpoint_prefix,
    )
    metrics_cb = TrainingMetricsCallback()

    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"  Phase:          {phase} ({checkpoint_prefix})")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Batch size:     {batch_size}")
    print(f"  N steps:        {n_steps}")
    print(f"  Entropy coef:   {ent_coef}")
    print(f"  Checkpoints:    {save_dir}/{checkpoint_prefix}_*")
    print(f"  TensorBoard:    {log_dir}")
    print(f"\nMonitor training with: tensorboard --logdir {log_dir}")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, metrics_cb],
        progress_bar=True,
    )

    # Save final model with phase prefix
    final_path = os.path.join(save_dir, f"{checkpoint_prefix}_final")
    model.save(final_path)
    print(f"\nTraining complete! Model saved to {final_path}")

    return model


def evaluate(model_path: str, server_url: str, n_games: int = 20):
    """Evaluate a trained model over multiple games with behavioral analysis."""
    env = make_env(server_url)
    model = MaskablePPO.load(model_path)

    ranks = []
    rewards = []
    stages = []
    all_breakdowns: list[dict] = []
    all_aggregates: list[dict] = []
    all_flags: list[list[tuple[str, str]]] = []

    for i in range(n_games):
        obs, info = env.reset()
        total_reward = 0
        done = False
        prev_breakdown: dict[str, float] = {}
        game_metrics = GameMetrics()
        prev_stage = info.get("stage", 1)
        last_action = -1

        while not done:
            action_masks = env.unwrapped.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            action = int(action)
            last_action = action
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # After END_TURN, capture a fight snapshot
            if action == 9:  # END_TURN
                cumulative = info.get("rewardBreakdown", {})
                fight_delta = {}
                for key, val in cumulative.items():
                    delta = val - prev_breakdown.get(key, 0.0)
                    if abs(delta) >= 0.0005:
                        fight_delta[key] = delta
                prev_breakdown = dict(cumulative)

                snapshot = FightSnapshot(info, reward_delta=fight_delta)
                game_metrics.add_fight(snapshot)

        # Game ended
        game_metrics.set_end(
            gold=info.get("money", info.get("gold", 0)),
            rank=info.get("rank", 8),
        )

        rank = info.get("rank", 8)
        stage = info.get("stage", 0)
        ranks.append(rank)
        rewards.append(total_reward)
        stages.append(stage)
        all_breakdowns.append(info.get("rewardBreakdown", {}))

        flags = game_metrics.generate_flags()
        all_flags.append(flags)
        aggregates = game_metrics.compute_aggregate()
        all_aggregates.append(aggregates)

        # Per-game one-liner
        warn_count = sum(1 for sev, _ in flags if sev == "WARN")
        flag_summary = f" [{warn_count} WARN]" if warn_count > 0 else ""
        print(
            f"  Game {i+1}/{n_games}: Rank={rank}, "
            f"Stage={stage}, Reward={total_reward:.2f}{flag_summary}"
        )

    # ── Summary ──────────────────────────────────────────────
    print(f"\nEvaluation Results ({n_games} games):")
    print(f"  Mean Rank:    {np.mean(ranks):.2f} +/- {np.std(ranks):.2f}")
    print(f"  Mean Reward:  {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Mean Stage:   {np.mean(stages):.1f}")
    print(f"  Win Rate:     {sum(1 for r in ranks if r == 1) / n_games:.1%}")
    print(f"  Top 4 Rate:   {sum(1 for r in ranks if r <= 4) / n_games:.1%}")

    # ── Mean Reward Breakdown ────────────────────────────────
    if all_breakdowns:
        all_keys = sorted(set(k for bd in all_breakdowns for k in bd))
        print(f"\n  --- Mean Reward Breakdown (per game, {n_games} games) ---")
        total_mean = 0.0
        for key in all_keys:
            vals = [bd.get(key, 0.0) for bd in all_breakdowns]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            if abs(mean_val) < 0.001 and std_val < 0.001:
                continue
            marker = "+" if mean_val > 0 else " "
            print(f"    {key:<24} {marker}{mean_val:>8.2f}  +/- {std_val:>6.2f}")
            total_mean += mean_val
        print(f"    {'─' * 47}")
        marker = "+" if total_mean > 0 else " "
        print(f"    {'TOTAL':<24} {marker}{total_mean:>8.2f}")

    # ── Behavioral Aggregates ────────────────────────────────
    if all_aggregates:
        agg_keys = sorted(set(k for agg in all_aggregates for k in agg))
        if agg_keys:
            print(f"\n  --- Behavioral Aggregates (mean across {n_games} games) ---")
            for key in agg_keys:
                vals = [agg.get(key, 0.0) for agg in all_aggregates if key in agg]
                if not vals:
                    continue
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                print(f"    {key:<28} {mean_val:>8.3f}  +/- {std_val:>6.3f}")

    # ── Behavioral Flag Frequency ────────────────────────────
    flag_counts: dict[str, int] = {}
    for flags in all_flags:
        for sev, msg in flags:
            # Normalize: strip stage numbers for counting
            tag = f"[{sev}] {msg.split(' at stage')[0].split(' for ')[0]}"
            flag_counts[tag] = flag_counts.get(tag, 0) + 1
    if flag_counts:
        print(f"\n  --- Behavioral Flag Frequency ({n_games} games) ---")
        for tag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            pct = count / n_games * 100
            print(f"    {tag:<50} {count:>3}/{n_games}  ({pct:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training for Pokemon Auto Chess")
    parser.add_argument("--server-url", default="http://localhost:9100", help="Training server URL")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--n-steps", type=int, default=1024, help="Steps per rollout")
    parser.add_argument("--ent-coef", type=float, default=0.10, help="Entropy coefficient")
    parser.add_argument("--save-dir", default="training/checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", default="training/logs", help="TensorBoard log directory")
    parser.add_argument("--run-name", default=None, help="Run name for namespacing logs/checkpoints (creates subdirs under save-dir and log-dir)")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument(
        "--phase", default="A", choices=["A", "B"],
        help="Training phase: A=dummy bots (curriculum), B=self-play. "
             "Affects checkpoint naming. When going A→B, use --resume with Phase A checkpoint."
    )
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--base-port", type=int, default=9100, help="Base port for training servers")
    parser.add_argument("--evaluate", default=None, help="Evaluate model at path instead of training")
    parser.add_argument("--eval-games", type=int, default=20, help="Number of evaluation games")
    parser.add_argument("--benchmark", action="store_true", help="Just run benchmark")

    args = parser.parse_args()

    # --run-name creates subdirectories so parallel runs don't collide
    save_dir = args.save_dir
    log_dir = args.log_dir
    if args.run_name:
        save_dir = os.path.join(save_dir, args.run_name)
        log_dir = os.path.join(log_dir, args.run_name)

    if args.benchmark:
        wait_for_server(args.server_url)
        benchmark_env(args.server_url)
    elif args.evaluate:
        wait_for_server(args.server_url)
        evaluate(args.evaluate, args.server_url, args.eval_games)
    else:
        train(
            server_url=args.server_url,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            ent_coef=args.ent_coef,
            save_dir=save_dir,
            log_dir=log_dir,
            resume_from=args.resume,
            phase=args.phase,
            num_envs=args.num_envs,
            base_port=args.base_port,
        )
