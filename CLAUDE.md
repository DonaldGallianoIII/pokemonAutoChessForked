# CLAUDE.md

## Project Overview

Pokemon Auto Chess — a browser-based auto-battler built with TypeScript (game server) and Python (ML training pipeline). The training pipeline uses gymnasium environments that communicate with Node.js game servers over HTTP.

## Repository Structure

- `app/` — TypeScript game server and client code
- `training/` — Python ML training pipeline (PPO via sb3-contrib)
  - `pac_env.py` — Single-agent gym environment (SubprocVecEnv compatible)
  - `selfplay_vec_env.py` — 8-player self-play VecEnv wrapper
  - `train_ppo.py` — PPO training script with MaskablePPO
  - `launch_servers.py` — Multi-server launcher for parallel training
  - `smoke_test.py` — End-to-end smoke tests
  - `replay.py` — Game replay utility

## Communication Style

When completing a task, always end with a summary listing:
- Which files were modified and why (the user downloads files from GitHub individually for local test bench spot-replacement)
- Any notable design decisions or edge cases handled
- Timeout/retry constants or other tunable values that were chosen

## Key Technical Context

- Training envs communicate with Node.js game servers via HTTP (`requests` library)
- All HTTP calls must have explicit `timeout=` parameters (15s for step/reset, 10s for health/init)
- `pac_env.py` is used inside `SubprocVecEnv` — a timeout in one env blocks ALL envs, so step() timeouts return a terminal state instead of raising
- `selfplay_vec_env.py` manages 8 player seats in one game via `/step-multi` batched calls
- MaskablePPO requires valid `action_masks` in info dicts at all times, including error/timeout paths

## Development Notes

- Do NOT modify TypeScript game server code when fixing training pipeline issues
- Do NOT add new Python dependencies without discussion
- Do NOT change PPO hyperparameters or SubprocVecEnv/VecEnv setup logic unless explicitly asked
- Health check endpoints already have 2s timeouts in `launch_servers.py` and `train_ppo.py`

## Training Progress & Tuning History

### Current best checkpoint: `current.zip`
- **Mean Rank: 2.82** | Win Rate: 14% | Top 4: 80% | Mean Stage: 26.4 (50-game eval vs 7 bots)
- Trained with 92-action space, auto-place OFF (agent controls unit placement)

### Placement rewards (in `training-config.ts`)
- Switched from linear formula `(9-rank)*SCALE - OFFSET` to a lookup table (`REWARD_PLACEMENT_TABLE`)
- Current values: 1st: +20, 2nd: +13, 3rd: +8, 4th: 0, 5th: -4, 6th: -9, 7th: -14, 8th: -20
- Only top-3 is rewarded; 4th is break-even. This was tuned to break a stalling pattern where the agent coasted in 2nd/3rd
- The wide spread (+20 to -20 = 40 range) means placement dominates over per-round shaped rewards (~0.02–0.75 range). If the agent stops caring about economy/synergy mid-game, bump the per-round rewards

### Learning rate findings
- **1e-4**: Stable, produced the current best checkpoint
- **3e-4**: Too aggressive — lost ~2 ranking places, avoid for fine-tuning
- **8e-5**: Selected for next overnight run (conservative fine-tune with new steeper rewards)
- When resuming from checkpoint, MUST set both `model.learning_rate` AND `model.lr_schedule = lambda _: lr` — SB3 ignores the scalar and uses the schedule callable internally

### Evaluation command
```
python training\run_parallel.py --num-envs 8 --evaluate "<path_to_checkpoint.zip>" --eval-games 50
```
