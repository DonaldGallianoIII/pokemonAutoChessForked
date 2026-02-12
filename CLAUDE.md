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
- All HTTP calls must have explicit `timeout=` parameters (30s for step/reset, 10s for health/init)
- `pac_env.py` is used inside `SubprocVecEnv` — a timeout in one env blocks ALL envs, so step() timeouts return a terminal state instead of raising
- `selfplay_vec_env.py` manages 8 player seats in one game via `/step-multi` batched calls
- MaskablePPO requires valid `action_masks` in info dicts at all times, including error/timeout paths

## Development Notes

- Do NOT modify TypeScript game server code when fixing training pipeline issues
- Do NOT add new Python dependencies without discussion
- Do NOT change PPO hyperparameters or SubprocVecEnv/VecEnv setup logic unless explicitly asked
- Health check endpoints already have 2s timeouts in `launch_servers.py` and `train_ppo.py`
