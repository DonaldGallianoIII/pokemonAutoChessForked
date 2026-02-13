# CLAUDE.md

## Project Overview

Pokemon Auto Chess — a browser-based auto-battler built with TypeScript (game server) and Python (ML training pipeline). The training pipeline uses gymnasium environments that communicate with Node.js game servers over HTTP.

## Repository Structure

- `app/` — TypeScript game server and client code
- `training/` — Python ML training pipeline (PPO via sb3-contrib)
  - `pac_env.py` — Single-agent gym environment (SubprocVecEnv compatible)
  - `selfplay_vec_env.py` — Multi-agent VecEnv wrapper (N RL agents per game, auto-detected from server)
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
- `selfplay_vec_env.py` manages N RL agent seats in one game via `/step-multi` batched calls (N auto-detected from server's `/num-agents` endpoint)
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

### Hybrid training mode (N RL agents + bots)
- **Purpose**: Transition step between single-agent (1v7) and full self-play (8v0)
- Agent trains against both itself AND bots, getting exposure to self-play dynamics without fully losing bot curriculum signal
- Controlled by two env vars / CLI flags:
  - **Server side**: `NUM_RL_AGENTS=2` env var (set before launching servers). Default=1 (unchanged classic mode)
  - **Python side**: `--num-agents 2` CLI flag on `train_ppo.py`. Must match the server's NUM_RL_AGENTS
- When `NUM_RL_AGENTS=1` (default), everything works exactly as before — zero behavioral change
- When `NUM_RL_AGENTS=2..7`, the server creates N RL agents + (8-N) bots, uses `/step-multi`
- `SelfPlayVecEnv` auto-detects N from the server's `/num-agents` endpoint
- Checkpoint prefix includes agent count: `phaseA_92act_2rl_*` for easy identification
- **How to launch hybrid 2v6**:
  ```bash
  # Terminal 1: server (2 RL agents + 6 bots)
  NUM_RL_AGENTS=2 npx ts-node app/training/training-server-main.ts
  # Terminal 2: trainer
  python training/train_ppo.py --num-agents 2 --resume training/checkpoints/current.zip
  ```
- **To revert to classic 1v7**: just omit the env vars (defaults to NUM_RL_AGENTS=1, --num-agents 1)
- Design notes:
  - `playerIds` contains ONLY RL agent IDs; bots are auto-ended each round in `resetTurnState()`
  - `stepBatch()` receives N actions (one per RL agent), bots don't participate in the action loop
  - Fight triggers when ALL alive players have ended turn (bots start pre-ended)
  - `advanceToNextPickPhase()` uses `player.isBot` checks — works identically in hybrid and single-agent mode

### Evaluation command
```
python training\run_parallel.py --num-envs 8 --evaluate "<path_to_checkpoint.zip>" --eval-games 50
```
