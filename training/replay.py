"""
Verbose Game Replay Script

Loads a trained checkpoint, plays ONE full game against the training server,
and prints every action the agent takes in human-readable format.

Prerequisites:
  1. Start the TypeScript training server:
       npx ts-node app/training/index.ts
  2. Install Python dependencies:
       pip install -r training/requirements.txt
  3. Run:
       python training/replay.py path/to/checkpoint.zip

For each step prints:
  Stage {s} | Turn {n} | Gold: {g} | Board: {b}/{max} | HP: {hp} | Action: {name} | {detail}

After each fight (END_TURN) prints a summary line:
  === FIGHT: Stage {s} | Result: WIN/LOSS/DRAW | HP: {hp} | Rank: {rank} ===

At game end prints final stats.
"""

import argparse
import sys
import time

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Observation layout constants (must match training-config.ts)
# ---------------------------------------------------------------------------
OBS_PLAYER_STATS = 14
OBS_SHOP_SLOTS = 6
OBS_SHOP_FEATURES = 9
OBS_BOARD_SLOTS = 32
OBS_BOARD_FEATURES = 12
OBS_HELD_ITEMS = 10
OBS_SYNERGIES = 31
OBS_GAME_INFO = 7
OBS_OPPONENT_COUNT = 7
OBS_OPPONENT_FEATURES = 10
OBS_PROPOSITION_SLOTS = 6
OBS_PROPOSITION_FEATURES = 7

# Derived offsets
OFF_PLAYER = 0
OFF_SHOP = OFF_PLAYER + OBS_PLAYER_STATS                          # 14
OFF_BOARD = OFF_SHOP + OBS_SHOP_SLOTS * OBS_SHOP_FEATURES         # 68
OFF_ITEMS = OFF_BOARD + OBS_BOARD_SLOTS * OBS_BOARD_FEATURES      # 452
OFF_SYNERGIES = OFF_ITEMS + OBS_HELD_ITEMS                        # 462
OFF_GAME = OFF_SYNERGIES + OBS_SYNERGIES                          # 493
OFF_OPPONENTS = OFF_GAME + OBS_GAME_INFO                          # 500
OFF_PROPOSITIONS = OFF_OPPONENTS + OBS_OPPONENT_COUNT * OBS_OPPONENT_FEATURES  # 570

# Reward thresholds for fight-result inference.
# Base rewards: WIN=+0.6, LOSS=-0.5, DRAW=0.0, plus a +0.12 survival bonus for
# every alive player.  Shaped bonuses (synergy delta, enemy kills, HP
# preservation, interest) add small positive amounts on top.
# Win floor  = 0.6 + 0.12 = 0.72, so >= 0.5 is reliably a win.
# Loss ceil  = -0.5 + 0.12 = -0.38, so <= -0.2 is reliably a loss.
_WIN_THRESHOLD = 0.5
_LOSS_THRESHOLD = -0.2


# ---------------------------------------------------------------------------
# Action decoding
# ---------------------------------------------------------------------------

def _grid_coords(cell: int) -> tuple[int, int]:
    """Convert cell index (0-31) to (x, y) board coordinates."""
    return cell % 8, cell // 8


def decode_action(action: int, obs: np.ndarray) -> tuple[str, str]:
    """Return (action_name, detail) for a given action index.

    Uses the observation vector to enrich BUY actions with the shop slot's
    species index (the best info available without a server-side name table).
    """
    if 0 <= action <= 5:
        slot = action
        # Shop slot species index lives at OFF_SHOP + slot*9 + 1
        species_raw = obs[OFF_SHOP + slot * OBS_SHOP_FEATURES + 1]
        cost_raw = obs[OFF_SHOP + slot * OBS_SHOP_FEATURES + 3]
        cost = int(round(cost_raw * 10))
        if species_raw > 0:
            detail = f"shop[{slot}] species={species_raw:.4f} cost={cost}g"
        else:
            detail = f"shop[{slot}] (empty)"
        return f"BUY slot {slot}", detail

    if action == 6:
        return "REFRESH", "reroll shop"
    if action == 7:
        return "LEVEL UP", "spend 4g for +4 exp"
    if action == 8:
        locked = obs[OFF_PLAYER + 11]  # shopLocked flag
        state = "locked" if locked > 0.5 else "unlocked"
        return "LOCK SHOP", f"toggle (was {state})"
    if action == 9:
        return "END TURN", "fight phase begins"

    if 10 <= action <= 41:
        cell = action - 10
        x, y = _grid_coords(cell)
        label = "board" if y >= 1 else "bench"
        return f"MOVE to ({x},{y})", f"place unit on {label}"

    if 42 <= action <= 73:
        cell = action - 42
        x, y = _grid_coords(cell)
        label = "board" if y >= 1 else "bench"
        # Check if there's a unit at that cell
        species_raw = obs[OFF_BOARD + cell * OBS_BOARD_FEATURES + 1]
        if species_raw > 0:
            stars = int(round(obs[OFF_BOARD + cell * OBS_BOARD_FEATURES + 2] * 3))
            detail = f"unit at ({x},{y}) {label} species={species_raw:.4f} {'*' * stars}"
        else:
            detail = f"({x},{y}) {label}"
        return f"SELL at ({x},{y})", detail

    if 74 <= action <= 79:
        slot = action - 74
        return f"REMOVE SHOP slot {slot}", f"discard shop[{slot}]"

    if 80 <= action <= 85:
        slot = action - 80
        # Proposition info at OFF_PROPOSITIONS + slot*7
        prop_base = OFF_PROPOSITIONS + slot * OBS_PROPOSITION_FEATURES
        species_raw = obs[prop_base]
        item_raw = obs[prop_base + 6]
        parts = []
        if species_raw > 0:
            parts.append(f"species={species_raw:.4f}")
        if item_raw > 0:
            parts.append(f"item={item_raw:.4f}")
        detail = " ".join(parts) if parts else "(empty)"
        return f"PICK proposition {slot}", detail

    if 86 <= action <= 91:
        pair = action - 86
        return f"COMBINE items pair {pair}", f"craft item pair #{pair}"

    return f"UNKNOWN({action})", ""


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def read_player(obs: np.ndarray) -> dict:
    """Extract player stats from observation."""
    return {
        "life": obs[0] * 100,
        "gold": obs[1] * 100,
        "level": obs[2] * 9,
        "board_size": obs[7] * 9,
        "rank": obs[6] * 8,
        "stage": obs[OFF_GAME] * 50,
        "max_team": obs[OFF_GAME + 6] * 9,
    }


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def replay(model_path: str, server_url: str, deterministic: bool = True):
    """Load checkpoint, play one full game, print every action."""

    from sb3_contrib import MaskablePPO
    from pac_env import PokemonAutoChessEnv
    from stable_baselines3.common.monitor import Monitor

    # Connect
    print(f"Connecting to {server_url} ...")
    for attempt in range(15):
        try:
            resp = requests.get(f"{server_url}/health", timeout=2)
            if resp.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        print("ERROR: could not reach training server")
        sys.exit(1)
    print("Server ready.\n")

    # Load model
    print(f"Loading checkpoint: {model_path}")
    env = PokemonAutoChessEnv(server_url=server_url)
    env = Monitor(env)
    model = MaskablePPO.load(model_path)
    print(f"Model loaded  (obs={env.observation_space.shape}, act={env.action_space.n})\n")

    # Reset
    obs, info = env.reset()
    done = False

    total_reward = 0.0
    total_steps = 0
    turn_action_num = 0        # actions within the current turn
    prev_life = info.get("life", 100)
    prev_stage = info.get("stage", 1)
    fights = 0
    wins = 0
    losses = 0
    draws = 0

    header = (
        f"{'Stage':>5} | {'Turn':>4} | {'Gold':>6} | {'Board':>9} | "
        f"{'HP':>5} | {'Action':<24} | Detail"
    )
    print(header)
    print("-" * len(header))

    while not done:
        masks = env.unwrapped.action_masks()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
        action = int(action)

        # Decode action BEFORE stepping (uses current obs for context)
        action_name, detail = decode_action(action, obs)

        # Read pre-step state from info (server provides these every step)
        stage = info.get("stage", prev_stage)
        life = info.get("life", prev_life)
        gold = info.get("money", info.get("gold", 0))
        board_size = info.get("boardSize", 0)
        p = read_player(obs)
        max_team = int(round(p["max_team"]))

        turn_action_num += 1

        print(
            f"  {stage:3d}   | {turn_action_num:4d} | {gold:5.0f}g | "
            f"{board_size:3d}/{max_team:<3d}   | {life:5.0f} | "
            f"{action_name:<24} | {detail}"
        )

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        done = terminated or truncated

        # After END_TURN the fight runs synchronously; detect result
        if action == 9:  # END_TURN
            new_life = info.get("life", life)
            new_stage = info.get("stage", stage)
            new_rank = info.get("rank", "?")
            fights += 1

            # Infer fight result from reward
            if reward >= _WIN_THRESHOLD:
                result = "WIN"
                wins += 1
            elif reward <= _LOSS_THRESHOLD:
                result = "LOSS"
                losses += 1
            else:
                result = "DRAW"
                draws += 1

            print(
                f"  === FIGHT: Stage {stage} | Result: {result} "
                f"(r={reward:+.3f}) | HP: {new_life:.0f} | Rank: {new_rank} ==="
            )
            print()

            prev_life = new_life
            prev_stage = new_stage
            turn_action_num = 0  # reset for next turn

    # ----- Game over summary -----
    final_rank = info.get("rank", "?")
    final_stage = info.get("stage", "?")
    final_life = info.get("life", 0)

    print()
    print("=" * 60)
    print("GAME OVER")
    print("=" * 60)
    print(f"  Final Rank:     {final_rank}")
    print(f"  Final Stage:    {final_stage}")
    print(f"  Final HP:       {final_life:.0f}")
    print(f"  Total Steps:    {total_steps}")
    print(f"  Total Reward:   {total_reward:+.3f}")
    print(f"  Fights:         {fights}  (W:{wins} / L:{losses} / D:{draws})")
    if fights > 0:
        print(f"  Win Rate:       {wins / fights:.1%}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay one full game from a trained checkpoint with verbose action logging."
    )
    parser.add_argument(
        "checkpoint",
        help="Path to MaskablePPO checkpoint (e.g. training/checkpoints/phase_A_92act_50000_steps.zip)",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:9100",
        help="Training server URL (default: http://localhost:9100)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of deterministic argmax",
    )

    args = parser.parse_args()
    replay(args.checkpoint, args.server_url, deterministic=not args.stochastic)
