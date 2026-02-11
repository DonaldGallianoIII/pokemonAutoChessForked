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
  Stage {s} | Turn {n} | Lv {lv} | Gold: {g} | Board: {b}/{max} | HP: {hp} | Action: {name} | {detail}

After each fight (END_TURN) prints board state, synergies, and a fight summary.
At game end prints final stats.
"""

import argparse
import sys
import time

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Reward thresholds for fight-result inference.
# Base rewards: WIN=+0.6, LOSS=-0.5, DRAW=0.0, plus a +0.12 survival bonus.
# Any positive reward is a win, any negative is a loss, zero is a draw.
# ---------------------------------------------------------------------------
_WIN_THRESHOLD = 0.0
_LOSS_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _star_str(stars: int) -> str:
    """Render star level as unicode stars."""
    return "\u2605" * stars


def _fmt_unit(u: dict, show_pos: bool = True) -> str:
    """Format a single unit dict from info.board / info.bench."""
    name = u["name"]
    stars = _star_str(u.get("stars", 1))
    items = u.get("items", [])
    parts = [f"{name} {stars}"]
    if show_pos:
        parts.append(f"({u['x']},{u['y']})")
    else:
        parts.append(f"(bench {u['x']})")
    if items:
        parts.append(f"[{', '.join(items)}]")
    return " ".join(parts)


def _print_board_state(info: dict):
    """Print a full snapshot of the agent's board, bench, shop, synergies, items."""
    level = info.get("level", "?")
    exp = info.get("exp", 0)
    exp_needed = info.get("expNeeded", 0)
    max_team = info.get("maxTeamSize", "?")
    board = info.get("board", [])
    bench = info.get("bench", [])
    shop = info.get("shop", [])
    synergies = info.get("synergies", [])
    items = info.get("items", [])

    print(f"  --- Board State (Lv {level}, exp {exp}/{exp_needed}, team cap {max_team}) ---")

    # Board units grouped by row (y=3 front, y=1 back)
    if board:
        by_row: dict[int, list[dict]] = {}
        for u in board:
            by_row.setdefault(u["y"], []).append(u)
        for y in sorted(by_row.keys(), reverse=True):
            row_label = {3: "front", 2: "mid  ", 1: "back "}.get(y, f"y={y}  ")
            units_str = "  ".join(_fmt_unit(u) for u in sorted(by_row[y], key=lambda u: u["x"]))
            print(f"    {row_label}: {units_str}")
    else:
        print("    (empty board)")

    # Bench
    if bench:
        bench_str = "  ".join(_fmt_unit(u, show_pos=False) for u in sorted(bench, key=lambda u: u["x"]))
        print(f"    bench: {bench_str}")

    # Active synergies
    if synergies:
        syn_parts = []
        for s in synergies:
            active = s["count"] >= s["threshold"] and s["threshold"] > 0
            marker = "+" if active else " "
            syn_parts.append(f"{s['name']}:{s['count']}/{s['threshold']}{marker}")
        print(f"    synergies: {', '.join(syn_parts)}")

    # Held items
    if items:
        print(f"    items: {', '.join(items)}")

    # Shop
    shop_parts = []
    for i, s in enumerate(shop):
        shop_parts.append(s if s else "---")
    print(f"    shop: [{' | '.join(shop_parts)}]")

    print()


def _print_opponent(info: dict):
    """Print the opponent's team the agent just fought against."""
    opp = info.get("opponent")
    if not opp:
        return

    name = opp.get("name", "???")
    opp_level = opp.get("level", "?")
    opp_life = opp.get("odLife", "?")
    opp_board = opp.get("board", [])

    print(f"  --- Enemy: {name} (Lv {opp_level}, HP {opp_life}) ---")

    if opp_board:
        by_row: dict[int, list[dict]] = {}
        for u in opp_board:
            by_row.setdefault(u["y"], []).append(u)
        for y in sorted(by_row.keys(), reverse=True):
            row_label = {3: "front", 2: "mid  ", 1: "back "}.get(y, f"y={y}  ")
            units_str = "  ".join(_fmt_unit(u) for u in sorted(by_row[y], key=lambda u: u["x"]))
            print(f"    {row_label}: {units_str}")
    else:
        if opp.get("level", 0) == 0:
            print("    (PVE encounter)")
        else:
            print("    (no board data)")

    print()


def _decode_action(action: int, info: dict) -> tuple[str, str]:
    """Return (action_name, detail) using rich info from the server."""
    shop = info.get("shop", [])
    board = info.get("board", [])
    bench = info.get("bench", [])

    # Build lookup of units by position for SELL
    unit_at: dict[tuple[int, int], dict] = {}
    for u in board:
        unit_at[(u["x"], u["y"])] = u
    for u in bench:
        unit_at[(u["x"], 0)] = u

    if 0 <= action <= 5:
        slot = action
        name = shop[slot] if slot < len(shop) and shop[slot] else "(empty)"
        return f"BUY slot {slot}", name

    if action == 6:
        return "REFRESH", "reroll shop"
    if action == 7:
        exp = info.get("exp", "?")
        exp_needed = info.get("expNeeded", "?")
        return "LEVEL UP", f"4g -> +4 exp ({exp}/{exp_needed})"
    if action == 8:
        return "LOCK SHOP", "toggle"
    if action == 9:
        return "END TURN", "-> fight"

    if 10 <= action <= 41:
        cell = action - 10
        x, y = cell % 8, cell // 8
        label = "board" if y >= 1 else "bench"
        return f"MOVE to ({x},{y})", label

    if 42 <= action <= 73:
        cell = action - 42
        x, y = cell % 8, cell // 8
        label = "board" if y >= 1 else "bench"
        u = unit_at.get((x, y))
        if u:
            return f"SELL at ({x},{y})", f"{u['name']} {_star_str(u.get('stars',1))} ({label})"
        return f"SELL at ({x},{y})", label

    if 74 <= action <= 79:
        slot = action - 74
        name = shop[slot] if slot < len(shop) and shop[slot] else "(empty)"
        return f"REMOVE SHOP slot {slot}", name

    if 80 <= action <= 85:
        slot = action - 80
        return f"PICK proposition {slot}", ""

    if 86 <= action <= 91:
        pair = action - 86
        items = info.get("items", [])
        return f"COMBINE items pair {pair}", f"held: {items}" if items else ""

    return f"UNKNOWN({action})", ""


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
    turn_action_num = 0
    prev_stage = info.get("stage", 1)
    fights = 0
    wins = 0
    losses = 0
    draws = 0

    # Track observation bounds across the entire game
    obs_global_min = obs.min()
    obs_global_max = obs.max()
    obs_clipped_count = 0  # steps where obs hits bounds

    print(f"  [OBS] reset: min={obs.min():.4f}  max={obs.max():.4f}  shape={obs.shape}")

    print(
        f"{'Stage':>5} | {'Turn':>4} | {'Lv':>2} | {'Gold':>5} | "
        f"{'Board':>9} | {'HP':>5} | {'Action':<24} | Detail"
    )
    print("-" * 95)

    while not done:
        masks = env.unwrapped.action_masks()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
        action = int(action)

        # Read pre-step state from info
        stage = info.get("stage", prev_stage)
        life = info.get("life", 0)
        gold = info.get("money", info.get("gold", 0))
        board_size = info.get("boardSize", 0)
        level = info.get("level", 1)
        max_team = info.get("maxTeamSize", level)

        # Decode action using rich info (names, not indices)
        action_name, detail = _decode_action(action, info)

        turn_action_num += 1

        print(
            f"  {stage:3d}   | {turn_action_num:4d} | {level:2d} | {gold:4.0f}g | "
            f"{board_size:3d}/{max_team:<3d}   | {life:5.0f} | "
            f"{action_name:<24} | {detail}"
        )

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        done = terminated or truncated

        # Track obs bounds
        step_min, step_max = obs.min(), obs.max()
        obs_global_min = min(obs_global_min, step_min)
        obs_global_max = max(obs_global_max, step_max)
        if step_min < -1.0 or step_max > 2.0:
            obs_clipped_count += 1
            # Find which indices are out of bounds
            oob_low = np.where(obs < -1.0)[0]
            oob_high = np.where(obs > 2.0)[0]
            if len(oob_low) > 0 or len(oob_high) > 0:
                print(f"  [OBS WARNING] min={step_min:.4f} max={step_max:.4f}  "
                      f"low_idx={oob_low.tolist()}  high_idx={oob_high.tolist()}")

        # After END_TURN: print board state + fight summary
        if action == 9:  # END_TURN
            new_life = info.get("life", life)
            new_stage = info.get("stage", stage)
            new_rank = info.get("rank", "?")
            new_level = info.get("level", level)
            fights += 1

            if reward > _WIN_THRESHOLD:
                result = "WIN"
                wins += 1
            elif reward < _LOSS_THRESHOLD:
                result = "LOSS"
                losses += 1
            else:
                result = "DRAW"
                draws += 1

            hp_delta = new_life - life
            hp_str = f"{hp_delta:+.0f}" if hp_delta != 0 else "0"

            print(
                f"  === FIGHT: Stage {stage} | Result: {result} "
                f"(r={reward:+.3f}) | HP: {new_life:.0f} ({hp_str}) "
                f"| Rank: {new_rank} ==="
            )

            # Obs snapshot: show raw values for key player stats
            # Indices: 0=life/100, 1=money/300, 2=level/9, 3=(streak+20)/40,
            #          4=interest/5, 7=boardSize/9, 12=totalMoney/500, 13=totalDmg/300
            print(f"  [OBS] life={obs[0]:.3f} money={obs[1]:.3f} level={obs[2]:.3f} "
                  f"streak={obs[3]:.3f} interest={obs[4]:.3f} boardSz={obs[7]:.3f} "
                  f"totalMoney={obs[12]:.3f} totalDmg={obs[13]:.3f} "
                  f"| raw gold~{obs[1]*300:.0f}g | min={obs.min():.3f} max={obs.max():.3f}")

            # Print opponent team, then our board state for the upcoming turn
            _print_opponent(info)
            _print_board_state(info)

            prev_stage = new_stage
            turn_action_num = 0

    # ----- Game over summary -----
    final_rank = info.get("rank", "?")
    final_stage = info.get("stage", "?")
    final_life = info.get("life", 0)
    final_level = info.get("level", "?")

    print()
    print("=" * 60)
    print("GAME OVER")
    print("=" * 60)
    print(f"  Final Rank:     {final_rank}")
    print(f"  Final Stage:    {final_stage}")
    print(f"  Final Level:    {final_level}")
    print(f"  Final HP:       {final_life:.0f}")
    print(f"  Total Steps:    {total_steps}")
    print(f"  Total Reward:   {total_reward:+.3f}")
    print(f"  Fights:         {fights}  (W:{wins} / L:{losses} / D:{draws})")

    # Observation bounds report
    print()
    print("  --- Observation Bounds Report ---")
    print(f"  Global min:     {obs_global_min:.4f}  (env low=-1.0)")
    print(f"  Global max:     {obs_global_max:.4f}  (env high=2.0)")
    print(f"  Steps clipped:  {obs_clipped_count} / {total_steps}")
    if obs_global_max > 1.05 or obs_global_min < -0.05:
        print(f"  ** Values outside [0,1] range detected! Check normalization **")
    if obs_global_max > 2.0 or obs_global_min < -1.0:
        print(f"  ** VALUES EXCEED ENV BOUNDS — SB3 IS SILENTLY CLIPPING! **")
    if fights > 0:
        print(f"  Win Rate:       {wins / fights:.1%}")

    # Final board state
    _print_board_state(info)

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
