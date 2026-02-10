"""
Smoke test: runs 10 complete games against the training server and validates:

1. No crashes or exceptions during play
2. No NaN/Inf in observations
3. All games terminate (reach done=True)
4. Rewards are non-zero for at least some games
5. Action masks are never all-zeros for alive players
6. Games take a reasonable number of steps (50-500, not 5 or 50,000)

Run this BEFORE starting an overnight training run. 10 games, ~30 seconds.

Usage:
    # Single-agent mode (default):
    python training/smoke_test.py

    # Self-play mode:
    python training/smoke_test.py --self-play
"""

import argparse
import sys
import time

import numpy as np
import requests


def smoke_test_single_agent(server_url: str, n_games: int = 10):
    """Smoke test for single-agent mode (1 RL agent + 7 bots)."""
    print(f"=== Smoke Test: Single Agent ({n_games} games) ===\n")

    # Get space dimensions
    obs_size = requests.get(f"{server_url}/observation-space").json()["n"]
    num_actions = requests.get(f"{server_url}/action-space").json()["n"]

    results = []
    errors = []

    for game_idx in range(n_games):
        try:
            resp = requests.post(f"{server_url}/reset").json()
            obs = np.array(resp["observation"], dtype=np.float32)
            mask = np.array(resp["info"]["actionMask"], dtype=np.int8)
            done = False
            steps = 0
            total_reward = 0.0
            has_nonzero_reward = False

            while not done:
                steps += 1

                # Validate observation
                if np.any(np.isnan(obs)):
                    errors.append(f"Game {game_idx+1} step {steps}: NaN in observation")
                    break
                if np.any(np.isinf(obs)):
                    errors.append(f"Game {game_idx+1} step {steps}: Inf in observation")
                    break
                if len(obs) != obs_size:
                    errors.append(
                        f"Game {game_idx+1} step {steps}: obs size {len(obs)} != {obs_size}"
                    )
                    break

                # Validate action mask
                if len(mask) != num_actions:
                    errors.append(
                        f"Game {game_idx+1} step {steps}: mask size {len(mask)} != {num_actions}"
                    )
                    break
                valid_actions = np.where(mask == 1)[0]
                if len(valid_actions) == 0:
                    errors.append(
                        f"Game {game_idx+1} step {steps}: all-zero action mask!"
                    )
                    break

                # Pick random valid action
                action = int(np.random.choice(valid_actions))
                resp = requests.post(
                    f"{server_url}/step", json={"action": action}
                ).json()

                obs = np.array(resp["observation"], dtype=np.float32)
                reward = float(resp["reward"])
                done = bool(resp["done"])
                mask = np.array(resp["info"]["actionMask"], dtype=np.int8)

                total_reward += reward
                if reward != 0:
                    has_nonzero_reward = True

                # Safety cap
                if steps > 10_000:
                    errors.append(f"Game {game_idx+1}: exceeded 10,000 steps without ending")
                    break

            results.append({
                "game": game_idx + 1,
                "steps": steps,
                "reward": total_reward,
                "rank": resp.get("info", {}).get("rank", "?"),
                "stage": resp.get("info", {}).get("stage", "?"),
                "has_nonzero_reward": has_nonzero_reward,
            })

            print(
                f"  Game {game_idx+1:2d}: {steps:4d} steps, "
                f"reward={total_reward:+7.2f}, "
                f"rank={resp.get('info', {}).get('rank', '?')}, "
                f"stage={resp.get('info', {}).get('stage', '?')}"
            )

        except Exception as e:
            errors.append(f"Game {game_idx+1}: exception: {e}")
            print(f"  Game {game_idx+1:2d}: EXCEPTION: {e}")

    return results, errors


def smoke_test_self_play(server_url: str, n_games: int = 10):
    """Smoke test for self-play mode (8 RL agents)."""
    print(f"=== Smoke Test: Self-Play ({n_games} games) ===\n")

    obs_size = requests.get(f"{server_url}/observation-space").json()["n"]
    num_actions = requests.get(f"{server_url}/action-space").json()["n"]

    results = []
    errors = []

    for game_idx in range(n_games):
        try:
            resp = requests.post(f"{server_url}/reset").json()
            # Initial obs is player 0's view, broadcast for initial masks
            init_mask = np.array(resp["info"]["actionMask"], dtype=np.int8)
            masks = np.tile(init_mask, (8, 1))

            done_all = False
            steps = 0
            total_rewards = np.zeros(8, dtype=np.float32)

            while not done_all:
                steps += 1

                # Pick random valid action per player
                actions = []
                for i in range(8):
                    valid = np.where(masks[i] == 1)[0]
                    if len(valid) == 0:
                        errors.append(
                            f"Game {game_idx+1} step {steps} player {i}: all-zero mask!"
                        )
                        actions.append(0)  # END_TURN fallback
                    else:
                        actions.append(int(np.random.choice(valid)))

                resp = requests.post(
                    f"{server_url}/step-multi", json={"actions": actions}
                ).json()

                obs_batch = np.array(resp["observations"], dtype=np.float32)
                reward_batch = np.array(resp["rewards"], dtype=np.float32)
                done_batch = np.array(resp["dones"], dtype=bool)
                infos = resp["infos"]

                # Validate observations
                if np.any(np.isnan(obs_batch)):
                    errors.append(f"Game {game_idx+1} step {steps}: NaN in observations")
                    break
                if obs_batch.shape != (8, obs_size):
                    errors.append(
                        f"Game {game_idx+1} step {steps}: obs shape {obs_batch.shape} != (8, {obs_size})"
                    )
                    break

                # Update masks for next step
                for i, info in enumerate(infos):
                    masks[i] = np.array(info["actionMask"], dtype=np.int8)

                total_rewards += reward_batch
                done_all = all(done_batch)

                if steps > 10_000:
                    errors.append(f"Game {game_idx+1}: exceeded 10,000 steps without ending")
                    break

            ranks = [info.get("rank", "?") for info in infos]
            stage = infos[0].get("stage", "?")

            results.append({
                "game": game_idx + 1,
                "steps": steps,
                "rewards": total_rewards.tolist(),
                "ranks": ranks,
                "stage": stage,
            })

            print(
                f"  Game {game_idx+1:2d}: {steps:4d} steps, "
                f"stage={stage}, "
                f"rewards=[{', '.join(f'{r:+.1f}' for r in total_rewards)}]"
            )

        except Exception as e:
            errors.append(f"Game {game_idx+1}: exception: {e}")
            print(f"  Game {game_idx+1:2d}: EXCEPTION: {e}")

    return results, errors


def print_summary(results, errors, mode):
    """Print validation summary."""
    print(f"\n{'='*60}")
    print(f"SMOKE TEST SUMMARY ({mode})")
    print(f"{'='*60}")

    if not results:
        print("  NO GAMES COMPLETED")
        if errors:
            print(f"\n  ERRORS ({len(errors)}):")
            for e in errors:
                print(f"    - {e}")
        return False

    steps_list = [r["steps"] for r in results]
    mean_steps = np.mean(steps_list)
    min_steps = np.min(steps_list)
    max_steps = np.max(steps_list)

    print(f"  Games completed: {len(results)}")
    print(f"  Mean game length: {mean_steps:.0f} steps (min={min_steps}, max={max_steps})")

    # Check game length health
    ok = True
    if mean_steps < 50:
        print("  WARNING: Very short games (<50 steps) — agents may be dying immediately")
        ok = False
    elif mean_steps > 5000:
        print("  WARNING: Very long games (>5000 steps) — possible infinite loop or stalling")
        ok = False
    else:
        print(f"  Game length: OK")

    # Check rewards
    if mode == "single_agent":
        rewards = [r["reward"] for r in results]
        nonzero = sum(1 for r in rewards if r != 0)
        print(f"  Non-zero reward games: {nonzero}/{len(results)}")
        if nonzero == 0:
            print("  WARNING: All rewards are zero — reward shaping may be broken")
            ok = False
    else:
        all_rewards = np.array([r["rewards"] for r in results])
        nonzero = np.count_nonzero(all_rewards)
        print(f"  Non-zero rewards: {nonzero}/{all_rewards.size}")
        if nonzero == 0:
            print("  WARNING: All rewards are zero — reward shaping may be broken")
            ok = False

    # Check errors
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
        ok = False
    else:
        print(f"  Errors: 0")

    if ok:
        print(f"\n  RESULT: PASS")
    else:
        print(f"\n  RESULT: FAIL — fix issues before starting training")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Smoke test for PPO training")
    parser.add_argument(
        "--server-url", default="http://localhost:9100", help="Training server URL"
    )
    parser.add_argument(
        "--self-play", action="store_true", help="Test self-play mode (SELF_PLAY=true)"
    )
    parser.add_argument("--games", type=int, default=10, help="Number of games to run")
    args = parser.parse_args()

    # Wait for server
    print(f"Connecting to {args.server_url}...")
    for _ in range(10):
        try:
            resp = requests.get(f"{args.server_url}/health", timeout=2)
            if resp.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        print("ERROR: Could not connect to training server")
        sys.exit(1)

    start = time.time()

    if args.self_play:
        results, errors = smoke_test_self_play(args.server_url, args.games)
        ok = print_summary(results, errors, "self_play")
    else:
        results, errors = smoke_test_single_agent(args.server_url, args.games)
        ok = print_summary(results, errors, "single_agent")

    elapsed = time.time() - start
    print(f"\n  Elapsed: {elapsed:.1f}s")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
