"""
SelfPlayVecEnv: A VecEnv wrapper for multi-agent PPO training.

Presents N player seats in a Pokemon Auto Chess game as N parallel
sub-environments to stable-baselines3. Supports two modes:

  - Full self-play (SELF_PLAY=true, N=8): All 8 seats are RL agents.
  - Hybrid mode (NUM_RL_AGENTS=2..7, N=2..7): N seats are RL agents,
    the remaining (8-N) seats are bots controlled by the game server.
    This is a transition step between single-agent (1v7) and full self-play.

The number of RL agents (N) is queried from the server's /num-agents
endpoint at startup. The wrapper sends N actions per step via /step-multi
and receives N observations/rewards/dones back.

Key design decisions (from design review):

Issue #1 (fight trigger timing): The TS server tracks turnEnded[] state that
persists across step calls within a round. The fight only triggers when the
LAST alive player ends their turn. Bots in hybrid mode are auto-ended by
the server. This wrapper doesn't need to handle this — it's all server-side.

Issue #2 (reward attribution): The /step-multi endpoint returns rewards for
ALL N RL agents after each fight, not just the triggering player. The wrapper
passes these through directly.

Issue #3 (VecEnv reset semantics): Dead players return done=False with
END_TURN-only action masks until the game actually ends. Only when the game
ends do ALL N players return done=True simultaneously. This prevents sb3's
VecEnv from trying to auto-reset individual sub-envs mid-game.

Usage:
    env = SelfPlayVecEnv(server_url="http://localhost:9100")
    # num_envs is auto-detected from server (8 for self-play, N for hybrid)
    obs = env.reset()
    obs, rewards, dones, infos = env.step(actions)
"""

import time
from urllib.parse import urlparse

import numpy as np
import requests
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

# Timeout constants (seconds)
STEP_TIMEOUT = 15    # for step-multi and reset calls — 15s is generous (normal <1s);
                     # lower than 30s to reduce stalls when a server hangs
HEALTH_TIMEOUT = 10  # for init/space-query calls
RESET_MAX_RETRIES = 3
RESET_BACKOFF_BASE = 2  # exponential backoff: 2s, 4s, 8s


def _port_from_url(url: str) -> str:
    """Extract port from server URL for logging."""
    parsed = urlparse(url)
    return str(parsed.port or "unknown")


class SelfPlayVecEnv(VecEnv):
    """
    VecEnv wrapper for multi-agent training (self-play or hybrid).

    Each of the N sub-envs is one RL-agent seat in a shared Pokemon Auto Chess
    game. Actions are batched into a single /step-multi HTTP call for efficiency
    (one round-trip per step, not N).

    N is auto-detected from the server's /num-agents endpoint:
      - SELF_PLAY=true  → N=8 (full self-play, no bots)
      - NUM_RL_AGENTS=2 → N=2 (hybrid: 2 RL agents + 6 bots)
      - etc.
    """

    def __init__(self, server_url: str = "http://localhost:9100"):
        self.server_url = server_url.rstrip("/")
        self._port = _port_from_url(self.server_url)

        # Query server for space dimensions and number of RL agents
        try:
            action_info = requests.get(
                f"{self.server_url}/action-space", timeout=HEALTH_TIMEOUT
            ).json()
            obs_info = requests.get(
                f"{self.server_url}/observation-space", timeout=HEALTH_TIMEOUT
            ).json()
            agents_info = requests.get(
                f"{self.server_url}/num-agents", timeout=HEALTH_TIMEOUT
            ).json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to training server at {self.server_url}. "
                "Make sure the server is running with SELF_PLAY=true or NUM_RL_AGENTS>=2"
            )

        self._num_actions = action_info["n"]
        self._obs_size = obs_info["n"]
        self._num_agents = agents_info["n"]

        print(
            f"SelfPlayVecEnv: server on port {self._port} has "
            f"{self._num_agents} RL agent(s)"
            f"{' (hybrid mode with bots)' if self._num_agents < 8 else ' (full self-play)'}"
        )

        observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(self._obs_size,), dtype=np.float32
        )
        action_space = spaces.Discrete(self._num_actions)

        super().__init__(
            num_envs=self._num_agents,
            observation_space=observation_space,
            action_space=action_space,
        )

        n = self._num_agents
        self._action_masks = np.ones((n, self._num_actions), dtype=np.int8)
        self._last_obs = np.zeros((n, self._obs_size), dtype=np.float32)
        self._pending_actions = None

    def reset(self) -> np.ndarray:
        n = self._num_agents
        last_error = None
        for attempt in range(RESET_MAX_RETRIES):
            try:
                response = requests.post(
                    f"{self.server_url}/reset", timeout=STEP_TIMEOUT
                ).json()
                obs = np.array(response["observation"], dtype=np.float32)

                # After reset, the server returns a single observation (player 0's view).
                # Broadcast the initial obs to all N sub-envs.
                # The first step_wait() call will get proper per-player observations.
                all_obs = np.tile(obs, (n, 1))

                info = response.get("info", {})
                mask = np.array(
                    info.get("actionMask", np.ones(self._num_actions)),
                    dtype=np.int8,
                )
                self._action_masks = np.tile(mask, (n, 1))
                self._last_obs = all_obs

                return all_obs
            except Exception as e:
                last_error = e
                backoff = RESET_BACKOFF_BASE ** (attempt + 1)
                print(
                    f"WARNING: Server on port {self._port} error during reset: "
                    f"{type(e).__name__}: {e} "
                    f"(attempt {attempt + 1}/{RESET_MAX_RETRIES}), retrying in {backoff}s"
                )
                time.sleep(backoff)

        raise ConnectionError(
            f"Server on port {self._port} timed out during reset after "
            f"{RESET_MAX_RETRIES} retries — server is unresponsive"
        ) from last_error

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = actions

    def step_wait(self):
        n = self._num_agents
        if self._pending_actions is None:
            raise RuntimeError("step_async must be called before step_wait")

        try:
            response = requests.post(
                f"{self.server_url}/step-multi",
                json={"actions": self._pending_actions.tolist()},
                timeout=STEP_TIMEOUT,
            ).json()
        except Exception as e:
            print(
                f"WARNING: Server on port {self._port} error during step: "
                f"{type(e).__name__}: {e}"
            )
            self._pending_actions = None
            # Return terminal state for all N agents — triggers auto-reset
            obs = self._last_obs
            rewards = np.zeros(n, dtype=np.float32)
            dones = np.array([True] * n, dtype=bool)
            infos = [
                {"action_masks": self._action_masks[i], "timeout": True}
                for i in range(n)
            ]
            # Auto-reset since all dones are True (matches normal path below)
            obs = self.reset()
            return obs, rewards, dones, infos

        self._pending_actions = None

        obs = np.array(response["observations"], dtype=np.float32)
        rewards = np.array(response["rewards"], dtype=np.float32)
        dones = np.array(response["dones"], dtype=bool)
        infos = response["infos"]

        # Inject action masks into infos for MaskablePPO.
        # sb3-contrib expects "action_masks" (plural 's') — NOT "actionMask" (TS camelCase).
        for i, info in enumerate(infos):
            mask = np.array(info.get("actionMask", np.ones(self._num_actions)), dtype=np.int8)
            info["action_masks"] = mask
            self._action_masks[i] = mask

        self._last_obs = obs

        # Issue #3: When ALL dones are True, the game has ended.
        # Auto-reset the environment for the next episode.
        if all(dones):
            obs = self.reset()

        return obs, rewards, dones, infos

    def action_masks(self) -> np.ndarray:
        """Return current action masks for all N sub-envs. Shape: (N, num_actions)."""
        return self._action_masks

    def close(self) -> None:
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        # sb3-contrib MaskablePPO calls env.env_method("action_masks") to get
        # per-sub-env masks when using a VecEnv. Dispatch to our action_masks().
        if method_name == "action_masks":
            masks = self.action_masks()
            target = indices if indices is not None else range(self.num_envs)
            return [masks[i] for i in target]
        raise NotImplementedError(f"env_method('{method_name}') not supported")

    def get_attr(self, attr_name, indices=None):
        if attr_name == "action_masks":
            return [self.action_masks]
        raise NotImplementedError(f"get_attr('{attr_name}') not supported")

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError(f"set_attr('{attr_name}') not supported")

    def seed(self, seed=None):
        pass
