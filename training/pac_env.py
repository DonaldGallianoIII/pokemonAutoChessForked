"""
Pokemon Auto Chess Gymnasium Environment

Wraps the TypeScript training server HTTP API as a standard Gymnasium
environment for use with stable-baselines3 PPO.

Usage:
    env = PokemonAutoChessEnv(server_url="http://localhost:9100")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import time
from urllib.parse import urlparse

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces

# Timeout constants (seconds)
STEP_TIMEOUT = 30    # for step() and reset() calls
HEALTH_TIMEOUT = 10  # for init/space-query calls
RESET_MAX_RETRIES = 3
RESET_BACKOFF_BASE = 2  # exponential backoff: 2s, 4s, 8s


def _port_from_url(url: str) -> str:
    """Extract port from server URL for logging."""
    parsed = urlparse(url)
    return str(parsed.port or "unknown")


class PokemonAutoChessEnv(gym.Env):
    """
    Gymnasium environment for Pokemon Auto Chess step-mode training.

    Observation space: Box(low=0.0, high=1.0, shape=(obs_size,))
    Action space: Discrete(num_actions)

    The agent acts during PICK phases only. Each step is one micro-action
    (buy, sell, reroll, level up, or end turn). When END_TURN is chosen,
    the fight phase runs instantly and the environment advances to the
    next pick phase.
    """

    metadata = {"render_modes": []}

    def __init__(self, server_url: str = "http://localhost:9100", use_action_mask: bool = True):
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.use_action_mask = use_action_mask
        self.port = _port_from_url(self.server_url)
        self.session = requests.Session()

        # Query server for space dimensions
        try:
            action_info = self.session.get(
                f"{self.server_url}/action-space", timeout=HEALTH_TIMEOUT
            ).json()
            obs_info = self.session.get(
                f"{self.server_url}/observation-space", timeout=HEALTH_TIMEOUT
            ).json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to training server at {self.server_url}. "
                "Make sure the TypeScript server is running: npx ts-node app/training/index.ts"
            )

        self.num_actions = action_info["n"]
        self.obs_size = obs_info["n"]

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_size,), dtype=np.float32
        )

        self._current_action_mask = np.ones(self.num_actions, dtype=np.int8)
        self._last_obs = np.zeros(self.obs_size, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        last_error = None
        for attempt in range(RESET_MAX_RETRIES):
            try:
                response = self.session.post(
                    f"{self.server_url}/reset", timeout=STEP_TIMEOUT
                ).json()
                obs = np.array(response["observation"], dtype=np.float32)
                info = response.get("info", {})
                self._current_action_mask = np.array(
                    info.get("actionMask", np.ones(self.num_actions)),
                    dtype=np.int8
                )
                # sb3-contrib MaskablePPO reads masks via env.action_masks() method.
                # Also inject into info with both key names for compatibility.
                info["action_masks"] = self._current_action_mask
                self._last_obs = obs
                return obs, info
            except Exception as e:
                last_error = e
                backoff = RESET_BACKOFF_BASE ** (attempt + 1)
                print(
                    f"WARNING: Server on port {self.port} error during reset: "
                    f"{type(e).__name__}: {e} "
                    f"(attempt {attempt + 1}/{RESET_MAX_RETRIES}), retrying in {backoff}s"
                )
                time.sleep(backoff)

        raise ConnectionError(
            f"Server on port {self.port} timed out during reset after "
            f"{RESET_MAX_RETRIES} retries — server is unresponsive"
        ) from last_error

    def step(self, action):
        action = int(action)
        try:
            response = self.session.post(
                f"{self.server_url}/step",
                json={"action": action},
                timeout=STEP_TIMEOUT,
            ).json()
        except Exception as e:
            print(
                f"WARNING: Server on port {self.port} error during step: "
                f"{type(e).__name__}: {e}"
            )
            # Return a terminal state so SubprocVecEnv will auto-reset this env
            obs = self._last_obs
            reward = 0.0
            terminated = True
            truncated = True
            info = {"action_masks": self._current_action_mask, "timeout": True}
            return obs, reward, terminated, truncated, info

        obs = np.array(response["observation"], dtype=np.float32)
        reward = float(response["reward"])
        terminated = bool(response["done"])
        truncated = False
        info = response.get("info", {})

        self._current_action_mask = np.array(
            info.get("actionMask", np.ones(self.num_actions)),
            dtype=np.int8
        )
        info["action_masks"] = self._current_action_mask
        self._last_obs = obs

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return current action mask for masked PPO."""
        return self._current_action_mask

    def close(self):
        self.session.close()
