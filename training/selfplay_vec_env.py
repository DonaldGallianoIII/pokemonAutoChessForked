"""
SelfPlayVecEnv: A VecEnv wrapper for 8-player self-play PPO training.

Presents the 8-player Pokemon Auto Chess game as 8 parallel sub-environments
to stable-baselines3. Each sub-env corresponds to one player seat.

Key design decisions (from design review):

Issue #1 (fight trigger timing): The TS server tracks turnEnded[] state that
persists across step calls within a round. The fight only triggers when the
LAST alive player ends their turn. This wrapper doesn't need to handle this
— it's all server-side.

Issue #2 (reward attribution): The /step-multi endpoint returns rewards for
ALL 8 players after each fight, not just the triggering player. The wrapper
passes these through directly.

Issue #3 (VecEnv reset semantics): Dead players return done=False with
END_TURN-only action masks until the game actually ends. Only when the game
ends do ALL 8 players return done=True simultaneously. This prevents sb3's
VecEnv from trying to auto-reset individual sub-envs mid-game.

Usage:
    env = SelfPlayVecEnv(server_url="http://localhost:9100")
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
    VecEnv wrapper for 8-player self-play.

    Each of the 8 sub-envs is one player seat in a shared Pokemon Auto Chess
    game. Actions are batched into a single /step-multi HTTP call for efficiency
    (one round-trip per step, not 8).
    """

    def __init__(self, server_url: str = "http://localhost:9100"):
        self.server_url = server_url.rstrip("/")
        self._port = _port_from_url(self.server_url)

        # Query server for space dimensions
        try:
            action_info = requests.get(
                f"{self.server_url}/action-space", timeout=HEALTH_TIMEOUT
            ).json()
            obs_info = requests.get(
                f"{self.server_url}/observation-space", timeout=HEALTH_TIMEOUT
            ).json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to training server at {self.server_url}. "
                "Make sure the server is running with SELF_PLAY=true"
            )

        self._num_actions = action_info["n"]
        self._obs_size = obs_info["n"]

        observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(self._obs_size,), dtype=np.float32
        )
        action_space = spaces.Discrete(self._num_actions)

        super().__init__(
            num_envs=8,
            observation_space=observation_space,
            action_space=action_space,
        )

        self._action_masks = np.ones((8, self._num_actions), dtype=np.int8)
        self._last_obs = np.zeros((8, self._obs_size), dtype=np.float32)
        self._pending_actions = None

    def reset(self) -> np.ndarray:
        last_error = None
        for attempt in range(RESET_MAX_RETRIES):
            try:
                response = requests.post(
                    f"{self.server_url}/reset", timeout=STEP_TIMEOUT
                ).json()
                obs = np.array(response["observation"], dtype=np.float32)

                # After reset, the server returns a single observation (player 0's view).
                # For self-play, we do a no-op step to get all 8 observations.
                # But actually, at stage 0, all players have propositions, so we need
                # the per-player observations from step-multi.
                # For now, broadcast the initial obs to all 8 sub-envs.
                # The first step_wait() call will get proper per-player observations.
                all_obs = np.tile(obs, (8, 1))

                info = response.get("info", {})
                mask = np.array(
                    info.get("actionMask", np.ones(self._num_actions)),
                    dtype=np.int8,
                )
                self._action_masks = np.tile(mask, (8, 1))
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
            # Return terminal state for all 8 players — triggers auto-reset
            obs = self._last_obs
            rewards = np.zeros(8, dtype=np.float32)
            dones = np.array([True] * 8, dtype=bool)
            infos = [
                {"action_masks": self._action_masks[i], "timeout": True}
                for i in range(8)
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
        """Return current action masks for all 8 sub-envs. Shape: (8, num_actions)."""
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
