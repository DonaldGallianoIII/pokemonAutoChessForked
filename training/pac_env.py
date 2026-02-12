"""
Pokemon Auto Chess Gymnasium Environment

Wraps the TypeScript training server HTTP API as a standard Gymnasium
environment for use with stable-baselines3 PPO.

Usage:
    env = PokemonAutoChessEnv(server_url="http://localhost:9100")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import os
import signal
import subprocess
import sys
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
RESTART_HEALTH_TIMEOUT = 60  # seconds to wait for restarted server

_IS_WINDOWS = sys.platform == "win32"


def _kill_proc(proc):
    """Kill a subprocess and all its children, cross-platform."""
    if _IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()


def _kill_port(port):
    """Best-effort kill of whatever process is listening on the given port."""
    if _IS_WINDOWS:
        try:
            result = subprocess.run(
                f"netstat -ano | findstr :{port} | findstr LISTENING",
                shell=True,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.split()
                if parts and parts[-1].isdigit():
                    subprocess.run(
                        ["taskkill", "/F", "/PID", parts[-1]],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
        except Exception:
            pass
    else:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
            )
            for pid_str in result.stdout.strip().split("\n"):
                if pid_str.strip().isdigit():
                    os.kill(int(pid_str), signal.SIGKILL)
        except Exception:
            pass


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

    def __init__(
        self,
        server_url: str = "http://localhost:9100",
        use_action_mask: bool = True,
        server_cmd: str = None,
        server_cwd: str = None,
    ):
        super().__init__()
        self.server_url = server_url.rstrip("/")
        self.use_action_mask = use_action_mask
        self.port = _port_from_url(self.server_url)
        self.session = requests.Session()

        # Server restart info (optional — enables self-healing on reset failure)
        self._server_cmd = server_cmd
        self._server_cwd = server_cwd
        self._server_proc = None

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

    def _restart_server(self):
        """Kill the old server and spawn a fresh one on the same port.

        Returns True if the server restarted and passed health check.
        Returns False if restart info was not provided or server failed to start.
        """
        if self._server_cmd is None:
            return False

        print(f"RESTART: Restarting server on port {self.port}...")

        # Kill existing server process
        if self._server_proc is not None:
            _kill_proc(self._server_proc)
            self._server_proc = None
        _kill_port(self.port)

        # Brief pause for port release
        time.sleep(1)

        # Spawn new server
        env = os.environ.copy()
        env["TRAINING_PORT"] = str(self.port)
        env["SKIP_MONGO"] = "true"

        popen_kwargs = {}
        if _IS_WINDOWS:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["preexec_fn"] = os.setsid

        self._server_proc = subprocess.Popen(
            self._server_cmd,
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._server_cwd,
            **popen_kwargs,
        )
        print(
            f"RESTART: Spawned new server on port {self.port} "
            f"(pid {self._server_proc.pid})"
        )

        # Wait for health check
        health_url = f"{self.server_url}/health"
        start = time.time()
        while time.time() - start < RESTART_HEALTH_TIMEOUT:
            if self._server_proc.poll() is not None:
                print(
                    f"RESTART: Server on port {self.port} crashed during startup "
                    f"(exit code {self._server_proc.returncode})"
                )
                self._server_proc = None
                return False
            try:
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    print(f"RESTART: Server on port {self.port} is healthy")
                    self.session.close()
                    self.session = requests.Session()
                    return True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass
            time.sleep(1)

        print(
            f"RESTART: Server on port {self.port} did not become healthy "
            f"within {RESTART_HEALTH_TIMEOUT}s"
        )
        _kill_proc(self._server_proc)
        self._server_proc = None
        return False

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

        # Exhausted retries — try server restart if we have launch info
        if self._restart_server():
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
                info["action_masks"] = self._current_action_mask
                self._last_obs = obs
                print(
                    f"RESTART: Server on port {self.port} reset successful "
                    f"after restart"
                )
                return obs, info
            except Exception as e:
                last_error = e
                print(
                    f"RESTART: Server on port {self.port} reset STILL failed "
                    f"after restart: {type(e).__name__}: {e}"
                )

        raise ConnectionError(
            f"Server on port {self.port} failed during reset after "
            f"{RESET_MAX_RETRIES} retries and server restart — "
            f"server is unresponsive"
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
        if self._server_proc is not None and self._server_proc.poll() is None:
            _kill_proc(self._server_proc)
            self._server_proc = None
