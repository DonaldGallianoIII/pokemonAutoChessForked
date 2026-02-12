"""
Launch N Node.js training server processes on sequential ports.

Spawns each server as a subprocess with its own TRAINING_PORT, waits for all
to pass health checks, and provides clean shutdown (no orphan processes).

Usage:
    # As a library:
    from launch_servers import launch_servers
    servers, cleanup = launch_servers(num_envs=16, base_port=9100)

    # Standalone test (launches 2 servers, waits 5s, shuts down):
    python training/launch_servers.py
"""

import atexit
import os
import signal
import subprocess
import sys
import time

import requests

_IS_WINDOWS = sys.platform == "win32"


def _kill_proc_tree(proc):
    """Kill a subprocess and all its children, cross-platform."""
    if _IS_WINDOWS:
        # taskkill /T kills the entire process tree, /F forces it
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


def launch_servers(
    num_envs: int = 16,
    base_port: int = 9100,
    health_timeout: int = 60,
) -> tuple:
    """
    Launch num_envs Node.js training servers on ports base_port..base_port+num_envs-1.

    Waits for all to be healthy (GET /health returns status: "ok").
    Returns (server_list, cleanup_function).

    server_list: list of (subprocess.Popen, int) tuples — (process, port).
    cleanup_function: callable that terminates all servers.
    """
    # Project root is one level up from training/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = "npx ts-node --transpile-only app/training/index.ts"

    servers: list[tuple[subprocess.Popen, int]] = []

    def cleanup():
        for proc, port in servers:
            if proc.poll() is None:
                _kill_proc_tree(proc)
        if servers:
            print(f"All {len(servers)} training servers shut down.")

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    # --- Spawn all processes ---
    # Platform-specific kwargs for process group isolation
    popen_kwargs: dict = {}
    if _IS_WINDOWS:
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    for i in range(num_envs):
        port = base_port + i
        env = os.environ.copy()
        env["TRAINING_PORT"] = str(port)
        env["SKIP_MONGO"] = "true"

        proc = subprocess.Popen(
            cmd,
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=project_root,
            **popen_kwargs,
        )
        servers.append((proc, port))
        print(f"Spawned server process for port {port} (pid {proc.pid})")

    # --- Wait for all to become healthy ---
    for i, (proc, port) in enumerate(servers):
        url = f"http://localhost:{port}/health"
        start = time.time()
        healthy = False

        while time.time() - start < health_timeout:
            # Check if process crashed
            if proc.poll() is not None:
                stderr_output = proc.stderr.read().decode() if proc.stderr else ""
                cleanup()
                raise RuntimeError(
                    f"Server on port {port} crashed during startup "
                    f"(exit code {proc.returncode}).\nstderr:\n{stderr_output}"
                )

            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    healthy = True
                    break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                pass

            time.sleep(1)

        if not healthy:
            stderr_output = ""
            if proc.stderr:
                if not _IS_WINDOWS:
                    import select
                    if select.select([proc.stderr], [], [], 0)[0]:
                        stderr_output = proc.stderr.read(4096).decode(errors="replace")
                # On Windows, skip non-blocking read (select doesn't work on pipes)
            cleanup()
            raise TimeoutError(
                f"Server on port {port} did not become healthy within "
                f"{health_timeout}s.\nstderr:\n{stderr_output}"
            )

        print(f"Server on port {port} ready ({i + 1}/{num_envs})")

    print(f"\nAll {num_envs} servers healthy on ports {base_port}-{base_port + num_envs - 1}")
    return servers, cleanup


if __name__ == "__main__":
    print("=== launch_servers.py standalone test ===")
    print("Launching 2 servers on ports 9100-9101...\n")

    servers, cleanup = launch_servers(num_envs=2, base_port=9100)

    print("\nWaiting 5 seconds to confirm stability...")
    time.sleep(5)

    # Quick sanity: hit both health endpoints one more time
    for _, port in servers:
        resp = requests.get(f"http://localhost:{port}/health", timeout=10)
        data = resp.json()
        print(f"  Port {port}: {data}")

    print("\nShutting down...")
    cleanup()

    # Verify processes are actually dead
    time.sleep(1)
    all_dead = all(proc.poll() is not None for proc, _ in servers)
    print(f"All processes exited cleanly: {all_dead}")
