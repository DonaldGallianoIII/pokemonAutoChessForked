# Design Doc: IPC Overhaul & GPU-Parallel Training

**Status:** DRAFT — Awaiting review
**Author:** Claude (Opus)
**Date:** 2026-02-11
**Scope:** Replace HTTP transport with Unix domain sockets + binary protocol; add GPU-accelerated multi-environment parallelism
**Prerequisite:** Current architecture (Phase 7.2) is stable and benchmarked

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current Architecture Inventory](#2-current-architecture-inventory)
3. [Performance Budget & Targets](#3-performance-budget--targets)
4. [Phase 1 — Binary IPC Protocol (Unix Sockets)](#phase-1--binary-ipc-protocol-unix-sockets)
5. [Phase 2 — Shared Memory Transport](#phase-2--shared-memory-transport)
6. [Phase 3 — Multi-Environment Parallelism (CPU)](#phase-3--multi-environment-parallelism-cpu)
7. [Phase 4 — GPU-Accelerated Inference & Batched Rollouts](#phase-4--gpu-accelerated-inference--batched-rollouts)
8. [Phase 5 — GPU-Parallel Combat Simulation](#phase-5--gpu-parallel-combat-simulation)
9. [Phase 6 — Integration, Benchmarking & Tuning](#phase-6--integration-benchmarking--tuning)
10. [Risk Register](#risk-register)
11. [Migration Strategy](#migration-strategy)
12. [Appendix A — Wire Protocol Specification](#appendix-a--wire-protocol-specification)
13. [Appendix B — Shared Memory Layout](#appendix-b--shared-memory-layout)
14. [Appendix C — File Inventory & Change Map](#appendix-c--file-inventory--change-map)

---

## 1. Problem Statement

### 1.1 — HTTP Overhead Per Step

The current training loop makes **one HTTP request per agent action**. Each `/step` call
incurs:

| Cost Component | Estimated Time |
|---|---|
| TCP connection overhead (even with keep-alive) | ~0.1–0.5 ms |
| HTTP header parsing (request + response) | ~0.1–0.3 ms |
| JSON serialization of 612-float observation + 92-int mask + info dict | ~0.2–0.5 ms |
| JSON deserialization on Python side | ~0.1–0.3 ms |
| Express.js middleware dispatch | ~0.05–0.1 ms |
| **Total per-step overhead** | **~0.6–1.7 ms** |

A typical game has ~100–300 steps. At ~1 ms overhead per step, that's **100–300 ms of pure
IPC overhead per game** — comparable to or exceeding the actual game simulation time.

With self-play (`/step-multi`), each call processes 8 players but still pays the HTTP+JSON
tax: 8 observations × 612 floats + 8 masks × 92 ints = **~40 KB of JSON per round-trip**.

### 1.2 — Single-Environment Bottleneck

The current architecture runs **one `TrainingEnv` instance per Node.js process**. The Python
trainer creates a single `PokemonAutoChessEnv` (or an 8-seat `SelfPlayVecEnv`), meaning:

- PPO's rollout buffer collects from **one game at a time**
- No overlap between environment stepping and policy inference
- CPU cores sit idle while waiting for HTTP responses

SB3's `SubprocVecEnv` could parallelize, but each sub-env would need its own Node.js server
on a different port — manual, fragile, and still HTTP-bound.

### 1.3 — GPU Underutilization

The MaskablePPO policy (256×256 MLP) runs on GPU for gradient updates but does **single-sample
CPU inference** during rollout collection. The GPU is idle ~95% of the time. Batching
inference across N parallel environments would amortize GPU kernel launch costs and saturate
memory bandwidth.

### 1.4 — Summary of Goals

| Metric | Current (HTTP) | Target (Phase 6) | Speedup |
|---|---|---|---|
| IPC overhead per step | ~1 ms | <0.01 ms | ~100× |
| Serialization per step | ~0.3 ms (JSON) | ~0.005 ms (binary memcpy) | ~60× |
| Parallel environments | 1 | 16–64 | 16–64× |
| GPU utilization during rollout | <5% | >60% | ~12× |
| **Effective training throughput** | **~1k steps/sec** | **~50k–200k steps/sec** | **50–200×** |

---

## 2. Current Architecture Inventory

### 2.1 — File Map

```
app/training/
├── index.ts                  # Entry point: MongoDB init → startTrainingServer()
├── training-server.ts        # Express HTTP server (9 endpoints, 210 lines)
├── training-env.ts           # Core game env (2,572 lines) — reset/step/stepBatch
├── training-config.ts        # Constants, obs/action space, reward shaping (221 lines)
├── headless-room.ts          # Mock GameRoom for simulation (175 lines)

training/
├── train_ppo.py              # SB3 MaskablePPO trainer (370 lines)
├── pac_env.py                # Gymnasium wrapper — HTTP client (101 lines)
├── selfplay_vec_env.py       # VecEnv for 8-player self-play (157 lines)
├── requirements.txt          # gymnasium, sb3, torch, etc.
├── OPTION_B_MASTER_PLAN.md   # Phase 0-7 design doc (existing)
```

### 2.2 — Data Flow (Current)

```
┌─────────────────────┐        HTTP/JSON          ┌────────────────────────┐
│  Python Trainer      │ ◄──────────────────────► │  Node.js Server        │
│                      │                           │                        │
│  train_ppo.py        │   POST /step {action:5}   │  training-server.ts    │
│  pac_env.py          │ ─────────────────────────►│    ↓                   │
│  MaskablePPO (SB3)   │                           │  training-env.ts       │
│                      │   {obs:[612], mask:[92],  │    .step(5)            │
│  GPU: gradient only  │    reward, done, info}    │    .getObservation()   │
│                      │ ◄─────────────────────────│    .getActionMask()    │
└─────────────────────┘                           └────────────────────────┘
         │                                                    │
         │ (idle during step)                                 │ (idle during inference)
         ▼                                                    ▼
      1 game at a time                               1 TrainingEnv instance
```

### 2.3 — Key Data Structures

| Structure | Size | Serialization Cost |
|---|---|---|
| Observation vector | 612 × float32 = 2,448 bytes | ~800 bytes JSON (compressed) |
| Action mask | 92 × int8 = 92 bytes | ~200 bytes JSON |
| Info dict (full) | ~2–5 KB JSON (board state, synergies, etc.) | ~2–5 KB |
| Action | 1 × int = 4 bytes | ~10 bytes JSON |
| StepResult (total) | ~3–8 KB JSON | — |
| Self-play batch (8×) | ~24–64 KB JSON | — |

### 2.4 — Existing Optimizations (Phase 7.2)

Already in `training-env.ts`:
- **Position grid cache** (line 167): `Map<number, Pokemon>` per player, O(1) lookups
- **Observation cache** (line 170): reused unless dirty-flagged
- **Item pair cache** (line 169): enumerated once per invalidation
- **Cache invalidation** (line 2120): dirty flags on board mutations

These are **in-process optimizations** — they reduce game logic time but don't touch IPC.

---

## 3. Performance Budget & Targets

### 3.1 — Step Time Breakdown (Target)

| Component | Current | Phase 1 | Phase 2 | Phase 4+ |
|---|---|---|---|---|
| IPC transport | ~0.5 ms | ~0.05 ms | ~0.005 ms | ~0.005 ms |
| Serialization | ~0.3 ms | ~0.02 ms | ~0.001 ms | ~0.001 ms |
| Game logic (step) | ~0.1 ms | ~0.1 ms | ~0.1 ms | ~0.1 ms |
| Observation build | ~0.05 ms | ~0.05 ms | ~0.01 ms | ~0.01 ms |
| Policy inference | ~0.5 ms | ~0.5 ms | ~0.5 ms | ~0.02 ms (batched GPU) |
| **Total per step** | **~1.5 ms** | **~0.7 ms** | **~0.6 ms** | **~0.14 ms** |
| **Steps/sec (1 env)** | **~670** | **~1,400** | **~1,600** | **~7,000** |
| **Steps/sec (32 envs)** | **N/A** | **N/A** | **N/A** | **~100k+** |

### 3.2 — Success Criteria

- **Phase 1 gate:** Single-env step latency < 0.2 ms IPC overhead (down from ~1 ms)
- **Phase 2 gate:** Zero-copy observation transfer measurable via benchmark
- **Phase 3 gate:** 16 parallel envs collecting rollouts independently
- **Phase 4 gate:** GPU inference batch size ≥ 16, GPU utilization > 50% during rollout
- **Phase 6 gate:** End-to-end 50× throughput improvement over HTTP baseline

---

## Phase 1 — Binary IPC Protocol (Unix Sockets)

**Goal:** Eliminate HTTP overhead and JSON serialization. Replace with a lightweight binary
protocol over Unix domain sockets.

**Estimated effort:** Medium
**Risk:** Low
**Speedup:** ~2× (IPC cost drops from ~1 ms to ~0.05 ms per step)

---

### 1.1 — Define Binary Wire Protocol

**What:** Design a fixed-format binary message protocol for all training RPCs.

**Why:** JSON parsing alone costs ~0.3 ms per step. A fixed binary layout can be read with
a single `memcpy` / `struct.unpack` on the Python side and a `Buffer.write*` on the Node side.

**Protocol design (see Appendix A for full spec):**

```
┌──────────────────────────────────────────────────────────┐
│  Message Frame                                           │
├──────────┬──────────┬──────────┬─────────────────────────┤
│ msg_type │ msg_id   │ body_len │ body (variable)         │
│ uint8    │ uint32   │ uint32   │ [body_len bytes]        │
│ 1 byte   │ 4 bytes  │ 4 bytes  │ ...                     │
└──────────┴──────────┴──────────┴─────────────────────────┘
```

**Message types:**

| msg_type | Name | Direction | Body Format |
|---|---|---|---|
| 0x01 | RESET_REQ | Python → Node | (empty) |
| 0x02 | RESET_RESP | Node → Python | `StepResultBinary` |
| 0x03 | STEP_REQ | Python → Node | `int32 action` |
| 0x04 | STEP_RESP | Node → Python | `StepResultBinary` |
| 0x05 | STEP_MULTI_REQ | Python → Node | `int32[8] actions` |
| 0x06 | STEP_MULTI_RESP | Node → Python | `StepResultBinary[8]` |
| 0x07 | OBSERVE_REQ | Python → Node | (empty) |
| 0x08 | OBSERVE_RESP | Node → Python | `ObserveBinary` |
| 0x09 | HEALTH_REQ | Python → Node | (empty) |
| 0x0A | HEALTH_RESP | Node → Python | `uint8 initialized` |
| 0x0B | SPACES_REQ | Python → Node | (empty) |
| 0x0C | SPACES_RESP | Node → Python | `int32 obs_n, int32 act_n` |

**StepResultBinary layout** (fixed size per player):

```
┌───────────────────────────────────────────────────┐
│ observation: float32[612]  =  2,448 bytes         │
│ action_mask: uint8[92]     =     92 bytes         │
│ reward:      float32       =      4 bytes         │
│ done:        uint8         =      1 byte          │
│ rank:        uint8         =      1 byte          │
│ life:        uint16        =      2 bytes         │
│ money:       uint16        =      2 bytes         │
│ stage:       uint16        =      2 bytes         │
│ board_size:  uint8         =      1 byte          │
│ level:       uint8         =      1 byte          │
│ synergy_ct:  uint8         =      1 byte          │
│ items_held:  uint8         =      1 byte          │
│ actions_turn:uint8         =      1 byte          │
│ ─────────────────────────────────────────────────  │
│ Total: 2,557 bytes per player                     │
│ (vs ~3,000–8,000 bytes JSON)                      │
└───────────────────────────────────────────────────┘
```

**Mini-steps:**

1. **1.1a** — Create `app/training/ipc-protocol.ts` with:
   - `enum MessageType` matching table above
   - `encodeStepResult(result: StepResult): Buffer` — writes fixed binary layout
   - `decodeAction(buf: Buffer): number` — reads int32 action
   - `encodeMultiStepResult(results: StepResult[]): Buffer` — 8× StepResultBinary
   - `HEADER_SIZE = 9` (1 + 4 + 4)
   - `STEP_RESULT_SIZE = 2557`
   - Unit tests: round-trip encode/decode for every message type

2. **1.1b** — Create `training/ipc_protocol.py` with:
   - `struct`-based `unpack_step_result(buf: bytes) -> (obs, mask, reward, done, info)`
   - `pack_action(action: int) -> bytes`
   - `pack_multi_action(actions: list[int]) -> bytes`
   - `HEADER_FORMAT = '<BII'` (msg_type, msg_id, body_len)
   - Unit tests: must match TypeScript encoder output byte-for-byte

3. **1.1c** — Write a cross-language conformance test:
   - TypeScript encodes a known StepResult → write to file
   - Python reads file → decodes → asserts values match
   - Run as CI check to prevent protocol drift

---

### 1.2 — Unix Socket Server (Node.js Side)

**What:** Replace Express HTTP server with a `net.Server` listening on a Unix domain socket.

**Why:** Unix sockets eliminate TCP overhead (no Nagle, no SYN/ACK, no port allocation).
Measured latency for Unix socket IPC is ~5–20 μs vs ~100–500 μs for localhost TCP.

**Mini-steps:**

1. **1.2a** — Create `app/training/ipc-server.ts`:
   - `net.createServer()` listening on `/tmp/pac-training.sock` (configurable via
     `TRAINING_SOCKET_PATH` env var)
   - Connection handler: accumulates incoming bytes into a frame buffer
   - Frame parser: reads header (9 bytes), waits for `body_len` bytes, dispatches
   - Dispatch table: `msg_type → handler(body, conn)` mapping
   - Response writer: `conn.write(encodeHeader(type, id, len) + body)`
   - Handles backpressure: if `conn.write()` returns false, pause reading until `drain`
   - Cleanup: `fs.unlinkSync(socketPath)` on exit/SIGTERM

2. **1.2b** — Wire up handlers to existing `TrainingEnv`:
   - `RESET_REQ → env.reset() → encodeStepResult → RESET_RESP`
   - `STEP_REQ → env.step(action) → encodeStepResult → STEP_RESP`
   - `STEP_MULTI_REQ → env.stepBatch(actions) → encodeMultiStepResult → STEP_MULTI_RESP`
   - `OBSERVE_REQ → env.getObservation() + env.getActionMask() → OBSERVE_RESP`
   - `HEALTH_REQ → HEALTH_RESP { initialized: 1 }`
   - `SPACES_REQ → SPACES_RESP { obs_n: 612, act_n: 92 }`

3. **1.2c** — Update `app/training/index.ts` to detect transport mode:
   - If `TRAINING_TRANSPORT=socket` (default going forward): start IPC server
   - If `TRAINING_TRANSPORT=http`: start existing Express server (backward compat)
   - Log which transport is active at startup

4. **1.2d** — Add graceful shutdown:
   - `process.on('SIGTERM', ...)` and `process.on('SIGINT', ...)` → close socket, cleanup file
   - Handle client disconnect: reset env state, log warning

---

### 1.3 — Unix Socket Client (Python Side)

**What:** New Python transport layer that connects to the Unix socket.

**Mini-steps:**

1. **1.3a** — Create `training/ipc_client.py`:
   - `class IPCClient`:
     - `__init__(self, socket_path="/tmp/pac-training.sock")`
     - `connect()` — `socket.socket(AF_UNIX, SOCK_STREAM)`, retry with backoff
     - `_send(msg_type, body)` — write header + body atomically
     - `_recv()` — read header, then body, return `(msg_type, msg_id, body)`
     - `reset() -> (obs, info)`
     - `step(action) -> (obs, reward, done, truncated, info)`
     - `step_multi(actions) -> (observations, rewards, dones, infos)`
     - `health() -> bool`
     - `spaces() -> (obs_n, act_n)`
   - Socket-level: `MSG_NOSIGNAL` flag, `SO_RCVBUF`/`SO_SNDBUF` tuned to 64 KB

2. **1.3b** — Create `training/pac_env_ipc.py`:
   - `class PokemonAutoChessIPCEnv(gym.Env)`:
     - Same interface as existing `PokemonAutoChessEnv` in `pac_env.py`
     - Replaces `requests.Session` with `IPCClient`
     - `observation_space`, `action_space` fetched via `SPACES_REQ` on init
     - `action_masks()` extracted from binary response (no JSON parsing)
   - Drop-in replacement: `train_ppo.py` switches env class based on config flag

3. **1.3c** — Create `training/selfplay_vec_env_ipc.py`:
   - Same semantics as `selfplay_vec_env.py` but uses `IPCClient.step_multi()`
   - Action masks: `(8, 92)` numpy array extracted directly from binary buffer
   - Auto-reset on game end (same logic as existing)

4. **1.3d** — Update `training/train_ppo.py`:
   - Add `--transport` flag: `http` (default for backward compat) or `socket`
   - When `socket`: instantiate `PokemonAutoChessIPCEnv` or `SelfPlayVecIPCEnv`
   - When `http`: use existing env classes (no change)

---

### 1.4 — Benchmark & Validate Phase 1

**What:** Prove correctness and measure speedup.

**Mini-steps:**

1. **1.4a** — Correctness test:
   - Run 100 games with HTTP transport, log (action, obs, reward, done) traces
   - Run same 100 games with socket transport (same random seed)
   - Assert traces are byte-identical (obs values, rewards, done flags)

2. **1.4b** — Latency benchmark:
   - Measure round-trip time for 10,000 consecutive `/step` calls
   - Compare HTTP vs socket: expect ~2× improvement
   - Log p50, p95, p99 latencies

3. **1.4c** — Throughput benchmark:
   - Run full training for 50k steps with both transports
   - Measure wall-clock time, steps/sec
   - Record in `training/benchmarks/phase1_results.md`

---

## Phase 2 — Shared Memory Transport

**Goal:** Eliminate all copy overhead by having Node.js and Python read/write the same
memory region directly.

**Estimated effort:** Medium-High
**Risk:** Medium (platform-specific, requires careful synchronization)
**Speedup:** ~1.5× over Phase 1 (serialization drops to near-zero)

---

### 2.1 — Design Shared Memory Layout

**What:** Allocate a POSIX shared memory region (`shm_open`) containing pre-allocated
buffers for observations, actions, and masks.

**Why:** Even with binary protocol over Unix sockets, we still `memcpy` ~2.5 KB per step.
With shared memory, the observation array lives in a `mmap`'d region that both processes
can read without any copy.

**Layout (see Appendix B for byte offsets):**

```
┌─────────────────────────────────────────────────────────────────┐
│  Shared Memory Region: /pac-training-shm                       │
│  Total size: ~328 KB (for 64 parallel environments)            │
├─────────────────────────────────────────────────────────────────┤
│  Control Block (64 bytes)                                      │
│  ├── magic:       uint32 = 0x50414354 ("PACT")                 │
│  ├── version:     uint32 = 1                                   │
│  ├── num_envs:    uint32                                       │
│  ├── obs_size:    uint32 = 612                                 │
│  ├── act_size:    uint32 = 92                                  │
│  ├── padding:     44 bytes                                     │
│  └── (cache-line aligned to 64 bytes)                          │
├─────────────────────────────────────────────────────────────────┤
│  Per-Environment Slots (× num_envs)                            │
│  Each slot (5,120 bytes, padded to 4KB page boundary):         │
│  ├── sem_action_ready:  uint32 (atomic flag)                   │
│  ├── sem_result_ready:  uint32 (atomic flag)                   │
│  ├── action:            int32                                  │
│  ├── reward:            float32                                │
│  ├── done:              uint8                                  │
│  ├── rank:              uint8                                  │
│  ├── padding:           2 bytes                                │
│  ├── observation:       float32[612] = 2,448 bytes             │
│  ├── action_mask:       uint8[92]    =    92 bytes             │
│  ├── info_compact:      (life, money, stage, etc.) 32 bytes    │
│  └── padding to 5,120 bytes                                    │
└─────────────────────────────────────────────────────────────────┘
```

**Mini-steps:**

1. **2.1a** — Document shared memory layout in `training/SHM_SPEC.md`
   - Byte offsets for every field
   - Alignment requirements (observation array must be 4-byte aligned for float32)
   - Endianness: little-endian (x86/ARM64 native)

2. **2.1b** — Define constants in both languages:
   - TypeScript: `app/training/shm-layout.ts` — `SHM_NAME`, `SLOT_SIZE`, field offsets
   - Python: `training/shm_layout.py` — matching constants, numpy dtype definitions

---

### 2.2 — Node.js Shared Memory Writer

**What:** Node.js creates the shared memory region and writes observation/mask directly
into the mapped buffer after each step.

**Mini-steps:**

1. **2.2a** — Create `app/training/shm-transport.ts`:
   - Use `shm_open` via `node-ffi-napi` or a small C++ addon (`node-addon-api`)
   - Alternative: use Node.js `SharedArrayBuffer` + file-backed `mmap` via native addon
   - `createSharedRegion(numEnvs)` → returns typed array views
   - `writeObservation(envIdx, obs: number[])` → `Float32Array.set()` into mapped region
   - `writeMask(envIdx, mask: number[])` → `Uint8Array.set()` into mapped region
   - `writeReward(envIdx, r: number)` → single float32 write
   - `setResultReady(envIdx)` → atomic store `sem_result_ready = 1`
   - `waitForAction(envIdx)` → spin-wait or `Atomics.wait()` on `sem_action_ready`

2. **2.2b** — Modify `getObservation()` to write directly to shared memory:
   - Currently (line 1743): `const obs: number[] = []` → push floats → return array
   - New: write floats directly into `Float32Array` view at correct offset
   - Eliminates intermediate array allocation + GC pressure
   - Keep the `number[]` path as fallback for HTTP/socket transports

3. **2.2c** — Modify `getActionMask()` similarly:
   - Currently (line 1955): `const mask = new Array(TOTAL_ACTIONS).fill(0)`
   - New: write directly into `Uint8Array` view at correct offset

---

### 2.3 — Python Shared Memory Reader

**What:** Python opens the same shared memory region and reads observations as numpy arrays
with zero copy.

**Mini-steps:**

1. **2.3a** — Create `training/shm_client.py`:
   - `multiprocessing.shared_memory.SharedMemory(name="pac-training-shm", create=False)`
   - `numpy.ndarray(shape=(612,), dtype=np.float32, buffer=shm.buf, offset=obs_offset)`
   - This gives a **zero-copy numpy view** — no deserialization at all
   - `read_action_mask(env_idx)` → `np.ndarray` view into mask region
   - `write_action(env_idx, action)` → single int32 write
   - `set_action_ready(env_idx)` → atomic store
   - `wait_for_result(env_idx)` → spin-wait with `time.sleep(0)` yield

2. **2.3b** — Create `training/pac_env_shm.py`:
   - `class PokemonAutoChessSHMEnv(gym.Env)`:
     - `__init__(env_idx, shm_name)` — opens shared memory, creates numpy views
     - `step(action)` → write action, set flag, wait for result, return numpy view
     - `action_masks()` → return mask numpy view directly (no copy!)
     - `reset()` → write reset flag, wait for result

3. **2.3c** — Synchronization protocol:
   - Use atomic flags (not mutexes) for lock-free signaling
   - Python writes action + sets `sem_action_ready = 1`
   - Node reads action, processes step, writes result, sets `sem_result_ready = 1`
   - Python spins on `sem_result_ready`, reads result, clears both flags
   - Spin-wait with `pause` instruction hint (Python: `time.sleep(0)`) to avoid burning CPU
   - Fallback: if spin exceeds 10 ms, switch to `select()`-based notification via a
     companion Unix socket (hybrid approach)

---

### 2.4 — Benchmark & Validate Phase 2

1. **2.4a** — Correctness: same trace comparison as Phase 1.4a
2. **2.4b** — Measure per-step overhead: should be <0.01 ms for IPC+serialization
3. **2.4c** — Measure GC pressure: Node.js `--trace-gc` before/after (expect fewer minor GCs
   due to eliminated `number[]` allocations in `getObservation`)

---

## Phase 3 — Multi-Environment Parallelism (CPU)

**Goal:** Run N independent `TrainingEnv` instances in parallel Node.js workers, so the
Python trainer can collect rollouts from many games simultaneously.

**Estimated effort:** Medium
**Risk:** Low-Medium
**Speedup:** Linear in N (16 envs → ~16× throughput)

---

### 3.1 — Node.js Worker Pool Architecture

**What:** Use `worker_threads` to run N `TrainingEnv` instances in parallel.

**Why:** Each `TrainingEnv` is independent (separate `GameState`, separate RNG). Workers
share the process memory space but each has its own event loop, avoiding the GIL-equivalent
problem of a single-threaded Node.js process.

```
┌─────────────────────────────────────────────────────────────┐
│  Main Thread (Coordinator)                                  │
│  ├── Manages shared memory region                           │
│  ├── Spawns N worker threads                                │
│  ├── Routes actions to correct worker                       │
│  └── Handles lifecycle (init, shutdown, error recovery)     │
├─────────────────────────────────────────────────────────────┤
│  Worker 0          │  Worker 1          │  Worker N-1       │
│  TrainingEnv #0    │  TrainingEnv #1    │  TrainingEnv #N-1 │
│  reads action[0]   │  reads action[1]   │  reads action[N-1]│
│  writes obs[0]     │  writes obs[1]     │  writes obs[N-1]  │
│  (independent)     │  (independent)     │  (independent)    │
└────────────────────┴────────────────────┴───────────────────┘
                              ▲
                              │ Shared Memory
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Python Trainer Process                                     │
│  ├── Reads obs[0..N-1] as numpy views (zero-copy)           │
│  ├── Batches N observations → GPU inference                 │
│  ├── Writes action[0..N-1]                                  │
│  └── Collects rollouts from all N envs simultaneously       │
└─────────────────────────────────────────────────────────────┘
```

**Mini-steps:**

1. **3.1a** — Create `app/training/env-worker.ts`:
   - Worker entry point: receives `{ envIdx, shmName }` via `workerData`
   - Creates `TrainingEnv` instance, calls `initialize()`, `reset()`
   - Main loop: wait for action signal → `env.step(action)` → write result → signal done
   - Error handling: catch exceptions in step, write error flag, log, auto-reset

2. **3.1b** — Create `app/training/worker-pool.ts`:
   - `class WorkerPool`:
     - `constructor(numWorkers: number)` — spawn workers, allocate shared memory
     - `initialize()` — wait for all workers to report ready
     - `shutdown()` — terminate workers, cleanup shared memory
   - Each worker gets its own `TrainingEnv` (separate game state, RNG)
   - Workers are pinned to shared memory slots by index

3. **3.1c** — Update `app/training/index.ts`:
   - New env var: `TRAINING_NUM_ENVS` (default: 1 for backward compat)
   - If `NUM_ENVS > 1`: create `WorkerPool` instead of single `TrainingEnv`
   - If `NUM_ENVS == 1`: keep existing single-threaded path (no regression)

4. **3.1d** — Handle MongoDB connection sharing:
   - MongoDB connection must be established in main thread
   - Bot data (`cachedBots`) serialized to workers via `postMessage()` during init
   - Workers don't connect to MongoDB themselves (avoid connection pool explosion)
   - Alternative: pre-fetch bots in main thread, write to shared buffer, workers read

---

### 3.2 — Python Multi-Env Collector

**What:** Python-side vectorized environment that reads from N shared memory slots.

**Mini-steps:**

1. **3.2a** — Create `training/parallel_vec_env.py`:
   - `class ParallelSHMVecEnv(VecEnv)`:
     - `num_envs = N`
     - `step_async(actions)` → write N actions to shared memory slots simultaneously
     - `step_wait()` → spin-wait until all N `sem_result_ready` flags are set
     - `reset()` → write reset flags for all N slots
     - `action_masks()` → return `(N, 92)` numpy array (views into shared memory)
   - Observations returned as `(N, 612)` numpy array — contiguous in shared memory

2. **3.2b** — Ensure SB3 compatibility:
   - `ParallelSHMVecEnv` must implement full `VecEnv` interface
   - `env_method("action_masks")` must work for `MaskablePPO`
   - `get_attr`, `set_attr`, `env_is_wrapped` stubs

3. **3.2c** — Handle auto-reset:
   - When a game finishes (`done=True`), the worker auto-resets and writes the
     new initial observation to the slot
   - Python sees `done=True` + the initial obs of the next game (standard VecEnv contract)
   - No extra round-trip needed for reset

---

### 3.3 — Self-Play × Multi-Env

**What:** Support N parallel self-play games (N × 8 = N×8 agents).

**Mini-steps:**

1. **3.3a** — Each worker runs one 8-player self-play game independently
   - Worker receives 8 actions per step (one per player seat)
   - Shared memory slot expanded: 8 × StepResult per slot
   - `SLOT_SIZE` grows to 8 × 5,120 = 40,960 bytes per environment

2. **3.3b** — Python `SelfPlayParallelVecEnv`:
   - `num_envs = N * 8` (each physical env exposes 8 sub-envs)
   - `step_async(actions)` → pack 8 actions per physical env slot
   - Internally groups actions by physical env, writes to shared memory
   - Returns flattened (N×8, 612) observation array

---

### 3.4 — Benchmark & Validate Phase 3

1. **3.4a** — Correctness: each worker produces valid game traces independently
2. **3.4b** — Scaling test: measure throughput at N = 1, 2, 4, 8, 16, 32
3. **3.4c** — Expected: near-linear scaling up to CPU core count, then plateau
4. **3.4d** — Memory usage: each `TrainingEnv` + `GameState` ≈ 10–50 MB; 32 envs ≈ 0.3–1.6 GB

---

## Phase 4 — GPU-Accelerated Inference & Batched Rollouts

**Goal:** Move policy inference to the GPU and batch across all N parallel environments,
so the GPU is utilized during rollout collection (not just gradient updates).

**Estimated effort:** Medium
**Risk:** Medium (requires custom rollout loop)
**Speedup:** ~10–20× for inference step (GPU batch vs CPU single)

---

### 4.1 — Custom Batched Rollout Collector

**What:** Replace SB3's default `collect_rollouts()` with a custom loop that batches
inference across all N environments.

**Why:** SB3's `SubprocVecEnv` already supports batched inference, but our shared-memory
`ParallelSHMVecEnv` is faster because it avoids process-level IPC. The key optimization is
ensuring the policy forward pass receives a batch of N observations in a single GPU call.

**Mini-steps:**

1. **4.1a** — Verify SB3 batched inference path:
   - SB3's `OnPolicyAlgorithm.collect_rollouts()` already calls
     `self.policy.predict(obs_batch)` where `obs_batch.shape = (n_envs, obs_dim)`
   - Confirm this works with `ParallelSHMVecEnv` — should be automatic
   - If SB3 iterates envs one-by-one, need custom collector

2. **4.1b** — Profile GPU utilization during rollout:
   - Use `torch.cuda.Event` timers around `policy.predict()`
   - With N=1: expect ~0.1 ms inference, GPU util <5%
   - With N=32: expect ~0.2 ms inference (batched), GPU util ~40–60%
   - Log: `inference_time_ms`, `env_step_time_ms`, `gpu_utilization`

3. **4.1c** — If SB3 doesn't batch properly, create `training/batched_collector.py`:
   - Custom rollout loop:
     ```
     for step in range(n_steps):
         obs_batch = vec_env.get_observations()       # (N, 612) from shared memory
         masks_batch = vec_env.action_masks()          # (N, 92) from shared memory
         with torch.no_grad():
             actions, values, log_probs = policy(obs_batch, masks_batch)  # single GPU call
         vec_env.step_async(actions.cpu().numpy())
         obs_new, rewards, dones, infos = vec_env.step_wait()
         rollout_buffer.add(obs_batch, actions, rewards, ...)
     ```
   - Key: `obs_batch` stays as `torch.Tensor` on GPU if possible (avoid CPU↔GPU copy)

---

### 4.2 — GPU-Resident Observation Tensors

**What:** Keep observation tensors on GPU to avoid CPU↔GPU transfer per step.

**Mini-steps:**

1. **4.2a** — Create CUDA-registered shared memory:
   - Use `torch.cuda.memory.caching_allocator` or `cuMemHostRegister` to pin the
     shared memory region
   - Pinned memory enables async DMA transfers: `torch.cuda.FloatTensor` can be
     constructed with zero-copy from pinned host memory
   - This eliminates the `numpy → torch.tensor().cuda()` copy chain

2. **4.2b** — Modify `ParallelSHMVecEnv` to return pinned tensors:
   - `get_observations()` returns `torch.FloatTensor` on CUDA device
   - `action_masks()` returns `torch.BoolTensor` on CUDA device
   - SB3's `MaskablePPO` needs patching to accept GPU tensors in `predict()`

3. **4.2c** — Fallback path:
   - If CUDA not available, fall back to standard numpy → torch → cuda transfer
   - Detect with `torch.cuda.is_available()`

---

### 4.3 — Hyperparameter Adjustments for Multi-Env

**What:** Tune PPO hyperparameters for the new multi-environment regime.

**Why:** With N parallel envs, the effective rollout size is `N × n_steps`. The batch size,
learning rate, and entropy coefficient need re-tuning.

**Mini-steps:**

1. **4.3a** — Adjust rollout/batch sizes:
   - Current: `n_steps=1024`, `batch_size=128`, `n_envs=1` → 1,024 transitions/rollout
   - With 32 envs: `n_steps=256`, `batch_size=512`, `n_envs=32` → 8,192 transitions/rollout
   - Larger batch → more stable gradients → can increase learning rate slightly

2. **4.3b** — Learning rate schedule:
   - Consider linear warmup (1k steps) → constant → linear decay
   - Higher LR may be stable with larger batches (linear scaling rule)

3. **4.3c** — Entropy coefficient:
   - With more diverse experience (32 games simultaneously), may need less entropy
     encouragement — reduce from 0.07 to 0.03–0.05
   - Monitor via TensorBoard: `entropy_loss` should stay above 1.0

---

### 4.4 — Benchmark & Validate Phase 4

1. **4.4a** — Measure end-to-end training throughput: steps/sec at N=1,4,8,16,32,64
2. **4.4b** — GPU utilization profiling: `nvidia-smi dmon` during training
3. **4.4c** — Training quality check: does 500k steps with 32 envs reach same reward as
   500k steps with 1 env? (Should be better due to more diverse experience)
4. **4.4d** — Memory profiling: GPU VRAM usage at each N level

---

## Phase 5 — GPU-Parallel Combat Simulation (Stretch Goal)

**Goal:** Port the combat simulation (`Simulation.update()`) to run on GPU using WebGPU
compute shaders or a CUDA kernel, enabling thousands of fights to run in parallel.

**Estimated effort:** Very High
**Risk:** High (complex game logic, many edge cases)
**Speedup:** ~10–50× for fight phase specifically

> **Note:** This phase is a stretch goal. Phases 1–4 provide the bulk of the speedup.
> Phase 5 is valuable only if fight simulation becomes the dominant bottleneck after
> Phases 1–4 eliminate IPC overhead.

---

### 5.1 — Profile Fight Phase Bottleneck

**What:** Determine if combat simulation is actually the bottleneck after IPC improvements.

**Mini-steps:**

1. **5.1a** — Instrument `runFightPhase()` (training-env.ts line 961):
   - Measure: time in `simulation.update()` loop vs time in reward computation
   - Current: `TRAINING_SIMULATION_DT = 50ms`, `TRAINING_MAX_FIGHT_STEPS = 2000`
   - A fight typically runs 50–200 substeps × N simulations (up to 4 PVP matchups)

2. **5.1b** — Collect fight duration statistics over 1000 games:
   - Mean, p50, p95, p99 fight duration (ms)
   - Mean substeps per fight
   - If fight phase < 20% of total step time after Phase 4: skip Phase 5

---

### 5.2 — Design GPU Simulation Kernel

**What:** Express the core combat loop as a data-parallel computation.

**Why:** Each fight is independent. With 32 parallel envs × 4 matchups per env = 128
simultaneous fights. Each fight has 2 teams of ~6 units each. This maps well to GPU warps.

**Architecture options:**

| Option | Pros | Cons |
|---|---|---|
| **A: CUDA kernel (C++)** | Maximum performance, full control | Hard to maintain parity with TS logic |
| **B: WebGPU compute shader** | Runs in Node.js via Dawn/wgpu | Immature tooling, limited debugging |
| **C: PyTorch custom op** | Stays in Python ecosystem | Still need to rewrite game logic |
| **D: Batched NumPy/JAX** | No GPU kernel authoring | Slower than native GPU, complex logic |

**Recommended: Option A (CUDA kernel) for maximum throughput, with Option D (JAX) as
a prototyping step.**

**Mini-steps:**

1. **5.2a** — Create a simplified combat model:
   - Strip Pokemon combat to essential mechanics: HP, ATK, DEF, SPE, range, types
   - Remove complex ability effects for v1 (add back incrementally)
   - Express as: `for each timestep: for each unit: find target → attack → apply damage`
   - This is embarrassingly parallel across (fights × units)

2. **5.2b** — Prototype in JAX:
   - `training/gpu_combat.py`:
     - `@jax.jit def simulate_fights(teams_blue, teams_red, max_steps) -> results`
     - `teams_blue.shape = (N_FIGHTS, MAX_UNITS, UNIT_FEATURES)`
     - `teams_red.shape = (N_FIGHTS, MAX_UNITS, UNIT_FEATURES)`
     - Inner loop: `jax.lax.while_loop` or `jax.lax.scan` over timesteps
   - Validate against TypeScript simulation: same inputs → same results (within float tol)

3. **5.2c** — Quantify speedup from JAX prototype:
   - 128 fights × 100 substeps on GPU vs Node.js sequential
   - If >10× speedup: proceed to CUDA kernel
   - If <5× speedup: JAX version may be sufficient

4. **5.2d** — (If needed) Write CUDA kernel:
   - `app/training/gpu/combat_kernel.cu`:
     - One CUDA thread per unit per fight
     - Shared memory for team state (fits in 48KB per block)
     - Synchronize within fight via `__syncthreads()`
   - Expose via PyTorch custom C++ extension or pybind11
   - Build with `setup.py` / `torch.utils.cpp_extension`

---

### 5.3 — Hybrid Fight Pipeline

**What:** Use GPU for combat, CPU for everything else (economy, shop, observations).

**Mini-steps:**

1. **5.3a** — Pre-fight: serialize board states to GPU-friendly tensors:
   - `encode_teams(player) -> (blue_tensor, red_tensor)` — fixed-size unit arrays
   - Transfer to GPU before fight starts

2. **5.3b** — Fight: run GPU simulation:
   - Returns: `(winner_per_fight, surviving_units, damage_dealt, kills)`

3. **5.3c** — Post-fight: transfer results back to CPU:
   - Update `GameState` from GPU results
   - Compute rewards (stays on CPU — complex logic, not perf-critical)

4. **5.3d** — Fallback: if GPU not available, use existing CPU simulation (no regression)

---

## Phase 6 — Integration, Benchmarking & Tuning

**Goal:** Polish, benchmark end-to-end, document, and ensure production readiness.

---

### 6.1 — End-to-End Integration

**Mini-steps:**

1. **6.1a** — Single launch script: `training/run_training.sh`:
   ```bash
   #!/bin/bash
   export TRAINING_TRANSPORT=shm        # shared memory
   export TRAINING_NUM_ENVS=${NUM_ENVS:-32}
   export SELF_PLAY=${SELF_PLAY:-false}

   # Start Node.js env server (worker pool)
   npx ts-node app/training/index.ts &
   NODE_PID=$!

   # Wait for ready
   sleep 2

   # Start Python trainer
   python training/train_ppo.py \
     --transport shm \
     --num-envs $TRAINING_NUM_ENVS \
     --timesteps ${TIMESTEPS:-1000000} \
     "$@"

   # Cleanup
   kill $NODE_PID
   ```

2. **6.1b** — Docker Compose for reproducible training:
   - Service 1: Node.js env server (CPU)
   - Service 2: Python trainer (GPU)
   - Shared volume for socket/shm files
   - GPU passthrough via `nvidia-docker`

3. **6.1c** — Configuration file `training/config.yaml`:
   ```yaml
   transport: shm        # http | socket | shm
   num_envs: 32
   self_play: false
   gpu_combat: false     # Phase 5 toggle
   hyperparams:
     learning_rate: 3e-4
     batch_size: 512
     n_steps: 256
     n_epochs: 10
     gamma: 0.99
     gae_lambda: 0.95
     clip_range: 0.2
     ent_coef: 0.05
   ```

---

### 6.2 — Comprehensive Benchmarking

**Mini-steps:**

1. **6.2a** — Benchmark matrix:

   | Config | Transport | Envs | GPU Inference | GPU Combat |
   |---|---|---|---|---|
   | Baseline | HTTP | 1 | No | No |
   | Phase 1 | Socket | 1 | No | No |
   | Phase 2 | SHM | 1 | No | No |
   | Phase 3 | SHM | 32 | No | No |
   | Phase 4 | SHM | 32 | Yes | No |
   | Phase 5 | SHM | 32 | Yes | Yes |

   Metrics per config:
   - Steps/second (wall clock)
   - Games/second
   - GPU utilization (%)
   - CPU utilization (%)
   - Memory usage (RSS)
   - GPU VRAM usage
   - Training reward at 500k steps (quality check)

2. **6.2b** — Automated benchmark runner:
   - `training/benchmark_suite.py` — runs each config, collects metrics, generates report
   - Output: `training/benchmarks/results_YYYYMMDD.md` with tables and charts

---

### 6.3 — Backward Compatibility

**Mini-steps:**

1. **6.3a** — HTTP transport remains functional:
   - All existing endpoints preserved in `training-server.ts`
   - `pac_env.py` still works with `--transport http`
   - No breaking changes to existing training scripts

2. **6.3b** — Graceful degradation:
   - If shared memory unavailable: fall back to socket
   - If socket unavailable: fall back to HTTP
   - If GPU unavailable: CPU inference only
   - If worker threads fail: single-threaded mode
   - Each fallback logged with clear warning message

3. **6.3c** — Feature detection:
   - Python auto-detects available transports on startup
   - Logs: "Using transport: shm (32 envs, GPU inference enabled)"

---

### 6.4 — Documentation & Cleanup

1. **6.4a** — Update `training/OPTION_B_MASTER_PLAN.md` Known Limitations section:
   - Remove limitation #2 ("HTTP-per-step overhead") — resolved
   - Add new section on transport options and GPU acceleration

2. **6.4b** — Architecture diagram in `training/ARCHITECTURE.md`
3. **6.4c** — Troubleshooting guide: common issues with shared memory, GPU, workers

---

## Risk Register

| ID | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Shared memory platform incompatibility (macOS vs Linux) | Medium | Medium | Unix sockets as fallback; test on both platforms |
| R2 | Worker thread memory explosion (32 × GameState) | High | Low | Profile memory per env; cap at available RAM / 50MB |
| R3 | GPU combat kernel diverges from TypeScript logic | High | High | Extensive conformance testing; gate on 99.9% match |
| R4 | SB3 doesn't batch inference properly with custom VecEnv | Medium | Medium | Custom rollout collector as backup (Phase 4.1c) |
| R5 | Binary protocol version mismatch (TS vs Python) | Medium | Low | Version field in header; CI conformance tests |
| R6 | Spin-wait CPU waste in shared memory synchronization | Low | Medium | Hybrid spin+sleep; companion notification socket |
| R7 | Node.js `worker_threads` GC pauses affect step latency | Medium | Medium | `--max-old-space-size` per worker; monitor GC timing |
| R8 | Training quality regression from parallel experience | Medium | Low | Compare reward curves at 500k steps; A/B test |
| R9 | CUDA kernel compilation requires specific GPU/driver | Medium | High | Fallback to CPU combat; document minimum requirements |

---

## Migration Strategy

### Incremental Rollout

Each phase is independently deployable and benchmarkable. The recommended order:

```
Phase 1 (Socket IPC)     ← Do first. Lowest risk, immediate 2× gain.
    │
    ▼
Phase 3 (Multi-Env CPU)  ← Do second. Biggest absolute gain (16-32×).
    │                        Can skip Phase 2 initially — socket IPC is
    │                        fast enough that multi-env dominates.
    ▼
Phase 2 (Shared Memory)  ← Do third. Polishes per-step overhead.
    │                        Most impactful when N is large.
    ▼
Phase 4 (GPU Inference)  ← Do fourth. Unlocks GPU during rollout.
    │
    ▼
Phase 5 (GPU Combat)     ← Stretch. Only if fights dominate after Phase 4.
    │
    ▼
Phase 6 (Integration)    ← Final polish, benchmarks, documentation.
```

### Feature Flags

All new functionality gated behind environment variables:

| Variable | Values | Default |
|---|---|---|
| `TRAINING_TRANSPORT` | `http`, `socket`, `shm` | `http` |
| `TRAINING_NUM_ENVS` | 1–64 | 1 |
| `TRAINING_GPU_INFERENCE` | `true`, `false` | `false` |
| `TRAINING_GPU_COMBAT` | `true`, `false` | `false` |
| `TRAINING_SOCKET_PATH` | file path | `/tmp/pac-training.sock` |
| `TRAINING_SHM_NAME` | shm name | `pac-training-shm` |

---

## Appendix A — Wire Protocol Specification

### Frame Format

All multi-byte integers are **little-endian**.

```
Offset  Size  Type    Field
0       1     uint8   msg_type
1       4     uint32  msg_id (monotonic counter for request/response correlation)
5       4     uint32  body_length
9       N     bytes   body
```

### Message Bodies

**STEP_REQ (0x03):**
```
Offset  Size  Type    Field
0       4     int32   action (0-91)
```

**STEP_RESP (0x04) / RESET_RESP (0x02):**
```
Offset  Size    Type        Field
0       2448    float32[612] observation
2448    92      uint8[92]    action_mask
2540    4       float32      reward
2544    1       uint8        done
2545    1       uint8        rank
2546    2       uint16       life
2548    2       uint16       money
2550    2       uint16       stage
2552    1       uint8        board_size
2553    1       uint8        level
2554    1       uint8        synergy_count
2555    1       uint8        items_held
2556    1       uint8        actions_this_turn
────────────────────────────────────────
Total: 2557 bytes
```

**STEP_MULTI_REQ (0x05):**
```
Offset  Size  Type       Field
0       32    int32[8]   actions (one per player)
```

**STEP_MULTI_RESP (0x06):**
```
Offset  Size     Type               Field
0       20456    StepResultBinary[8] results (8 × 2557 bytes)
```

**SPACES_RESP (0x0C):**
```
Offset  Size  Type    Field
0       4     int32   obs_n (612)
4       4     int32   act_n (92)
```

---

## Appendix B — Shared Memory Layout

### Control Block (offset 0, 64 bytes)

```
Offset  Size  Type    Field
0       4     uint32  magic (0x50414354 = "PACT")
4       4     uint32  version (1)
8       4     uint32  num_envs
12      4     uint32  obs_size (612)
16      4     uint32  act_size (92)
20      4     uint32  slot_size (5120)
24      40    bytes   reserved/padding
```

### Per-Environment Slot (offset 64 + envIdx × 5120, 5120 bytes each)

```
Offset  Size    Type        Field
0       4       uint32      sem_action_ready (atomic: 0=idle, 1=action written)
4       4       uint32      sem_result_ready (atomic: 0=idle, 1=result written)
8       4       uint32      command (0=step, 1=reset, 2=shutdown)
12      4       int32       action
16      4       float32     reward
20      1       uint8       done
21      1       uint8       rank
22      2       uint16      life
24      2       uint16      money
26      2       uint16      stage
28      1       uint8       board_size
29      1       uint8       level
30      1       uint8       synergy_count
31      1       uint8       items_held
32      2448    float32[612] observation
2480    92      uint8[92]   action_mask
2572    2548    bytes       reserved/padding (to 5120)
```

### Total Region Size

- Control block: 64 bytes
- Per-env slots: `num_envs × 5120` bytes
- 32 envs: 64 + 32 × 5120 = **163,904 bytes (~160 KB)**
- 64 envs: 64 + 64 × 5120 = **327,744 bytes (~320 KB)**

---

## Appendix C — File Inventory & Change Map

### New Files

| File | Phase | Description |
|---|---|---|
| `app/training/ipc-protocol.ts` | 1 | Binary message encode/decode |
| `app/training/ipc-server.ts` | 1 | Unix socket server |
| `app/training/shm-layout.ts` | 2 | Shared memory constants |
| `app/training/shm-transport.ts` | 2 | Node.js shared memory writer |
| `app/training/env-worker.ts` | 3 | Worker thread entry point |
| `app/training/worker-pool.ts` | 3 | Worker thread coordinator |
| `training/ipc_protocol.py` | 1 | Binary message decode/encode |
| `training/ipc_client.py` | 1 | Unix socket client |
| `training/pac_env_ipc.py` | 1 | Gymnasium env (socket transport) |
| `training/selfplay_vec_env_ipc.py` | 1 | Self-play VecEnv (socket transport) |
| `training/shm_layout.py` | 2 | Shared memory constants |
| `training/shm_client.py` | 2 | Shared memory reader |
| `training/pac_env_shm.py` | 2 | Gymnasium env (SHM transport) |
| `training/parallel_vec_env.py` | 3 | Multi-env VecEnv (SHM) |
| `training/batched_collector.py` | 4 | Custom GPU-batched rollout |
| `training/gpu_combat.py` | 5 | JAX/CUDA combat simulation |
| `training/run_training.sh` | 6 | Launch script |
| `training/config.yaml` | 6 | Unified configuration |
| `training/benchmark_suite.py` | 6 | Automated benchmark runner |

### Modified Files

| File | Phase | Changes |
|---|---|---|
| `app/training/index.ts` | 1,3 | Transport detection, worker pool init |
| `app/training/training-env.ts` | 2 | Direct-write to shared memory in getObservation/getActionMask |
| `app/training/training-config.ts` | 1,3 | New env var constants |
| `training/train_ppo.py` | 1,3,4 | Transport flag, multi-env support, HP tuning |
| `training/requirements.txt` | 4,5 | Add jax, cuda deps |

### Unchanged Files

| File | Reason |
|---|---|
| `app/training/headless-room.ts` | Pure game logic, no IPC |
| `training/pac_env.py` | Preserved for HTTP backward compat |
| `training/selfplay_vec_env.py` | Preserved for HTTP backward compat |
| `training/OPTION_B_MASTER_PLAN.md` | Historical reference |
