# Training Environment Patch Notes

## Option B Master Plan v2 — Align Headless PPO Training Env to Agent-IO API

**Branch:** `claude/align-training-env-api-1rxmb`
**Target:** 92-action, 612-feature observation space matching the agent-io extension API bible.

---

### Phase 0 — Scaffolding Helpers (`e4d4c3f1c`)

**File:** `app/training/training-config.ts`

- Added grid constants: `GRID_WIDTH=8`, `GRID_HEIGHT=4`, `GRID_CELLS=32`, `SHOP_SLOTS=6`, `MAX_COMBINE_PAIRS=6`
- Added helper functions: `cellToXY()`, `xyToCell()`, `enumerateItemPairs()`, `findRecipeResult()`
- Added lookup tables with normalized indices: `getPkmSpeciesIndex()`, `getSynergyIndex()`, `getItemIndex()`, `getWeatherIndex()`

---

### Phase 1 — Action Space Index Remap (`9fe86c018`)

**File:** `app/training/training-config.ts`

- Expanded `TrainingAction` enum from 22 entries to 92 entries
- New layout aligned 1:1 with agent-io API: BUY 0-5, REFRESH 6, LEVEL_UP 7, LOCK_SHOP 8, END_TURN 9, MOVE 10-41, SELL 42-73, REMOVE_SHOP 74-79, PICK 80-85, COMBINE 86-91
- Set `TOTAL_ACTIONS = 92`

---

### Phase 2 — Implement All 7 New Action Groups (`255bf2442`)

**File:** `app/training/training-env.ts`

- Implemented 7 new action handlers in `executeAction()`:
  - **MOVE** (10-41): `moveUnitToCell()` — finds first available unit, moves to target cell with bench/board boundary checks
  - **SELL** (42-73): `sellPokemonAtCell()` — sells unit at grid position, returns items, updates synergies
  - **REMOVE_SHOP** (74-79): `removeFromShop()` — removes shop slot (gold-gated, no deduction)
  - **PICK** (80-85): `pickProposition()` — full proposition pick with duo support, evolution check, item pairing
  - **COMBINE** (86-91): `combineItems()` — combines held item pairs using recipe lookup
  - **LOCK_SHOP** (8): Toggle `player.shopLocked`
  - **LEVEL_UP** (7): `levelUp()` — 4 gold for 4 XP
- Updated `getActionMask()` with validity checks for all 92 actions
- Added `findFirstAvailableUnit()` helper for MOVE semantics

---

### Phase 3A — Observation Expansion Part 1 (`9bce14cc9`)

**File:** `app/training/training-env.ts`, `app/training/training-config.ts`

- Expanded player stats from 6 to 14 features: added `expNeeded`, `shopFreeRolls`, `rerollCount`, `shopLocked`, `totalMoneyEarned`, `totalPlayerDamageDealt`, `boardSize`, `rank`
- Expanded shop features from 4 to 9 per slot: added `speciesIndex`, `rarity`, `cost`, `type1-4`, `isEvoPossible`
- Expanded board features from 6 to 12 per slot: added `speciesIndex`, `stars`, `rarity`, `type1-4`, `atk`, `hp`, `range`, `numItems`
- Added held items encoding (10 normalized item indices)

### Phase 3B — Observation Expansion Part 2 (`5db5ba969`)

**File:** `app/training/training-env.ts`, `app/training/training-config.ts`

- Added synergy vector: 31 synergy values (normalized by /10)
- Added game info block (7 features): `stageLevel`, `phase`, `playersAlive`, `hasPropositions`, `weatherIndex`, `isPVE`, `maxTeamSize`
- Added opponent stats (7 opponents x 10 features = 70): `life`, `rank`, `level`, `gold`, `streak`, `boardSize`, `topSyn1Idx`, `topSyn1Count`, `topSyn2Idx`, `topSyn2Count`
- Added proposition encoding (6 slots x 7 features = 42): `speciesIndex`, `rarity`, `type1-4`, `itemIndex`
- **Total observation: 14 + 54 + 384 + 10 + 31 + 7 + 70 + 42 = 612 features**

---

### Phase 4 — Simplified Minigame Phases (`0226a1f30`)

**File:** `app/training/training-env.ts`

- Added `pickItemProposition()` for item-only propositions (no pokemon attached)
- Item carousel stages (4, 12, 17, 22, 27, 34): present 3 random items as propositions in `advanceToNextPickPhase()`
- PVE reward handling in `runFightPhase()`: direct rewards auto-given, reward propositions set for agent to pick
- Updated `executeAction()`, `getActionMask()`, and `getObservation()` to handle item-only proposition state
- Portal/unique/legendary propositions verified working via existing PICK actions

---

### Phase 5 — Python Training Pipeline (`d46e7a416`)

**File:** `training/train_ppo.py`, `training/requirements.txt`

- Switched from `stable_baselines3.PPO` to `sb3_contrib.MaskablePPO` for native action masking during rollout collection
- Deleted manual `ActionMaskCallback` class (no longer needed)
- Updated network architecture: `net_arch` from `[128, 128]` to `[256, 256]` (both policy and value networks)
- Updated hyperparameter defaults: `ent_coef` 0.01 -> 0.02, `n_steps` 512 -> 1024, `batch_size` 64 -> 128
- Updated `evaluate()` to pass `action_masks` to `model.predict()`
- Expanded `TrainingMetricsCallback` with 4 new TensorBoard curves: `mean_gold`, `mean_board_size`, `synergy_count`, `items_held`
- Added `StepResult.info` fields: `gold`, `boardSize`, `synergyCount`, `itemsHeld`
- Added `sb3-contrib>=2.2.0` to requirements.txt

---

### Phase 6 — Reward Tuning (`ad4815cfa`)

**File:** `app/training/training-env.ts`, `app/training/training-config.ts`

Four shaped reward signals added on top of base win/loss rewards:

1. **Economy interest bonus (6.1):** `REWARD_INTEREST_BONUS = 0.05` per interest gold earned, gated by board guard (`boardSize >= maxTeamSize - 2`) to prevent pure hoarding
2. **Synergy activation delta (6.2):** `REWARD_SYNERGY_THRESHOLD = 0.1` per newly activated synergy threshold. Uses `countActiveSynergyThresholds()` with `SynergyTriggers` data. Only positive deltas rewarded (no penalty for losing synergies). Snapshot taken at start of each pick phase.
3. **Combat damage (6.3):** `REWARD_PER_ENEMY_KILL = 0.02` per enemy unit killed. Required refactoring simulation lifecycle into 3 loops: `onFinish()` all -> extract kill data -> `stop()` all (because `stop()` clears team MapSchemas).
4. **HP preservation (6.4):** `REWARD_HP_SCALE = 0.005` bonus on wins, scaled by `player.life / 100`.

---

### Phase 7 — Integration & Final Verification (`b35881177`)

**Files:** `app/training/training-config.ts`, `app/training/training-env.ts`, `app/training/verify-actions.ts`

1. **Auto-place off + bench penalty (7.1):**
   - Flipped `TRAINING_AUTO_PLACE = false` — agent must use MOVE actions to place units on board
   - Added `REWARD_BENCH_PENALTY = -0.01` per bench unit when board has open slots at turn end
   - Applied in both `step()` (single-agent) and `stepBatch()` (self-play) paths

2. **Benchmark optimizations (7.2):** Three cache layers for hot-path performance:
   - **Position grid cache:** `Map<cellKey, Pokemon>` per player for O(1) `findPokemonAt()` (was O(n) board scan)
   - **Item pair cache:** Cached `enumerateItemPairs()` per player, shared by `combineItems()` and `getActionMask()`
   - **Observation cache:** Full 612-dim obs vector cached with dirty flag, skips recompute when unchanged
   - All caches invalidated after board mutations and phase transitions via `invalidatePlayerCaches()`

3. **Verification script (7.3):** `app/training/verify-actions.ts`
   - Asserts all 92 action indices match the API bible layout
   - Verifies `cellToXY` round-trip for all 32 cells
   - Verifies observation breakdown sums to 612
   - Prints formatted action/observation breakdown tables
   - Run: `npx ts-node --transpile-only app/training/verify-actions.ts`

---

### Files Modified (Cumulative)

| File | Description |
|------|-------------|
| `app/training/training-config.ts` | All constants, enums, helpers, reward tuning params |
| `app/training/training-env.ts` | Main Gym-like env: 92 actions, 612 obs, shaped rewards, caching |
| `app/training/training-server.ts` | HTTP API server (unchanged in this work) |
| `app/training/verify-actions.ts` | Action/observation space verification script (new) |
| `training/train_ppo.py` | MaskablePPO, 256x256 arch, expanded metrics |
| `training/requirements.txt` | Added sb3-contrib |

### Ground Truth Constants

| Constant | Value |
|----------|-------|
| Total actions | 92 |
| Total observation features | 612 |
| Grid | 8x4 (32 cells), y=0 bench, y=1-3 board |
| Shop slots | 6 |
| Synergy types | 31 |
| Pokemon species | 1139 |
| Max types per pokemon | 3 (+ 1 reserved) |
| Max items per pokemon | 3 |
| Item components | 10 |
| Craftable items | 56 |
| Max combine pairs | 6 |
| Proposition slots | 6 |
