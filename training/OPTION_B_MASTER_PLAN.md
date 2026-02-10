# Option B Master Plan v2: Align Headless Training Env to Agent-IO API (v5)

**Status:** Approved plan — ready for implementation
**Scope:** Expand the headless PPO training environment to match the 92-action, rich-observation agent-io API
**Audited by:** Cassy (Claude Opus) — all audit findings incorporated
**Implementation notes audit:** All 4 minor items verified (see Implementation Notes appendix)

---

## Verified Ground Truth

These constants are tested against live gameplay. Use as source of truth.

| Constant | Value | Source |
|---|---|---|
| Action space size | 92 | API bible |
| Grid dimensions | 8 wide x 4 rows (y=0 bench, y=1-3 board) | API bible + `BOARD_SIDE_HEIGHT=4` |
| Total grid cells | 32 | 8 x 4 |
| Shop slots | 6 | `SHOP_SIZE = 6` in `app/config/game/shop.ts` |
| Synergy count | 31 | `Synergy` enum in `app/types/enum/Synergy.ts` |
| Total Pkm entries | 1,139 | `Pkm` enum in `app/types/enum/Pokemon.ts` |
| Max base types per pokemon | 3 | Verified (e.g., Steelix: Rock/Ground/Steel) |
| Max items per pokemon | 3 | `this.items.size >= 3` check in `pokemon-entity.ts` |
| Max types per pokemon | 6 | 3 base + 3 stone items |
| Item components | 10 | `ItemComponents` array |
| Craftable items | 56 | `ItemRecipe` entries |
| Remove-from-shop cost | **0 gold** (requires gold >= buy price as gate, does NOT deduct) | `game-commands.ts:206-211` |
| Remove-from-shop side effects | Locks shop + releases pokemon to pool | Same source |
| Refresh cost | 1g (or 0 if shopFreeRolls > 0) | Game code |
| Level up cost | 4g | Game code |
| Interest | 1g per 10g saved, max 5g | Game code |
| Base income | 5g/round | Game code |
| Streak bonus 2/3/4/5+ | +1/+2/+3/+5g | Game code |
| Damage formula | ceil(stageLevel/2) + surviving opponent units | Game code |

---

## Final Observation Space (612 features)

```
Player stats:      14
Shop:              54   (6 slots x 9 features)
Board:            384   (32 cells x 12 features)
Held items:        10
Synergies:         31
Game info:          7
Opponents:         70   (7 opponents x 10 features)
Propositions:      42   (6 slots x 7 features)
──────────────────────
TOTAL:            612
```

---

## Final Action Space (92 discrete actions)

```
 0- 5   Buy shop slot 0-5
 6      Refresh shop
 7      Level up
 8      Lock/unlock shop
 9      End turn (pass)
10-41   Move first-available unit to grid cell 0-31
42-73   Sell unit at grid cell 0-31
74-79   Remove pokemon from shop slot 0-5
80-85   Pick proposition 0-5
86-91   Combine held items (pair index 0-5)
```

Grid cell mapping: `cell = y * 8 + x`, where y=0 is bench, y=1-3 is board.

---

## Known Limitations (Accepted for v1)

1. **MOVE actions pick "first available unit"** — the agent controls WHERE to place, not WHICH unit. The heuristic: scan bench left-to-right (y=0, x=0..7), then board row by row. For v2, consider hierarchical action space or direct `reposition` commands.

2. **HTTP-per-step overhead** — sb3 calls `/step` once per action via HTTP. Expect ~50-100ms per game vs ~5ms for pure in-process. The `/batch-step` endpoint exists but sb3 doesn't use it natively. For v2, consider unix sockets, stdin/stdout pipes, or porting to pure Python.

3. **Carousel/portal minigames simplified to discrete picks** — no spatial pursuit movement in training. Items/portals are presented as propositions the agent picks from using PICK actions.

---

## Phase 0: Scaffolding & Helpers

No behavior changes. Add utilities that later phases depend on.

**Files touched:** `app/training/training-config.ts`

### 0.1 — Grid helper functions

Add to `training-config.ts`:

```ts
export const GRID_WIDTH = 8
export const GRID_HEIGHT = 4   // y=0 bench, y=1-3 board
export const GRID_CELLS = GRID_WIDTH * GRID_HEIGHT  // 32
export const SHOP_SLOTS = 6
export const MAX_COMBINE_PAIRS = 6
export const MAX_HELD_ITEMS = 10  // observation encoding cap
export const NUM_SYNERGIES = 31
export const NUM_PKM_SPECIES = 1139

export function cellToXY(cell: number): [number, number] {
  return [cell % GRID_WIDTH, Math.floor(cell / GRID_WIDTH)]
}

export function xyToCell(x: number, y: number): number {
  return y * GRID_WIDTH + x
}
```

**Acceptance:** `cellToXY(0)=[0,0]`, `cellToXY(9)=[1,1]`, `cellToXY(31)=[7,3]`, `xyToCell(3,2)=19`.

---

### 0.2 — Item-pair enumeration helper

Add to `training-config.ts`:

```ts
export function enumerateItemPairs(items: string[]): [number, number][] {
  const pairs: [number, number][] = []
  for (let i = 0; i < items.length && pairs.length < MAX_COMBINE_PAIRS; i++) {
    for (let j = i + 1; j < items.length && pairs.length < MAX_COMBINE_PAIRS; j++) {
      pairs.push([i, j])
    }
  }
  return pairs
}
```

**Acceptance:** `enumerateItemPairs(["A","B","C"])` returns `[[0,1],[0,2],[1,2]]`. Empty input returns `[]`. More than 6 pairs truncated to 6.

---

### 0.3 — Item recipe lookup helper

Add to `training-config.ts` (import `Item`, `ItemRecipe` from `app/types/enum/Item`):

```ts
export function findRecipeResult(itemA: Item, itemB: Item): Item | null {
  for (const [result, ingredients] of Object.entries(ItemRecipe)) {
    if ((ingredients[0] === itemA && ingredients[1] === itemB) ||
        (ingredients[0] === itemB && ingredients[1] === itemA)) {
      return result as Item
    }
  }
  return null
}
```

**Acceptance:** `findRecipeResult(FOSSIL_STONE, MYSTIC_WATER)` returns `WATER_STONE`. Invalid pair returns `null`.

---

### 0.4 — Pokemon species-index lookup helper

Add to `training-config.ts` (import `Pkm` from `app/types/enum/Pokemon`):

```ts
const PkmValues = Object.values(Pkm)
const PkmToIndex = new Map<string, number>()
PkmValues.forEach((pkm, i) => PkmToIndex.set(pkm, i))

export function getPkmSpeciesIndex(pkm: Pkm): number {
  return (PkmToIndex.get(pkm) ?? 0) / NUM_PKM_SPECIES
}
```

This gives a unique 0-1 normalized value per pokemon species. Used in shop, board, and proposition observations.

**Acceptance:** `getPkmSpeciesIndex(Pkm.PIKACHU)` returns a value between 0 and 1. `getPkmSpeciesIndex(Pkm.DEFAULT)` returns 0.

---

### 0.5 — Synergy-index lookup helper

Add to `training-config.ts` (import `Synergy`, `SynergyArray` from `app/types/enum/Synergy`):

```ts
const SynergyToIndex = new Map<string, number>()
SynergyArray.forEach((syn, i) => SynergyToIndex.set(syn, i))

export function getSynergyIndex(synergy: Synergy): number {
  return ((SynergyToIndex.get(synergy) ?? 0) + 1) / NUM_SYNERGIES
  // +1 so index 0 (NORMAL) doesn't collide with "no synergy" = 0.0
}
```

**Acceptance:** Each synergy maps to a unique value in (0, 1]. `getSynergyIndex(Synergy.NORMAL)` > 0.

---

## Phase 1: Redefine Action Space (config only, no new behavior)

Goal: Change enum and constants to 92. Remap existing logic to new indices. New actions default to no-op / masked off. **Training works identically after this phase.**

**Files touched:** `app/training/training-config.ts`, `app/training/training-env.ts`, `training/pac_env.py`, `training/train_ppo.py`

### 1.1 — Replace TrainingAction enum with 92-action layout

**File:** `app/training/training-config.ts`

Replace the existing enum entirely:

```ts
export enum TrainingAction {
  BUY_0 = 0, BUY_1 = 1, BUY_2 = 2, BUY_3 = 3, BUY_4 = 4, BUY_5 = 5,
  REFRESH = 6,
  LEVEL_UP = 7,
  LOCK_SHOP = 8,
  END_TURN = 9,
  // 10-41: Move first-available unit to grid cell 0-31
  MOVE_0 = 10,   // through MOVE_31 = 41
  // 42-73: Sell unit at grid cell 0-31
  SELL_0 = 42,   // through SELL_31 = 73
  // 74-79: Remove from shop slot 0-5
  REMOVE_SHOP_0 = 74, REMOVE_SHOP_1 = 75, REMOVE_SHOP_2 = 76,
  REMOVE_SHOP_3 = 77, REMOVE_SHOP_4 = 78, REMOVE_SHOP_5 = 79,
  // 80-85: Pick proposition 0-5
  PICK_0 = 80, PICK_1 = 81, PICK_2 = 82,
  PICK_3 = 83, PICK_4 = 84, PICK_5 = 85,
  // 86-91: Combine items pair 0-5
  COMBINE_0 = 86, COMBINE_1 = 87, COMBINE_2 = 88,
  COMBINE_3 = 89, COMBINE_4 = 90, COMBINE_5 = 91,
}
export const TOTAL_ACTIONS = 92
```

Use intermediate comment blocks for the 32-element MOVE and SELL ranges. You don't need to list all 32 individually — the math `action - TrainingAction.MOVE_0` handles it.

**Acceptance:** Enum compiles. `TOTAL_ACTIONS === 92`. `TrainingAction.SELL_0 + 31 === 73`.

---

### 1.2 — Remap getActionMask() to new indices

**File:** `app/training/training-env.ts` — method `getActionMask()`

Changes:
- Array size: `new Array(TOTAL_ACTIONS).fill(0)` (now 92)
- END_TURN fallback: `mask[TrainingAction.END_TURN]` = index 9
- BUY: loop `i < 5` using `TrainingAction.BUY_0 + i`. **BUY_5 stays masked 0** (gated until Phase 3.2 adds its observation).
- SELL: loop `x < 8` using `TrainingAction.SELL_0 + x` (bench cells 0-7 only for now). Board cells stay masked 0.
- REFRESH: `mask[TrainingAction.REFRESH]`
- LEVEL_UP: `mask[TrainingAction.LEVEL_UP]`
- PICK: `mask[TrainingAction.PICK_0 + i]`
- All new ranges (LOCK_SHOP, MOVE_*, non-bench SELL_*, REMOVE_SHOP_*, COMBINE_*): leave as 0

**Acceptance:** `/observe` returns `actionMask` of length 92. Existing valid actions map to correct new indices.

---

### 1.3 — Remap executeAction() switch to new indices

**File:** `app/training/training-env.ts` — method `executeAction()`

Remap the switch statement:
- `BUY_0..BUY_4`: `shopIndex = action - TrainingAction.BUY_0`, call `buyPokemon`
- `BUY_5`: return false (no-op until Phase 2.2)
- `REFRESH` (6): call `rerollShop`
- `LEVEL_UP` (7): call `levelUp`
- `LOCK_SHOP` (8): return false (no-op until Phase 2.1)
- `END_TURN` (9): return true
- `MOVE_0..MOVE_31` (10-41): return false (no-op until Phase 2.4)
- `SELL_0..SELL_31` (42-73): compute `[x,y] = cellToXY(action - TrainingAction.SELL_0)`. If y===0, call `sellPokemonAtBench(agent, x)`. Else return false (until Phase 2.3).
- `REMOVE_SHOP_0..5` (74-79): return false (no-op until Phase 2.5)
- `PICK_0..PICK_5` (80-85): call `pickProposition(agent, action - TrainingAction.PICK_0)`
- `COMBINE_0..5` (86-91): return false (no-op until Phase 2.6)

Also update the proposition check in `step()` to reference `TrainingAction.PICK_0` / `TrainingAction.PICK_5`.

**Acceptance:** POST `/step` with old-equivalent actions works identically. Buy slot 1 = action 1, refresh = action 6, end turn = action 9, pick 0 = action 80.

---

### 1.4 — Verify Python side auto-adapts

**Files:** `training/pac_env.py`, `training/train_ppo.py`

Both query `/action-space` and `/observation-space` dynamically. Verify:
1. `pac_env.py`: `action_space` is now `Discrete(92)`, mask length 92
2. `train_ppo.py`: Update comments/docstring only (says "16 actions" and "64x64" — stale)

**Acceptance:** `GET /action-space` returns `{n: 92}`. Python env initializes with `Discrete(92)`.

---

## Phase 2: Implement New Actions (one at a time)

Each chunk adds one action group with both execution logic AND mask logic. After each, the action works and is properly gated.

**Files touched:** `app/training/training-env.ts`, `app/training/training-config.ts`

### 2.1 — Lock shop (action 8)

**executeAction:** `action === TrainingAction.LOCK_SHOP` → toggle `player.shopLocked = !player.shopLocked`. Return true.

**getActionMask:** `mask[TrainingAction.LOCK_SHOP] = 1` — always valid during normal shop phase (not during propositions).

**Acceptance:** Lock → reroll → shop unchanged. Unlock → reroll → shop changes. Mask is 1 during shop, 0 during propositions.

---

### 2.2 — Buy 6th shop slot (action 5)

**executeAction:** `BUY_5` calls `buyPokemon(agent, 5)` (same logic as 0-4).

**getActionMask:** **Keep BUY_5 masked to 0 for now.** Add a comment: `// BUY_5 mask gated until obs expansion in Phase 3.2`. This prevents the agent from acting on a slot it can't see.

**Acceptance:** BUY_5 executes correctly when called directly. Mask stays 0 until Phase 3.2 explicitly enables it.

---

### 2.3 — Sell unit at any grid cell (actions 42-73)

Add new method `sellPokemonAtCell(player, x, y)`:
1. Find pokemon at `(x, y)` via `findPokemonAt(player, x, y)`
2. If none, return false
3. Delete from `player.board`
4. Call `this.state.shop.releasePokemon(pokemon.name, player, this.state)`
5. Refund: `player.addMoney(getSellPrice(pokemon, ...), false, null)`
6. Return items: `pokemon.items.forEach(item => player.items.push(item))`
7. Call `player.updateSynergies()` and update `player.boardSize`
8. Return true

**executeAction:** For 42-73: `[x,y] = cellToXY(action - TrainingAction.SELL_0)`, call `sellPokemonAtCell`.

**getActionMask:** For all 32 cells: check `findPokemonAt(player, x, y) !== null`. If yes, `mask[TrainingAction.SELL_0 + cell] = 1`.

**Remove:** The old bench-only `sellPokemonAtBench` method (now subsumed by cells 0-7 which ARE bench y=0).

**Acceptance:** Sell a bench unit via action 42+x. Sell a board unit via action 42+cell. Items returned. Gold refunded.

---

### 2.4 — Move unit to grid cell (actions 10-41)

Add new method `moveUnitToCell(player, targetX, targetY)`:
1. If target cell is occupied (`findPokemonAt(player, targetX, targetY) !== null`), return false
2. Find "first available unit": scan bench left-to-right `(x=0..7, y=0)`, then board rows `(y=1, x=0..7)`, `(y=2, x=0..7)`, `(y=3, x=0..7)`. First pokemon found is the one that moves.
3. If no unit found, return false
4. If target is board (y>=1): check `boardSize < maxTeamSize` OR source is also board (repositioning, not adding). If adding and board full, return false.
5. Set `pokemon.positionX = targetX`, `pokemon.positionY = targetY`
6. Call `pokemon.onChangePosition(targetX, targetY, player, this.state)`
7. Update synergies and boardSize
8. Return true

**getActionMask:** For each of 32 cells:
- Cell must be empty
- At least one unit exists anywhere on the board/bench to move
- If target is board (y>=1) and source would be bench: `boardSize < maxTeamSize`

**executeAction:** For 10-41: `[x,y] = cellToXY(action - TrainingAction.MOVE_0)`, call `moveUnitToCell`.

**Acceptance:** Buy pokemon (goes to bench). Move action 18 (cell 10 → x=2, y=1). Unit now at board position (2,1). Board size incremented.

> **Known limitation:** Agent controls WHERE, not WHICH unit moves. Documented in Known Limitations section above.

---

### 2.5 — Remove from shop (actions 74-79)

Add new method `removeFromShop(player, shopIndex)`:
1. `name = player.shop[shopIndex]`
2. If `!name || name === Pkm.DEFAULT`, return false
3. `cost = getBuyPrice(name, this.state.specialGameRule)`
4. If `player.money < cost`, return false
5. **DO NOT deduct gold** (matches real game — gold is a gate check only)
6. `player.shop[shopIndex] = Pkm.DEFAULT`
7. `player.shopLocked = true`
8. `this.state.shop.releasePokemon(name, player, this.state)`
9. Return true

**getActionMask:** For each of 6 slots: `shop[i] !== Pkm.DEFAULT AND player.money >= getBuyPrice(shop[i], ...)`. Mask = 1 if both true.

**executeAction:** For 74-79: `shopIndex = action - TrainingAction.REMOVE_SHOP_0`, call `removeFromShop`.

**Acceptance:** With 3g and a 1g common in slot 2: remove succeeds, slot is DEFAULT, **gold stays at 3g**, shop is locked, pokemon returned to pool.

---

### 2.6 — Combine items (actions 86-91)

Add new method `combineItems(player, pairIndex)`:
1. `pairs = enumerateItemPairs(values(player.items))`
2. If `pairIndex >= pairs.length`, return false
3. `[i, j] = pairs[pairIndex]`
4. `itemA = player.items[i]`, `itemB = player.items[j]`
5. `result = findRecipeResult(itemA, itemB)`
6. If `!result`, return false
7. Remove both items from `player.items` (remove higher index first to avoid shifting)
8. Push `result` to `player.items`
9. Return true

**getActionMask:** Call `enumerateItemPairs(values(player.items))`. For each pair index 0-5: check `findRecipeResult(itemA, itemB) !== null`. Mask = 1 if valid recipe.

**executeAction:** For 86-91: `pairIndex = action - TrainingAction.COMBINE_0`, call `combineItems`.

**Acceptance:** Player with FOSSIL_STONE + MYSTIC_WATER. Combine pair 0. Player now has WATER_STONE, both components removed.

---

### 2.7 — Add TRAINING_AUTO_PLACE config flag

**File:** `app/training/training-config.ts`

```ts
export const TRAINING_AUTO_PLACE = true  // flip to false in Phase 7.1
```

**File:** `app/training/training-env.ts` — method `step()`, the `shouldEndTurn` block

Gate the existing `autoPlaceTeam` call:

```ts
if (TRAINING_AUTO_PLACE) {
  this.autoPlaceTeam(agent)
}
```

**Acceptance:** With flag true: behavior unchanged (units auto-placed before fight). With flag false: units stay where the agent put them.

---

## Phase 3: Expand Observation Space

Each chunk expands one observation section. Update `TOTAL_OBS_SIZE` at the end in 3.8.

**Files touched:** `app/training/training-config.ts`, `app/training/training-env.ts`

### 3.1 — Expand player stats (8 → 14 features)

**Config:** `OBS_PLAYER_STATS = 14`

**Observation encoding (14 features):**

| # | Feature | Normalization |
|---|---|---|
| 1 | life | / 100 |
| 2 | money | / 100 |
| 3 | level | / 9 |
| 4 | streak | / 10 |
| 5 | interest | / 5 |
| 6 | alive | 0 or 1 |
| 7 | rank | / 8 |
| 8 | boardSize | / 9 |
| 9 | expNeeded | / 32 |
| 10 | shopFreeRolls | / 3 |
| 11 | rerollCount | / 20 |
| 12 | shopLocked | 0 or 1 |
| 13 | totalMoneyEarned | / 200 |
| 14 | totalPlayerDamageDealt | / 100 |

**Acceptance:** Observation starts with 14 values. All in [0, ~2] range.

---

### 3.2 — Expand shop observations (5 rarity → 6 slots x 9 features = 54)

**Config:**
```ts
export const OBS_SHOP_SLOTS = 6
export const OBS_SHOP_FEATURES = 9
```

**Per-slot encoding (9 features):**

| # | Feature | Normalization | Notes |
|---|---|---|---|
| 1 | hasUnit | 0 or 1 | 0 if slot is Pkm.DEFAULT |
| 2 | speciesIndex | getPkmSpeciesIndex(name) | Unique per pokemon species |
| 3 | rarity | 0-1 map | Same rarity scale as before |
| 4 | cost | / 10 | getBuyPrice() |
| 5 | type1 | getSynergyIndex(types[0]) | First base type |
| 6 | type2 | getSynergyIndex(types[1]) | Second base type, or 0 |
| 7 | type3 | getSynergyIndex(types[2]) | Third base type, or 0 |
| 8 | type4 | 0 | Always 0 for shop pokemon (no items yet) |
| 9 | isEvoPossible | 0 or 1 | Would buying trigger a 3-merge? |

For `isEvoPossible`: create the pokemon temporarily, check `CountEvolutionRule.canEvolveIfGettingOne(pokemon, player)`.

**After completing this chunk:** Enable BUY_5 in the action mask (was gated in Phase 2.2). Add the 6th slot buy mask check in `getActionMask()`.

**Acceptance:** Shop observation is exactly 54 floats. Species indices differ for Horsea vs Squirtle (both common/water). Type slots encode up to 3 base types.

---

### 3.3 — Expand board observations (fix grid + richer features: 32 cells x 12 = 384)

**Config:**
```ts
export const OBS_BOARD_SLOTS = 32   // FIX: was 40 (y=4 bug)
export const OBS_BOARD_FEATURES_PER_SLOT = 12
```

**Grid iteration:** `for y = 0..3, for x = 0..7` (32 cells, NOT the old y=1..4).

**Per-cell encoding (12 features):**

| # | Feature | Normalization | Notes |
|---|---|---|---|
| 1 | hasUnit | 0 or 1 | |
| 2 | speciesIndex | getPkmSpeciesIndex(name) | Unique pokemon identity |
| 3 | stars | / 3 | Evolution stage |
| 4 | rarity | 0-1 map | |
| 5 | type1 | getSynergyIndex | Base types only (see impl note) |
| 6 | type2 | getSynergyIndex | |
| 7 | type3 | getSynergyIndex | |
| 8 | type4 | 0 | Always 0 during PICK phase (see impl note) |
| 9 | atk | / 50 | |
| 10 | hp | / 500 | Current max HP |
| 11 | range | / 4 | |
| 12 | numItems | / 3 | Items equipped on this unit |

**Reading types from board units:** Use `values(pokemon.types)` to read the SetSchema. During PICK phase, this contains **base types only** (up to 3). Stone-added synergies are only applied to `PokemonEntity` during combat simulation, not to `Pokemon` instances on the board during shop phase. The 4th type slot will be 0 during observations. The agent can infer stone synergy potential from the held items observation (Phase 3.4).

> **Implementation note (verified):** `PokemonEntity.addItem()` mutates `types` directly but requires `this.simulation` to exist. `Pokemon` instances on `player.board` during PICK have no simulation. Items are stored in `player.items` (inventory), not equipped to board pokemon. Type encoding is base-types-only during PICK, which is correct and sufficient.

**Acceptance:** Board observation is exactly 384 floats. Row y=4 no longer encoded. Type slots encode base types (up to 3). 4th slot is 0 during PICK phase.

---

### 3.4 — Add held items observation (new section: 10 features)

**Config:** `export const OBS_HELD_ITEMS = 10`

Encode the first 10 items in `player.items` as normalized category indices.

Create a pre-computed map of all Item enum values to sequential indices:

```ts
const AllItems = Object.values(Item)
const ItemToIndex = new Map<string, number>()
AllItems.forEach((item, i) => ItemToIndex.set(item, i))
const NUM_ITEMS = AllItems.length

function getItemIndex(item: Item): number {
  return ((ItemToIndex.get(item) ?? 0) + 1) / NUM_ITEMS
}
```

For each of the 10 observation slots:
- If `i < player.items.length`: `getItemIndex(player.items[i])`
- Else: 0

**Acceptance:** Player with 3 items → first 3 slots nonzero, rest 0. Different items produce different values.

---

### 3.5 — Expand opponent observations (16 → 70 features)

**Config:**
```ts
export const OBS_OPPONENT_FEATURES = 10
export const OBS_OPPONENT_COUNT = 7
export const OBS_OPPONENT_STATS = OBS_OPPONENT_COUNT * OBS_OPPONENT_FEATURES  // 70
```

**Per-opponent encoding (10 features):**

| # | Feature | Normalization |
|---|---|---|
| 1 | life | / 100 |
| 2 | rank | / 8 |
| 3 | level | / 9 |
| 4 | gold | / 100 |
| 5 | streak | / 10 |
| 6 | boardSize | / 9 |
| 7 | topSynergy1Index | getSynergyIndex() |
| 8 | topSynergy1Count | / 10 |
| 9 | topSynergy2Index | getSynergyIndex() |
| 10 | topSynergy2Count | / 10 |

**Top synergies:** Iterate `player.synergies`, find the two with highest count. If tied, order doesn't matter.

Filter to 7 opponents (exclude self). If fewer alive, pad remaining slots with zeros.

**Acceptance:** 70 floats. Top synergies are correctly identified (opponent with Water=4, Fire=2 → topSyn1=Water/4, topSyn2=Fire/2).

---

### 3.6 — Expand proposition observations (18 → 42 features)

**Config:**
```ts
export const OBS_PROPOSITION_FEATURES = 7
```

**Per-slot encoding (7 features):**

| # | Feature | Normalization | Notes |
|---|---|---|---|
| 1 | speciesIndex | getPkmSpeciesIndex() | Pokemon identity |
| 2 | rarity | 0-1 map | |
| 3 | type1 | getSynergyIndex | |
| 4 | type2 | getSynergyIndex | |
| 5 | type3 | getSynergyIndex | |
| 6 | type4 | 0 | Always 0 (propositions have no items) |
| 7 | hasItem | 0 or 1 | Is there a bonus item attached? |

6 slots x 7 features = 42.

**Acceptance:** During pick phase, propositions encode species identity and all base types. Horsea and Squirtle produce different speciesIndex values.

---

### 3.7 — Expand game info (4 → 7 features)

**Config:** `export const OBS_GAME_INFO = 7`

**Game info encoding (7 features):**

| # | Feature | Normalization | Notes |
|---|---|---|---|
| 1 | stageLevel | / 50 | |
| 2 | phase | / 2 | Same as before |
| 3 | playersAlive | / 8 | |
| 4 | hasPropositions | 0 or 1 | |
| 5 | weatherIndex | index / numWeathers | Encode current weather |
| 6 | isPVE | 0 or 1 | Is current stage a PVE round? |
| 7 | maxTeamSize | / 9 | Team size cap at current level |

For weather: create a `WeatherToIndex` map similar to SynergyToIndex.

**Acceptance:** 7 floats. PVE rounds show `isPVE=1`. Weather varies.

---

### 3.8 — Update TOTAL_OBS_SIZE and synergy count

**File:** `app/training/training-config.ts`

Fix synergy obs count: `OBS_SYNERGIES = 31` (was 32 — there are exactly 31 synergies).

Recalculate:
```ts
export const TOTAL_OBS_SIZE =
  OBS_PLAYER_STATS +                                    // 14
  OBS_SHOP_SLOTS * OBS_SHOP_FEATURES +                  // 54
  OBS_BOARD_SLOTS * OBS_BOARD_FEATURES_PER_SLOT +       // 384
  OBS_HELD_ITEMS +                                       // 10
  OBS_SYNERGIES +                                        // 31
  OBS_GAME_INFO +                                        // 7
  OBS_OPPONENT_STATS +                                   // 70
  OBS_PROPOSITION_SLOTS * OBS_PROPOSITION_FEATURES       // 42
  // = 612
```

**Acceptance:** `TOTAL_OBS_SIZE === 612`. Observation arrays are exactly length 612.

---

### 3.9 — Verify Python observation space

**File:** `training/pac_env.py`

The env queries `/observation-space` dynamically, so `obs_size` auto-updates. Verify:
- `observation_space.shape == (612,)`
- Box bounds `low=-1.0, high=2.0` still adequate (all normalized values stay in [0, 1])

**Acceptance:** Python env initializes. `env.observation_space.shape == (612,)`.

---

## Phase 4: Simplified Minigame Phases

Full carousel/portal pursuit doesn't fit discrete actions. Simplify to picks.

**Files touched:** `app/training/training-env.ts`

### 4.1 — Item carousel as discrete pick

**Current behavior** (in `advanceToNextPickPhase`): At `ItemCarouselStages`, a random item is pushed directly to `player.items`.

**New behavior:** Instead, present 3-5 random items as `itemsProposition`:

```ts
if (ItemCarouselStages.includes(this.state.stageLevel)) {
  this.state.players.forEach((player) => {
    if (!player.isBot && player.alive) {
      const itemPool = this.state.stageLevel >= 20
        ? CraftableItemsNoScarves
        : ItemComponentsNoFossilOrScarf
      const choices = pickNRandomIn(itemPool, 3)
      choices.forEach(item => player.itemsProposition.push(item))
    }
  })
}
```

**Also modify `pickProposition`:** Handle the case where `pokemonsProposition` is empty but `itemsProposition` is not empty. When agent picks PICK_0..5 during an item-only proposition:
1. Get `item = player.itemsProposition[propositionIndex]`
2. Push to `player.items`
3. Clear `itemsProposition`

**Mask:** During item-only propositions, PICK_0..2 are valid (or however many items offered). All other actions masked 0.

**Acceptance:** At an item carousel stage, agent sees 3 item propositions. Picks one. Item added to inventory. Others discarded.

---

### 4.2 — Verify portal/unique/legendary propositions work

**Current behavior:** `assignUniquePropositions` at stages 10 and 20 already creates pokemon propositions that the agent picks via PICK actions.

**Verify this is end-to-end correct:** At stage 10, unique pokemon propositions appear. Agent picks via PICK_0..5. Pokemon placed on bench with bonus item.

No code changes needed — just verification.

**Acceptance:** Stage 10 unique picks and stage 20 legendary picks work correctly.

---

### 4.3 — PVE reward item picks

After PVE battles where the agent wins, the game awards item rewards.

**In `runFightPhase()`**, after PVE simulation completes and agent's battle result is WIN, but BEFORE returning from `runFightPhase()`:
1. Generate 3 random items from `ItemComponentsNoFossilOrScarf` (or appropriate pool)
2. Push items to `agent.itemsProposition`
3. The next pick phase will present them for PICK actions

**Mask handling:** Same as 4.1 — item-only propositions allow PICK_0..2.

> **Implementation note (verified):** `assignShop()` does NOT touch `itemsProposition`. `advanceToNextPickPhase()` does NOT clear `itemsProposition` either. Items set in `runFightPhase()` will survive into the next pick phase. Additional pick stages (5, 8, 11) only `.push()` to `itemsProposition` without clearing, but these stages never overlap with PVE stages, so no collision. Order of operations: set propositions in `runFightPhase()` → `advanceToNextPickPhase()` runs → propositions persist → agent picks.

**Acceptance:** After winning a PVE round, agent sees item propositions and picks one.

---

## Phase 5: Python Training Pipeline

**Files touched:** `training/requirements.txt`, `training/train_ppo.py`

### 5.1 — Switch to MaskablePPO from sb3-contrib

**File:** `training/requirements.txt` — add `sb3-contrib>=2.2.0`

**File:** `training/train_ppo.py`

```python
# OLD
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, ...)

# NEW
from sb3_contrib import MaskablePPO
model = MaskablePPO("MlpPolicy", env, ...)
```

MaskablePPO automatically calls `env.action_masks()` during rollout collection. The existing `action_masks()` method on `PokemonAutoChessEnv` already returns the correct format.

**Delete** the empty `ActionMaskCallback` class entirely — MaskablePPO handles masking internally.

Update `evaluate()` to use `MaskablePPO.load()` and pass `action_masks` during prediction:
```python
action_masks = env.action_masks()
action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
```

**Acceptance:** Training starts. Model only samples valid actions (no mask violations in logs).

---

### 5.2 — Update network architecture for 612-dim input

**File:** `training/train_ppo.py`

```python
policy_kwargs=dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)
```

With 612 input features and 92 action outputs, 128-wide layers are undersized. 256x256 is appropriate.

**Acceptance:** Model summary shows `[612] → 256 → 256 → [92]` for policy, similar for value.

---

### 5.3 — Adjust hyperparameters for larger space

**File:** `training/train_ppo.py`

Starting values (will need tuning):
- `ent_coef`: `0.01` → `0.02` (more exploration for 92 actions)
- `n_steps`: `512` → `1024` (longer rollouts for richer episodes)
- `batch_size`: `64` → `128` (more samples per gradient step)

**Acceptance:** Training runs without NaN or reward collapse.

---

### 5.4 — Expand training metrics logging

**File:** `training/train_ppo.py` — `TrainingMetricsCallback`

Add logging for:
- `training/mean_gold` — average gold at end of game
- `training/mean_board_size` — is the agent fielding units?
- `training/synergy_count` — number of active synergies
- `training/items_held` — economy management signal

Pull from the `info` dict returned by env (extend `getInfo()` in training-env.ts to include these).

**Acceptance:** TensorBoard shows all new metric curves.

---

## Phase 6: Reward Tuning

### 6.1 — Add economy shaping reward (with hoarding guard)

**Config:**
```ts
export const REWARD_INTEREST_BONUS = 0.05     // per interest gold earned
```

**Implementation** (in `runFightPhase()`, after income calculation):

```ts
if (player.id === this.agentId) {
  // Only reward interest if agent is actually fielding a team
  const maxTeam = getMaxTeamSize(player.experienceManager.level, this.state.specialGameRule)
  if (player.boardSize >= maxTeam - 2) {
    reward += player.interest * REWARD_INTEREST_BONUS
  }
}
```

The `boardSize >= maxTeam - 2` guard prevents the agent from learning to hoard gold and never buy anything. It must field at least a nearly-full team to earn interest rewards.

**Acceptance:** Agent with 5/6 board and 4 interest gets +0.20. Agent with 0/6 board and 5 interest gets +0 (no reward — hoarding).

---

### 6.2 — Add synergy activation reward (delta-based)

**Config:**
```ts
export const REWARD_SYNERGY_THRESHOLD = 0.1
```

**State tracking:** Add `prevActiveSynergyCount: number = 0` to `TrainingEnv`.

**Implementation:** At the START of each turn (in `advanceToNextPickPhase` or beginning of `step` when a new pick phase starts):
```ts
this.prevActiveSynergyCount = this.countActiveSynergyThresholds(agent)
```

At the END of each turn (just before fight):
```ts
const currentThresholds = this.countActiveSynergyThresholds(agent)
const delta = currentThresholds - this.prevActiveSynergyCount
if (delta > 0) {
  reward += delta * REWARD_SYNERGY_THRESHOLD
}
// Only reward POSITIVE deltas. Don't penalize losing synergies from selling —
// that's already penalized by combat outcomes.
```

Helper `countActiveSynergyThresholds(player)`: iterate synergies, count how many are at or above their activation threshold (reference the synergy threshold data).

**Acceptance:** Agent that goes from 1 active synergy to 3 active synergies gets +0.2. Agent that sells a unit and drops from 3 to 2 gets no penalty from this reward (only combat results penalize).

---

### 6.3 — Add combat damage reward (enemy kills)

**Config:**
```ts
export const REWARD_PER_ENEMY_KILL = 0.02
```

**Implementation** (in `runFightPhase()`):

> **Implementation note (verified):** Dead units are REMOVED from the MapSchema entirely during combat — `unit.life <= 0` won't work because dead units are gone. After `simulation.stop()`, teams are CLEARED completely. Must capture initial team sizes BEFORE fight, then count survivors AFTER `onFinish()` but BEFORE `stop()`.

**Step 1:** Before running simulations, record initial enemy team sizes per simulation:

```ts
// Before the simulation loop
const initialEnemySizes = new Map<string, number>()
this.state.simulations.forEach((sim, id) => {
  const agentIsBlue = agent.team === Team.BLUE_TEAM
  const enemyTeam = agentIsBlue ? sim.redTeam : sim.blueTeam
  initialEnemySizes.set(id, enemyTeam.size)
})
```

**Step 2:** After `onFinish()` for the agent's simulation, but BEFORE `stop()`:

```ts
// After onFinish, before stop
const agentSim = this.state.simulations.get(agent.simulationId)
if (agentSim) {
  const agentIsBlue = agent.team === Team.BLUE_TEAM
  const enemyTeam = agentIsBlue ? agentSim.redTeam : agentSim.blueTeam
  const initialSize = initialEnemySizes.get(agent.simulationId) ?? 0
  const survivingEnemies = enemyTeam.size
  const enemyDeaths = initialSize - survivingEnemies
  reward += enemyDeaths * REWARD_PER_ENEMY_KILL
}

// THEN call stop() which clears teams
this.state.simulations.forEach((simulation) => {
  simulation.stop()
})
```

**Important:** The current code calls `onFinish()` and `stop()` in a single loop. Refactor to: (1) run sims to completion, (2) call `onFinish()` on all, (3) extract reward data, (4) THEN call `stop()` on all.

This provides immediate board-quality signal even on losses. "Lost but killed 5 of 6" is much better signal than just "lost".

**Acceptance:** Agent that kills 4 of 6 enemy units in a loss gets: -0.3 (loss) + 0.08 (kills) = -0.22. Better than a 0-kill loss (-0.30).

---

### 6.4 — Add HP preservation reward

**Config:**
```ts
export const REWARD_HP_SCALE = 0.005
```

**Implementation** (in `runFightPhase()`):
```ts
if (this.lastBattleResult === BattleResult.WIN && agent.alive) {
  reward += (agent.life / 100) * REWARD_HP_SCALE
}
```

Tiny bonus for winning with more HP — proxy for board strength.

**Acceptance:** Agent at 90HP after a win gets +0.0045. At 30HP, gets +0.0015.

---

### 6.5 — Tune final placement reward (after training runs)

This is a tuning step, not a code step. After running training for 50k-100k steps with all new rewards:

1. Check TensorBoard reward curves
2. If per-round rewards dominate final placement reward, increase `REWARD_PLACEMENT_SCALE`
3. If final placement dominates and per-round rewards are noise, decrease `REWARD_PLACEMENT_SCALE`
4. Target: per-round rewards contribute ~30-40% of total episode reward, placement ~60-70%

Current formula: `(9 - rank) * 2.0 - 6.0` → Rank 1 = +10, Rank 8 = -4.

**Acceptance:** Reward curves show both per-round learning signal and placement optimization.

---

## Phase 7: Integration & Testing

### 7.1 — Flip TRAINING_AUTO_PLACE to false

**File:** `app/training/training-config.ts`

```ts
export const TRAINING_AUTO_PLACE = false
```

**Risk mitigation:** Train with auto-place for 200k+ steps first (Phases 0-6). Save that checkpoint. Then flip to false and fine-tune from the checkpoint. The agent already understands economy — now it needs to learn placement.

**Add bench penalty** to incentivize placement:
```ts
export const REWARD_BENCH_PENALTY = -0.01  // per bench unit that COULD be on board
```

At end of turn (before fight), count units on bench that could legally be placed (bench units where boardSize < maxTeamSize). Apply penalty.

**Acceptance:** With auto-place off, agent learns to use MOVE actions. Bench penalty visible in TensorBoard as it decreases over training.

---

### 7.2 — Benchmark final environment

Run `/benchmark` with full 92-action space and 612-dim observations.

**Targets:**
- Steps/sec: > 500 (viable for training)
- Game completes correctly (rank 1-8, all stages)
- No NaN in observations

**If too slow**, profile and optimize:
- Cache `getObservation()` results within the same turn (many steps per turn, state only changes per-step)
- Cache `getActionMask()` similarly
- Pre-compute item pairs once per turn
- Pre-compute pokemon lookups (findPokemonAt) into a grid cache

**Acceptance:** Benchmark prints speed, final rank, and reward without errors.

---

### 7.3 — Action space mapping verification test

Write a simple test script (TypeScript or as part of training-server) that verifies:
1. For each of the 92 actions, the training env's interpretation matches the API bible
2. Specifically:
   - Action 0 = buy slot 0
   - Action 5 = buy slot 5
   - Action 6 = refresh
   - Action 10 = move to cell 0 = (0,0) = bench slot 0
   - Action 18 = move to cell 8 = (0,1) = front-left board
   - Action 42 = sell at cell 0 = (0,0) = bench slot 0
   - Action 50 = sell at cell 8 = (0,1) = front-left board
   - Action 74 = remove shop slot 0
   - Action 80 = pick proposition 0
   - Action 86 = combine item pair 0
3. Grid cell math: `cellToXY(action - 10)` and `cellToXY(action - 42)` produce correct coordinates

**This ensures** a model trained in the headless env maps 1:1 onto agent-io commands with zero translation.

**Acceptance:** All 92 actions verified. Ready for deployment through agent-io bridge.

---

## Dependency Graph

```
Phase 0 (0.1-0.5: helpers)
  └──→ Phase 1 (1.1-1.4: action config remap)
         ├──→ Phase 2.1 (lock shop)
         ├──→ Phase 2.2 (buy slot 5) ──→ [enable mask after 3.2]
         ├──→ Phase 2.3 (sell any cell)
         ├──→ Phase 2.4 (move to cell)
         ├──→ Phase 2.5 (remove from shop)
         ├──→ Phase 2.6 (combine items) ──→ needs 0.2, 0.3
         └──→ Phase 2.7 (auto-place flag)

Phase 0 (helpers)
  └──→ Phase 3.1-3.7 (observation expansion) ←── parallel with Phase 2
         └──→ Phase 3.8 (recalculate total)
                └──→ Phase 3.9 (python verify)

Phase 2 + Phase 3 done
  └──→ Phase 4 (minigame phases)
         └──→ Phase 5 (python pipeline)
                └──→ Phase 6 (reward tuning)
                       └──→ Phase 7 (integration)
```

**Phases 2 and 3 can be done in parallel.** Phase 2 chunks (2.1-2.7) are also independent of each other (except 2.6 needs helpers 0.2 + 0.3). Phase 3 chunks (3.1-3.7) are independent of each other.

---

## Chunk Summary

| Phase | Chunk | Description | Depends On |
|---|---|---|---|
| 0 | 0.1 | Grid helpers (cellToXY, xyToCell) | — |
| 0 | 0.2 | Item-pair enumeration helper | — |
| 0 | 0.3 | Item recipe lookup helper | — |
| 0 | 0.4 | Pokemon species-index helper | — |
| 0 | 0.5 | Synergy-index helper | — |
| 1 | 1.1 | Replace TrainingAction enum (92 actions) | 0.1 |
| 1 | 1.2 | Remap getActionMask() to new indices | 1.1 |
| 1 | 1.3 | Remap executeAction() to new indices | 1.1, 1.2 |
| 1 | 1.4 | Verify Python auto-adapts | 1.1-1.3 |
| 2 | 2.1 | Lock shop action | 1.3 |
| 2 | 2.2 | Buy 6th shop slot (mask gated) | 1.3 |
| 2 | 2.3 | Sell at any grid cell | 1.3 |
| 2 | 2.4 | Move unit to grid cell | 1.3 |
| 2 | 2.5 | Remove from shop (no gold cost) | 1.3 |
| 2 | 2.6 | Combine items | 1.3, 0.2, 0.3 |
| 2 | 2.7 | Auto-place config flag | 1.3 |
| 3 | 3.1 | Expand player stats (8→14) | — |
| 3 | 3.2 | Expand shop obs (6x9=54) + enable BUY_5 mask | 0.4, 0.5, 2.2 |
| 3 | 3.3 | Expand board obs (32x12=384, fix y=4 bug) | 0.4, 0.5 |
| 3 | 3.4 | Add held items obs (10) | — |
| 3 | 3.5 | Expand opponent obs (7x10=70) | 0.5 |
| 3 | 3.6 | Expand proposition obs (6x7=42) | 0.4, 0.5 |
| 3 | 3.7 | Expand game info (4→7) | — |
| 3 | 3.8 | Recalculate TOTAL_OBS_SIZE = 612 | 3.1-3.7 |
| 3 | 3.9 | Verify Python obs space | 3.8 |
| 4 | 4.1 | Item carousel as discrete pick | 2.*, 3.* |
| 4 | 4.2 | Verify portal/unique/legendary picks | 2.*, 3.* |
| 4 | 4.3 | PVE reward item picks | 2.*, 3.* |
| 5 | 5.1 | Switch to MaskablePPO | 4.* |
| 5 | 5.2 | Update network (256x256) | 5.1 |
| 5 | 5.3 | Adjust hyperparameters | 5.2 |
| 5 | 5.4 | Expand training metrics | 5.1 |
| 6 | 6.1 | Economy reward (with hoarding guard) | 5.* |
| 6 | 6.2 | Synergy reward (delta-based) | 5.* |
| 6 | 6.3 | Combat damage reward (enemy kills) | 5.* |
| 6 | 6.4 | HP preservation reward | 5.* |
| 6 | 6.5 | Tune placement reward (manual) | 6.1-6.4 |
| 7 | 7.1 | Disable auto-place + bench penalty | 6.* |
| 7 | 7.2 | Benchmark final environment | 7.1 |
| 7 | 7.3 | Action space mapping verification | 7.1 |

**Total: 33 chunks** (5 more than original 28, due to audit additions)

Each chunk is ~30-60 minutes of focused implementation work.

---

## Appendix: Implementation Notes (Verified)

These items were flagged as "verify during implementation" and have been verified against the source code.

### A. Board type reading during PICK phase (affects Phase 3.3)

**Verified in:** `app/core/pokemon-entity.ts:788-816`, `app/models/colyseus-models/pokemon.ts:66`

During PICK phase, board pokemon are `Pokemon` instances (not `PokemonEntity`). Items live in `player.items` (inventory), not equipped to individual pokemon. Stone synergies are only added by `PokemonEntity.addItem()` which runs during combat simulation and requires `this.simulation` to exist.

**Consequence:** `pokemon.types` during PICK phase contains **base types only** (max 3). The 4th type slot in observations will always be 0. This is correct — the agent infers stone synergy potential from the held items observation (Phase 3.4).

### B. Move heuristic and fresh mask per step (affects Phase 2.4)

**Verified in:** `app/training/training-env.ts:201-291`

`getActionMask()` is called via `getInfo()` which is called at the end of every `step()`. The mask is always fresh after each action. When the agent calls MOVE repeatedly, each subsequent mask reflects the updated cell occupancy. No caching issue.

### C. Enemy kill counting uses team size delta, not HP check (affects Phase 6.3)

**Verified in:** `app/core/simulation.ts:1416, 1511`

Dead units are **removed from the MapSchema entirely** during combat (the battle ends when `team.size === 0`). After `simulation.stop()`, teams are **cleared completely**. The correct approach:
1. Record initial enemy team size before simulation loop
2. After `onFinish()` but before `stop()`, read `enemyTeam.size` (surviving enemies)
3. `deadEnemies = initialSize - survivingEnemies`
4. Only then call `stop()`

The existing `runFightPhase()` calls `onFinish()` and `stop()` in the same loop — must be refactored to separate them.

### D. PVE reward proposition persistence (affects Phase 4.3)

**Verified in:** `app/training/training-env.ts:780-861`, `app/rooms/game-room.ts:1753-1766`

`assignShop()` does NOT touch `itemsProposition`. `advanceToNextPickPhase()` does NOT clear `itemsProposition`. Items pushed to `agent.itemsProposition` in `runFightPhase()` will survive into the next pick phase.

Additional pick stages (5, 8, 11) only `.push()` without clearing, but these never overlap with PVE stages — no collision risk.

The real game uses `resetArraySchema()` (destructive clear + replace) in `stopFightingPhase()`, but the training env doesn't need this because it handles PVE rewards inline.
