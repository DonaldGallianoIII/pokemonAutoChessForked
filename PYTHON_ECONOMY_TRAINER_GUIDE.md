# Python Economy Trainer — Full Implementation Guide

> **Goal**: Build a pure-Python gymnasium environment that simulates the *economy phase only* (buy, sell, level, reroll, bench/board management) so we can train the RL agent's economy skills 100-1000× faster than the full TypeScript game loop. Fights are replaced by a fast power-score estimator.

---

## Table of Contents

1. [Architecture Overview](#phase-0-architecture-overview)
2. [Phase 1 — Static Data Layer](#phase-1-static-data-layer)
3. [Phase 2 — Unit & Board Model](#phase-2-unit--board-model)
4. [Phase 3 — Shop & Pool System](#phase-3-shop--pool-system)
5. [Phase 4 — Economy Engine](#phase-4-economy-engine)
6. [Phase 5 — Fight Power Estimator](#phase-5-fight-power-estimator)
7. [Phase 6 — Bot AI (Opponents)](#phase-6-bot-ai-opponents)
8. [Phase 7 — Observation Builder](#phase-7-observation-builder)
9. [Phase 8 — Action Dispatcher & Mask Builder](#phase-8-action-dispatcher--mask-builder)
10. [Phase 9 — Reward Calculator](#phase-9-reward-calculator)
11. [Phase 10 — Gymnasium Env Wrapper](#phase-10-gymnasium-env-wrapper)
12. [Phase 11 — Integration & Validation](#phase-11-integration--validation)
13. [Phase 12 — Training Script](#phase-12-training-script)

---

## Phase 0 — Architecture Overview

### What we're building

```
┌─────────────────────────────────────┐
│  Python Gymnasium Environment       │
│                                     │
│  ┌──────────┐  ┌────────────────┐   │
│  │ Shop     │  │ Board/Bench    │   │
│  │ Pool     │  │ 8×4 grid       │   │
│  └──────────┘  └────────────────┘   │
│  ┌──────────┐  ┌────────────────┐   │
│  │ Economy  │  │ Power Score    │   │
│  │ Engine   │  │ Estimator      │   │
│  └──────────┘  └────────────────┘   │
│  ┌──────────┐  ┌────────────────┐   │
│  │ Bot AI   │  │ Obs/Act/Reward │   │
│  │ (7 bots) │  │ Builders       │   │
│  └──────────┘  └────────────────┘   │
└─────────────────────────────────────┘
         ↕ step(action) / reset()
┌─────────────────────────────────────┐
│  MaskablePPO (sb3-contrib)          │
│  SubprocVecEnv (N parallel)        │
└─────────────────────────────────────┘
```

### What we're NOT building
- No WebSocket / Colyseus / HTTP communication
- No fight simulation (replaced by power-score estimator)
- No animation, rendering, or client code
- No items system initially (Phase 1 can stub items as stat bonuses)
- No weather system
- No flower pots / eggs / special mechanics

### Key constraint
The observation vector must be **identical** to the TypeScript server's 612-float layout so checkpoints transfer directly between the Python economy trainer and the full TS game.

---

## Phase 1 — Static Data Layer

### Purpose
Extract all Pokemon data, synergy definitions, and game constants from the TypeScript codebase into Python-friendly formats. This is the foundation everything else builds on.

### Step 1.1 — Export Pokemon Data CSV

The file `app/models/precomputed/pokemons-data.csv` already exists. Verify it contains these columns (or generate a fresh export):

| Column | Example | Notes |
|--------|---------|-------|
| `index` | `"0006"` | PkmIndex string |
| `name` | `"CHARIZARD"` | Pkm enum name |
| `rarity` | `"COMMON"` | One of: COMMON, UNCOMMON, RARE, EPIC, ULTRA, UNIQUE, LEGENDARY, HATCH, SPECIAL |
| `hp` | `220` | Base HP |
| `atk` | `18` | Base attack |
| `def` | `5` | Base defense |
| `speDef` | `5` | Base special defense |
| `range` | `1` | Attack range (1=melee, 2-4=ranged) |
| `speed` | `57` | Attack speed (unused in economy sim but needed for obs) |
| `types` | `"FIRE,FLYING"` | Comma-separated synergy types |
| `tier` | `3` | Star level (1, 2, or 3) |
| `evolution` | `""` | Name of evolution (empty if final form) |
| `stars` | `3` | Same as tier |
| `skill` | `"BLAST_BURN"` | Ability name (unused in economy sim) |

**What to build**: A Python dataclass or dict:

```
@dataclass
class PokemonData:
    index: str          # "0006"
    name: str           # "CHARIZARD"
    rarity: str         # "COMMON"
    hp: int
    atk: int
    def_: int
    spe_def: int
    range: int
    speed: int
    types: list[str]    # ["FIRE", "FLYING"]
    tier: int           # 1, 2, or 3
    evolution: str      # next evolution name, or ""
    cost: int           # derived from rarity
```

**Derived values:**
```
RARITY_COST = {
    "COMMON": 1, "UNCOMMON": 2, "RARE": 3, "EPIC": 4,
    "ULTRA": 5, "UNIQUE": 10, "LEGENDARY": 20,
    "HATCH": 3, "SPECIAL": 0
}
```

### Step 1.2 — Evolution Chain Mapping

Build a dict mapping each base-form Pokemon to its full evolution chain:

```python
# evolution_chains["CHARMANDER"] = ["CHARMANDER", "CHARMELEON", "CHARIZARD"]
# evolution_chains["CHARMELEON"] = ["CHARMANDER", "CHARMELEON", "CHARIZARD"]
# evolution_chains["CHARIZARD"] = ["CHARMANDER", "CHARMELEON", "CHARIZARD"]

# Also: family_of["CHARMELEON"] = "CHARMANDER" (base form)
```

The rule is: 3 copies of a tier-N unit merge into 1 tier-(N+1) unit. The `evolution` field in the CSV tells you what each unit evolves into.

Walk the CSV: for each unit where `evolution != ""`, link it forward. Group by base form.

### Step 1.3 — Synergy Constants

```python
SYNERGY_LIST = [
    "NORMAL", "GRASS", "FIRE", "WATER", "ELECTRIC", "FIGHTING",
    "PSYCHIC", "DARK", "STEEL", "GROUND", "POISON", "DRAGON",
    "FIELD", "MONSTER", "HUMAN", "AQUATIC", "BUG", "FLYING",
    "FLORA", "ROCK", "GHOST", "FAIRY", "ICE", "FOSSIL",
    "SOUND", "ARTIFICIAL", "BABY", "LIGHT", "WILD", "AMORPHOUS",
    "GOURMET"
]  # 31 synergies, indices 0-30

SYNERGY_TRIGGERS = {
    "NORMAL":     [3, 5, 7, 9],
    "GRASS":      [3, 5, 7, 9],
    "FIRE":       [2, 4, 6, 8],
    "WATER":      [3, 6, 9],
    "ELECTRIC":   [3, 5, 7],
    "FIGHTING":   [2, 4, 6, 8],
    "PSYCHIC":    [3, 5, 7],
    "DARK":       [3, 5, 7],
    "STEEL":      [2, 4, 6, 8],
    "GROUND":     [2, 4, 6, 8],
    "POISON":     [3, 5, 7],
    "DRAGON":     [3, 5, 7],
    "FIELD":      [3, 6, 9],
    "MONSTER":    [2, 4, 6, 8],
    "HUMAN":      [2, 4, 6],
    "AQUATIC":    [2, 4, 6, 8],
    "BUG":        [2, 4, 6, 8],
    "FLYING":     [2, 4, 6, 8],
    "FLORA":      [3, 4, 5, 6],
    "ROCK":       [2, 4, 6],
    "GHOST":      [2, 4, 6, 8],
    "FAIRY":      [2, 4, 6, 8],
    "ICE":        [2, 4, 6, 8],
    "FOSSIL":     [2, 4, 6],
    "SOUND":      [2, 4, 6],
    "ARTIFICIAL": [2, 4, 6],
    "BABY":       [3, 5, 7],
    "LIGHT":      [2, 3, 4, 5],
    "WILD":       [2, 4, 6, 9],
    "AMORPHOUS":  [3, 5, 7],
    "GOURMET":    [3, 4, 5],
}
```

### Step 1.4 — Rarity Roll Probabilities

```python
# RARITY_PROB[player_level] = [common, uncommon, rare, epic, ultra]
RARITY_PROB = {
    1: [1.00, 0.00, 0.00, 0.00, 0.00],
    2: [1.00, 0.00, 0.00, 0.00, 0.00],
    3: [0.70, 0.30, 0.00, 0.00, 0.00],
    4: [0.50, 0.40, 0.10, 0.00, 0.00],
    5: [0.36, 0.42, 0.20, 0.02, 0.00],
    6: [0.25, 0.40, 0.30, 0.05, 0.00],
    7: [0.16, 0.33, 0.35, 0.15, 0.01],
    8: [0.11, 0.27, 0.35, 0.22, 0.05],
    9: [0.05, 0.20, 0.35, 0.30, 0.10],
}
RARITY_ORDER = ["COMMON", "UNCOMMON", "RARE", "EPIC", "ULTRA"]
```

### Step 1.5 — Pool Sizes

```python
# POOL_SIZE[rarity] = number of copies of EACH species in the shared pool
POOL_SIZE = {
    "COMMON":    27,
    "UNCOMMON":  22,
    "RARE":      18,
    "EPIC":      14,
    "ULTRA":     10,
    "UNIQUE":    1,
    "LEGENDARY": 1,
}
```

### Step 1.6 — XP Table

```python
EXP_TABLE = {
    1: 0,    # Need 0 total XP for level 1 (start)
    2: 2,
    3: 6,
    4: 10,
    5: 22,
    6: 34,
    7: 52,
    8: 72,
    9: 255,  # Effectively unreachable (cap)
}
# XP needed to go FROM level L to L+1 = EXP_TABLE[L+1] - EXP_TABLE[L]
# Level-up buy: costs 4 gold, gives 4 XP
# Per-round passive: +2 XP
```

### Step 1.7 — Stage Constants

```python
# PVE stages (no PvP matchup, fight against scripted enemies)
PVE_STAGES = {1, 2, 3, 9, 14, 19, 24, 29}

# Portal carousel stages (unique/legendary picks)
PORTAL_STAGES = {0, 10, 20}

# Item carousel stages
ITEM_CAROUSEL_STAGES = {4, 12, 17, 22, 27, 34}

# Additional pick stages
ADDITIONAL_PICK_STAGES = {5, 8, 11}

MAX_STAGE = 40  # Games rarely go past 35
```

### Deliverable
One Python file: `training/py_eco/data.py` containing all of the above as module-level constants plus `load_pokemon_data()` that reads the CSV and returns a dict of `PokemonData` keyed by name.

### Validation
- Assert 31 synergies in SYNERGY_LIST
- Assert every Pokemon in CSV has valid rarity
- Assert every evolution chain terminates (no cycles)
- Assert rarity probabilities sum to 1.0 per level
- Print summary: "Loaded N pokemon across R rarities"

---

## Phase 2 — Unit & Board Model

### Purpose
Model the bench (8 slots), board (8×3 grid), and unit instances with their stats and positions.

### Step 2.1 — Unit Instance

```python
@dataclass
class Unit:
    species: str        # "CHARIZARD"
    data: PokemonData   # Reference to static data
    stars: int          # 1, 2, or 3
    items: list[str]    # Item names (max 3), initially empty
    position: tuple[int, int]  # (x, y) where y=0 is bench, y=1-3 is board
    uid: int            # Unique instance ID (for tracking buy-then-sell)
```

Key properties (derived):
- `unit.types` → `unit.data.types` (synergies)
- `unit.rarity` → `unit.data.rarity`
- `unit.cost` → `RARITY_COST[unit.rarity]` (buy price)
- `unit.sell_price` → `unit.cost * unit.stars` (standard formula)
- `unit.family` → `family_of[unit.species]` (base form name)
- `unit.on_board` → `unit.position[1] > 0`
- `unit.on_bench` → `unit.position[1] == 0`

### Step 2.2 — Grid Model

```
Layout (y-axis, 0=bench):

y=0: [ bench_0 ] [ bench_1 ] [ bench_2 ] [ bench_3 ] [ bench_4 ] [ bench_5 ] [ bench_6 ] [ bench_7 ]
y=1: [ board_0 ] [ board_1 ] [ board_2 ] [ board_3 ] [ board_4 ] [ board_5 ] [ board_6 ] [ board_7 ]
y=2: [ board_8 ] [ board_9 ] [board_10 ] [board_11 ] [board_12 ] [board_13 ] [board_14 ] [board_15 ]
y=3: [board_16 ] [board_17 ] [board_18 ] [board_19 ] [board_20 ] [board_21 ] [board_22 ] [board_23 ]

Cell index: cell = y * 8 + x  (0-31)
- Cells 0-7: bench (y=0)
- Cells 8-15: back row (y=1)
- Cells 16-23: mid row (y=2)
- Cells 24-31: front row (y=3)
```

**Board class**:
```python
class Board:
    cells: list[Unit | None]  # 32 slots (index 0-31)

    def get(self, cell_idx) -> Unit | None
    def set(self, cell_idx, unit)
    def remove(self, cell_idx) -> Unit
    def swap(self, idx_a, idx_b)

    def bench_units(self) -> list[Unit]      # cells 0-7
    def board_units(self) -> list[Unit]      # cells 8-31
    def bench_count(self) -> int
    def board_count(self) -> int
    def bench_has_space(self) -> bool
    def first_empty_bench(self) -> int | None  # first None in 0-7
    def first_available_unit(self) -> int | None  # first occupied cell, bench then board

    def is_board_cell(self, idx) -> bool     # idx >= 8
    def is_bench_cell(self, idx) -> bool     # idx < 8
```

### Step 2.3 — Synergy Calculator

Count synergies from board units only (not bench):

```python
def compute_synergies(board: Board) -> dict[str, int]:
    """Returns {synergy_name: count} for all board units."""
    type_counts = defaultdict(set)  # synergy -> set of family base forms
    for unit in board.board_units():
        for t in unit.types:
            type_counts[t].add(unit.family)
    # Count unique families per synergy (not raw unit count)
    return {syn: len(families) for syn, families in type_counts.items()}
```

**Important**: Synergy count is based on **unique families**, not raw unit count. Two Charmanders on board contribute only 1 to FIRE synergy. Two different FIRE-type species contribute 2.

Wait — actually, re-reading the TypeScript: synergies count unique *Pokemon names* on the board, NOT families. Each distinct species name on the board that has a given type adds 1 to that synergy. Multiple copies of the same species don't stack. But a Charmander (FIRE) and Charmeleon (FIRE) from the same family would each contribute 1 to FIRE.

**Correction**: Count unique species names on board per synergy type. Actually, the standard auto-chess approach is that each unit on the board contributes its types, but duplicates of the same species don't double-count. Verify against the TypeScript `updateSynergies()` — it iterates board units and for each type, counts how many board units have that type (using unique species check OR raw count depending on implementation).

**Action item**: Read `updateSynergies()` precisely during implementation. For this guide, implement as: each unique species on board contributes its types once. If you have 2 Charmanders, FIRE gets +1 not +2.

### Step 2.4 — Evolution Logic

When a player acquires a 3rd copy of the same species (all same star level), they auto-merge:

```python
def check_evolution(board: Board, species: str, star: int) -> bool:
    """Check if we have 3 units of `species` at `star` level."""
    count = sum(
        1 for u in board.all_units()
        if u.species == species and u.stars == star
    )
    return count >= 3

def execute_evolution(board: Board, species: str, star: int) -> Unit | None:
    """Merge 3 copies into evolved form. Returns the evolved unit or None."""
    # 1. Find all 3 matching units
    matches = [i for i, u in enumerate(board.cells)
               if u and u.species == species and u.stars == star]
    if len(matches) < 3:
        return None

    # 2. Look up evolution
    evolved_species = POKEMON_DATA[species].evolution
    if not evolved_species:
        return None  # Already max star, no evolution

    # 3. Collect items from all 3 units
    all_items = []
    for idx in matches:
        all_items.extend(board.cells[idx].items)

    # 4. Remove 3 units, place evolved unit at first match's position
    pos = matches[0]
    for idx in matches:
        board.cells[idx] = None

    evolved = Unit(
        species=evolved_species,
        data=POKEMON_DATA[evolved_species],
        stars=star + 1,
        items=all_items[:3],  # Max 3 items
        position=idx_to_xy(pos),
        uid=next_uid(),
    )
    board.cells[pos] = evolved

    # 5. Check for chain evolution (3 of the evolved form might now exist)
    if check_evolution(board, evolved_species, star + 1):
        return execute_evolution(board, evolved_species, star + 1)

    return evolved
```

### Deliverable
File: `training/py_eco/board.py` with `Unit`, `Board`, `compute_synergies()`, `check_evolution()`, `execute_evolution()`.

### Validation
- Place 3 Charmanders, verify auto-merge to Charmeleon
- Place 9 Charmanders (3+3+3), verify chain-evolve: 3 Charmander → 3 Charmeleon → 1 Charizard
- Synergy counts: 1 Charmander = FIRE:1, 1 Charmander + 1 Vulpix = FIRE:2 (not FIRE:1)
- Bench units don't count for synergies

---

## Phase 3 — Shop & Pool System

### Purpose
Simulate the shared pokemon pool and per-player shop rolls, identical to the TypeScript server.

### Step 3.1 — Shared Pool

```python
class SharedPool:
    """Tracks remaining copies of each species across all 8 players."""

    def __init__(self, pokemon_by_rarity: dict[str, list[str]]):
        # pool["CHARMANDER"] = 27  (COMMON has 27 copies)
        self.pool: dict[str, int] = {}
        for rarity, species_list in pokemon_by_rarity.items():
            count = POOL_SIZE.get(rarity, 0)
            for species in species_list:
                self.pool[species] = count

    def take(self, species: str) -> bool:
        """Remove one copy from pool. Returns False if empty."""
        if self.pool.get(species, 0) <= 0:
            return False
        self.pool[species] -= 1
        return True

    def return_unit(self, species: str, count: int = 1):
        """Return copies to pool (on sell or player elimination)."""
        self.pool[species] = self.pool.get(species, 0) + count
```

**When units return to pool:**
- Player sells a unit → return copies (1 for 1-star, 3 for 2-star, 9 for 3-star)
- Player is eliminated → return ALL their units (bench + board)
- Shop is rerolled → return the 6 offered units that weren't bought

**When units leave pool:**
- A shop slot is rolled → 1 copy removed from pool
- Actually: copies leave pool when **shown in shop**, not when bought. If not bought and shop is rerolled, they return.

### Step 3.2 — Shop Roll

```python
SHOP_SIZE = 6  # 6 slots per roll

def roll_shop(player_level: int, pool: SharedPool, rng: np.random.Generator) -> list[str]:
    """Generate 6 shop slots based on player level and available pool."""
    probs = RARITY_PROB[player_level]  # [common%, uncommon%, rare%, epic%, ultra%]
    shop = []

    for _ in range(SHOP_SIZE):
        # 1. Pick rarity
        rarity = rng.choice(RARITY_ORDER, p=probs)

        # 2. Get available species of this rarity with remaining copies
        available = [
            sp for sp in POKEMON_BY_RARITY[rarity]
            if pool.pool.get(sp, 0) > 0
            and POKEMON_DATA[sp].tier == 1  # Shop only offers tier-1 (1-star) units
        ]

        if not available:
            shop.append(None)  # Empty slot (pool exhausted for this rarity)
            continue

        # 3. Pick uniformly from available species
        species = rng.choice(available)
        pool.take(species)
        shop.append(species)

    return shop
```

**Critical detail**: The shop ONLY offers **tier-1 (1-star)** base forms. You never see a Charmeleon in the shop — only Charmander. Evolution happens after buying 3 copies.

**Critical detail**: UNIQUE and LEGENDARY are NOT in the normal shop pool. They only appear during portal carousel stages (0, 10, 20) as propositions.

### Step 3.3 — Shop Operations

```python
class Shop:
    slots: list[str | None]  # 6 slots, species name or None
    locked: bool             # If True, don't reroll on new round
    free_rerolls: int        # Free rerolls remaining this round
    lock_used_this_turn: bool  # Lock toggle limit (once per turn)

    def reroll(self, player_level, pool, rng):
        """Return current shop to pool, roll new shop."""
        for species in self.slots:
            if species:
                pool.return_unit(species)
        self.slots = roll_shop(player_level, pool, rng)

    def buy(self, slot_idx) -> str | None:
        """Buy from slot. Returns species name. Does NOT return to pool."""
        species = self.slots[slot_idx]
        self.slots[slot_idx] = None
        return species

    def remove(self, slot_idx):
        """Remove from shop without buying. Returns to pool."""
        species = self.slots[slot_idx]
        if species:
            # Note: in TS, REMOVE_SHOP doesn't refund — it's like buying and discarding
            # But it DOES remove from pool permanently (doesn't return)
            self.slots[slot_idx] = None
```

### Step 3.4 — Round Start Shop Behavior

At the start of each new round (after fight resolves):
1. If `shop.locked == False`: reroll shop (return old offers to pool, generate new ones)
2. If `shop.locked == True`: keep current shop, unlock it (one-time freeze)
3. Reset `free_rerolls` based on stage (usually 0, some special stages grant free rerolls)
4. Reset `lock_used_this_turn = False`

### Deliverable
File: `training/py_eco/shop.py` with `SharedPool`, `Shop`, `roll_shop()`.

### Validation
- Roll 1000 shops at level 5, verify rarity distribution matches probabilities (±5%)
- Buy a unit, verify pool decremented
- Sell a unit, verify pool incremented
- Reroll, verify old offers returned to pool
- Lock shop, advance round, verify shop preserved
- Exhaust a species (buy all 27 Commons), verify it stops appearing

---

## Phase 4 — Economy Engine

### Purpose
Manage gold, XP, income, streak, interest, and the round lifecycle.

### Step 4.1 — Player State

```python
@dataclass
class PlayerState:
    # Identity
    player_id: int         # 0-7
    is_bot: bool
    alive: bool = True

    # Economy
    money: int = 5         # Starting gold
    level: int = 1
    experience: int = 0
    life: int = 100
    streak: int = 0        # Positive = win streak, negative = loss streak
    rank: int = 8          # Current standing (1=first, 8=last)

    # Tracking
    total_money_earned: int = 0
    total_player_damage_dealt: int = 0
    reroll_count: int = 0

    # Per-turn state
    actions_this_turn: int = 0
    lock_used_this_turn: bool = False
    last_buy_species: str | None = None   # For buy-then-sell detection
    consecutive_moves: int = 0
    last_move_target: int = -1            # For oscillation prevention

    # Components
    board: Board           # 32-cell grid
    shop: Shop
    items: list[str]       # Held items (max 10)

    # Opponent tracking
    history: list[dict]    # [{opponent_id, result}]
```

### Step 4.2 — Interest & Income Calculation

```python
def calculate_income(player: PlayerState, is_pve: bool) -> int:
    """Calculate gold income after a fight. Returns total income."""
    # Interest: floor(money / 10), capped at 5
    max_interest = 5
    interest = min(max_interest, player.money // 10)
    player.interest = interest  # Store for obs/reward

    income = interest

    # Streak bonus (PvP only)
    if not is_pve:
        streak_bonus = min(5, abs(player.streak))  # Cap at 5
        income += streak_bonus

    # Base income
    income += 5

    return income

def apply_income(player: PlayerState, is_pve: bool):
    """Give income and XP after fight."""
    income = calculate_income(player, is_pve)
    player.money += income
    player.total_money_earned += income

    # Passive XP (+2 per round)
    add_experience(player, 2)
```

### Step 4.3 — Level-Up / Buy XP

```python
def get_exp_for_next_level(level: int) -> int:
    """XP needed to go from `level` to `level+1`."""
    if level >= 9:
        return 999  # Can't level past 9
    return EXP_TABLE[level + 1] - EXP_TABLE[level]

def add_experience(player: PlayerState, amount: int):
    """Add XP, auto-level if threshold met."""
    player.experience += amount
    while player.level < 9:
        needed = get_exp_for_next_level(player.level)
        if player.experience >= needed:
            player.experience -= needed
            player.level += 1
        else:
            break

def buy_experience(player: PlayerState) -> bool:
    """Buy 4 XP for 4 gold. Returns success."""
    if player.money < 4 or player.level >= 9:
        return False
    player.money -= 4
    add_experience(player, 4)
    return True
```

### Step 4.4 — Buy / Sell Units

```python
def buy_unit(player: PlayerState, shop_slot: int, pool: SharedPool) -> dict:
    """Buy from shop slot. Returns info dict for reward calc."""
    species = player.shop.slots[shop_slot]
    if not species:
        return {"success": False}

    cost = RARITY_COST[POKEMON_DATA[species].rarity]
    if player.money < cost:
        return {"success": False}

    bench_slot = player.board.first_empty_bench()
    if bench_slot is None:
        # Check if buying would trigger evolution (freeing a slot)
        # Count existing copies on bench+board
        existing = sum(1 for u in player.board.all_units()
                       if u.species == species and u.stars == 1)
        if existing < 2:  # Need 2 existing + this buy = 3 for evolution
            return {"success": False}
        # Evolution will free slots, allow the buy

    # Execute buy
    player.money -= cost
    player.shop.slots[shop_slot] = None  # Already taken from pool on roll

    # Create unit and place on bench
    unit = Unit(
        species=species,
        data=POKEMON_DATA[species],
        stars=1,
        items=[],
        position=(bench_slot % 8, 0),
        uid=next_uid(),
    )
    player.board.cells[bench_slot] = unit

    # Check evolution
    evolved = None
    if check_evolution(player.board, species, 1):
        evolved = execute_evolution(player.board, species, 1)

    # Duplicate / evolution detection
    existing_count = sum(1 for u in player.board.all_units()
                         if u and u.family == family_of.get(species, species))

    result = {
        "success": True,
        "species": species,
        "cost": cost,
        "evolved": evolved is not None,
        "is_duplicate": existing_count >= 2,  # Had 1+ before this buy
        "rarity": POKEMON_DATA[species].rarity,
    }

    player.last_buy_species = species
    return result


def sell_unit(player: PlayerState, cell_idx: int, pool: SharedPool) -> dict:
    """Sell unit at cell_idx. Returns info dict."""
    unit = player.board.cells[cell_idx]
    if not unit:
        return {"success": False}

    # Calculate sell price
    sell_price = RARITY_COST[unit.rarity] * unit.stars

    # Return copies to pool
    copies_returned = 3 ** (unit.stars - 1)  # 1-star=1, 2-star=3, 3-star=9
    base_species = get_base_form(unit.species)
    pool.return_unit(base_species, copies_returned)

    # Remove from board
    player.board.cells[cell_idx] = None
    player.money += sell_price

    # Buy-then-sell detection
    is_buy_then_sell = (player.last_buy_species == unit.species)

    result = {
        "success": True,
        "species": unit.species,
        "sell_price": sell_price,
        "stars": unit.stars,
        "is_evolved": unit.stars >= 2,
        "is_buy_then_sell": is_buy_then_sell,
    }

    player.last_buy_species = None  # Reset after sell
    return result
```

### Step 4.5 — Streak Tracking

```python
def update_streak(player: PlayerState, result: str):
    """Update win/loss streak after fight."""
    if result == "WIN":
        if player.streak > 0:
            player.streak += 1
        else:
            player.streak = 1
    elif result == "LOSS":
        if player.streak < 0:
            player.streak -= 1
        else:
            player.streak = -1
    # DRAW: streak unchanged
```

### Step 4.6 — Damage Calculation (on loss)

```python
# Damage to player HP on loss = sum of surviving enemy unit stars + stage bonus
def calculate_player_damage(surviving_enemy_units: list[Unit], stage: int) -> int:
    """Damage dealt to loser's HP."""
    unit_damage = sum(u.stars for u in surviving_enemy_units)

    # Stage bonus (increases over time)
    if stage <= 5:
        stage_damage = 0
    elif stage <= 10:
        stage_damage = 1
    elif stage <= 15:
        stage_damage = 2
    elif stage <= 20:
        stage_damage = 5
    elif stage <= 25:
        stage_damage = 8
    else:
        stage_damage = 12

    return unit_damage + stage_damage
```

### Deliverable
File: `training/py_eco/economy.py` with `PlayerState`, income/buy/sell/level-up functions.

### Validation
- Start with 5 gold, buy a 1-cost unit → 4 gold remaining
- Sell a 2-star rare unit → +6 gold (3 × 2)
- Win 3 fights in a row → streak = 3
- At 50 gold: interest = 5, income = 5 + 5 + streak
- Level from 1→2 needs 2 XP. Buy XP (4g) gives 4 XP → jumps to level 3
- Verify a player going from 45g → round income → should get 4 interest

---

## Phase 5 — Fight Power Estimator

### Purpose
Instead of simulating fights (which is the bottleneck), estimate the fight outcome using a power-score heuristic. This is the KEY speedup — we trade fight accuracy for 100-1000× faster training.

### Step 5.1 — Unit Power Score

```python
def unit_power(unit: Unit) -> float:
    """Estimate combat power of a single unit."""
    d = unit.data

    # Star scaling (exponential — 2-stars are ~3x, 3-stars are ~9x)
    star_mult = {1: 1.0, 2: 3.0, 3: 9.0}[unit.stars]

    # Raw combat stats
    offensive = d.atk * (1 + d.speed / 100)  # DPS proxy
    defensive = d.hp * (1 + (d.def_ + d.spe_def) / 50)  # EHP proxy

    # Range bonus (ranged units survive longer)
    range_mult = 1.0 + 0.1 * max(0, d.range - 1)

    # Base power
    power = (offensive * 0.4 + defensive * 0.6) * star_mult * range_mult

    # Item bonus (flat per item, simple approximation)
    power *= (1 + 0.15 * len(unit.items))

    return power
```

### Step 5.2 — Synergy Power Bonus

```python
def synergy_power_bonus(synergies: dict[str, int]) -> float:
    """Bonus multiplier from active synergies."""
    bonus = 0.0
    for syn_name, count in synergies.items():
        triggers = SYNERGY_TRIGGERS.get(syn_name, [])
        tiers_hit = sum(1 for t in triggers if count >= t)
        if tiers_hit > 0:
            # Each tier hit adds ~5% team power
            bonus += 0.05 * tiers_hit
    return bonus
```

### Step 5.3 — Team Power Score

```python
def team_power(board: Board) -> float:
    """Total team power score."""
    units = board.board_units()
    if not units:
        return 0.0

    raw = sum(unit_power(u) for u in units)
    synergies = compute_synergies(board)
    syn_bonus = synergy_power_bonus(synergies)

    return raw * (1 + syn_bonus)
```

### Step 5.4 — Fight Outcome Estimator

```python
def estimate_fight(player_power: float, opponent_power: float,
                   rng: np.random.Generator) -> tuple[str, int, int]:
    """
    Estimate fight outcome.

    Returns: (result, player_units_killed, enemy_units_killed)
    - result: "WIN", "LOSS", or "DRAW"
    - Killed counts used for damage calculation and reward signals
    """
    if player_power == 0 and opponent_power == 0:
        return ("DRAW", 0, 0)

    total = player_power + opponent_power
    win_prob = player_power / total

    # Add noise (±10%) to prevent deterministic outcomes
    noise = rng.normal(0, 0.05)
    win_prob = np.clip(win_prob + noise, 0.02, 0.98)

    roll = rng.random()

    # Draw zone: if powers are very close (within 5%), small draw chance
    draw_threshold = 0.05 * (1 - abs(win_prob - 0.5) * 2)

    if roll < draw_threshold:
        result = "DRAW"
        player_killed = 0
        enemy_killed = 0
    elif roll < draw_threshold + win_prob * (1 - draw_threshold):
        result = "WIN"
        # Estimate kills: proportional to power ratio
        kill_ratio = min(1.0, player_power / max(1, opponent_power))
        enemy_killed = max(1, int(len(board.board_units()) * kill_ratio))  # placeholder
        player_killed = max(0, int(len(board.board_units()) * (1 - kill_ratio) * 0.5))
    else:
        result = "LOSS"
        kill_ratio = min(1.0, opponent_power / max(1, player_power))
        player_killed = max(1, int(kill_ratio * 3))  # survivors that deal damage
        enemy_killed = max(0, int((1 - kill_ratio) * 2))

    return (result, player_killed, enemy_killed)
```

### Step 5.5 — Tuning the Estimator

The power estimator doesn't need to be perfect — it needs to produce **plausible rank distributions** that teach the agent correct economy timing. Key properties:

1. **Stronger teams should win more often** (monotonic in power score)
2. **Stars matter a lot** (a 2-star beats most 1-stars)
3. **Synergies matter** (6 FIRE units with synergy active should beat 6 random units)
4. **Late-game power should scale** (level 8 teams should crush level 5 teams)

**Calibration method**: Run 1000 simulated games with the estimator, compare rank distribution to real TS server games. Tune `star_mult`, `syn_bonus`, and noise until distributions are similar. The exact numbers don't need to match — the agent will recalibrate when transferred to the full server.

### Deliverable
File: `training/py_eco/power.py` with `unit_power()`, `team_power()`, `estimate_fight()`.

### Validation
- 6 random 1-star commons < 3 random 2-star rares (power score)
- Synergy bonus should be 10-30% for a typical board
- Run 100 fights at equal power → ~45-55% win rate (noise)
- Run 100 fights at 2:1 power → ~80-90% win rate

---

## Phase 6 — Bot AI (Opponents)

### Purpose
Simulate 7 bot opponents with realistic economy behavior. These don't need to be sophisticated — they just need to generate plausible game states that the RL agent can learn from.

### Step 6.1 — Bot Decision Tree

Each bot follows a simple priority-based strategy per round:

```python
class BotAI:
    """Simple rule-based bot for economy training."""

    strategy: str  # "econ", "aggro", "balanced" (randomly assigned)

    def take_turn(self, player: PlayerState, stage: int, pool: SharedPool, rng):
        """Execute one full turn of bot decisions."""

        if self.strategy == "econ":
            self._econ_turn(player, stage, pool, rng)
        elif self.strategy == "aggro":
            self._aggro_turn(player, stage, pool, rng)
        else:
            self._balanced_turn(player, stage, pool, rng)

    def _econ_turn(self, player, stage, pool, rng):
        """Conservative: save to 50g ASAP, only buy upgrades."""
        # 1. Level up if XP is 2 away from next level (efficient)
        # 2. Buy only if it completes a pair or triple
        # 3. Never reroll below 50g
        # 4. Sell bench units that don't fit synergies
        ...

    def _aggro_turn(self, player, stage, pool, rng):
        """Aggressive: spend to stay strong, level fast."""
        # 1. Level up if affordable
        # 2. Buy any unit that fits synergies
        # 3. Reroll 1-2 times looking for upgrades
        # 4. Fill board to max team size
        ...

    def _balanced_turn(self, player, stage, pool, rng):
        """Middle ground: econ early, aggro late."""
        # Before stage 10: econ behavior
        # After stage 10: aggro behavior
        ...
```

### Step 6.2 — Bot Level-Up Timing

```python
BOT_LEVEL_SCHEDULE = {
    # stage: target_level
    3: 3,
    5: 4,
    9: 5,
    13: 6,
    17: 7,
    21: 8,
    25: 9,
}
# Bots buy XP to match this schedule (±1 level variance for diversity)
```

### Step 6.3 — Bot Board Management

Bots need to:
1. Place bought units on board (prefer filling synergies)
2. Sell bench units that don't contribute to synergies
3. Swap units when better ones are available (higher star, better synergy)

```python
def bot_place_units(player: PlayerState):
    """Move bench units to board if space available."""
    while player.board.bench_count() > 0 and player.board.board_count() < player.level:
        # Find best bench unit (highest power score or synergy contribution)
        best_bench = None
        best_score = -1
        for i in range(8):
            unit = player.board.cells[i]
            if unit:
                score = unit_power(unit) + synergy_contribution(unit, player.board)
                if score > best_score:
                    best_score = score
                    best_bench = i

        if best_bench is not None:
            # Find empty board cell
            empty = first_empty_board_cell(player.board)
            if empty is not None:
                player.board.swap(best_bench, empty)
            else:
                break
        else:
            break
```

### Step 6.4 — Bot Diversity

Randomly assign each bot one of 3-4 strategy archetypes at game start. Also add variance:
- Starting gold: 5 (same as player)
- Level-up timing: ±1 stage from schedule
- Reroll frequency: ±30% from base
- Synergy focus: randomly pick 2-3 preferred synergies, bias purchases toward them

This creates opponent diversity without needing the full bot scenario system.

### Deliverable
File: `training/py_eco/bots.py` with `BotAI`, `bot_place_units()`.

### Validation
- Run 100 games with 7 bots only → verify all bots level to 8-9 by stage 25
- Verify bots maintain 40-60g economy in mid-game
- Verify bot boards have 2-3 active synergies by stage 15
- No bot should go bankrupt (0g with empty board)

---

## Phase 7 — Observation Builder

### Purpose
Build the exact 612-float observation vector, matching the TypeScript server's layout bit-for-bit. This is critical for checkpoint transfer.

### Step 7.1 — Observation Layout (612 floats)

```
Offset  Count  Section
──────  ─────  ───────
0       14     Player stats
14      54     Shop (6 slots × 9 features)
68      384    Board/Bench (32 cells × 12 features)
452     10     Held items
462     31     Synergies
493     7      Game info
500     70     Opponents (7 × 10 features)
570     42     Propositions (6 × 7 features)
──────  ─────
Total:  612
```

### Step 7.2 — Player Stats (indices 0-13)

```python
def build_player_obs(player: PlayerState) -> list[float]:
    return [
        player.life / 100,                          # [0]
        player.money / 300,                          # [1]
        player.level / 9,                            # [2]
        (player.streak + 20) / 40,                   # [3]
        player.interest / 5,                         # [4]
        1.0 if player.alive else 0.0,                # [5]
        player.rank / 8,                             # [6]
        player.board.board_count() / 9,              # [7]
        get_exp_for_next_level(player.level) / 32,   # [8]  — expNeeded, not current exp
        player.shop.free_rerolls / 3,                # [9]
        player.reroll_count / 50,                    # [10]
        1.0 if player.shop.locked else 0.0,          # [11]
        player.total_money_earned / 500,             # [12]
        player.total_player_damage_dealt / 300,      # [13]
    ]
```

### Step 7.3 — Shop (indices 14-67)

```python
RARITY_FLOAT = {
    "COMMON": 0.1, "UNCOMMON": 0.2, "RARE": 0.4, "EPIC": 0.6,
    "ULTRA": 0.8, "UNIQUE": 0.9, "LEGENDARY": 1.0,
    "HATCH": 0.3, "SPECIAL": 0.5,
}

def build_shop_obs(player: PlayerState) -> list[float]:
    obs = []
    for slot_idx in range(6):
        species = player.shop.slots[slot_idx] if slot_idx < len(player.shop.slots) else None
        if species is None:
            obs.extend([0.0] * 9)
        else:
            d = POKEMON_DATA[species]
            types = d.types + [""] * (3 - len(d.types))  # Pad to 3

            # Check if evolution possible (2+ copies already owned)
            owned_count = sum(1 for u in player.board.all_units()
                             if u.species == species and u.stars == 1)

            obs.extend([
                1.0,                                        # hasUnit
                get_species_index(species),                 # speciesIndex (normalized)
                RARITY_FLOAT.get(d.rarity, 0.0),           # rarity
                RARITY_COST.get(d.rarity, 0) / 20,         # cost
                get_synergy_index(types[0]),                # type1
                get_synergy_index(types[1]),                # type2
                get_synergy_index(types[2]),                # type3
                0.0,                                        # type4 (always 0 for shop)
                1.0 if owned_count >= 2 else 0.0,           # isEvoPossible
            ])
    return obs
```

### Step 7.4 — Board/Bench (indices 68-451)

```python
def build_board_obs(player: PlayerState) -> list[float]:
    obs = []
    for cell_idx in range(32):  # y=0..3, x=0..7
        unit = player.board.cells[cell_idx]
        if unit is None:
            obs.extend([0.0] * 12)
        else:
            d = unit.data
            types = d.types + [""] * (4 - len(d.types))
            obs.extend([
                1.0,                                  # hasUnit
                get_species_index(unit.species),      # speciesIndex
                unit.stars / 3,                       # stars
                RARITY_FLOAT.get(d.rarity, 0.0),     # rarity
                get_synergy_index(types[0]),          # type1
                get_synergy_index(types[1]),          # type2
                get_synergy_index(types[2]),          # type3
                get_synergy_index(types[3]),          # type4
                d.atk / 100,                          # atk
                d.hp / 1000,                          # hp
                d.range / 5,                          # range
                len(unit.items) / 3,                  # numItems
            ])
    return obs
```

### Step 7.5 — Held Items (indices 452-461)

```python
def build_items_obs(player: PlayerState) -> list[float]:
    obs = [0.0] * 10
    for i, item in enumerate(player.items[:10]):
        obs[i] = get_item_index(item)  # Normalized item index
    return obs
```

**Note**: Items are mostly irrelevant in the economy trainer. Stub with zeros initially. When items are added later, each item gets a normalized index.

### Step 7.6 — Synergies (indices 462-492)

```python
def build_synergy_obs(player: PlayerState) -> list[float]:
    synergies = compute_synergies(player.board)
    obs = []
    for syn_name in SYNERGY_LIST:  # Fixed order, 31 synergies
        count = synergies.get(syn_name, 0)
        obs.append(min(count / 10, 1.0))
    return obs
```

### Step 7.7 — Game Info (indices 493-499)

```python
def build_game_info_obs(game_state) -> list[float]:
    return [
        game_state.stage / 50,                              # [493]
        0.0,                     # phase: always 0 (PICK)   # [494]
        game_state.players_alive / 8,                       # [495]
        1.0 if game_state.has_propositions else 0.0,        # [496]
        0.0,                     # weather: stub at 0       # [497]
        1.0 if game_state.stage in PVE_STAGES else 0.0,     # [498]
        game_state.rl_player.level / 9,                     # [499] maxTeamSize
    ]
```

### Step 7.8 — Opponents (indices 500-569)

```python
def build_opponents_obs(game_state, rl_player_id: int) -> list[float]:
    obs = []
    opponents = [p for p in game_state.players if p.alive and p.player_id != rl_player_id]
    opponents.sort(key=lambda p: p.rank)  # Sort by rank

    for i in range(7):  # Always 7 slots
        if i < len(opponents):
            opp = opponents[i]
            synergies = compute_synergies(opp.board)
            top_syns = sorted(synergies.items(), key=lambda x: -x[1])[:2]

            obs.extend([
                opp.life / 100,
                opp.rank / 8,
                opp.level / 9,
                opp.money / 300,
                (opp.streak + 20) / 40,
                opp.board.board_count() / 9,
                get_synergy_index(top_syns[0][0]) if len(top_syns) > 0 else 0.0,
                top_syns[0][1] / 10 if len(top_syns) > 0 else 0.0,
                get_synergy_index(top_syns[1][0]) if len(top_syns) > 1 else 0.0,
                top_syns[1][1] / 10 if len(top_syns) > 1 else 0.0,
            ])
        else:
            obs.extend([0.0] * 10)
    return obs
```

### Step 7.9 — Propositions (indices 570-611)

```python
def build_propositions_obs(game_state) -> list[float]:
    # Stub: propositions are stage-0/10/20 portal picks
    # For economy trainer, can be zeros initially
    return [0.0] * 42
```

### Step 7.10 — Helper Functions

```python
def get_species_index(species: str) -> float:
    """Normalize species name to 0-1 float."""
    # Must match TypeScript's PRECOMPUTED_POKEMONS_PER_TYPE_AND_CATEGORY index
    # Use a fixed sorted list of all species names
    return SPECIES_TO_INDEX.get(species, 0.0) / len(ALL_SPECIES)

def get_synergy_index(synergy_name: str) -> float:
    """Normalize synergy name to 0-1 float."""
    if not synergy_name:
        return 0.0
    idx = SYNERGY_LIST.index(synergy_name) if synergy_name in SYNERGY_LIST else 0
    return (idx + 1) / len(SYNERGY_LIST)  # +1 so index 0 maps to nonzero

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
```

**Critical**: `get_species_index()` and `get_synergy_index()` MUST produce identical values to the TypeScript server. Extract the exact ordering from the TS code and hardcode it in Python.

### Deliverable
File: `training/py_eco/observation.py` with `build_observation()` that returns `np.ndarray` of shape `(612,)`.

### Validation
- Build obs for a known game state, compare with TS server's `/step` response
- Assert obs.shape == (612,)
- Assert all values in [0, 1]
- Assert player stats at correct indices
- Spot-check: set money=150, verify obs[1] = 150/300 = 0.5

---

## Phase 8 — Action Dispatcher & Mask Builder

### Purpose
Map each of the 92 actions to game operations and build valid action masks.

### Step 8.1 — Action Enum

```python
# Action indices (must match TypeScript TrainingAction enum exactly)
BUY_0, BUY_1, BUY_2, BUY_3, BUY_4, BUY_5 = 0, 1, 2, 3, 4, 5
REFRESH = 6
LEVEL_UP = 7
LOCK_SHOP = 8
END_TURN = 9
MOVE_BASE = 10   # MOVE_0=10, MOVE_1=11, ..., MOVE_31=41
SELL_BASE = 42   # SELL_0=42, SELL_1=43, ..., SELL_31=73
REMOVE_SHOP_BASE = 74  # REMOVE_SHOP_0=74, ..., REMOVE_SHOP_5=79
PICK_BASE = 80   # PICK_0=80, ..., PICK_5=85
COMBINE_BASE = 86  # COMBINE_0=86, ..., COMBINE_5=91

NUM_ACTIONS = 92
MAX_ACTIONS_PER_TURN = 15
```

### Step 8.2 — Action Dispatcher

```python
def execute_action(action: int, player: PlayerState, game_state, pool: SharedPool) -> dict:
    """Execute action and return info dict for reward calculation."""
    player.actions_this_turn += 1
    info = {"action": action, "success": False}

    # BUY (0-5)
    if 0 <= action <= 5:
        info = buy_unit(player, action, pool)
        if info["success"]:
            player.consecutive_moves = 0
            player.last_move_target = -1

    # REFRESH (6)
    elif action == REFRESH:
        cost = 0 if player.shop.free_rerolls > 0 else 1
        if player.money >= cost:
            player.money -= cost
            if player.shop.free_rerolls > 0:
                player.shop.free_rerolls -= 1
            player.shop.reroll(player.level, pool, game_state.rng)
            player.reroll_count += 1
            info = {"success": True, "action": "refresh", "cost": cost}
        player.consecutive_moves = 0

    # LEVEL_UP (7)
    elif action == LEVEL_UP:
        old_level = player.level
        success = buy_experience(player)
        info = {
            "success": success, "action": "level_up",
            "old_level": old_level, "new_level": player.level,
            "money_after": player.money,
        }
        player.consecutive_moves = 0

    # LOCK_SHOP (8)
    elif action == LOCK_SHOP:
        player.shop.locked = not player.shop.locked
        player.lock_used_this_turn = True
        info = {"success": True, "action": "lock"}
        player.consecutive_moves = 0

    # END_TURN (9)
    elif action == END_TURN:
        info = {"success": True, "action": "end_turn"}

    # MOVE (10-41)
    elif MOVE_BASE <= action <= MOVE_BASE + 31:
        target_cell = action - MOVE_BASE
        source_cell = player.board.first_available_unit()
        if source_cell is not None and player.board.cells[target_cell] is None:
            # Board-full check: bench→board only if board not full
            source_is_bench = source_cell < 8
            target_is_board = target_cell >= 8
            if source_is_bench and target_is_board:
                if player.board.board_count() >= player.level:
                    info = {"success": False, "reason": "board_full"}
                    return info

            player.board.swap(source_cell, target_cell)
            player.consecutive_moves += 1
            player.last_move_target = target_cell
            info = {
                "success": True, "action": "move",
                "from": source_cell, "to": target_cell,
                "consecutive": player.consecutive_moves,
            }
        else:
            info = {"success": False, "action": "move"}

    # SELL (42-73)
    elif SELL_BASE <= action <= SELL_BASE + 31:
        cell_idx = action - SELL_BASE
        info = sell_unit(player, cell_idx, pool)
        player.consecutive_moves = 0

    # REMOVE_SHOP (74-79)
    elif REMOVE_SHOP_BASE <= action <= REMOVE_SHOP_BASE + 5:
        slot_idx = action - REMOVE_SHOP_BASE
        player.shop.remove(slot_idx)
        info = {"success": True, "action": "remove_shop"}
        player.consecutive_moves = 0

    # PICK (80-85)
    elif PICK_BASE <= action <= PICK_BASE + 5:
        pick_idx = action - PICK_BASE
        info = handle_pick(player, pick_idx, game_state)
        player.consecutive_moves = 0

    # COMBINE (86-91)
    elif COMBINE_BASE <= action <= COMBINE_BASE + 5:
        # Items system — stub for now
        info = {"success": False, "action": "combine", "reason": "not_implemented"}

    return info
```

### Step 8.3 — Action Mask Builder

```python
def build_action_mask(player: PlayerState, game_state) -> np.ndarray:
    """Build 92-element binary mask. 1=valid, 0=invalid."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)

    # Dead or not in pick phase → only END_TURN
    if not player.alive or game_state.phase != "PICK":
        mask[END_TURN] = 1
        return mask

    # --- Proposition override ---
    if game_state.has_pokemon_propositions:
        # Only PICK actions valid
        for i in range(6):
            if i < len(game_state.pokemon_propositions):
                # Valid if bench has space OR triggers evolution
                if player.board.bench_has_space() or \
                   _pick_triggers_evolution(player, game_state.pokemon_propositions[i]):
                    mask[PICK_BASE + i] = 1
        if mask.sum() == 0:
            mask[END_TURN] = 1  # Fallback
        return mask

    if game_state.has_item_propositions:
        for i in range(len(game_state.item_propositions)):
            mask[PICK_BASE + i] = 1
        return mask

    # --- Normal pick phase ---

    # END_TURN: always valid
    mask[END_TURN] = 1

    # LOCK_SHOP: valid if not used this turn
    if not player.lock_used_this_turn:
        mask[LOCK_SHOP] = 1

    # BUY (0-5): valid if shop has unit, enough gold, bench has space (or evo)
    for i in range(6):
        species = player.shop.slots[i] if i < len(player.shop.slots) else None
        if species:
            cost = RARITY_COST[POKEMON_DATA[species].rarity]
            if player.money >= cost:
                if player.board.bench_has_space():
                    mask[i] = 1  # BUY_0..5 = indices 0..5
                else:
                    # Check if buying triggers evolution (frees bench space)
                    owned = sum(1 for u in player.board.all_units()
                               if u.species == species and u.stars == 1)
                    if owned >= 2:
                        mask[i] = 1

    # SELL (42-73): valid if cell has unit
    for cell_idx in range(32):
        if player.board.cells[cell_idx] is not None:
            mask[SELL_BASE + cell_idx] = 1

    # MOVE (10-41): valid if target empty, source exists, board constraints
    source = player.board.first_available_unit()
    if source is not None:
        for target in range(32):
            if player.board.cells[target] is None:
                # Bench→board: check board size
                source_is_bench = source < 8
                target_is_board = target >= 8
                if source_is_bench and target_is_board:
                    if player.board.board_count() >= player.level:
                        continue  # Board full

                # Oscillation prevention
                if player.consecutive_moves >= 2 and target == player.last_move_target:
                    continue

                mask[MOVE_BASE + target] = 1

    # REFRESH (6): valid if can afford
    roll_cost = 0 if player.shop.free_rerolls > 0 else 1
    if player.money >= roll_cost:
        mask[REFRESH] = 1

    # LEVEL_UP (7): valid if can afford and not max level
    if player.money >= 4 and player.level < 9:
        mask[LEVEL_UP] = 1

    # REMOVE_SHOP (74-79): valid if shop slot has unit and can afford
    for i in range(6):
        species = player.shop.slots[i] if i < len(player.shop.slots) else None
        if species:
            cost = RARITY_COST[POKEMON_DATA[species].rarity]
            if player.money >= cost:
                mask[REMOVE_SHOP_BASE + i] = 1

    # COMBINE (86-91): stub — no items yet
    # When items are implemented, check valid recipes

    return mask
```

### Deliverable
File: `training/py_eco/actions.py` with `execute_action()`, `build_action_mask()`.

### Validation
- At game start (5g, level 1, empty board): BUY for 1-cost units valid, LEVEL_UP valid, SELL invalid (nothing to sell), MOVE invalid (nothing to move)
- With 0g: BUY invalid, REFRESH invalid, LEVEL_UP invalid
- With full bench (8 units): BUY invalid unless triggers evolution
- After END_TURN: verify action count resets

---

## Phase 9 — Reward Calculator

### Purpose
Reproduce the TypeScript reward signals exactly. This is critical for checkpoint transfer — if rewards differ, the transferred policy will behave incorrectly.

### Step 9.1 — Reward Breakdown Dict

```python
@dataclass
class RewardTracker:
    breakdown: dict[str, float] = field(default_factory=dict)

    def add(self, signal: str, value: float):
        self.breakdown[signal] = self.breakdown.get(signal, 0) + value

    @property
    def total(self) -> float:
        return sum(self.breakdown.values())

    def reset(self):
        self.breakdown.clear()
```

### Step 9.2 — Per-Step Rewards (during pick phase)

```python
def compute_step_reward(action: int, action_info: dict, player: PlayerState,
                        game_state, reward: RewardTracker):
    """Compute rewards for a single action within the pick phase."""
    stage = game_state.stage

    # --- Buy rewards ---
    if action_info.get("action") == "buy" and action_info["success"]:
        if action_info["evolved"]:
            r = 0.30 if stage > 20 else 0.20
            reward.add("buy_evolution", r)

            # Evo-from-reroll bonus (if previous action was REFRESH)
            if player.prev_action == REFRESH:
                evo_bonus = {
                    "COMMON": 0.50, "UNCOMMON": 0.75, "RARE": 1.00,
                    "EPIC": 1.50, "ULTRA": 2.00, "UNIQUE": 2.00,
                    "LEGENDARY": 2.00, "HATCH": 0.75, "SPECIAL": 1.00,
                }
                reward.add("evo_from_reroll", evo_bonus.get(action_info["rarity"], 0.5))

        elif action_info["is_duplicate"]:
            r = 0.12 if stage > 20 else 0.08
            reward.add("buy_duplicate", r)

    # --- Sell penalties ---
    if action_info.get("action") == "sell" and action_info["success"]:
        if action_info["is_evolved"]:
            reward.add("sell_evolved", -0.15)
        if action_info["is_buy_then_sell"]:
            reward.add("buy_then_sell", -1.0)

    # --- Move fidget ---
    if action_info.get("action") == "move" and action_info["success"]:
        if action_info["consecutive"] > 2:
            reward.add("move_fidget", -0.08)

    # --- Level-up ---
    if action_info.get("action") == "level_up" and action_info["success"]:
        new_level = action_info["new_level"]
        money_after = action_info["money_after"]

        if new_level == 5 and stage == 9:
            reward.add("level_up", 0.10)  # Exception: stage 9 → level 5
        elif money_after >= 50:
            reward.add("level_up_rich", 0.0)  # Neutral
        else:
            reward.add("level_up_penalty", -0.15)

    # --- Reroll penalty (below 50g, no gold pressure) ---
    if action_info.get("action") == "refresh" and action_info["success"]:
        interest = min(5, player.money // 10)
        interest_signal = interest * 0.06
        if player.money < 50 and not _is_gold_pressure(player, stage):
            penalty = max(-(interest_signal / 2), -0.06)
            reward.add("reroll_eco_penalty", penalty)

    # --- Empty turn penalty ---
    if action == END_TURN and player.actions_this_turn == 1:
        if stage >= 16 and player.money >= 50:
            reward.add("empty_turn", -0.15)

    # --- Bench penalty (on end turn) ---
    if action == END_TURN:
        open_board_slots = player.level - player.board.board_count()
        if open_board_slots > 0:
            penalized = min(player.board.bench_count(), open_board_slots)
            if penalized > 0:
                reward.add("bench_penalty", penalized * -0.01)

    player.prev_action = action
```

### Step 9.3 — Per-Round Rewards (after fight)

```python
def compute_round_reward(player: PlayerState, fight_result: str,
                         enemy_killed: int, game_state, reward: RewardTracker):
    """Compute rewards after fight resolution."""
    stage = game_state.stage

    # --- Battle result (stage-scaled) ---
    win_scale = _get_win_scaling(stage)
    loss_scale = _get_loss_scaling(stage)

    if fight_result == "WIN":
        reward.add("battle", 0.75 * (1 + win_scale))
    elif fight_result == "LOSS":
        reward.add("battle", -0.5 * (1 + loss_scale))
    # DRAW: 0

    # --- Enemy kills ---
    reward.add("enemy_kills", enemy_killed * 0.02)

    # --- HP preservation (win only) ---
    if fight_result == "WIN":
        reward.add("hp_preservation", (player.life / 100) * 0.005)

    # --- Keep unique/legendary ---
    for unit in player.board.board_units():
        if unit.rarity == "UNIQUE":
            reward.add("keep_unique", 0.04)
        elif unit.rarity == "LEGENDARY":
            reward.add("keep_legendary", 0.04)

    # --- Synergy depth ---
    synergies = compute_synergies(player.board)
    syn_reward = 0.0
    active_count = 0
    for syn_name, count in synergies.items():
        triggers = SYNERGY_TRIGGERS.get(syn_name, [])
        max_tiers = len(triggers)
        tiers_hit = sum(1 for t in triggers if count >= t)
        if tiers_hit > 0:
            syn_reward += 0.075 * tiers_hit * (tiers_hit / max_tiers)
            active_count += 1
    syn_reward *= (1 + 0.1 * active_count)
    reward.add("synergy_depth", syn_reward)

    # --- Income signals ---
    interest = min(5, player.money // 10)  # Pre-income interest
    board_guard = player.board.board_count() >= player.level - 2

    if board_guard:
        reward.add("interest_bonus", interest * 0.06)
        if interest >= 5:
            reward.add("gold_standard", 0.30)

    # --- Gold excess penalty ---
    excess = max(0, player.money - 55)
    if excess > 0:
        raw = 0.015 * excess * (excess + 1) / 2
        reward.add("gold_excess", max(-raw, -10))

    # --- Gold pressure penalty ---
    if stage >= 5:
        _apply_gold_pressure(player, stage, reward)

    # --- Bench dead-weight ---
    _apply_bench_deadweight(player, stage, reward)

    # --- Unit quality ---
    _apply_unit_quality(player, stage, reward)


def _get_win_scaling(stage: int) -> float:
    if stage <= 5: return -0.75
    if stage <= 10: return -0.50
    if stage <= 15: return 0.0
    if stage <= 20: return 1.0
    return 2.0

def _get_loss_scaling(stage: int) -> float:
    if stage <= 5: return -0.50
    if stage <= 10: return -0.25
    if stage <= 15: return 0.0
    if stage <= 20: return 0.50
    return 1.0
```

### Step 9.4 — Placement Reward (game end)

```python
REWARD_PLACEMENT_TABLE = [28, 15, 8, -5, -9, -14, -19, -26]

def compute_placement_reward(rank: int) -> float:
    """Final reward based on placement. rank is 1-indexed."""
    return REWARD_PLACEMENT_TABLE[rank - 1]
```

### Deliverable
File: `training/py_eco/rewards.py` with `RewardTracker`, per-step and per-round reward functions.

### Validation
- Buy 3rd copy → +0.20 reward
- Buy then immediately sell → -1.0 penalty
- Win at stage 15 → +0.75 reward
- 50g with full board → +0.30 gold standard
- 1st place → +28 total
- Verify all reward signals match TypeScript names (for tensorboard comparison)

---

## Phase 10 — Gymnasium Env Wrapper

### Purpose
Wrap everything into a standard `gymnasium.Env` compatible with `SubprocVecEnv` and `MaskablePPO`.

### Step 10.1 — Environment Class

```python
class PacEcoEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int = None):
        super().__init__()
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(612,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(92)
        self.rng = np.random.default_rng(seed)
        self.game_state = None

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.game_state = GameState(self.rng)
        self.game_state.initialize()  # Create 8 players, roll initial shops

        obs = build_observation(self.game_state)
        info = {
            "action_masks": build_action_mask(
                self.game_state.rl_player, self.game_state
            ),
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        player = self.game_state.rl_player
        reward_tracker = RewardTracker()

        # Execute action
        action_info = execute_action(action, player, self.game_state, self.game_state.pool)

        # Per-step reward
        compute_step_reward(action, action_info, player, self.game_state, reward_tracker)

        # Check end-of-turn
        terminated = False
        truncated = False

        if action == END_TURN or player.actions_this_turn >= MAX_ACTIONS_PER_TURN:
            # --- Bot turns ---
            for bot in self.game_state.bots:
                bot.take_turn(self.game_state)

            # --- Fight phase (power estimator) ---
            self._resolve_fights(reward_tracker)

            # --- Post-fight: income, XP, shop ---
            self._advance_round()

            # --- Check game over ---
            if self.game_state.is_game_over():
                terminated = True
                rank = self.game_state.get_player_rank(player)
                reward_tracker.add("placement", compute_placement_reward(rank))

            # Reset turn state
            player.actions_this_turn = 0
            player.lock_used_this_turn = False
            player.consecutive_moves = 0
            player.last_move_target = -1
            player.last_buy_species = None

        obs = build_observation(self.game_state)
        reward = reward_tracker.total
        info = {
            "action_masks": build_action_mask(player, self.game_state),
            "rewardBreakdown": reward_tracker.breakdown,
            "stage": self.game_state.stage,
            "rank": self.game_state.get_player_rank(player),
            "life": player.life,
            "money": player.money,
        }

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """For MaskablePPO compatibility."""
        return build_action_mask(self.game_state.rl_player, self.game_state)
```

### Step 10.2 — Game State Manager

```python
class GameState:
    def __init__(self, rng):
        self.rng = rng
        self.stage = 0
        self.phase = "PICK"
        self.players: list[PlayerState] = []
        self.pool: SharedPool = None
        self.rl_player: PlayerState = None
        self.bots: list[BotAI] = []

    def initialize(self):
        # 1. Create shared pool
        self.pool = SharedPool(POKEMON_BY_RARITY)

        # 2. Create 8 players (1 RL + 7 bots)
        for i in range(8):
            player = PlayerState(player_id=i, is_bot=(i > 0))
            player.board = Board()
            player.shop = Shop()
            player.shop.reroll(player.level, self.pool, self.rng)
            self.players.append(player)

        self.rl_player = self.players[0]

        # 3. Create bot AIs
        strategies = ["econ", "aggro", "balanced", "econ", "balanced", "aggro", "balanced"]
        for i, player in enumerate(self.players[1:]):
            bot = BotAI(strategy=strategies[i])
            self.bots.append(bot)

    @property
    def players_alive(self) -> int:
        return sum(1 for p in self.players if p.alive)

    def is_game_over(self) -> bool:
        return self.players_alive <= 1 or not self.rl_player.alive
```

### Step 10.3 — Fight Resolution

```python
def _resolve_fights(self, reward_tracker):
    """Resolve all fights for the round using power estimator."""
    stage = self.game_state.stage
    is_pve = stage in PVE_STAGES

    # Simple matchmaking: random opponent for each player
    alive_players = [p for p in self.game_state.players if p.alive]

    if is_pve:
        # PvE: fight against fixed power level (scales with stage)
        pve_power = 50 + stage * 30  # Simple scaling
        for player in alive_players:
            player_power = team_power(player.board)
            result, _, enemy_killed = estimate_fight(
                player_power, pve_power, self.game_state.rng
            )
            self._apply_fight_result(player, result, enemy_killed, stage)
            if player == self.game_state.rl_player:
                compute_round_reward(player, result, enemy_killed,
                                     self.game_state, reward_tracker)
    else:
        # PvP: pair up players
        shuffled = list(alive_players)
        self.game_state.rng.shuffle(shuffled)

        for i in range(0, len(shuffled) - 1, 2):
            p1, p2 = shuffled[i], shuffled[i + 1]
            power1 = team_power(p1.board)
            power2 = team_power(p2.board)
            result, p1_killed, p2_killed = estimate_fight(
                power1, power2, self.game_state.rng
            )

            self._apply_fight_result(p1, result, p2_killed, stage)
            opp_result = {"WIN": "LOSS", "LOSS": "WIN", "DRAW": "DRAW"}[result]
            self._apply_fight_result(p2, opp_result, p1_killed, stage)

            if p1 == self.game_state.rl_player:
                compute_round_reward(p1, result, p2_killed,
                                     self.game_state, reward_tracker)
            elif p2 == self.game_state.rl_player:
                compute_round_reward(p2, opp_result, p1_killed,
                                     self.game_state, reward_tracker)

def _apply_fight_result(self, player, result, enemy_killed, stage):
    """Apply fight outcome to player state."""
    update_streak(player, result)

    if result == "LOSS":
        # Calculate damage
        damage = 2 + stage // 3  # Simplified damage formula
        player.life -= damage
        if player.life <= 0:
            player.life = 0
            player.alive = False
            # Return all units to pool
            for unit in player.board.all_units():
                copies = 3 ** (unit.stars - 1)
                base = get_base_form(unit.species)
                self.game_state.pool.return_unit(base, copies)

    player.history.append({"result": result})
```

### Step 10.4 — Round Advancement

```python
def _advance_round(self):
    """Advance to next round after fight."""
    is_pve = self.game_state.stage in PVE_STAGES

    # Give income to all alive players
    for player in self.game_state.players:
        if player.alive:
            apply_income(player, is_pve)

            # Roll new shop (unless locked)
            if not player.shop.locked:
                player.shop.reroll(player.level, self.game_state.pool, self.game_state.rng)
            else:
                player.shop.locked = False  # Unlock after one freeze round

    # Update ranks
    self._update_ranks()

    # Advance stage
    self.game_state.stage += 1
```

### Deliverable
File: `training/py_eco/env.py` with `PacEcoEnv`.

### Validation
- `env = PacEcoEnv(); obs, info = env.reset()` → obs.shape == (612,)
- `obs, reward, term, trunc, info = env.step(9)` → processes END_TURN
- Run 100 random-action episodes → no crashes, all terminate
- Average episode length: 25-40 stages (expect games to end by stage 30-35)
- `info["action_masks"]` always has at least one 1-bit

---

## Phase 11 — Integration & Validation

### Purpose
Verify the Python env produces realistic game dynamics and that observations transfer correctly.

### Step 11.1 — Smoke Tests

```python
def test_basic_game():
    """Full game plays out without errors."""
    env = PacEcoEnv(seed=42)
    obs, info = env.reset()
    total_reward = 0
    steps = 0

    while True:
        mask = info["action_masks"]
        valid_actions = np.where(mask == 1)[0]
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    assert steps > 50, "Game too short"
    assert steps < 2000, "Game too long"
    print(f"Game ended: {steps} steps, reward={total_reward:.2f}, rank={info['rank']}")
```

### Step 11.2 — Observation Consistency Test

```python
def test_obs_consistency():
    """Verify observation values are in valid ranges."""
    env = PacEcoEnv(seed=42)
    obs, _ = env.reset()

    for step_i in range(500):
        assert obs.shape == (612,)
        assert np.all(obs >= -0.01), f"Negative obs at step {step_i}: {obs.min()}"
        assert np.all(obs <= 1.01), f"Obs > 1 at step {step_i}: {obs.max()}"

        mask = env.action_masks()
        assert mask.sum() > 0, f"No valid actions at step {step_i}"

        action = np.random.choice(np.where(mask == 1)[0])
        obs, _, terminated, _, _ = env.step(action)

        if terminated:
            obs, _ = env.reset()
```

### Step 11.3 — Economy Sanity Checks

```python
def test_economy_flow():
    """Verify gold/XP/level progression is realistic."""
    env = PacEcoEnv(seed=42)
    obs, info = env.reset()

    gold_history = []
    level_history = []

    for _ in range(1000):
        action = END_TURN  # Do nothing, just collect income
        obs, _, terminated, _, info = env.step(action)

        gold_history.append(info["money"])
        level_history.append(env.game_state.rl_player.level)

        if terminated:
            break

    # Should reach 50g within 10 rounds (5 base income + interest growth)
    assert max(gold_history[:40]) >= 50, "Never reached 50g economy"
    # Should level up naturally from passive XP
    assert max(level_history) >= 3, "Never leveled past 3"
```

### Step 11.4 — Cross-Validation with TypeScript Server

This is the most important validation. Run the same sequence of actions through both the Python env and the TS server, compare observations:

```python
def test_cross_validation():
    """Compare Python obs vs TypeScript server obs for same action sequence."""
    # 1. Start TS server
    # 2. Reset both envs
    # 3. For each step:
    #    a. Take same action in both
    #    b. Compare obs vectors (allow ±0.01 tolerance for float rounding)
    #    c. Compare reward signals
    #    d. Compare action masks

    # This test requires the TS server running.
    # Document expected discrepancies:
    #   - Fight outcomes will differ (estimator vs simulation)
    #   - Bot behavior will differ (simplified vs full bot scenarios)
    #   - Item-related obs will be zeros in Python
    #
    # Focus comparison on:
    #   - Player stats (gold, level, life, streak)
    #   - Shop contents
    #   - Board/bench unit placement
    #   - Synergy counts
    #   - Action mask agreement for economy actions
```

### Step 11.5 — Performance Benchmark

```python
def benchmark_steps_per_second():
    """Measure training throughput."""
    env = PacEcoEnv(seed=42)
    obs, info = env.reset()

    import time
    start = time.time()
    steps = 0
    episodes = 0

    while time.time() - start < 10:  # Run for 10 seconds
        mask = info["action_masks"]
        action = np.random.choice(np.where(mask == 1)[0])
        obs, _, terminated, _, info = env.step(action)
        steps += 1
        if terminated:
            obs, info = env.reset()
            episodes += 1

    elapsed = time.time() - start
    print(f"Single env: {steps/elapsed:.0f} steps/sec, {episodes/elapsed:.1f} episodes/sec")

    # Target: >10,000 steps/sec per env (vs ~100 with TS server)
```

### Deliverable
File: `training/py_eco/tests.py` with all validation tests.

### Validation
- All smoke tests pass
- Steps/sec > 5,000 (single env, should be ~10,000+)
- Economy flow produces realistic gold/level curves
- Observation vector shape and ranges correct

---

## Phase 12 — Training Script

### Purpose
Wire the Python env into the existing `train_ppo.py` infrastructure.

### Step 12.1 — Env Factory

```python
# In train_ppo.py or new file training/train_eco.py

def make_eco_env(rank: int, seed: int):
    """Factory for SubprocVecEnv."""
    def _init():
        env = PacEcoEnv(seed=seed + rank)
        return env
    return _init

def create_eco_vec_env(num_envs: int, seed: int = 42):
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env_fns = [make_eco_env(i, seed) for i in range(num_envs)]
    return SubprocVecEnv(env_fns)
```

### Step 12.2 — Training Launch

```python
# training/train_eco.py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vec_env = create_eco_vec_env(args.num_envs, args.seed)

    if args.resume:
        model = MaskablePPO.load(args.resume, env=vec_env)
        model.learning_rate = args.lr
        model.lr_schedule = lambda _: args.lr
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=512,
            n_epochs=4,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./eco_tb_logs/",
        )

    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=True,
    )

    model.save("training/checkpoints/eco_latest.zip")
```

### Step 12.3 — Training Workflow

```
# Phase A: Economy pre-training (Python only, fast)
python training/train_eco.py --num-envs 16 --total-timesteps 10_000_000

# Phase B: Fine-tune on full game (TypeScript server, slow but accurate)
python training/train_ppo.py --resume training/checkpoints/eco_latest.zip --lr 8e-5

# Phase C: Evaluate
python training/run_parallel.py --num-envs 8 --evaluate training/checkpoints/eco_latest.zip --eval-games 50
```

### Step 12.4 — Transfer Checkpoint Checklist

Before transferring an economy-trained checkpoint to the full TS server:

1. **Observation compatibility**: Verify `obs_space.shape == (612,)` matches
2. **Action compatibility**: Verify `action_space.n == 92` matches
3. **Policy architecture**: Must use same `MlpPolicy` config (layer sizes, activation)
4. **Learning rate**: Drop to 8e-5 for fine-tuning (economy-only training may overfit econ signals)
5. **Expected behavior on transfer**:
   - Economy decisions (buy/sell/level/reroll) should transfer well
   - Board positioning may be random (no fight feedback in economy trainer)
   - Fight-dependent rewards (synergy depth, kill bonuses) will recalibrate quickly

### Deliverable
File: `training/train_eco.py`.

---

## File Summary

| File | Purpose |
|------|---------|
| `training/py_eco/__init__.py` | Package init |
| `training/py_eco/data.py` | Static data: Pokemon, synergies, constants |
| `training/py_eco/board.py` | Unit, Board, synergy calc, evolution |
| `training/py_eco/shop.py` | SharedPool, Shop, roll mechanics |
| `training/py_eco/economy.py` | PlayerState, income, buy/sell, level-up |
| `training/py_eco/power.py` | Fight power estimator |
| `training/py_eco/bots.py` | Bot AI opponents |
| `training/py_eco/observation.py` | 612-float observation builder |
| `training/py_eco/actions.py` | Action dispatcher + mask builder |
| `training/py_eco/rewards.py` | Reward calculator (all signals) |
| `training/py_eco/env.py` | Gymnasium env wrapper |
| `training/py_eco/tests.py` | Validation suite |
| `training/train_eco.py` | Training script entry point |

---

## Implementation Order & Dependencies

```
Phase 1 (data.py)           ← No dependencies, do first
    ↓
Phase 2 (board.py)          ← Depends on data.py
    ↓
Phase 3 (shop.py)           ← Depends on data.py
    ↓
Phase 4 (economy.py)        ← Depends on board.py, shop.py
    ↓
Phase 5 (power.py)          ← Depends on board.py
    ↓
Phase 6 (bots.py)           ← Depends on economy.py, power.py
    ↓
Phase 7 (observation.py)    ← Depends on board.py, economy.py
    ↓
Phase 8 (actions.py)        ← Depends on economy.py, shop.py
    ↓
Phase 9 (rewards.py)        ← Depends on economy.py, board.py
    ↓
Phase 10 (env.py)           ← Depends on ALL above
    ↓
Phase 11 (tests.py)         ← Depends on env.py
    ↓
Phase 12 (train_eco.py)     ← Depends on env.py
```

Phases 1-3 can be done in parallel. Phases 5-6 can be done in parallel. Phase 7-9 can be done in parallel once Phase 4 is done.

---

## Key Design Decisions to Make

1. **Power estimator fidelity**: How complex should fight estimation be? Start simple (Step 5.1), tune after seeing rank distributions. You can always make it more complex later.

2. **Bot sophistication**: Simple rule-based bots are fine for economy training. The agent needs opponents that create realistic game states (someone to lose to, someone to beat), not optimal play.

3. **Items**: Stub initially (all item obs = 0, COMBINE actions always masked). Add items in a later phase. The agent can learn economy without items and fine-tune item usage on the TS server.

4. **Propositions**: Stage 0/10/20 portal picks are important for unique/legendary acquisition. Implement as simple random offers initially, or skip entirely and let the agent learn picks during TS fine-tuning.

5. **Damage formula**: The simplified `2 + stage // 3` damage on loss is a placeholder. Tune it so games end at realistic stages (25-35). If games end too fast, reduce damage. If too slow, increase.

6. **Species index normalization**: Must match the TS server EXACTLY. Extract the sorted species list from the TS code and hardcode it in `data.py`. Any mismatch here will corrupt the transferred policy.

---

## Tunable Constants Summary

| Constant | Location | Default | Notes |
|----------|----------|---------|-------|
| `STAR_MULT` | power.py | {1:1, 2:3, 3:9} | Star power scaling |
| `SYN_BONUS_PER_TIER` | power.py | 0.05 | Synergy power bonus |
| `FIGHT_NOISE_STD` | power.py | 0.05 | Fight outcome randomness |
| `PVE_POWER_BASE` | env.py | 50 | PvE difficulty base |
| `PVE_POWER_PER_STAGE` | env.py | 30 | PvE difficulty scaling |
| `LOSS_DAMAGE_BASE` | env.py | 2 | Base HP damage on loss |
| `LOSS_DAMAGE_STAGE_DIV` | env.py | 3 | Stage damage scaling |
| `BOT_LEVEL_SCHEDULE` | bots.py | {3:3, 5:4, ...} | Bot leveling targets |
| All reward constants | rewards.py | (match TS) | Must match training-config.ts |
