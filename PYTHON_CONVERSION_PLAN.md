# Plan: Convert Game Engine to Pure Python for Training

## Why This Makes Training Faster

**Current architecture:** Python (RL) --HTTP--> Node.js (game engine) --HTTP--> Python
**Target architecture:** Python (RL) --direct function call--> Python (game engine)

| Bottleneck | Current | After Conversion |
|------------|---------|-----------------|
| Per-step latency | ~10-50ms (HTTP round-trip + JSON serialize/deserialize) | ~0.01-0.1ms (direct function call) |
| Per-game overhead | ~500-2500 actions × 10-50ms = 5-125s per game | ~500-2500 actions × 0.1ms = 0.05-0.25s per game |
| Parallelism | Need N Node.js processes (each ~100MB RAM) | Single process with vectorized NumPy, or lightweight multiprocessing |
| Serialization | 612-float obs + 92-int mask serialized to JSON every step | Zero-copy NumPy arrays |
| Process management | Cross-platform subprocess spawning, port cleanup, health checks, server rotation every 200 episodes | None |

**Conservative estimate: 100-500x faster step throughput**, which translates to training runs completing in minutes/hours instead of hours/days.

---

## Scope Assessment

### What exists today

| Layer | Location | Lines | Role |
|-------|----------|-------|------|
| Python training pipeline | `training/` | ~2,500 | RL loop, env wrappers, metrics — **kept as-is** |
| TypeScript training env | `app/training/` | ~4,100 | Observation building, reward shaping, action masking |
| TypeScript game engine | `app/core/` | ~27,000 | Combat simulation, abilities, effects, board, shop |
| TypeScript data models | `app/models/`, `app/types/`, `app/config/` | ~15,000 | Pokemon stats, items, synergies, enums, configs |
| Static data | `app/models/precomputed/pokemons-data.csv` | 1,131 rows | Base stats, types, abilities per Pokemon |

### What needs converting (game logic only — not UI, networking, or client)

| Subsystem | TS Lines | Complexity | Notes |
|-----------|----------|------------|-------|
| Abilities (`abilities.ts`) | 16,571 | Very High | 530 unique abilities, conditional chains, status effects |
| Combat simulation (`simulation.ts`, state files) | ~3,400 | Very High | State machine, pathfinding, damage calc, mana, knockback |
| Effects system (`effects/`) | 3,459 | High | Items (1,273), passives (1,390), synergies (427), base (369) |
| Pokemon entity model | 1,836 | High | 100+ properties, state transitions |
| Training env (obs/reward/mask) | 3,036 | Medium | Observation vector, reward shaping, action dispatch |
| Shop/economy | 779 | Medium | Rarity pools, pricing, Ditto/Eevee rates |
| Board logic | 666 | Medium | 8x4 grid, distance, pathfinding |
| Evolution rules | 408 | Medium | Count-based, item-based, divergent |
| Bot logic | ~415 | Low | Decision trees for opponent bots |
| Matchmaking | 182 | Low | Opponent selection |
| Data enums & configs | ~5,500 | Low | Can be auto-generated from CSV + enums |
| Training config | 414 | Low | Constants, reward tables |

**Total game logic to port: ~36,000 lines of TypeScript**

---

## Phased Conversion Plan

### Phase 0: Foundation & Data Layer
**Goal:** Python package structure + all static game data accessible in Python.

| Step | What | Details | Depends On |
|------|------|---------|------------|
| 0.1 | Create package structure | `pyengine/` package with submodules: `data/`, `core/`, `effects/`, `training/` | — |
| 0.2 | Auto-generate Pokemon enums | Script to parse `Pokemon.ts` → Python enum (1,167 entries) | — |
| 0.3 | Auto-generate Ability, Passive, Item, Synergy enums | Parse TS enums → Python enums (530 + 188 + 200 + 32 entries) | — |
| 0.4 | Import Pokemon CSV data | Load `pokemons-data.csv` → dataclass registry: `PokemonData(name, hp, atk, def, spd, range, types[], ability, passive, rarity, evolution, stages)` | 0.2, 0.3 |
| 0.5 | Port config constants | Reward tables, shop rates, pool sizes, rarity costs, stage thresholds from `training-config.ts`, `shop.ts`, `pools.ts` | — |
| 0.6 | Port type system | Synergy breakpoints, type effectiveness (if any), weather enums, status enums | 0.3 |

**Validation:** Unit tests that every Pokemon from CSV loads correctly, has valid types/ability/rarity.

---

### Phase 1: Board & Entity Model
**Goal:** Pokemon entities can exist on a board with stats, items, and positions.

| Step | What | Details | Depends On |
|------|------|---------|------------|
| 1.1 | Board class | 8x4 grid (24 board + 8 bench cells), cell occupancy, distance/orientation helpers | 0.1 |
| 1.2 | PokemonEntity class | All combat-relevant properties: hp, atk, def, spdef, speed, range, mana, crit, items[], types[], status effects, position, team | 0.4 |
| 1.3 | Player class | life, gold, level, exp, streak, shop[], board (Board), bench, synergies{}, rank, alive flag | 1.1, 1.2 |
| 1.4 | Item model | Item enum + stat modifiers + recipe pairs → crafted item mapping | 0.3 |
| 1.5 | Synergy calculator | Count types on board → active synergies with breakpoint tiers (2/4/6/8) | 1.2, 0.6 |

**Validation:** Create a Player with 6 units on board, verify synergy counts match expected breakpoints.

---

### Phase 2: Shop & Economy
**Goal:** Full economy loop — buy, sell, refresh, level up, interest, evolution.

| Step | What | Details | Depends On |
|------|------|---------|------------|
| 2.1 | Shop generation | Rarity-weighted random pool per player level, stage-dependent probabilities | 0.5, 1.3 |
| 2.2 | Buy/sell mechanics | Buy from shop → bench, sell from bench/board → gold, pricing formulas | 1.3, 2.1 |
| 2.3 | Evolution system | CountEvolutionRule (3 copies → star up), ItemEvolutionRule, stat scaling on evolve | 1.2, 0.4 |
| 2.4 | Level up & XP | Exp costs per level, max team size scaling, natural XP per round | 1.3 |
| 2.5 | Interest & income | Base income + streak bonus + interest (gold/10, cap 5) | 1.3 |
| 2.6 | Special shop mechanics | Ditto rate, Eevee branching, shop lock, free rerolls | 2.1 |

**Validation:** Simulate 30 rounds of economy-only (no combat), verify gold/level/board trajectories match TS server output for same RNG seed.

---

### Phase 3: Combat Simulation (Core)
**Goal:** Two teams fight to resolution. This is the hardest phase.

| Step | What | Details | Depends On |
|------|------|---------|------------|
| 3.1 | State machine | Idle → Moving → Attacking → Dead transitions per entity | 1.2 |
| 3.2 | Target selection | Find nearest enemy, range check, retarget on kill | 3.1, 1.1 |
| 3.3 | Movement/pathfinding | Grid-based movement toward target, collision avoidance | 1.1, 3.2 |
| 3.4 | Basic attack loop | ATK damage formula: `atk × (100 - def) / 100`, apply to HP, mana generation on hit | 3.1 |
| 3.5 | Damage modifiers | Crit chance/power, STAB bonus, type interactions | 3.4 |
| 3.6 | Status effects | Burn (-ATK), Poison (DOT), Freeze (skip turn), Paralyze (speed), Confusion (self-hit), Silence (no ability) | 3.4 |
| 3.7 | Mana & ability trigger | Mana fills on attack/receive, trigger ability at full mana, ability costs | 3.4 |
| 3.8 | Simulation loop | Tick all entities per 50ms step, detect fight end (one team eliminated), max 2000 steps | 3.1-3.7 |
| 3.9 | Fight result | Winner determination, HP damage to loser, kill counts | 3.8 |

**Validation:** Run 100 fights with known board states, compare win rates and average fight duration to TS server. Exact reproduction not required (RNG), but statistical distribution should match.

---

### Phase 4: Effects System
**Goal:** Items, passives, and synergies modify combat through event hooks.

| Step | What | Details | Depends On |
|------|------|---------|------------|
| 4.1 | Effect architecture | Event hook system: OnAttack, OnHit, OnDamageDealt, OnDamageReceived, OnDeath, OnKill, OnSpawn, OnSimStart | 3.8 |
| 4.2 | Synergy effects | 32 synergy bonuses at breakpoint tiers (stat buffs, damage multipliers, special mechanics) | 4.1, 1.5 |
| 4.3 | Item effects — stat items | ~30 simple stat-boosting items (Muscle Band, Assault Vest, etc.) | 4.1, 1.4 |
| 4.4 | Item effects — complex items | ~40 items with triggered effects (Scope Lens crit, Smoke Ball evasion, Amulet Coin gold) | 4.1 |
| 4.5 | Passive effects — simple | ~100 passives that are stat mods or simple triggers | 4.1 |
| 4.6 | Passive effects — complex | ~88 passives with unique mechanics (form changes, weather, terrain, stacking) | 4.1 |

**Validation:** Unit tests for each synergy tier, each item effect, and the most impactful passives. Cross-reference with TS implementation for correctness.

---

### Phase 5: Abilities
**Goal:** All 530 active abilities implemented. This is the largest single subsystem.

This phase is best tackled in priority order — implement abilities used by common/uncommon Pokemon first (the ones that appear most in training games), then work outward to rarer ones.

| Step | What | Details | Depends On |
|------|------|---------|------------|
| 5.1 | Ability framework | Base class with `cast(caster, target, board)`, mana cost, cooldown, AoE helpers | 3.7, 4.1 |
| 5.2 | Direct damage abilities | ~80 abilities: Fire Blast, Hydro Pump, Thunder, etc. (damage + optional status) | 5.1 |
| 5.3 | Buff/debuff abilities | ~60 abilities: Dragon Dance, Nasty Plot, Charm, Torment, etc. | 5.1 |
| 5.4 | AoE/positional abilities | ~50 abilities: Earthquake, Discharge, Surf, etc. (area targeting) | 5.1, 1.1 |
| 5.5 | Healing/shield abilities | ~40 abilities: Wish, Recover, Protect, Light Screen, etc. | 5.1 |
| 5.6 | Utility/special abilities | ~50 abilities: Teleport, Substitute, Transform, Trick Room, etc. | 5.1 |
| 5.7 | Summoning abilities | ~20 abilities that spawn additional units mid-fight | 5.1, 1.2 |
| 5.8 | Legendary/unique abilities | ~50 complex abilities for legendary Pokemon (Roar of Time, Spacial Rend, etc.) | 5.1-5.6 |
| 5.9 | Remaining abilities | Mop up anything missed, edge cases, Hidden Power variants | 5.1-5.8 |

**Pragmatic shortcut:** For initial training viability, abilities can be implemented as simplified versions (correct damage/status, approximate targeting) and refined iteratively. The agent won't notice subtle ability differences until it's already quite skilled.

**Validation:** Per-ability unit tests for the top 50 most-used abilities. Statistical combat comparison for full games.

---

### Phase 6: Game Loop & Training Integration
**Goal:** Full game runs in Python, produces observations/rewards/masks compatible with existing training pipeline.

| Step | What | Details | Depends On |
|------|------|---------|------------|
| 6.1 | Game state manager | 8 players, round progression (PvE → PvP), stage advancement, player elimination | 1.3, 3.9 |
| 6.2 | Matchmaking | Opponent selection algorithm (avoid repeats, ghost boards for eliminated players) | 6.1 |
| 6.3 | PvE rounds | Predefined PvE encounters at stages 1, 2, 3, 10, 15, 20 (neutral creeps) | 6.1, 3.8 |
| 6.4 | Bot AI | Decision-tree bot logic for 7 opponent bots (buy/sell/position heuristics) | 6.1, 2.1-2.6 |
| 6.5 | Observation builder | 612-float vector matching current TS format exactly | 6.1, 1.3 |
| 6.6 | Action dispatcher | 92-action dispatch (buy/sell/move/refresh/level/lock/end/pick/combine) | 6.1, 2.1-2.6 |
| 6.7 | Action mask builder | Valid action mask per state, matching TS logic exactly | 6.6 |
| 6.8 | Reward calculator | Placement table + per-round shaped rewards, matching `training-config.ts` | 6.1 |
| 6.9 | Gymnasium env wrapper | `PythonGameEnv(gymnasium.Env)` — drop-in replacement for `PokemonAutoChessEnv` | 6.5-6.8 |
| 6.10 | Multi-agent wrapper | VecEnv wrapper for N RL agents (replaces `SelfPlayVecEnv`) | 6.9 |

**Validation:** Run existing `smoke_test.py` against Python env (swap HTTP calls for direct calls). Run 50-game eval with existing checkpoint — mean rank should be within 0.5 of TS server result.

---

### Phase 7: Optimization & Parity Testing
**Goal:** Performance tuning and full behavioral parity with TS engine.

| Step | What | Details | Depends On |
|------|------|---------|------------|
| 7.1 | Profiling | Profile full game in Python, identify hot paths | 6.9 |
| 7.2 | NumPy vectorization | Vectorize observation building, damage calculations, board queries | 7.1 |
| 7.3 | Optional: Cython/Numba | JIT-compile combat simulation inner loop if needed | 7.1 |
| 7.4 | Parity test suite | Run 1000 games on both TS and Python engines, compare: mean rank distribution, game length distribution, economy curves, synergy activation rates | 6.9 |
| 7.5 | Checkpoint compatibility | Verify existing MaskablePPO checkpoints work with Python env (same obs/action space) | 6.9 |
| 7.6 | Regression CI | Automated test that Python env produces valid obs/mask/reward for 100 random games | 7.4 |

---

## Summary: Phase Effort Estimates

| Phase | Description | TS Lines Covered | Relative Size |
|-------|-------------|-----------------|---------------|
| **0** | Foundation & data layer | ~5,500 (enums, CSV, configs) | Small — largely auto-generated |
| **1** | Board & entity model | ~2,500 (board, entity, player) | Small-Medium |
| **2** | Shop & economy | ~1,200 (shop, evolution, income) | Medium |
| **3** | Combat simulation (core) | ~3,400 (sim, states, damage) | Large — high complexity |
| **4** | Effects system | ~3,500 (items, passives, synergies) | Large — many individual effects |
| **5** | Abilities | ~16,500 (all 530 abilities) | Very Large — biggest single phase |
| **6** | Game loop & training integration | ~7,000 (game state, obs, rewards, env) | Large — integration complexity |
| **7** | Optimization & parity | N/A | Medium — profiling & tuning |

---

## Suggested Approach: MVP First

Rather than porting all 530 abilities and 200 items before training, a **Minimum Viable Engine** approach gets you training faster:

### MVP Scope (Phases 0-3 + partial 4-6)
- Full economy loop (buy/sell/level/evolve)
- Combat with basic attacks only (no abilities, no item effects, no passives)
- Synergy stat bonuses only (no triggered effects)
- Observation/reward/mask matching existing format
- Bot AI (simplified)

This MVP lets you **start training immediately** with the speed benefit. The agent learns economy, positioning, team composition, and basic combat. Abilities and effects are layered in incrementally.

### Incremental Ability Porting
After MVP, add abilities in tiers:
1. **Tier 1:** Top 30 most-common abilities (covers Common/Uncommon Pokemon)
2. **Tier 2:** Next 50 (Rare/Epic Pokemon)
3. **Tier 3:** Next 50 (Ultra/Unique)
4. **Tier 4:** Remaining 400 (Legendary, niche, variants)

Each tier improves combat fidelity without blocking training progress.

---

## Risk & Tradeoffs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Behavioral drift from TS engine | Agent trained in Python may not transfer to TS game | Parity test suite (Phase 7.4), same obs/action space |
| 530 abilities is a massive porting effort | Delays full completion | MVP approach — train with basic combat first |
| Combat RNG differences | Fight outcomes differ between engines | Use same PRNG seeding approach, validate distributions not exact outcomes |
| Python combat may be slower than expected | Negates some speed benefit | NumPy vectorization (7.2), Cython fallback (7.3) |
| Existing checkpoints may behave differently | Wasted prior training | Checkpoint compatibility testing (7.5) before retraining |

---

## Files Modified

- **Created:** `PYTHON_CONVERSION_PLAN.md` — this document

## Design Decisions

- **MVP-first approach** recommended over full port — gets training speed benefit sooner
- **Auto-generation** for enums/data from existing CSV and TS sources — reduces manual error
- **Same obs/action/reward format** — existing checkpoints and training scripts work unchanged
- **No TS server changes** — the TS server remains available as ground-truth reference
