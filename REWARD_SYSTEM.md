# Reward & Punishment System — Complete Reference

> **Purpose**: Full inventory of every reward signal the RL agent receives, how they interact, their relative magnitudes, and where to find them in code. Written for an overhaul.

**Source files**:
- Constants: `app/training/training-config.ts` (lines 102-329)
- Calculations: `app/training/training-env.ts` (step() lines 418-698, runFightPhase() lines 1388-1723)

---

## Table of Contents

1. [Signal Timing Overview](#1-signal-timing-overview)
2. [Placement Reward (Game End)](#2-placement-reward)
3. [Battle Result (Per-Round)](#3-battle-result)
4. [Economy Rewards (Per-Round)](#4-economy-rewards)
5. [Synergy & Board Composition (Per-Round)](#5-synergy--board-composition)
6. [Purchase Rewards (Per-Action)](#6-purchase-rewards)
7. [Action Penalties (Per-Action)](#7-action-penalties)
8. [Gold Management Penalties (Per-Round)](#8-gold-management-penalties)
9. [Board Quality Penalties (Per-Round)](#9-board-quality-penalties)
10. [Turn Efficiency (Per-Turn)](#10-turn-efficiency)
11. [Disabled / Removed Signals](#11-disabled--removed-signals)
12. [Full Magnitude Table](#12-full-magnitude-table)
13. [Per-Game Accumulation Estimates](#13-per-game-accumulation-estimates)
14. [Ratio Analysis & Interactions](#14-ratio-analysis--interactions)
15. [Known Issues & Overhaul Targets](#15-known-issues--overhaul-targets)

---

## 1. Signal Timing Overview

Rewards fire at three different cadences. Understanding which cadence a signal uses is critical because it determines how many times it accumulates per game.

| Cadence | Fires when | ~Occurrences/game | Signals |
|---------|-----------|-------------------|---------|
| **Per-action** | Every step in PICK phase | ~150-250 actions | Buy dup/evo, move fidget, sell penalties, level-up, reroll eco |
| **Per-round** (fight) | Once after each combat | ~25-30 rounds | Battle result, synergy, kills, keep unique/legend, HP preservation, gold excess, gold pressure, bench dead-weight, unit quality, interest, gold standard |
| **Per-game** | Once at game end | 1 | Placement table |

**Key insight**: A per-round signal of 0.30 fires ~27 times = ~8.1/game. A per-action signal of 0.08 might fire ~10 times = ~0.8/game. The cadence matters more than the raw constant.

---

## 2. Placement Reward

**When**: Game end (player eliminated or game finishes)
**Config**: `REWARD_PLACEMENT_TABLE` (training-config.ts:130-139)
**Code**: `computeFinalReward()` (training-env.ts:2551-2555)

```
Rank  Reward   Delta from next
1st   +28.0    +13.0 over 2nd
2nd   +15.0    +7.0 over 3rd
3rd    +8.0    +13.0 over 4th
4th    -5.0    -4.0 from 5th
5th    -9.0    -5.0 from 6th
6th   -14.0    -5.0 from 7th
7th   -19.0    -7.0 from 8th
8th   -26.0
```

**Total spread**: 54 points (28 to -26).
**Break-even**: Between 3rd (+8) and 4th (-5). Only top-3 is positive.
**Design**: This is the dominant signal. All shaped rewards combined typically sum to +5 to +15 per game for a competent agent, so placement dwarfs everything.

### Delta analysis

The 1st-to-2nd gap (+13) is almost as large as the 3rd-to-4th gap (+13). This means the agent is incentivized to push for 1st rather than coast at 2nd/3rd — but the diminishing returns between 4th-8th mean the agent doesn't differentiate much between "losing badly" and "losing slightly." If you want the agent to fight harder to avoid 7th vs 8th, the bottom spread needs widening.

---

## 3. Battle Result

**When**: Per-round, after each fight
**Config**: `REWARD_PER_WIN`, `REWARD_PER_LOSS`, `REWARD_PER_DRAW` + stage scaling tables (training-config.ts:103-125)
**Code**: `runFightPhase()` (training-env.ts:1388-1427)

### Base values

| Outcome | Base |
|---------|------|
| Win     | +0.75 |
| Loss    | -0.50 |
| Draw    | 0.00  |

### Stage scaling

Formula: `base × (1 + scaling)` where scaling is a lookup:

| Stage range | Win multiplier | Effective win | Loss multiplier | Effective loss |
|-------------|---------------|---------------|-----------------|----------------|
| 1-5 (PVE)  | ×0.25         | +0.19         | ×0.50           | -0.25          |
| 6-10        | ×0.50         | +0.375        | ×0.75           | -0.375         |
| 11-15       | ×1.00         | +0.75         | ×1.00           | -0.50          |
| 16-20       | ×2.00         | +1.50         | ×1.50           | -0.75          |
| 21+         | ×3.00         | +2.25         | ×2.00           | -1.00          |

### Per-game accumulation

Assuming agent wins 50% of ~27 rounds with average stage ~14:
- Wins: ~13 × 0.75 avg = ~9.75
- Losses: ~13 × -0.50 avg = ~-6.50
- **Net**: ~+3.25 for an average player

A strong agent winning 65% late-game:
- Late wins: 8 × 2.25 = +18.0
- Late losses: 4 × -1.00 = -4.0
- Early: ~+1.0
- **Net**: ~+15.0

**Asymmetry**: Win rewards scale faster than loss penalties (×3.0 vs ×2.0 at stage 21+). This means winning late matters much more than losing early.

---

## 4. Economy Rewards

### 4.1 Interest Bonus

**When**: Per-round, during income phase
**Config**: `REWARD_INTEREST_BONUS = 0.06` (training-config.ts:142)
**Code**: training-env.ts:1678-1687
**Guard**: Board size >= maxTeamSize - 2 (must field a near-full team)

| Gold held | Interest | Reward/round | Over 15 rounds |
|-----------|----------|-------------|----------------|
| 0-9       | 0        | 0           | 0              |
| 10-19     | 1        | 0.06        | 0.90           |
| 20-29     | 2        | 0.12        | 1.80           |
| 30-39     | 3        | 0.18        | 2.70           |
| 40-49     | 4        | 0.24        | 3.60           |
| 50+       | 5        | 0.30        | 4.50           |

**Board guard**: If board is more than 2 slots below max team size, interest reward is zeroed. Prevents bench-stacking to farm interest.

### 4.2 Gold Standard

**When**: Per-round, during income phase (stacks with interest)
**Config**: `REWARD_GOLD_STANDARD = 0.30` (training-config.ts:147)
**Code**: training-env.ts:1689-1697
**Condition**: `player.interest >= 5` (had >= 50g before income)

This is a flat +0.30/round **on top of** the interest bonus. At 50g the combined signal is:
- Interest: 5 × 0.06 = 0.30
- Gold standard: 0.30
- **Total: 0.60/round**

Over 15 rounds at 50g = **+9.0 total** from economy alone.

### Economy reward interaction

The 50g breakpoint creates a sharp cliff:
- 49g → 0.24/round (interest only)
- 50g → 0.60/round (interest + gold standard)

That's a **+150% jump** for 1 gold. This makes 50g the strongest attractor in the reward landscape. Every gold spent below 50g costs 0.36/round in lost signal.

---

## 5. Synergy & Board Composition

### 5.1 Depth-Based Synergy Reward

**When**: Per-round, after each fight
**Config**: `SYNERGY_DEPTH_BASE = 0.075`, `SYNERGY_ACTIVE_COUNT_BONUS = 0.1` (training-config.ts:158-161)
**Code**: training-env.ts:1434-1461

**Formula per synergy**:
```
per_synergy = 0.075 × tier_hit × (tier_hit / max_tiers)
```
Where `tier_hit` = number of breakpoints reached, `max_tiers` = total breakpoints for that synergy.

**Final total**:
```
total = sum_all_synergies × (1 + 0.1 × active_synergy_count)
```

**Worked examples**:

| Synergy | Breakpoints | Units fielded | tier_hit | max_tiers | Per-synergy reward |
|---------|-------------|--------------|----------|-----------|-------------------|
| Water (3/6/9) | 3 | 3 units | 1 | 3 | 0.075 × 1 × (1/3) = 0.025 |
| Water (3/6/9) | 3 | 6 units | 2 | 3 | 0.075 × 2 × (2/3) = 0.100 |
| Water (3/6/9) | 3 | 9 units | 3 | 3 | 0.075 × 3 × (3/3) = 0.225 |
| Fire (2/4/6) | 3 | 4 units | 2 | 3 | 0.075 × 2 × (2/3) = 0.100 |
| Psychic (2/4) | 2 | 4 units | 2 | 2 | 0.075 × 2 × (2/2) = 0.150 |

**Active count multiplier**: With 5 active synergies, the multiplier is `1 + 0.1 × 5 = 1.5` (50% bonus).

**Typical range**: A mid-game board with 3-4 active synergies at tier 1-2 yields ~0.3-0.6/round. A late-game board with 5-6 active synergies at higher tiers yields ~0.8-1.5/round.

**Over 27 rounds**: ~8-25 total (highly variable).

### 5.2 Keep Unique/Legendary Bonus

**When**: Per-round, after each fight
**Config**: `REWARD_KEEP_UNIQUE = 0.04`, `REWARD_KEEP_LEGENDARY = 0.04` (training-config.ts:316-317)
**Code**: training-env.ts:1474-1490
**Condition**: Unit must be on board (positionY > 0), not bench

Flat +0.04 per unique or legendary unit on board per round. With 2 such units over 15 late rounds = +1.2 total. Minor signal.

### 5.3 Enemy Kill Reward

**When**: Per-round, after each fight
**Config**: `REWARD_PER_ENEMY_KILL = 0.02` (training-config.ts:148)
**Code**: training-env.ts:1463-1471

Killing 5 enemy units in a fight = +0.10. Over 27 rounds with ~4 kills/round = ~2.16 total. Small but consistent signal that rewards aggressive board composition.

### 5.4 HP Preservation Bonus

**When**: Per-round, only on wins
**Config**: `REWARD_HP_SCALE = 0.005` (training-config.ts:149)
**Code**: training-env.ts:1492-1501

Formula: `(player.life / 100) × 0.005`

At 80 HP: +0.004 per win. Over 13 wins = +0.05. **Negligibly small** — functionally a dead signal. If you want HP preservation to matter, this needs to be 10-50x larger.

---

## 6. Purchase Rewards

All fire per-action during PICK phase.

### 6.1 Buy Duplicate (2nd copy of species)

**Config**: `REWARD_BUY_DUPLICATE = 0.08`, `REWARD_BUY_DUPLICATE_LATEGAME = 0.12` (training-config.ts:320,323)
**Code**: training-env.ts:471-473
**Late game**: Stage > 20

Encourages collecting pairs. Typical game might have 5-10 duplicate purchases = +0.40-0.80 early, +0.60-1.20 late.

### 6.2 Buy Evolution (3rd copy, triggers star-up)

**Config**: `REWARD_BUY_EVOLUTION = 0.20`, `REWARD_BUY_EVOLUTION_LATEGAME = 0.30` (training-config.ts:321,324)
**Code**: training-env.ts:459-461
**Late game**: Stage > 20

Strongest per-action positive signal. Typical 3-5 evolutions per game = +0.60-1.50.

### 6.3 Evolution from Reroll (star-up immediately after refreshing shop)

**Config**: `REWARD_EVO_FROM_REROLL` rarity table (training-config.ts:300-310)
**Code**: training-env.ts:463-469
**Condition**: Previous action was REFRESH **and** current BUY triggers evolution

| Rarity    | Bonus |
|-----------|-------|
| Common    | +0.50 |
| Uncommon  | +0.75 |
| Rare      | +1.00 |
| Epic      | +1.50 |
| Ultra     | +2.00 |
| Unique    | +2.00 |
| Legendary | +2.00 |
| Hatch     | +0.75 |
| Special   | +1.00 |

**Additive** on top of the base evolution reward. A rare evo-from-reroll = 0.20 (base) + 1.00 (bonus) = **+1.20 total**.

This is a very strong signal — possibly too strong. A single lucky reroll can equal 4 rounds of gold standard income. Maybe 1-2 of these per game = +1.0-4.0 total, but highly variance-dependent.

---

## 7. Action Penalties

### 7.1 Move Fidget

**Config**: `MOVE_FIDGET_GRACE = 2`, `REWARD_MOVE_FIDGET = -0.08` (training-config.ts:235-236)
**Code**: training-env.ts:476-484

First 2 consecutive MOVE actions are free. Each additional consecutive MOVE costs -0.08. Resets when any non-MOVE action is taken.

Prevents oscillation loops (moving a unit back and forth). Typical impact: -0.16 to -0.48/game if the agent fidgets.

### 7.2 Sell Evolved Unit

**Config**: `REWARD_SELL_EVOLVED = -0.15` (training-config.ts:239)
**Code**: training-env.ts:494-499
**Condition**: Sold unit had 2+ stars

Selling a 2-star unit you invested in is almost always wrong. -0.15 per occurrence. Should be rare in a well-trained agent.

### 7.3 Buy-Then-Immediately-Sell

**Config**: `REWARD_BUY_THEN_SELL = -1.0` (training-config.ts:243)
**Code**: training-env.ts:502-507
**Condition**: SELL is the very next action after BUY, and the sold unit's species index matches the bought unit

Heaviest per-action penalty. Pure gold waste — the agent should use REMOVE_SHOP to clear unwanted shop slots. -1.0 is comparable to a late-game battle loss.

### 7.4 Level-Up Penalty

**Config**: `REWARD_LEVEL_UP_PENALTY = -0.15`, `REWARD_LEVEL_UP_STAGE9_TO5 = +0.10`, `LEVEL_UP_RICH_THRESHOLD = 50` (training-config.ts:262-264)
**Code**: training-env.ts:525-543

| Condition | Reward |
|-----------|--------|
| Level to 5 at stage 9 (PVE Gyarados) | **+0.10** |
| Gold >= 50 after leveling | **0.00** (neutral) |
| Gold < 50 after leveling (all other cases) | **-0.15** |

**Interaction with economy signals**: Breaking from 50g to 46g loses:
- Gold standard: -0.30/round
- Interest tier drop (5→4): -0.06/round
- Level-up penalty: -0.15 (one-time)
- **Total cost**: -0.15 immediate + -0.36/round ongoing

This makes premature leveling very expensive. Over 5 rounds the ongoing cost alone is -1.80.

### 7.5 Reroll Below Economy

**Config**: `REROLL_ECO_PENALTY_DIVISOR = 2`, `REROLL_ECO_PENALTY_FLOOR = -0.06` (training-config.ts:283-284)
**Code**: training-env.ts:557-581
**Condition**: Gold < 50 AND gold pressure tier = SAFE

Formula: `penalty = -(current_interest × REWARD_INTEREST_BONUS) / 2`

| Gold before reroll | Interest | Penalty |
|-------------------|----------|---------|
| 0-9               | 0        | -0.06 (floor) |
| 10-19             | 1        | -0.06 (floor) |
| 20-29             | 2        | -0.06 |
| 30-39             | 3        | -0.09 |
| 40-49             | 4        | -0.12 |

**Disabled when gold pressure is active** (non-SAFE tier) — if you're dying, spending is correct.

---

## 8. Gold Management Penalties

### 8.1 Gold Excess

**When**: Per-round, after each fight
**Config**: `GOLD_EXCESS_GRACE = 55`, `GOLD_EXCESS_BASE_RATE = 0.015`, `GOLD_EXCESS_MAX_PENALTY = -10.0` (training-config.ts:166-168)
**Code**: training-env.ts:1503-1514

Formula: `penalty = -0.015 × excess × (excess + 1) / 2` (capped at -10.0)

| Gold | Excess (over 55) | Penalty/round |
|------|-------------------|---------------|
| 55   | 0                 | 0             |
| 60   | 5                 | -0.225        |
| 65   | 10                | -0.825        |
| 70   | 15                | -1.80         |
| 80   | 25                | -4.875        |
| 90   | 35                | -9.45         |
| 95+  | 40+               | -10.0 (cap)   |

**Quadratic growth** means holding 70g is 8x worse than holding 60g. The 55g grace allows slight buffer above max interest threshold.

### 8.2 Gold Pressure System

**When**: Per-round, after each fight (stages 5+)
**Config**: `GOLD_PRESSURE_AVG_DAMAGE`, `GOLD_PRESSURE_TIERS`, `GOLD_PRESSURE_MAX_PENALTY = -10.0` (training-config.ts:174-186)
**Code**: training-env.ts:1516-1548

**Step 1**: Compute lives remaining = `floor(HP / avgDamage)`

| Stage range | Avg damage per loss |
|-------------|-------------------|
| 5-10        | 8 HP              |
| 11-16       | 12 HP             |
| 17-22       | 16 HP             |
| 23+         | 18 HP             |

**Step 2**: Determine tier from lives remaining

| Tier   | Lives needed | Base rate | Free gold (no penalty below this) |
|--------|-------------|-----------|----------------------------------|
| SAFE   | 4+          | 0         | Infinity (no penalty ever)       |
| MINOR  | 3           | 0.005     | 50                               |
| MEDIUM | 2           | 0.020     | 30                               |
| ALERT  | 1 or 0      | 0.030     | 10                               |

**Step 3**: Penalty = `tier.baseRate × overage × (overage + 1) / 2` where `overage = max(0, gold - freeGold)`

**Worked examples at stage 17 (avgDamage = 16)**:

| HP  | Lives | Tier   | Gold | Overage | Penalty/round |
|-----|-------|--------|------|---------|---------------|
| 80  | 5     | SAFE   | 60   | 0       | 0             |
| 48  | 3     | MINOR  | 60   | 10      | -0.275        |
| 32  | 2     | MEDIUM | 60   | 30      | -9.30         |
| 16  | 1     | ALERT  | 60   | 50      | -10.0 (cap)   |
| 16  | 1     | ALERT  | 10   | 0       | 0 (under free)|

**Interaction with economy rewards**: At ALERT tier, the agent's free gold is only 10g. Holding 50g at ALERT costs:
- Gold pressure: `0.03 × 40 × 41 / 2 = -24.6` → capped at **-10.0/round**
- Meanwhile interest bonus gives +0.60/round
- **Net: -9.4/round** — economy rewards are completely overwhelmed

This creates the correct behavioral switch: hoard gold when healthy, spend everything when dying.

---

## 9. Board Quality Penalties

### 9.1 Bench Dead-Weight

**When**: Per-round, after each fight
**Config**: `BENCH_DEAD_WEIGHT_BY_STAGE` (training-config.ts:191-196)
**Code**: training-env.ts:1550-1583

A bench unit is "dead weight" if its evolution family has no members on the board (positionY > 0).

| Stage range | Penalty per dead-weight unit |
|-------------|----------------------------|
| 1-10        | 0 (free experimentation)   |
| 11-15       | -0.02                      |
| 16-20       | -0.05                      |
| 21+         | -0.10                      |

3 dead-weight bench units at stage 22 = -0.30/round × ~7 rounds = **-2.10 total**. Moderate signal.

### 9.2 Unit Quality (1-Star Board Penalty)

**When**: Per-round, after each fight
**Config**: `UNIT_QUALITY_STAGES`, `UNIT_QUALITY_RARITY_DISCOUNT` (training-config.ts:203-222)
**Code**: training-env.ts:1585-1624

Penalizes **1-star** units on the board (positionY > 0). 2+ star units are never penalized.

**Base rates by stage** (per 1-star unit):

| Stage range | With active synergy | Without active synergy |
|-------------|--------------------|-----------------------|
| 1-8         | 0                  | 0                     |
| 9-13        | -0.01              | -0.03                 |
| 14-18       | -0.03              | -0.08                 |
| 19+         | -0.08              | -0.18                 |

**Rarity discount** applied as: `final = base × (1 - discount)`

| Rarity    | Discount | Effective penalty (stage 19+, no synergy) |
|-----------|----------|------------------------------------------|
| Common    | 0%       | -0.180                                   |
| Uncommon  | 10%      | -0.162                                   |
| Rare      | 30%      | -0.126                                   |
| Epic      | 50%      | -0.090                                   |
| Ultra     | 70%      | -0.054                                   |
| Unique    | 80%      | -0.036                                   |
| Legendary | 80%      | -0.036                                   |
| Hatch     | 20%      | -0.144                                   |
| Special   | 40%      | -0.108                                   |

**Worst case**: 6 common 1-star units with no synergy at stage 19+ = 6 × -0.18 = **-1.08/round**. Over 8 rounds = -8.64. This is a very strong late-game pressure to upgrade.

**With synergy**: Same board but all contributing to active synergies = 6 × -0.08 = -0.48/round (55% reduction).

---

## 10. Turn Efficiency

### 10.1 Empty Turn Penalty

**Config**: `REWARD_EMPTY_TURN_PENALTY = -0.15`, `EMPTY_TURN_MIN_STAGE = 16`, `EMPTY_TURN_GOLD_FLOOR = 50` (training-config.ts:290-292)
**Code**: training-env.ts:618-627
**Condition**: END_TURN is the very first action taken AND stage >= 16 AND gold >= 50

Only fires when the agent does literally nothing with a full wallet in the late game. Narrow trigger. Probably fires 0-3 times/game.

### 10.2 Bench Open Slots Penalty

**Config**: `REWARD_BENCH_PENALTY = -0.01` (training-config.ts:232)
**Code**: training-env.ts:643-658
**When**: Turn end, only when `TRAINING_AUTO_PLACE = false`
**Condition**: Board has open slots AND bench has units

Penalty = `min(benchCount, openSlots) × -0.01` per turn end.

Very light. 3 bench units with 3 open board slots = -0.03/turn. Over 27 turns = -0.81 total. This is a nudge, not a hammer.

---

## 11. Disabled / Removed Signals

These exist in the code but are set to 0 or commented out:

| Signal | Old value | Why removed |
|--------|-----------|-------------|
| `REWARD_REROLL` | was +0.03 | Agent spammed 50 rerolls/game for free reward |
| `REWARD_REROLL_LATEGAME` | was +0.05 | Same reward hacking issue |
| `REWARD_PER_SURVIVE_ROUND` | was +0.12 | Free +2.2/game that rewarded coasting |
| Reroll boost layers (gold/level) | ×2/×4 | Still in code but multiply 0, so no effect |

---

## 12. Full Magnitude Table

Sorted by typical per-game impact (magnitude × frequency):

| # | Signal | Per-instance | Frequency | Est. total/game | Type |
|---|--------|-------------|-----------|-----------------|------|
| 1 | **Placement** | -26 to +28 | 1/game | -26 to +28 | Reward/Penalty |
| 2 | **Battle result** (late wins) | +2.25 | ~8 late rounds | +18.0 | Reward |
| 3 | **Gold standard** | +0.30/round | ~15 rounds | +4.5 | Reward |
| 4 | **Interest bonus** | 0.06-0.30/round | ~20 rounds | +4.5 | Reward |
| 5 | **Battle result** (losses) | -0.25 to -1.0 | ~13 rounds | -6.5 | Penalty |
| 6 | **Synergy depth** | 0.3-1.5/round | ~27 rounds | +8-25 | Reward |
| 7 | **Unit quality** | -0.18 to 0/unit | ~15 rounds | -2 to -8 | Penalty |
| 8 | **Evo from reroll** | +0.50 to +2.00 | 1-2/game | +1-4 | Reward |
| 9 | **Gold pressure** | 0 to -10/round | 0-5 rounds | 0 to -10 | Penalty |
| 10 | **Gold excess** | 0 to -10/round | 0-3 rounds | 0 to -5 | Penalty |
| 11 | **Bench dead-weight** | -0.02 to -0.10/unit | ~10 rounds | -0.5 to -3 | Penalty |
| 12 | **Enemy kills** | +0.02/kill | ~110 kills | +2.2 | Reward |
| 13 | **Buy evolution** | +0.20 to +0.30 | 3-5/game | +0.6-1.5 | Reward |
| 14 | **Keep unique/legend** | +0.04/unit | ~15 rounds | +0.6-1.2 | Reward |
| 15 | **Bench open slots** | -0.01/unit | ~27 turns | -0.3-0.8 | Penalty |
| 16 | **Buy duplicate** | +0.08 to +0.12 | 5-10/game | +0.4-1.2 | Reward |
| 17 | **Reroll eco penalty** | -0.06 to -0.12 | 5-15/game | -0.3-1.8 | Penalty |
| 18 | **Level-up penalty** | -0.15 | 2-4/game | -0.3-0.6 | Penalty |
| 19 | **Move fidget** | -0.08 | 2-6/game | -0.16-0.48 | Penalty |
| 20 | **Sell evolved** | -0.15 | 0-2/game | 0-0.3 | Penalty |
| 21 | **Empty turn** | -0.15 | 0-3/game | 0-0.45 | Penalty |
| 22 | **Buy-then-sell** | -1.0 | 0-1/game | 0-1.0 | Penalty |
| 23 | **HP preservation** | +0.004 | ~13 wins | +0.05 | Reward |
| 24 | **Level-up stage 9** | +0.10 | 0-1/game | 0-0.10 | Reward |

---

## 13. Per-Game Accumulation Estimates

### Scenario A: Competent agent, finishes 2nd

| Category | Estimated total |
|----------|----------------|
| Placement | +15.0 |
| Battle results (net) | +8.0 |
| Economy (interest + gold std) | +8.0 |
| Synergy depth | +12.0 |
| Purchase rewards | +3.0 |
| Enemy kills | +2.0 |
| Keep unique/legend | +0.8 |
| Board quality penalties | -3.0 |
| Reroll eco penalties | -0.5 |
| Level-up penalties | -0.3 |
| Bench penalties | -0.5 |
| **TOTAL** | **~+44.5** |

### Scenario B: Weak agent, finishes 7th

| Category | Estimated total |
|----------|----------------|
| Placement | -19.0 |
| Battle results (net) | -4.0 |
| Economy (poor eco) | +3.0 |
| Synergy depth (weak board) | +5.0 |
| Purchase rewards | +1.0 |
| Enemy kills | +1.0 |
| Unit quality penalties (many 1-stars) | -6.0 |
| Gold pressure (dying + holding gold) | -5.0 |
| Misc penalties | -1.5 |
| **TOTAL** | **~-25.5** |

### Spread: +44.5 vs -25.5 = **70-point range**

Placement alone accounts for 34 of those 70 points (49%). The shaped rewards account for the other ~36 points (51%).

---

## 14. Ratio Analysis & Interactions

### 14.1 Placement vs Shaped Rewards

- Placement spread: 54 points (28 to -26)
- Shaped rewards typical range: -10 to +25
- **Ratio: ~2:1 placement-dominant**

This means finishing 1 rank higher is worth more than all mid-game optimization combined. If the agent learns to coast at 3rd place with perfect economy, shaped rewards can't overcome the +7 from pushing to 2nd.

**Concern**: If shaped rewards are too weak relative to placement, the agent may ignore economy/synergies entirely and just focus on whatever random strategy gets top-3. If too strong, the agent optimizes economy and ignores winning.

### 14.2 Economy vs Combat Signals

At 50g with a strong board:
- Economy signals: +0.60/round (interest + gold standard)
- Battle win (stage 15): +0.75/round
- Synergy depth: +0.50/round average

Economy = 32% of per-round signal. Combat (win + synergy) = 68%.

At 50g with a dying HP bar (ALERT tier):
- Economy signals: +0.60/round
- Gold pressure penalty: up to -10.0/round
- **Economy is annihilated** — correct behavior

### 14.3 Evo-from-Reroll vs Economy

A single rare evo-from-reroll = +1.20 (base evo + bonus).
This equals 2 full rounds of gold standard income (2 × 0.60 = 1.20).

For the agent to "break even" on a 1g reroll that finds a rare evolution:
- Cost: 1g reroll = lose 0.06 interest signal
- Gain: +1.20 from evo-from-reroll
- **Ratio: 20:1 in favor of the reroll**

Even at a 5% hit rate on finding an evolution, expected value = 0.05 × 1.20 = 0.06, which exactly breaks even with the lost interest signal. This seems correctly calibrated for rare evolutions but may over-reward common evolutions (0.05 × 0.70 = 0.035, barely below break-even).

### 14.4 Unit Quality vs Synergy Depth

A 1-star common unit at stage 19+ with no synergy costs -0.18/round.
A 1-star common unit at stage 19+ WITH active synergy costs -0.08/round.

The synergy depth reward for having that synergy active (1 tier of a 3-tier synergy) = +0.025/round.

So the synergy benefit of keeping the unit (+0.025 synergy + 0.10 reduced quality penalty = +0.125) is worth more than removing it (-0 quality penalty but -0.025 synergy loss = -0.025). **The math says keep it** — but the signal is close. A 2-tier synergy hit changes the calculus significantly.

### 14.5 Level-Up Decision Points

**Scenario: Agent at 50g, considering spending 4g to level**

If gold drops to 46g:
- Lose gold standard: -0.30/round ongoing
- Lose 1 interest tier: -0.06/round ongoing
- Level-up penalty: -0.15 one-time
- Total cost over 5 rounds: -0.15 + 5 × 0.36 = **-1.95**

Benefit of the level: 1 extra team slot → probably +0.25-0.50/round in combat improvement.
Over 5 rounds: +1.25 to +2.50.

**Break-even**: The level-up is worth it only if the extra unit adds >0.36/round in combat value. This is roughly correct for mid-game but may be too restrictive for late-game power spikes.

### 14.6 Gold Pressure Thresholds

The transition from SAFE to MINOR is very sharp:

At stage 17 (avgDamage=16):
- 65 HP (4 lives): SAFE, no penalty, free to hoard
- 48 HP (3 lives): MINOR, penalty on gold >50

Losing 17 HP (roughly 1 loss) can flip the entire economy incentive. The agent has one loss of warning before MINOR kicks in. At MEDIUM (32 HP, 2 lives), free gold drops to 30 — the agent should already be spending.

### 14.7 Synergy Depth Scaling

The quadratic depth formula `tier_hit × (tier_hit / max_tiers)` means:

| Tiers hit / Total tiers | Multiplier |
|--------------------------|-----------|
| 1/3 | 0.33 |
| 2/3 | 1.33 |
| 3/3 | 3.00 |

Going from tier 1 to tier 2 is a **4x increase** in reward (0.33 → 1.33). Going from tier 2 to tier 3 is a **2.25x increase** (1.33 → 3.00). This heavily rewards committing to a synergy rather than splashing many at tier 1.

---

## 15. Known Issues & Overhaul Targets

### 15.1 HP Preservation is Useless
At +0.005/win maximum, this signal accumulates to ~0.05/game. It's 1000x weaker than placement. Either remove it or boost to 0.05-0.10 range.

### 15.2 Synergy Depth Variance
Synergy depth can range from +5 to +25 per game depending on board composition. This 20-point variance is almost as large as the placement spread (54 points). A lucky synergy game can make up for bad placement, which may not be intended.

### 15.3 Evo-from-Reroll Spike
A single legendary evo-from-reroll gives +2.20 (0.20 base + 2.00 bonus). This is larger than the per-round battle win at stage 21+ (+2.25). One purchase event shouldn't rival a late-game combat result.

### 15.4 Economy Cliff at 50g
The 150% jump from 49g to 50g creates a very sharp attractor. The agent may over-prioritize reaching 50g and under-invest in board strength. Consider smoothing the gold standard threshold or making it progressive.

### 15.5 Bottom Placement Flatness
The gap between 5th (-9) and 8th (-26) is 17 points. The gap between 1st (+28) and 4th (-5) is 33 points. The bottom half has roughly half the gradient of the top half. If the agent doesn't differentiate between 5th-8th, it won't learn to fight for survival once it's behind.

### 15.6 Stage 1-8 Free Zone
No unit quality penalties, no bench dead-weight penalties, no reroll eco penalty context in early stages. The agent gets ~8 rounds of unguided play. If early decisions matter for late-game outcomes, the agent is flying blind during the most formative turns.

### 15.7 Gold Pressure vs Economy Deadlock
At MINOR tier (3 lives), free gold is 50. But the economy rewards also peak at 50g. This means MINOR tier has zero behavioral impact if the agent is already at 50g — the pressure only kicks in if they go above. MINOR should probably have a lower free gold threshold (40?) to start pulling the agent toward spending.

### 15.8 Bench Open Slots Too Weak
At -0.01/unit, it takes 27 rounds × 3 units = -0.81 total game impact. Meanwhile a single buy evolution gives +0.20. The agent can ignore empty board slots for multiple turns with no meaningful consequence. Consider 5-10x increase.

### 15.9 Disabled Reroll Reward Creates Dead Zone
With `REWARD_REROLL = 0`, the only positive signal for rerolling is evo-from-reroll (which requires finding an evolution). Productive rerolls that find duplicates (2nd copies) give +0.08 via buy duplicate, but the reroll itself costs 1g (lost interest signal). The agent may under-reroll in spots where fishing for duplicates is correct play.
