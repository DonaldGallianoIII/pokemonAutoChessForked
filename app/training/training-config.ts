/**
 * Configuration for step-mode training.
 * Games run synchronously with no real-time waiting.
 */

import { Item, ItemRecipe } from "../types/enum/Item"
import { Pkm } from "../types/enum/Pokemon"
import { Synergy, SynergyArray } from "../types/enum/Synergy"

// Number of bot opponents in training games
export const TRAINING_NUM_OPPONENTS = 7

// Max actions the RL agent can take per PICK phase before auto-advancing
export const TRAINING_MAX_ACTIONS_PER_TURN = 30

// Delta time (ms) per simulation sub-step during FIGHT phase
export const TRAINING_SIMULATION_DT = 50

// Max simulation steps per fight before force-finishing (safety cap)
export const TRAINING_MAX_FIGHT_STEPS = 2000

// Max proposition slots (NB_UNIQUE_PROPOSITIONS = 6 is the max)
export const MAX_PROPOSITIONS = 6

// Observation space dimensions
export const OBS_PLAYER_STATS = 8 // life, money, level, streak, interest, alive, rank, boardSize
export const OBS_SHOP_SLOTS = 5 // 5 shop slots, each encoded as pokemon rarity
export const OBS_BOARD_SLOTS = 40 // 8 bench + 32 board cells (4x8)
export const OBS_BOARD_FEATURES_PER_SLOT = 3 // hasUnit, stars, rarity
export const OBS_SYNERGIES = 32 // 32 synergy types (padded, actual is 31)
export const OBS_GAME_INFO = 4 // stageLevel, phase, playersAlive, hasPropositions
export const OBS_OPPONENT_STATS = 16 // 2 features per opponent (life, rank) * 8 max opponents
export const OBS_PROPOSITION_SLOTS = MAX_PROPOSITIONS // 6 proposition slots
export const OBS_PROPOSITION_FEATURES = 3 // rarity, numTypes, hasItem

export const TOTAL_OBS_SIZE =
  OBS_PLAYER_STATS +
  OBS_SHOP_SLOTS +
  OBS_BOARD_SLOTS * OBS_BOARD_FEATURES_PER_SLOT +
  OBS_SYNERGIES +
  OBS_GAME_INFO +
  OBS_OPPONENT_STATS +
  OBS_PROPOSITION_SLOTS * OBS_PROPOSITION_FEATURES

// Action space (22-action v1)
//
// ACTION INDEX MAPPING — training env vs future agent-io extension:
//
// These indices are internal to the training env. They do NOT match the
// planned 92-action agent-io extension layout (OPTION_B_MASTER_PLAN.md),
// which uses: BUY 0-5, REFRESH 6, LEVEL 7, LOCK 8, END_TURN 9, MOVE 10-41,
// SELL 42-73, REMOVE_SHOP 74-79, PICK 80-85, COMBINE_ITEMS 86-91.
//
// When bridging a trained model to the extension, you MUST translate indices.
// After Phase 1 expansion to 92 actions, the training env should switch to
// match the extension layout 1:1 to eliminate translation overhead.
export enum TrainingAction {
  END_TURN = 0, // End pick phase, advance to fight
  BUY_0 = 1,
  BUY_1 = 2,
  BUY_2 = 3,
  BUY_3 = 4,
  BUY_4 = 5,
  SELL_0 = 6, // Sell pokemon at bench position 0
  SELL_1 = 7,
  SELL_2 = 8,
  SELL_3 = 9,
  SELL_4 = 10,
  SELL_5 = 11,
  SELL_6 = 12,
  SELL_7 = 13,
  REROLL = 14,
  LEVEL_UP = 15,
  // Pick from proposition slots (starters, uniques, legendaries, additional picks)
  PICK_PROPOSITION_0 = 16,
  PICK_PROPOSITION_1 = 17,
  PICK_PROPOSITION_2 = 18,
  PICK_PROPOSITION_3 = 19,
  PICK_PROPOSITION_4 = 20,
  PICK_PROPOSITION_5 = 21
}

export const TOTAL_ACTIONS = 22

// Reward shaping
export const REWARD_PER_WIN = 0.5
export const REWARD_PER_LOSS = -0.3
export const REWARD_PER_DRAW = 0.0
export const REWARD_PER_KILL = -2.0 // penalty when agent dies
export const REWARD_PLACEMENT_SCALE = 2.0 // final reward = (9 - rank) * scale - offset
export const REWARD_PLACEMENT_OFFSET = 6.0

// Self-play mode: when true, all 8 players are RL agents controlled via /step-multi.
// When false (default), 1 RL agent plays against 7 bots (Phase A curriculum training).
// Toggle via environment variable: SELF_PLAY=true
export const SELF_PLAY = process.env.SELF_PLAY === "true"

// HTTP server port for training API
export const TRAINING_API_PORT = parseInt(process.env.TRAINING_PORT ?? "9100")

// ─── Phase 0: Grid & Helper Constants ────────────────────────────────

export const GRID_WIDTH = 8
export const GRID_HEIGHT = 4 // y=0 bench, y=1-3 board
export const GRID_CELLS = GRID_WIDTH * GRID_HEIGHT // 32
export const SHOP_SLOTS = 6
export const MAX_COMBINE_PAIRS = 6
export const MAX_HELD_ITEMS = 10 // observation encoding cap
export const NUM_PKM_SPECIES = Object.values(Pkm).length
export const NUM_SYNERGIES = SynergyArray.length

export function cellToXY(cell: number): [number, number] {
  return [cell % GRID_WIDTH, Math.floor(cell / GRID_WIDTH)]
}

export function xyToCell(x: number, y: number): number {
  return y * GRID_WIDTH + x
}

// ─── Item-pair enumeration ───────────────────────────────────────────

export function enumerateItemPairs(items: string[]): [number, number][] {
  const pairs: [number, number][] = []
  for (let i = 0; i < items.length && pairs.length < MAX_COMBINE_PAIRS; i++) {
    for (let j = i + 1; j < items.length && pairs.length < MAX_COMBINE_PAIRS; j++) {
      pairs.push([i, j])
    }
  }
  return pairs
}

// ─── Item recipe lookup ──────────────────────────────────────────────

export function findRecipeResult(itemA: Item, itemB: Item): Item | null {
  for (const [result, ingredients] of Object.entries(ItemRecipe)) {
    if (
      (ingredients![0] === itemA && ingredients![1] === itemB) ||
      (ingredients![0] === itemB && ingredients![1] === itemA)
    ) {
      return result as Item
    }
  }
  return null
}

// ─── Pokemon species-index lookup ────────────────────────────────────

const PkmValues = Object.values(Pkm)
const PkmToIndex = new Map<string, number>()
PkmValues.forEach((pkm, i) => PkmToIndex.set(pkm, i))

export function getPkmSpeciesIndex(pkm: Pkm): number {
  return (PkmToIndex.get(pkm) ?? 0) / NUM_PKM_SPECIES
}

// ─── Synergy-index lookup ────────────────────────────────────────────

const SynergyToIndex = new Map<string, number>()
SynergyArray.forEach((syn, i) => SynergyToIndex.set(syn, i))

export function getSynergyIndex(synergy: Synergy): number {
  return ((SynergyToIndex.get(synergy) ?? 0) + 1) / NUM_SYNERGIES
}
