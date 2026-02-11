/**
 * Configuration for step-mode training.
 * Games run synchronously with no real-time waiting.
 */

import { Item, ItemRecipe } from "../types/enum/Item"
import { Pkm } from "../types/enum/Pokemon"
import { Synergy, SynergyArray } from "../types/enum/Synergy"
import { Weather } from "../types/enum/Weather"

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
export const OBS_PLAYER_STATS = 14 // life, money, level, streak, interest, alive, rank, boardSize, expNeeded, shopFreeRolls, rerollCount, shopLocked, totalMoneyEarned, totalPlayerDamageDealt
export const OBS_SHOP_SLOTS = 6 // 6 shop slots
export const OBS_SHOP_FEATURES = 9 // per slot: hasUnit, species, rarity, cost, type1-4, isEvoPossible
export const OBS_BOARD_SLOTS = 32 // 8x4 grid (y=0 bench, y=1-3 board)
export const OBS_BOARD_FEATURES_PER_SLOT = 12 // hasUnit, species, stars, rarity, type1-4, atk, hp, range, numItems
export const OBS_HELD_ITEMS = 10 // up to 10 held items (normalized item indices)
export const OBS_SYNERGIES = 31 // 31 synergy types (matches SynergyArray.length)
export const OBS_GAME_INFO = 7 // stageLevel, phase, playersAlive, hasPropositions, weatherIndex, isPVE, maxTeamSize
export const OBS_OPPONENT_COUNT = 7
export const OBS_OPPONENT_FEATURES = 10 // life, rank, level, gold, streak, boardSize, topSyn1Idx, topSyn1Count, topSyn2Idx, topSyn2Count
export const OBS_OPPONENT_STATS = OBS_OPPONENT_COUNT * OBS_OPPONENT_FEATURES // 70
export const OBS_PROPOSITION_SLOTS = 6
export const OBS_PROPOSITION_FEATURES = 7 // species, rarity, type1-4, itemIndex

export const TOTAL_OBS_SIZE =
  OBS_PLAYER_STATS +                                  // 14
  OBS_SHOP_SLOTS * OBS_SHOP_FEATURES +                // 54
  OBS_BOARD_SLOTS * OBS_BOARD_FEATURES_PER_SLOT +     // 384
  OBS_HELD_ITEMS +                                     // 10
  OBS_SYNERGIES +                                      // 31
  OBS_GAME_INFO +                                      // 7
  OBS_OPPONENT_STATS +                                 // 70
  OBS_PROPOSITION_SLOTS * OBS_PROPOSITION_FEATURES     // 42
  // = 612

// Action space (92-action layout, aligned with agent-io API)
//
// These indices match the agent-io extension layout 1:1.
// BUY 0-5, REFRESH 6, LEVEL_UP 7, LOCK_SHOP 8, END_TURN 9,
// MOVE 10-41, SELL 42-73, REMOVE_SHOP 74-79, PICK 80-85, COMBINE 86-91.
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

// Reward shaping
export const REWARD_PER_WIN = 0.6
export const REWARD_PER_LOSS = -0.5
export const REWARD_PER_DRAW = 0.0
export const REWARD_PER_KILL = -2.0 // penalty when agent dies
export const REWARD_PLACEMENT_SCALE = 2.0 // final reward = (9 - rank) * scale - offset
export const REWARD_PLACEMENT_OFFSET = 6.0

// Shaped rewards (Phase 6)
export const REWARD_INTEREST_BONUS = 0.05   // per interest gold earned (with board guard)
export const REWARD_SYNERGY_THRESHOLD = 0.3 // per newly activated synergy threshold
export const REWARD_PER_ENEMY_KILL = 0.02   // per enemy unit killed in combat
export const REWARD_HP_SCALE = 0.005        // HP preservation bonus on win
export const REWARD_PER_SURVIVE_ROUND = 0.12 // bonus for every alive player each round

// Self-play mode: when true, all 8 players are RL agents controlled via /step-multi.
// When false (default), 1 RL agent plays against 7 bots (Phase A curriculum training).
// Toggle via environment variable: SELF_PLAY=true
export const SELF_PLAY = process.env.SELF_PLAY === "true"

// HTTP server port for training API
export const TRAINING_API_PORT = parseInt(process.env.TRAINING_PORT ?? "9100")

// Auto-place: when false, agent must use MOVE actions to place units on board.
// Flipped to false in Phase 7.1 — agent controls placement directly.
export const TRAINING_AUTO_PLACE = false

// Bench penalty: applied per bench unit when board has open slots at turn end
export const REWARD_BENCH_PENALTY = -0.01

// Move fidget penalty: applied per move after MOVE_FIDGET_GRACE free moves in a row
export const MOVE_FIDGET_GRACE = 2
export const REWARD_MOVE_FIDGET = -0.08  // was -0.03, increased to punish oscillation loops

// Sell penalty: penalize selling evolved (2-3 star) units
export const REWARD_SELL_EVOLVED = -0.15

// Level-up reward: small incentive to invest gold in levels
export const REWARD_LEVEL_UP = 0.10

// Per-step bonus for keeping unique/legendary units on board (not bench, not sold)
export const REWARD_KEEP_UNIQUE = 0.007
export const REWARD_KEEP_LEGENDARY = 0.007

// Reward for buying a unit whose species already exists on the board/bench (encourages evolutions)
export const REWARD_BUY_DUPLICATE = 0.08     // buying 2nd copy
export const REWARD_BUY_EVOLUTION = 0.20     // buying 3rd copy (triggers evolution)

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

// ─── Item-index lookup ───────────────────────────────────────────────

const AllItems = Object.values(Item)
const ItemToIndex = new Map<string, number>()
AllItems.forEach((item, i) => ItemToIndex.set(item, i))
const NUM_ITEMS = AllItems.length

export function getItemIndex(item: Item): number {
  return ((ItemToIndex.get(item) ?? 0) + 1) / NUM_ITEMS
}

// ─── Weather-index lookup ────────────────────────────────────────────

const WeatherValues = Object.values(Weather)
const WeatherToIndex = new Map<string, number>()
WeatherValues.forEach((w, i) => WeatherToIndex.set(w, i))
const NUM_WEATHERS = WeatherValues.length

export function getWeatherIndex(weather: Weather): number {
  return ((WeatherToIndex.get(weather) ?? 0) + 1) / NUM_WEATHERS
}
