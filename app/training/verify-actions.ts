/**
 * verify-actions.ts — Phase 7.3
 *
 * Asserts that all 92 action indices, grid math, and totals match
 * the agent-io API bible layout. Run with:
 *
 *   npx ts-node --transpile-only app/training/verify-actions.ts
 */

import {
  cellToXY,
  GRID_CELLS,
  GRID_HEIGHT,
  GRID_WIDTH,
  SHOP_SLOTS,
  MAX_COMBINE_PAIRS,
  TOTAL_ACTIONS,
  TOTAL_OBS_SIZE,
  OBS_PLAYER_STATS,
  OBS_SHOP_SLOTS,
  OBS_SHOP_FEATURES,
  OBS_BOARD_SLOTS,
  OBS_BOARD_FEATURES_PER_SLOT,
  OBS_HELD_ITEMS,
  OBS_SYNERGIES,
  OBS_GAME_INFO,
  OBS_OPPONENT_STATS,
  OBS_PROPOSITION_SLOTS,
  OBS_PROPOSITION_FEATURES,
  TrainingAction
} from "./training-config"

let passed = 0
let failed = 0

function assert(condition: boolean, message: string): void {
  if (condition) {
    passed++
  } else {
    failed++
    console.error(`FAIL: ${message}`)
  }
}

// ── Grid constants ──────────────────────────────────────────────────────

assert(GRID_WIDTH === 8, `GRID_WIDTH should be 8, got ${GRID_WIDTH}`)
assert(GRID_HEIGHT === 4, `GRID_HEIGHT should be 4, got ${GRID_HEIGHT}`)
assert(GRID_CELLS === 32, `GRID_CELLS should be 32, got ${GRID_CELLS}`)
assert(SHOP_SLOTS === 6, `SHOP_SLOTS should be 6, got ${SHOP_SLOTS}`)
assert(MAX_COMBINE_PAIRS === 6, `MAX_COMBINE_PAIRS should be 6, got ${MAX_COMBINE_PAIRS}`)

// ── cellToXY round-trip ─────────────────────────────────────────────────

for (let cell = 0; cell < GRID_CELLS; cell++) {
  const [x, y] = cellToXY(cell)
  const reconstructed = y * GRID_WIDTH + x
  assert(
    reconstructed === cell,
    `cellToXY(${cell}) = [${x},${y}] → ${reconstructed}, expected ${cell}`
  )
  assert(x >= 0 && x < GRID_WIDTH, `cellToXY(${cell}).x = ${x} out of range`)
  assert(y >= 0 && y < GRID_HEIGHT, `cellToXY(${cell}).y = ${y} out of range`)
}

// ── Action index layout (API bible) ────────────────────────────────────
// BUY 0-5, REFRESH 6, LEVEL_UP 7, LOCK_SHOP 8, END_TURN 9,
// MOVE 10-41, SELL 42-73, REMOVE_SHOP 74-79, PICK 80-85, COMBINE 86-91.

assert(TrainingAction.BUY_0 === 0, `BUY_0 should be 0`)
assert(TrainingAction.BUY_5 === 5, `BUY_5 should be 5`)
assert(TrainingAction.REFRESH === 6, `REFRESH should be 6`)
assert(TrainingAction.LEVEL_UP === 7, `LEVEL_UP should be 7`)
assert(TrainingAction.LOCK_SHOP === 8, `LOCK_SHOP should be 8`)
assert(TrainingAction.END_TURN === 9, `END_TURN should be 9`)
assert(TrainingAction.MOVE_0 === 10, `MOVE_0 should be 10`)

// MOVE spans 32 cells: 10..41
const moveEnd = TrainingAction.MOVE_0 + GRID_CELLS - 1
assert(moveEnd === 41, `MOVE end should be 41, got ${moveEnd}`)

// SELL starts at 42
assert(TrainingAction.SELL_0 === 42, `SELL_0 should be 42`)
const sellEnd = TrainingAction.SELL_0 + GRID_CELLS - 1
assert(sellEnd === 73, `SELL end should be 73, got ${sellEnd}`)

// REMOVE_SHOP starts at 74
assert(TrainingAction.REMOVE_SHOP_0 === 74, `REMOVE_SHOP_0 should be 74`)
assert(TrainingAction.REMOVE_SHOP_5 === 79, `REMOVE_SHOP_5 should be 79`)

// PICK starts at 80
assert(TrainingAction.PICK_0 === 80, `PICK_0 should be 80`)
assert(TrainingAction.PICK_5 === 85, `PICK_5 should be 85`)

// COMBINE starts at 86
assert(TrainingAction.COMBINE_0 === 86, `COMBINE_0 should be 86`)
assert(TrainingAction.COMBINE_5 === 91, `COMBINE_5 should be 91`)

// ── Total action count ──────────────────────────────────────────────────

// Count all action indices
const actionRanges = [
  { name: "BUY", start: 0, count: SHOP_SLOTS },         // 0-5 = 6
  { name: "REFRESH", start: 6, count: 1 },               // 6 = 1
  { name: "LEVEL_UP", start: 7, count: 1 },              // 7 = 1
  { name: "LOCK_SHOP", start: 8, count: 1 },             // 8 = 1
  { name: "END_TURN", start: 9, count: 1 },              // 9 = 1
  { name: "MOVE", start: 10, count: GRID_CELLS },        // 10-41 = 32
  { name: "SELL", start: 42, count: GRID_CELLS },        // 42-73 = 32
  { name: "REMOVE_SHOP", start: 74, count: SHOP_SLOTS }, // 74-79 = 6
  { name: "PICK", start: 80, count: SHOP_SLOTS },        // 80-85 = 6
  { name: "COMBINE", start: 86, count: MAX_COMBINE_PAIRS } // 86-91 = 6
]

// Verify contiguous and non-overlapping
let expectedNext = 0
for (const range of actionRanges) {
  assert(
    range.start === expectedNext,
    `${range.name} start should be ${expectedNext}, got ${range.start}`
  )
  expectedNext = range.start + range.count
}

const totalFromRanges = actionRanges.reduce((sum, r) => sum + r.count, 0)
assert(
  totalFromRanges === 92,
  `Sum of all action ranges should be 92, got ${totalFromRanges}`
)
assert(
  TOTAL_ACTIONS === 92,
  `TOTAL_ACTIONS should be 92, got ${TOTAL_ACTIONS}`
)
assert(
  totalFromRanges === TOTAL_ACTIONS,
  `Sum of ranges (${totalFromRanges}) should equal TOTAL_ACTIONS (${TOTAL_ACTIONS})`
)

// ── Observation space ───────────────────────────────────────────────────

const obsBreakdown = {
  playerStats: OBS_PLAYER_STATS,                              // 14
  shop: OBS_SHOP_SLOTS * OBS_SHOP_FEATURES,                  // 54
  board: OBS_BOARD_SLOTS * OBS_BOARD_FEATURES_PER_SLOT,      // 384
  heldItems: OBS_HELD_ITEMS,                                  // 10
  synergies: OBS_SYNERGIES,                                   // 31
  gameInfo: OBS_GAME_INFO,                                    // 7
  opponents: OBS_OPPONENT_STATS,                              // 70
  propositions: OBS_PROPOSITION_SLOTS * OBS_PROPOSITION_FEATURES // 42
}

const obsTotal = Object.values(obsBreakdown).reduce((a, b) => a + b, 0)

assert(OBS_PLAYER_STATS === 14, `OBS_PLAYER_STATS should be 14, got ${OBS_PLAYER_STATS}`)
assert(OBS_SHOP_SLOTS === 6, `OBS_SHOP_SLOTS should be 6, got ${OBS_SHOP_SLOTS}`)
assert(OBS_SHOP_FEATURES === 9, `OBS_SHOP_FEATURES should be 9, got ${OBS_SHOP_FEATURES}`)
assert(OBS_BOARD_SLOTS === 32, `OBS_BOARD_SLOTS should be 32, got ${OBS_BOARD_SLOTS}`)
assert(OBS_BOARD_FEATURES_PER_SLOT === 12, `OBS_BOARD_FEATURES_PER_SLOT should be 12, got ${OBS_BOARD_FEATURES_PER_SLOT}`)
assert(OBS_HELD_ITEMS === 10, `OBS_HELD_ITEMS should be 10, got ${OBS_HELD_ITEMS}`)
assert(OBS_SYNERGIES === 31, `OBS_SYNERGIES should be 31, got ${OBS_SYNERGIES}`)
assert(OBS_GAME_INFO === 7, `OBS_GAME_INFO should be 7, got ${OBS_GAME_INFO}`)
assert(OBS_OPPONENT_STATS === 70, `OBS_OPPONENT_STATS should be 70, got ${OBS_OPPONENT_STATS}`)
assert(OBS_PROPOSITION_SLOTS === 6, `OBS_PROPOSITION_SLOTS should be 6, got ${OBS_PROPOSITION_SLOTS}`)
assert(OBS_PROPOSITION_FEATURES === 7, `OBS_PROPOSITION_FEATURES should be 7, got ${OBS_PROPOSITION_FEATURES}`)

assert(obsTotal === 612, `Observation total should be 612, got ${obsTotal}`)
assert(TOTAL_OBS_SIZE === 612, `TOTAL_OBS_SIZE should be 612, got ${TOTAL_OBS_SIZE}`)
assert(
  obsTotal === TOTAL_OBS_SIZE,
  `Computed obs total (${obsTotal}) should equal TOTAL_OBS_SIZE (${TOTAL_OBS_SIZE})`
)

// ── Summary ─────────────────────────────────────────────────────────────

console.log(`\nAction space breakdown:`)
for (const range of actionRanges) {
  const end = range.start + range.count - 1
  console.log(`  ${range.name.padEnd(14)} ${String(range.start).padStart(2)}-${String(end).padStart(2)}  (${range.count})`)
}
console.log(`  ${"TOTAL".padEnd(14)}      = ${TOTAL_ACTIONS}`)

console.log(`\nObservation space breakdown:`)
for (const [key, value] of Object.entries(obsBreakdown)) {
  console.log(`  ${key.padEnd(16)} ${String(value).padStart(4)}`)
}
console.log(`  ${"TOTAL".padEnd(16)} ${String(TOTAL_OBS_SIZE).padStart(4)}`)

console.log(`\n${passed} passed, ${failed} failed`)
if (failed > 0) {
  process.exit(1)
} else {
  console.log("All action/observation space assertions passed!")
}
