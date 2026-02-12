/**
 * TrainingEnv: A Gym-like environment for PPO training.
 *
 * Drives the game synchronously in step mode:
 * - PICK phase: RL agent takes micro-actions (buy/sell/reroll/level/end_turn)
 * - FIGHT phase: runs simulation synchronously until done
 * - TOWN phase: auto-skipped
 *
 * Games complete in milliseconds instead of minutes.
 */
import { MapSchema } from "@colyseus/schema"
import { nanoid } from "nanoid"
import {
  AdditionalPicksStages,
  FIGHTING_PHASE_DURATION,
  ItemCarouselStages,
  PortalCarouselStages,
  StageDuration,
  SynergyTriggers
} from "../config"
import { selectMatchups } from "../core/matchmaking"
import Simulation from "../core/simulation"
import Player from "../models/colyseus-models/player"
import { Pokemon } from "../models/colyseus-models/pokemon"
import PokemonFactory from "../models/pokemon-factory"
import { getPokemonData } from "../models/precomputed/precomputed-pokemon-data"
import { PRECOMPUTED_POKEMONS_PER_RARITY } from "../models/precomputed/precomputed-rarity"
import { getBuyPrice, getAdditionalsTier1, getSellPrice } from "../models/shop"
import GameState from "../rooms/states/game-state"
import { Role, Transfer } from "../types"
import {
  BattleResult,
  GameMode,
  GamePhaseState,
  Rarity,
  Team
} from "../types/enum/Game"
import {
  CraftableItemsNoScarves,
  Item,
  ItemComponents,
  ItemComponentsNoFossilOrScarf,
  ItemComponentsNoScarf
} from "../types/enum/Item"
import { Pkm, PkmDuo, PkmDuos, PkmIndex, PkmProposition } from "../types/enum/Pokemon"
import { SpecialGameRule } from "../types/enum/SpecialGameRule"
import { Synergy, SynergyArray } from "../types/enum/Synergy"
import { Weather } from "../types/enum/Weather"
import { getAvatarString } from "../utils/avatar"
import {
  getFirstAvailablePositionInBench,
  getFirstAvailablePositionOnBoard,
  getFreeSpaceOnBench,
  isOnBench
} from "../utils/board"
import { max } from "../utils/number"
import { pickNRandomIn, pickRandomIn, shuffleArray } from "../utils/random"
import { values } from "../utils/schemas"
import { HeadlessRoom } from "./headless-room"
import {
  cellToXY,
  enumerateItemPairs,
  findRecipeResult,
  getItemIndex,
  getPkmSpeciesIndex,
  getSynergyIndex,
  getWeatherIndex,
  GRID_CELLS,
  GRID_HEIGHT,
  GRID_WIDTH,
  MAX_PROPOSITIONS,
  MOVE_FIDGET_GRACE,
  OBS_HELD_ITEMS,
  OBS_OPPONENT_COUNT,
  OBS_OPPONENT_FEATURES,
  OBS_PROPOSITION_FEATURES,
  OBS_PROPOSITION_SLOTS,
  GOLD_EXCESS_THRESHOLD,
  GOLD_LATEGAME_STAGE,
  GOLD_LATEGAME_TIER1_THRESHOLD,
  GOLD_LATEGAME_TIER2_THRESHOLD,
  GOLD_MIN_TARGETS,
  REWARD_BENCH_PENALTY,
  REWARD_BUY_DUPLICATE,
  REWARD_BUY_DUPLICATE_LATEGAME,
  REWARD_BUY_EVOLUTION,
  REWARD_BUY_EVOLUTION_LATEGAME,
  REWARD_GOLD_EXCESS_PENALTY,
  REWARD_GOLD_LATEGAME_TIER1,
  REWARD_GOLD_LATEGAME_TIER2,
  REWARD_GOLD_LATEGAME_TIER3,
  REWARD_GOLD_LOW_PENALTY,
  REWARD_LEVEL_UP,
  REWARD_MOVE_FIDGET,
  REWARD_REROLL,
  REWARD_REROLL_LATEGAME,
  REWARD_HP_SCALE,
  REWARD_KEEP_LEGENDARY,
  REWARD_KEEP_UNIQUE,
  REWARD_INTEREST_BONUS,
  REWARD_SELL_EVOLVED,
  REWARD_PER_DRAW,
  REWARD_PER_ENEMY_KILL,
  REWARD_PER_KILL,
  REWARD_PER_LOSS,
  REWARD_PER_SURVIVE_ROUND,
  REWARD_PER_WIN,
  REWARD_PLACEMENT_OFFSET,
  REWARD_PLACEMENT_SCALE,
  REWARD_SYNERGY_THRESHOLD,
  REWARD_SYNERGY_SUSTAINED,
  REWARD_SYNERGY_MULTI_BONUS,
  REWARD_SYNERGY_MULTI_CAP,
  SELF_PLAY,
  TOTAL_ACTIONS,
  TOTAL_OBS_SIZE,
  TRAINING_AUTO_PLACE,
  TRAINING_MAX_ACTIONS_PER_TURN,
  TRAINING_MAX_FIGHT_STEPS,
  TRAINING_NUM_OPPONENTS,
  TRAINING_SIMULATION_DT,
  TrainingAction
} from "./training-config"
import { BotV2, IBot } from "../models/mongo-models/bot-v2"
import { CountEvolutionRule } from "../core/evolution-rules"
import { getMaxTeamSize } from "../utils/board"
import { PVEStages } from "../models/pve-stages"
import { getWeather } from "../utils/weather"

export interface StepResult {
  observation: number[]
  reward: number
  done: boolean
  info: {
    stage: number
    phase: string
    rank: number
    life: number
    money: number
    actionsThisTurn: number
    actionMask: number[]
    gold: number
    boardSize: number
    synergyCount: number
    itemsHeld: number
    level: number
    maxTeamSize: number
    exp: number
    expNeeded: number
    shop: (string | null)[]
    board: { name: string; x: number; y: number; stars: number; items: string[] }[]
    bench: { name: string; x: number; stars: number; items: string[] }[]
    synergies: { name: string; count: number; threshold: number }[]
    items: string[]
    opponent: {
      name: string
      odLife: number
      level: number
      board: { name: string; x: number; y: number; stars: number; items: string[] }[]
    } | null
  }
}

export class TrainingEnv {
  state!: GameState
  room!: HeadlessRoom
  agentId!: string
  actionsThisTurn = 0
  consecutiveMoves = 0
  lastMoveCell = -1  // last MOVE target cell, used to prevent back-and-forth oscillation
  lastSoldStars = 0  // stars of the last sold unit, for sell penalty
  lockShopUsedThisTurn = false  // prevent LOCK_SHOP spam (toggle once per turn max)
  totalSteps = 0
  lastBattleResult: BattleResult | null = null
  prevActiveSynergyCount = 0
  cachedBots: IBot[] = []
  botSynergies = new Map<string, Synergy>() // each dummy bot's team synergy theme
  additionalUncommonPool: Pkm[] = []
  additionalRarePool: Pkm[] = []
  additionalEpicPool: Pkm[] = []

  // 7.2: Caches for hot-path optimization
  private positionGridCache = new Map<string, Map<number, Pokemon>>() // playerId → (cellKey → Pokemon)
  private positionGridDirty = new Set<string>() // playerIds whose grid needs rebuild
  private itemPairCache = new Map<string, [number, number][]>() // playerId → cached pairs
  private observationCache = new Map<string, number[]>() // playerId → cached obs
  private observationDirty = new Set<string>() // playerIds whose obs needs rebuild

  // Self-play state: tracks per-player turn status across step calls within a round.
  // Issue #1: turnEnded MUST persist across step calls — fight triggers only when
  // the LAST alive player ends their turn, not on the first END_TURN.
  playerIds: string[] = []
  turnEnded: Map<string, boolean> = new Map()
  actionsPerPlayer: Map<string, number> = new Map()

  async initialize() {
    // Pre-fetch bots from DB so we don't need to query every reset
    if (this.cachedBots.length === 0) {
      try {
        const mongoose = await import("mongoose")
        if (mongoose.connection.readyState === 1) {
          const bots = await BotV2.find({ approved: true }).lean()
          if (bots.length > 0) {
            this.cachedBots = bots as IBot[]
          }
        }
      } catch (err) {
        // MongoDB not available - training will use empty bot pool
        this.cachedBots = []
      }
    }
  }

  reset(): StepResult {
    // Create fresh game state
    this.state = new GameState(
      nanoid(),
      "training",
      true, // noElo
      GameMode.CUSTOM_LOBBY,
      null,
      null,
      null
    )

    this.room = new HeadlessRoom(this.state)

    // Initialize additional pick pools (same as GameRoom.onCreate)
    this.additionalUncommonPool = getAdditionalsTier1(
      PRECOMPUTED_POKEMONS_PER_RARITY.UNCOMMON
    )
    this.additionalRarePool = getAdditionalsTier1(
      PRECOMPUTED_POKEMONS_PER_RARITY.RARE
    )
    this.additionalEpicPool = getAdditionalsTier1(
      PRECOMPUTED_POKEMONS_PER_RARITY.EPIC
    )
    shuffleArray(this.additionalUncommonPool)
    shuffleArray(this.additionalRarePool)
    shuffleArray(this.additionalEpicPool)

    this.playerIds = []

    if (SELF_PLAY) {
      // Self-play: create 8 RL agent players (no bots)
      this.createSelfPlayAgents()
      this.agentId = this.playerIds[0] // backwards compat for getObservation etc.

      // All players get propositions at stage 0
      this.state.players.forEach((player) => {
        this.state.shop.assignUniquePropositions(player, this.state, [])
      })

      // Assign shops for all players
      this.state.players.forEach((player) => {
        this.state.shop.assignShop(player, false, this.state)
      })
    } else {
      // Single-agent mode: 1 RL agent + 7 bots (Phase A curriculum training)
      this.agentId = "rl-agent-" + nanoid(6)
      const agentPlayer = new Player(
        this.agentId,
        "RL-Agent",
        1000, // elo
        0, // games
        getAvatarString(PkmIndex[Pkm.PIKACHU], false),
        false, // isBot
        1, // rank
        new Map(),
        "",
        Role.BASIC,
        this.state
      )
      this.state.players.set(this.agentId, agentPlayer)
      this.playerIds.push(this.agentId)

      // Create bot opponents
      this.createBotPlayers()

      // Track bot player IDs
      this.state.players.forEach((_player, id) => {
        if (id !== this.agentId) this.playerIds.push(id)
      })

      // Initialize shop for agent
      this.state.shop.assignShop(agentPlayer, false, this.state)

      // Stage 0: Portal carousel gives starter pokemon propositions
      // Agent gets propositions to choose from; bots auto-pick
      this.state.players.forEach((player) => {
        if (!player.isBot) {
          this.state.shop.assignUniquePropositions(player, this.state, [])
        }
      })
      // Auto-pick for bots only; agent propositions stay pending
      this.autoPickPropositionsForBots()
    }

    // Start the game
    this.state.gameLoaded = true
    this.state.stageLevel = 0

    // Stay at stage 0 with propositions — agent(s) pick via PICK action
    this.state.phase = GamePhaseState.PICK
    this.state.time =
      (StageDuration[this.state.stageLevel] ?? StageDuration.DEFAULT) * 1000

    this.actionsThisTurn = 0
    this.consecutiveMoves = 0
    this.lastMoveCell = -1
    this.lastSoldStars = 0
    this.lockShopUsedThisTurn = false
    this.totalSteps = 0
    this.lastBattleResult = null
    this.prevActiveSynergyCount = 0
    this.positionGridCache.clear()
    this.positionGridDirty.clear()
    this.itemPairCache.clear()
    this.observationCache.clear()
    this.observationDirty.clear()
    this.resetTurnState()

    return {
      observation: this.getObservation(),
      reward: 0,
      done: false,
      info: this.getInfo()
    }
  }

  step(action: number): StepResult {
    if (this.state.gameFinished) {
      return {
        observation: this.getObservation(),
        reward: 0,
        done: true,
        info: this.getInfo()
      }
    }

    this.totalSteps++
    let reward = 0

    const agent = this.state.players.get(this.agentId)
    if (!agent || !agent.alive) {
      return {
        observation: this.getObservation(),
        reward: REWARD_PER_KILL,
        done: true,
        info: this.getInfo()
      }
    }

    if (this.state.phase === GamePhaseState.PICK) {
      // Snapshot shop name + existing copy count BEFORE buy (evolution merges copies)
      const isBuy = action >= TrainingAction.BUY_0 && action <= TrainingAction.BUY_5
      let preBuyCopies = 0
      if (isBuy) {
        const shopIdx = action - TrainingAction.BUY_0
        const shopName = agent.shop[shopIdx]
        if (shopName && shopName !== Pkm.DEFAULT) {
          const targetIndex = getPokemonData(shopName)?.index
          if (targetIndex) {
            preBuyCopies = values(agent.board).filter(
              (p) => p.index === targetIndex
            ).length
          }
        }
      }

      // Execute the agent's action
      const actionExecuted = this.executeAction(action, agent)
      this.actionsThisTurn++

      // Reward for buying duplicates (encourages building toward evolutions)
      // Check pre-buy count: 1 existing = 2nd copy, 2 existing = 3rd copy (evolution!)
      // After stage 20, boost rewards to encourage spending gold on upgrades.
      const lateGame = this.state.stageLevel > 20
      if (isBuy && actionExecuted && preBuyCopies >= 2) {
        reward += lateGame ? REWARD_BUY_EVOLUTION_LATEGAME : REWARD_BUY_EVOLUTION
      } else if (isBuy && actionExecuted && preBuyCopies === 1) {
        reward += lateGame ? REWARD_BUY_DUPLICATE_LATEGAME : REWARD_BUY_DUPLICATE
      }

      // Move fidget penalty: 2 free moves, then penalty per consecutive move
      const isMove = action >= TrainingAction.MOVE_0 && action <= TrainingAction.MOVE_0 + 31
      if (isMove && actionExecuted) {
        // Track the target cell so we can block oscillation in the action mask
        const moveCell = action - TrainingAction.MOVE_0
        this.lastMoveCell = moveCell
        this.consecutiveMoves++
        if (this.consecutiveMoves > MOVE_FIDGET_GRACE) {
          reward += REWARD_MOVE_FIDGET
        }
      } else if (isMove && !actionExecuted) {
        // Failed move still counts as consecutive
        this.consecutiveMoves++
      } else {
        this.consecutiveMoves = 0
        this.lastMoveCell = -1
      }

      // Sell penalty: penalize selling evolved (2+ star) units
      const isSell = action >= TrainingAction.SELL_0 && action <= TrainingAction.SELL_0 + 31
      if (isSell && actionExecuted) {
        // Check if the unit that was just sold had 2+ stars
        // (unit is gone now, but we can check from pre-sell snapshot)
        // We'll detect this from the sell action itself
        reward += this.lastSoldStars >= 2 ? REWARD_SELL_EVOLVED : 0
      }

      // Level-up reward: only when board is reasonably filled
      // Prevents "level to 9 with 3 units" — no reward if more than 2 slots empty
      if (action === TrainingAction.LEVEL_UP && actionExecuted) {
        const maxTeamSize = getMaxTeamSize(
          agent.experienceManager.level,
          this.state.specialGameRule
        )
        if (agent.boardSize >= maxTeamSize - 2) {
          reward += REWARD_LEVEL_UP
        }
      }

      // Reroll reward: unconditional incentive to refresh shop (boosted late game)
      if (action === TrainingAction.REFRESH && actionExecuted) {
        reward += lateGame ? REWARD_REROLL_LATEGAME : REWARD_REROLL
      }

      // Per-step bonus for keeping unique/legendary units on board
      agent.board.forEach((pokemon) => {
        if (pokemon.positionY > 0) {
          const rarity = getPokemonData(pokemon.name).rarity
          if (rarity === Rarity.UNIQUE) reward += REWARD_KEEP_UNIQUE
          else if (rarity === Rarity.LEGENDARY) reward += REWARD_KEEP_LEGENDARY
        }
      })

      // If agent just picked a proposition, check if we need to advance from stage 0
      if (
        actionExecuted &&
        action >= TrainingAction.PICK_0 &&
        action <= TrainingAction.PICK_0 + 5
      ) {
        // After picking, if still at stage 0, advance to stage 1
        if (this.state.stageLevel === 0) {
          this.state.stageLevel = 1
          this.state.botManager.updateBots()
          this.state.shop.assignShop(agent, false, this.state)
          this.actionsThisTurn = 0
          this.consecutiveMoves = 0
          this.lastMoveCell = -1
          this.lockShopUsedThisTurn = false
        }
        // If at a later stage with propositions (uniques, additionals), just continue the turn
        // The propositions have been cleared so normal PICK actions resume
        return {
          observation: this.getObservation(),
          reward,
          done: false,
          info: this.getInfo()
        }
      }

      // Check if turn should end
      const shouldEndTurn =
        action === TrainingAction.END_TURN ||
        this.actionsThisTurn >= TRAINING_MAX_ACTIONS_PER_TURN

      if (shouldEndTurn) {
        // Safety: if propositions are still pending at turn end, auto-pick randomly
        if (agent.pokemonsProposition.length > 0 || agent.itemsProposition.length > 0) {
          this.autoPickForAgent(agent)
        }

        // Auto-place pokemon on board if there's room
        if (TRAINING_AUTO_PLACE) {
          this.autoPlaceTeam(agent)
        }

        // Auto-equip held items onto board units before combat
        this.autoEquipItems(agent)

        // 7.1: Bench penalty — penalize units left on bench when board has open slots
        if (!TRAINING_AUTO_PLACE) {
          const maxTeamSize = getMaxTeamSize(
            agent.experienceManager.level,
            this.state.specialGameRule
          )
          if (agent.boardSize < maxTeamSize) {
            let benchCount = 0
            agent.board.forEach((p) => {
              if (isOnBench(p)) benchCount++
            })
            if (benchCount > 0) {
              const openSlots = maxTeamSize - agent.boardSize
              const penaltyUnits = Math.min(benchCount, openSlots)
              reward += penaltyUnits * REWARD_BENCH_PENALTY
            }
          }
        }

        // Run the fight phase synchronously
        // Issue #2: runFightPhase returns per-player rewards so all players
        // get their combat reward, not just the player who triggered the fight.
        const fightRewards = this.runFightPhase()
        reward += fightRewards.get(this.agentId) ?? 0

        // Check game end
        if (this.state.gameFinished || !agent.alive) {
          const finalReward = this.computeFinalReward(agent)
          return {
            observation: this.getObservation(),
            reward: reward + finalReward,
            done: true,
            info: this.getInfo()
          }
        }

        // Advance to next PICK phase
        this.advanceToNextPickPhase()
        this.actionsThisTurn = 0
        this.consecutiveMoves = 0
        this.lastMoveCell = -1
        this.lockShopUsedThisTurn = false
      }
    }

    return {
      observation: this.getObservation(),
      reward,
      done: false,
      info: this.getInfo()
    }
  }

  private executeAction(action: number, agent: Player): boolean {
    // If agent has pokemon propositions pending, only allow PICK actions
    if (agent.pokemonsProposition.length > 0) {
      if (action >= TrainingAction.PICK_0 && action <= TrainingAction.PICK_0 + 5) {
        return this.pickProposition(agent, action - TrainingAction.PICK_0)
      }
      return false // no other actions allowed during proposition phase
    }

    // Item-only propositions (item carousel, PVE rewards)
    if (agent.itemsProposition.length > 0) {
      if (action >= TrainingAction.PICK_0 && action <= TrainingAction.PICK_0 + 5) {
        return this.pickItemProposition(agent, action - TrainingAction.PICK_0)
      }
      return false // no other actions allowed during item proposition phase
    }

    // BUY_0..BUY_5 (0-5)
    if (action >= TrainingAction.BUY_0 && action <= TrainingAction.BUY_5) {
      return this.buyPokemon(agent, action - TrainingAction.BUY_0)
    }
    // REFRESH (6)
    if (action === TrainingAction.REFRESH) return this.rerollShop(agent)
    // LEVEL_UP (7)
    if (action === TrainingAction.LEVEL_UP) return this.levelUp(agent)
    // LOCK_SHOP (8)
    if (action === TrainingAction.LOCK_SHOP) {
      agent.shopLocked = !agent.shopLocked
      this.lockShopUsedThisTurn = true
      return true
    }
    // END_TURN (9)
    if (action === TrainingAction.END_TURN) return true
    // MOVE_0..MOVE_31 (10-41)
    if (action >= TrainingAction.MOVE_0 && action <= TrainingAction.MOVE_0 + 31) {
      const [x, y] = cellToXY(action - TrainingAction.MOVE_0)
      return this.moveUnitToCell(agent, x, y)
    }
    // SELL_0..SELL_31 (42-73)
    if (action >= TrainingAction.SELL_0 && action <= TrainingAction.SELL_0 + 31) {
      const [x, y] = cellToXY(action - TrainingAction.SELL_0)
      return this.sellPokemonAtCell(agent, x, y)
    }
    // REMOVE_SHOP_0..5 (74-79)
    if (action >= TrainingAction.REMOVE_SHOP_0 && action <= TrainingAction.REMOVE_SHOP_0 + 5) {
      return this.removeFromShop(agent, action - TrainingAction.REMOVE_SHOP_0)
    }
    // PICK_0..PICK_5 (80-85)
    if (action >= TrainingAction.PICK_0 && action <= TrainingAction.PICK_0 + 5) {
      return this.pickProposition(agent, action - TrainingAction.PICK_0)
    }
    // COMBINE_0..5 (86-91)
    if (action >= TrainingAction.COMBINE_0 && action <= TrainingAction.COMBINE_0 + 5) {
      return this.combineItems(agent, action - TrainingAction.COMBINE_0)
    }

    return false
  }

  /**
   * Pick a pokemon proposition for the agent.
   * Replicates the logic from GameRoom.pickPokemonProposition().
   */
  private pickProposition(player: Player, propositionIndex: number): boolean {
    if (player.pokemonsProposition.length === 0) return false
    if (propositionIndex >= player.pokemonsProposition.length) return false

    const pkm = player.pokemonsProposition[propositionIndex] as PkmProposition
    if (!pkm) return false

    // Handle duos (e.g., Latios/Latias)
    const pokemonsObtained: Pokemon[] = (
      pkm in PkmDuos ? PkmDuos[pkm as PkmDuo] : [pkm as Pkm]
    ).map((p) => PokemonFactory.createPokemonFromName(p, player))

    const pokemon = pokemonsObtained[0]
    const isEvolution =
      pokemon.evolutionRule &&
      pokemon.evolutionRule instanceof CountEvolutionRule &&
      pokemon.evolutionRule.canEvolveIfGettingOne(pokemon, player)

    const freeSpace = getFreeSpaceOnBench(player.board)
    if (freeSpace < pokemonsObtained.length && !isEvolution) return false

    const selectedIndex = propositionIndex
    player.pokemonsProposition.clear()

    // Add to additional pool for additional pick stages
    if (AdditionalPicksStages.includes(this.state.stageLevel)) {
      this.state.shop.addAdditionalPokemon(pkm, this.state)
    }

    // Give corresponding item (starters and additional picks)
    if (
      AdditionalPicksStages.includes(this.state.stageLevel) ||
      this.state.stageLevel <= 1
    ) {
      if (player.itemsProposition.length > 0) {
        const selectedItem = player.itemsProposition[selectedIndex]
        if (selectedItem != null) {
          player.items.push(selectedItem)
          this.autoEquipItems(player)
        }
        player.itemsProposition.clear()
      }
    }

    // Track first partner for stage 0
    if (this.state.stageLevel <= 1) {
      player.firstPartner = pokemonsObtained[0].name
    }

    // Place all obtained pokemon on bench
    pokemonsObtained.forEach((pkmn) => {
      const freeCellX = getFirstAvailablePositionInBench(player.board)
      if (isEvolution) {
        pkmn.positionX = freeCellX ?? -1
        pkmn.positionY = 0
        player.board.set(pkmn.id, pkmn)
        pkmn.onAcquired(player)
        this.room.checkEvolutionsAfterPokemonAcquired(player.id)
      } else if (freeCellX !== null) {
        pkmn.positionX = freeCellX
        pkmn.positionY = 0
        player.board.set(pkmn.id, pkmn)
        pkmn.onAcquired(player)
      } else {
        // No space — sell for money
        const sellPrice = getSellPrice(pkmn, this.state.specialGameRule)
        player.addMoney(sellPrice, true, null)
      }
    })

    this.invalidatePlayerCaches(player.id)
    return true
  }

  /**
   * Pick an item from item-only propositions (item carousel, PVE rewards).
   * When pokemonsProposition is empty but itemsProposition has items.
   */
  private pickItemProposition(player: Player, propositionIndex: number): boolean {
    if (player.itemsProposition.length === 0) return false
    if (propositionIndex >= player.itemsProposition.length) return false

    const item = player.itemsProposition[propositionIndex]
    if (item == null) return false

    player.items.push(item)
    this.autoEquipItems(player)
    player.itemsProposition.clear()
    this.invalidatePlayerCaches(player.id)
    return true
  }

  private buyPokemon(player: Player, shopIndex: number): boolean {
    const name = player.shop[shopIndex]
    if (!name || name === Pkm.DEFAULT) return false

    const pokemon = PokemonFactory.createPokemonFromName(name, player)
    const isEvolution =
      pokemon.evolutionRule &&
      pokemon.evolutionRule instanceof CountEvolutionRule &&
      pokemon.evolutionRule.canEvolveIfGettingOne(pokemon, player)

    const cost = getBuyPrice(name, this.state.specialGameRule)
    const freeSpace = getFreeSpaceOnBench(player.board)
    const hasSpace = freeSpace > 0 || isEvolution

    if (player.money < cost || !hasSpace) return false

    player.money -= cost
    const x = getFirstAvailablePositionInBench(player.board)
    pokemon.positionX = x !== null ? x : -1
    pokemon.positionY = 0
    player.board.set(pokemon.id, pokemon)
    pokemon.onAcquired(player)
    player.shop[shopIndex] = Pkm.DEFAULT

    this.room.checkEvolutionsAfterPokemonAcquired(this.agentId)
    this.invalidatePlayerCaches(player.id)
    return true
  }

  private sellPokemonAtCell(player: Player, x: number, y: number): boolean {
    const targetPokemon = this.findPokemonAt(player, x, y)
    if (!targetPokemon) return false

    let targetId: string | null = null
    player.board.forEach((pokemon, key) => {
      if (pokemon === targetPokemon) targetId = key
    })
    if (!targetId) return false

    // Track stars for sell penalty reward
    this.lastSoldStars = targetPokemon.stars ?? 1

    player.board.delete(targetId)
    this.state.shop.releasePokemon(targetPokemon.name, player, this.state)

    const sellPrice = getSellPrice(targetPokemon, this.state.specialGameRule)
    player.addMoney(sellPrice, false, null)
    targetPokemon.items.forEach((it: Item) => {
      player.items.push(it)
    })
    this.autoEquipItems(player)

    player.updateSynergies()
    player.boardSize = this.room.getTeamSize(player.board)
    this.invalidatePlayerCaches(player.id)
    return true
  }

  private rerollShop(player: Player): boolean {
    const rollCost = player.shopFreeRolls > 0 ? 0 : 1
    if (player.money < rollCost) return false

    player.rerollCount++
    player.money -= rollCost
    if (player.shopFreeRolls > 0) {
      player.shopFreeRolls--
    }
    this.state.shop.assignShop(player, true, this.state)
    this.invalidatePlayerCaches(player.id)
    return true
  }

  private levelUp(player: Player): boolean {
    const cost = 4 // standard level up cost
    if (player.money < cost || !player.experienceManager.canLevelUp())
      return false

    player.addExperience(4)
    player.money -= cost
    this.invalidatePlayerCaches(player.id)
    return true
  }

  /**
   * Find the "first available unit" by scanning bench left-to-right,
   * then board rows top-to-bottom. Used by MOVE actions.
   */
  private findFirstAvailableUnit(player: Player): Pokemon | null {
    // Scan bench left-to-right (y=0, x=0..7)
    for (let x = 0; x < GRID_WIDTH; x++) {
      const pokemon = this.findPokemonAt(player, x, 0)
      if (pokemon) return pokemon
    }
    // Scan board row by row (y=1..3, x=0..7)
    for (let y = 1; y < GRID_HEIGHT; y++) {
      for (let x = 0; x < GRID_WIDTH; x++) {
        const pokemon = this.findPokemonAt(player, x, y)
        if (pokemon) return pokemon
      }
    }
    return null
  }

  private moveUnitToCell(player: Player, targetX: number, targetY: number): boolean {
    // Target must be empty
    if (this.findPokemonAt(player, targetX, targetY)) return false

    const pokemon = this.findFirstAvailableUnit(player)
    if (!pokemon) return false

    const sourceY = pokemon.positionY
    const maxTeamSize = getMaxTeamSize(
      player.experienceManager.level,
      this.state.specialGameRule
    )

    // Moving bench → board: check board not full
    if (targetY >= 1 && sourceY === 0) {
      if (player.boardSize >= maxTeamSize) return false
    }

    pokemon.positionX = targetX
    pokemon.positionY = targetY

    // Update board size if crossing bench/board boundary
    if (sourceY === 0 && targetY >= 1) {
      player.boardSize++
    } else if (sourceY >= 1 && targetY === 0) {
      player.boardSize--
    }

    if (typeof pokemon.onChangePosition === "function") {
      pokemon.onChangePosition(targetX, targetY, player, this.state)
    }
    player.updateSynergies()
    this.invalidatePlayerCaches(player.id)
    return true
  }

  private removeFromShop(player: Player, shopIndex: number): boolean {
    const name = player.shop[shopIndex]
    if (!name || name === Pkm.DEFAULT) return false

    const cost = getBuyPrice(name, this.state.specialGameRule)
    if (player.money < cost) return false

    // DO NOT deduct gold — gold is a gate check only (matches real game)
    player.shop[shopIndex] = Pkm.DEFAULT
    player.shopLocked = true
    this.state.shop.releasePokemon(name, player, this.state)
    this.invalidatePlayerCaches(player.id)
    return true
  }

  private combineItems(player: Player, pairIndex: number): boolean {
    const items = Array.from(player.items.values()) as Item[]
    const pairs = this.getCachedItemPairs(player.id, items)
    if (pairIndex >= pairs.length) return false

    const [i, j] = pairs[pairIndex]
    const itemA = items[i]
    const itemB = items[j]
    const result = findRecipeResult(itemA, itemB)
    if (!result) return false

    // Remove higher index first to avoid shifting
    player.items.splice(j, 1)
    player.items.splice(i, 1)
    player.items.push(result)
    this.autoEquipItems(player)
    this.invalidatePlayerCaches(player.id)
    return true
  }

  /**
   * Auto-equip all held items onto board units.
   * Prioritizes units on the board (y>=1) over bench (y=0),
   * and within each group, units with fewer items first.
   */
  private autoEquipItems(player: Player): void {
    if (player.items.length === 0) return

    // Get all units that can hold items, sorted: board first, then by fewest items
    const units = values(player.board)
      .filter((p) => p.canHoldItems && p.items.size < 3)
      .sort((a, b) => {
        // Board units (y>=1) before bench units (y=0)
        const aOnBoard = a.positionY >= 1 ? 0 : 1
        const bOnBoard = b.positionY >= 1 ? 0 : 1
        if (aOnBoard !== bOnBoard) return aOnBoard - bOnBoard
        // Fewer items first (spread items across units)
        return a.items.size - b.items.size
      })

    if (units.length === 0) return

    // Take all items out of player inventory, try to equip each
    const itemsToEquip = Array.from(player.items.values()) as Item[]
    const remaining: Item[] = []

    for (const item of itemsToEquip) {
      // Re-sort to account for items we just added
      units.sort((a, b) => {
        const aOnBoard = a.positionY >= 1 ? 0 : 1
        const bOnBoard = b.positionY >= 1 ? 0 : 1
        if (aOnBoard !== bOnBoard) return aOnBoard - bOnBoard
        return a.items.size - b.items.size
      })

      // Find first unit with room
      const target = units.find((u) => u.items.size < 3)
      if (!target) {
        remaining.push(item)
        continue
      }

      // Item combination: if incoming item is a basic component and the unit
      // already has a basic component, combine them into a crafted item
      // (matches real game behavior from OnDragDropItemCommand)
      const isBasic = ItemComponents.includes(item)
      if (isBasic) {
        const existingComponent = Array.from(target.items.values()).find(
          (i) => ItemComponents.includes(i as Item)
        ) as Item | undefined
        if (existingComponent) {
          const crafted = findRecipeResult(item, existingComponent)
          if (crafted) {
            target.items.delete(existingComponent)
            target.addItem(crafted, player)
            continue
          }
        }
      }

      target.addItem(item, player)
    }

    // Replace player inventory with only unequipped items
    player.items.clear()
    remaining.forEach((item) => player.items.push(item))
  }

  private autoPlaceTeam(player: Player): void {
    const teamSize = this.room.getTeamSize(player.board)
    const maxTeamSize = getMaxTeamSize(
      player.experienceManager.level,
      this.state.specialGameRule
    )

    if (teamSize < maxTeamSize) {
      const numToPlace = maxTeamSize - teamSize
      for (let i = 0; i < numToPlace; i++) {
        const pokemon = values(player.board)
          .filter((p) => isOnBench(p) && p.canBePlaced)
          .sort((a, b) => a.positionX - b.positionX)[0]

        if (pokemon) {
          const coords = getFirstAvailablePositionOnBoard(
            player.board,
            pokemon.types.has(Synergy.DARK) && pokemon.range === 1
              ? 3
              : pokemon.range
          )
          if (coords) {
            pokemon.positionX = coords[0]
            pokemon.positionY = coords[1]
            pokemon.onChangePosition(
              coords[0],
              coords[1],
              player,
              this.state
            )
          }
        }
      }
      player.updateSynergies()
      player.boardSize = this.room.getTeamSize(player.board)
      this.invalidatePlayerCaches(player.id)
    }
  }

  /**
   * Runs the fight phase synchronously and returns per-player rewards.
   *
   * Issue #2 (reward attribution): Returns a Map of playerId → reward so that
   * ALL players receive their combat reward after fights, not just the player
   * whose END_TURN action happened to trigger the fight.
   */
  private runFightPhase(): Map<string, number> {
    this.state.phase = GamePhaseState.FIGHT
    this.state.time = FIGHTING_PHASE_DURATION
    this.state.simulations.clear()

    // Update dummy bot teams before combat (only in non-self-play mode)
    if (!SELF_PLAY && this.cachedBots.length === 0) {
      this.updateDummyBotTeams()
    }

    // Register played pokemons
    this.state.players.forEach((player) => {
      if (player.alive) player.registerPlayedPokemons()
    })

    const isPVE = this.state.stageLevel in PVEStages

    if (isPVE) {
      // PVE battle
      const pveStage = PVEStages[this.state.stageLevel]
      if (pveStage) {
        this.state.players.forEach((player) => {
          if (!player.alive) return
          player.opponentId = "pve"
          player.opponentName = pveStage.name
          player.opponentAvatar = getAvatarString(
            PkmIndex[pveStage.avatar],
            false,
            pveStage.emotion
          )
          player.opponentTitle = "WILD"
          player.team = Team.BLUE_TEAM

          const pveBoard = PokemonFactory.makePveBoard(
            pveStage,
            false,
            this.state.townEncounter
          )
          const weather = getWeather(player, null, pveBoard)
          const simulation = new Simulation(
            nanoid(),
            this.room as any,
            player.board,
            pveBoard,
            player,
            undefined,
            this.state.stageLevel,
            weather
          )
          player.simulationId = simulation.id
          this.state.simulations.set(simulation.id, simulation)
          simulation.start()
        })
      }
    } else {
      // PVP battles
      const matchups = selectMatchups(this.state)
      matchups.forEach((matchup) => {
        const { bluePlayer, redPlayer, ghost } = matchup
        const weather = getWeather(
          bluePlayer,
          redPlayer,
          redPlayer.board,
          ghost
        )
        const simulationId = nanoid()

        bluePlayer.simulationId = simulationId
        bluePlayer.team = Team.BLUE_TEAM
        bluePlayer.opponents.set(
          redPlayer.id,
          (bluePlayer.opponents.get(redPlayer.id) ?? 0) + 1
        )
        bluePlayer.opponentId = redPlayer.id
        bluePlayer.opponentName = ghost
          ? `Ghost of ${redPlayer.name}`
          : redPlayer.name
        bluePlayer.opponentAvatar = redPlayer.avatar
        bluePlayer.opponentTitle = redPlayer.title ?? ""

        if (!ghost) {
          redPlayer.simulationId = simulationId
          redPlayer.team = Team.RED_TEAM
          redPlayer.opponents.set(
            bluePlayer.id,
            (redPlayer.opponents.get(bluePlayer.id) ?? 0) + 1
          )
          redPlayer.opponentId = bluePlayer.id
          redPlayer.opponentName = bluePlayer.name
          redPlayer.opponentAvatar = bluePlayer.avatar
          redPlayer.opponentTitle = bluePlayer.title ?? ""
        }

        const simulation = new Simulation(
          simulationId,
          this.room as any,
          bluePlayer.board,
          redPlayer.board,
          bluePlayer,
          redPlayer,
          this.state.stageLevel,
          weather,
          ghost
        )
        this.state.simulations.set(simulation.id, simulation)
        simulation.start()
      })
    }

    // Capture initial enemy team sizes for kill counting (6.3)
    const initialEnemySizes = new Map<string, number>()
    this.state.players.forEach((player) => {
      if (!player.alive) return
      const sim = this.state.simulations.get(player.simulationId)
      if (!sim) return
      const enemyTeam = player.team === Team.BLUE_TEAM ? sim.redTeam : sim.blueTeam
      initialEnemySizes.set(player.id, enemyTeam.size)
    })

    // Run all simulations synchronously
    let steps = 0
    let allFinished = false
    const wallClockStart = Date.now()
    const WALL_CLOCK_LIMIT_MS = 15_000 // 15s hard cap — well under the 30s HTTP timeout
    while (!allFinished && steps < TRAINING_MAX_FIGHT_STEPS) {
      if (steps % 100 === 0 && Date.now() - wallClockStart > WALL_CLOCK_LIMIT_MS) {
        console.error(
          `[Training] Simulation wall-clock timeout after ${steps} steps ` +
          `(${Date.now() - wallClockStart}ms), forcing all to DRAW`
        )
        this.state.simulations.forEach((simulation) => {
          if (!simulation.finished) {
            simulation.finished = true
            simulation.winnerId = ""
          }
        })
        break
      }
      allFinished = true
      this.state.simulations.forEach((simulation) => {
        if (!simulation.finished) {
          try {
            simulation.update(TRAINING_SIMULATION_DT)
          } catch (err) {
            console.error(
              `[Training] Simulation ${simulation.id} threw during update (step ${steps}), forcing DRAW:`,
              err
            )
            simulation.finished = true
            simulation.winnerId = ""
          }
          if (!simulation.finished) {
            allFinished = false
          }
        }
      })
      steps++
    }

    // Force-finish any remaining simulations (split from stop for data extraction)
    this.state.simulations.forEach((simulation) => {
      if (!simulation.finished) {
        simulation.onFinish()
      }
    })

    // Extract enemy kill counts BEFORE stop() clears teams (6.3)
    const enemyKills = new Map<string, number>()
    this.state.players.forEach((player) => {
      if (!player.alive) return
      const sim = this.state.simulations.get(player.simulationId)
      if (!sim) return
      const enemyTeam = player.team === Team.BLUE_TEAM ? sim.redTeam : sim.blueTeam
      const initial = initialEnemySizes.get(player.id) ?? 0
      const surviving = enemyTeam.size
      enemyKills.set(player.id, Math.max(0, initial - surviving))
    })

    // Now stop all simulations (clears teams)
    this.state.simulations.forEach((simulation) => {
      simulation.stop()
    })

    // Compute streak
    if (!isPVE) {
      this.state.players.forEach((player) => {
        if (!player.alive) return
        const [prev, last] = player.history
          .filter(
            (s) => s.id !== "pve" && s.result !== BattleResult.DRAW
          )
          .map((s) => s.result)
          .slice(-2)
        if (last === BattleResult.DRAW) {
          // preserve streak
        } else if (last !== prev) {
          player.streak = 0
        } else {
          player.streak += 1
        }
      })
    }

    // Check death
    this.state.players.forEach((player) => {
      if (player.life <= 0 && player.alive) {
        player.alive = false
        player.spectatedPlayerId = player.id
      }
    })

    // Check end game
    const playersAlive = values(this.state.players).filter((p) => p.alive)
    if (playersAlive.length <= 1) {
      this.state.gameFinished = true
    }

    // Issue #2: Compute per-player rewards so ALL players get their combat reward.
    // In single-agent mode only the agent's reward matters, but we compute for all
    // to support self-play where all 8 players need rewards attributed correctly.
    const rewards = new Map<string, number>()
    this.state.players.forEach((player, id) => {
      if (!player.alive) {
        rewards.set(id, 0)
        return
      }
      const lastHistory = player.history.at(-1)
      if (lastHistory) {
        const result = lastHistory.result as BattleResult
        if (result === BattleResult.WIN) {
          rewards.set(id, REWARD_PER_WIN)
        } else if (result === BattleResult.DEFEAT) {
          rewards.set(id, REWARD_PER_LOSS)
        } else {
          rewards.set(id, REWARD_PER_DRAW)
        }
      } else {
        rewards.set(id, 0)
      }
    })

    // Survival bonus: every alive player gets a flat bonus each round
    this.state.players.forEach((player, id) => {
      if (!player.alive) return
      rewards.set(id, (rewards.get(id) ?? 0) + REWARD_PER_SURVIVE_ROUND)
    })

    // ── Shaped rewards (Phase 6) ──────────────────────────────────────

    // 6.2: Synergy activation delta — reward positive synergy growth
    this.state.players.forEach((player, id) => {
      if (!player.alive || player.isBot) return
      const currentThresholds = this.countActiveSynergyThresholds(player)
      if (id === this.agentId) {
        const delta = currentThresholds - this.prevActiveSynergyCount
        if (delta > 0) {
          rewards.set(id, (rewards.get(id) ?? 0) + delta * REWARD_SYNERGY_THRESHOLD)
        }
      }
    })

    // 6.2b: Sustained synergy reward — ongoing bonus for maintaining strong comps.
    // Uses real SynergyTriggers breakpoints per synergy type. Higher tiers and
    // stacking multiple synergies at tier 2+ earn a multiplicative bonus.
    this.state.players.forEach((player, id) => {
      if (!player.alive || player.isBot) return
      if (id === this.agentId) {
        const { totalTiers, qualifyingCount } = this.computeSynergyTierInfo(player)
        if (totalTiers > 0) {
          const basePoints = totalTiers * REWARD_SYNERGY_SUSTAINED
          const multiplier = Math.min(
            1.0 + qualifyingCount * REWARD_SYNERGY_MULTI_BONUS,
            REWARD_SYNERGY_MULTI_CAP
          )
          rewards.set(id, (rewards.get(id) ?? 0) + basePoints * multiplier)
        }
      }
    })

    // 6.3: Combat damage — reward enemy kills
    this.state.players.forEach((player, id) => {
      if (!player.alive || player.isBot) return
      const kills = enemyKills.get(id) ?? 0
      if (kills > 0) {
        rewards.set(id, (rewards.get(id) ?? 0) + kills * REWARD_PER_ENEMY_KILL)
      }
    })

    // 6.4: HP preservation — small bonus for winning with high HP
    this.state.players.forEach((player, id) => {
      if (!player.alive || player.isBot) return
      const lastHistory = player.history.at(-1)
      if (lastHistory?.result === BattleResult.WIN) {
        rewards.set(id, (rewards.get(id) ?? 0) + (player.life / 100) * REWARD_HP_SCALE)
      }
    })

    // 6.5: Gold hoarding penalty — discourage holding gold beyond interest cap.
    // Before stage 21: flat -0.04 per gold above 70.
    // After stage 21 (tiered): >50g -0.01, >60g -0.04, >70g -0.07 per gold in each bracket.
    // Applied before income so it reflects the agent's spending decisions.
    this.state.players.forEach((player, id) => {
      if (!player.alive || player.isBot) return
      const gold = player.money
      let penalty = 0
      if (this.state.stageLevel >= GOLD_LATEGAME_STAGE) {
        // Late game: tiered brackets (each tier only penalizes gold within its range)
        if (gold > GOLD_EXCESS_THRESHOLD) {
          penalty += (gold - GOLD_EXCESS_THRESHOLD) * REWARD_GOLD_LATEGAME_TIER3
        }
        if (gold > GOLD_LATEGAME_TIER2_THRESHOLD) {
          penalty += (Math.min(gold, GOLD_EXCESS_THRESHOLD) - GOLD_LATEGAME_TIER2_THRESHOLD) * REWARD_GOLD_LATEGAME_TIER2
        }
        if (gold > GOLD_LATEGAME_TIER1_THRESHOLD) {
          penalty += (Math.min(gold, GOLD_LATEGAME_TIER2_THRESHOLD) - GOLD_LATEGAME_TIER1_THRESHOLD) * REWARD_GOLD_LATEGAME_TIER1
        }
      } else {
        // Early game: flat penalty above 70
        if (gold > GOLD_EXCESS_THRESHOLD) {
          penalty += (gold - GOLD_EXCESS_THRESHOLD) * REWARD_GOLD_EXCESS_PENALTY
        }
      }
      if (penalty < 0) {
        rewards.set(id, (rewards.get(id) ?? 0) + penalty)
      }
    })

    // 6.6: Low-gold penalty — teach the agent to save toward interest thresholds.
    // The minimum gold target ramps up with stage progression (no penalty before stage 5).
    this.state.players.forEach((player, id) => {
      if (!player.alive || player.isBot) return
      const target = GOLD_MIN_TARGETS.find(
        ([stage]) => this.state.stageLevel >= stage
      )
      if (target) {
        const deficit = target[1] - player.money
        if (deficit > 0) {
          rewards.set(id, (rewards.get(id) ?? 0) + deficit * REWARD_GOLD_LOW_PENALTY)
        }
      }
    })

    // Track agent's last battle result for logging
    const agent = this.state.players.get(this.agentId)
    if (agent) {
      const lastHistory = agent.history.at(-1)
      if (lastHistory) {
        this.lastBattleResult = lastHistory.result as BattleResult
      }
    }

    // PVE reward handling: auto-give direct rewards, set propositions for picks
    if (isPVE) {
      const pveStage = PVEStages[this.state.stageLevel]
      if (pveStage) {
        this.state.players.forEach((player) => {
          if (!player.alive) return
          const lastHist = player.history.at(-1)
          if (lastHist?.result !== BattleResult.WIN) return

          // Auto-give direct rewards (getRewards)
          const directRewards = pveStage.getRewards?.(player) ?? []
          directRewards.forEach((item) => player.items.push(item))
          if (directRewards.length > 0) this.autoEquipItems(player)

          // Set reward propositions for agent to pick (getRewardsPropositions)
          const rewardPropositions = pveStage.getRewardsPropositions?.(player) ?? []
          if (rewardPropositions.length > 0 && !player.isBot) {
            rewardPropositions.forEach((item) => player.itemsProposition.push(item))
          } else if (rewardPropositions.length > 0 && player.isBot) {
            // Bots auto-pick a random reward proposition
            player.items.push(pickRandomIn(rewardPropositions))
            this.autoEquipItems(player)
          }
        })
      }
    }

    if (!this.state.gameFinished) {
      this.state.stageLevel += 1

      // Compute income
      this.state.players.forEach((player) => {
        if (!player.alive) return
        let income = 0
        player.interest = max(player.maxInterest)(
          Math.floor(player.money / 10)
        )
        income += player.interest
        if (!isPVE) income += max(5)(player.streak)
        income += 5
        player.addMoney(income, true, null)
        player.addExperience(2)

        // 6.1: Interest bonus with board guard — must field nearly full team
        if (!player.isBot && player.interest > 0) {
          const maxTeam = getMaxTeamSize(
            player.experienceManager.level,
            this.state.specialGameRule
          )
          if (player.boardSize >= maxTeam - 2) {
            rewards.set(
              player.id,
              (rewards.get(player.id) ?? 0) + player.interest * REWARD_INTEREST_BONUS
            )
          }
        }
      })

      // Update bot levels
      this.state.players.forEach((player) => {
        if (player.isBot && player.alive) {
          player.experienceManager.level = max(9)(
            Math.round(this.state.stageLevel / 2)
          )
        }
      })

      // Update bots
      this.state.botManager.updateBots()
    }

    // Rank players
    this.room.rankPlayers()

    // 7.2: Invalidate all caches after fight (board state changed)
    this.state.players.forEach((_player, id) => {
      this.invalidatePlayerCaches(id)
    })

    return rewards
  }

  /**
   * After fight, advance to the next PICK phase.
   * Handles all skipped TOWN-phase events:
   *   - Portal carousel stages (0, 10, 20): unique/legendary pokemon propositions
   *   - Item carousel stages (4, 12, 17, 22, 27, 34): random item components
   *   - Additional pick stages (5, 8, 11): uncommon/rare/epic pokemon propositions
   */
  private advanceToNextPickPhase(): void {
    // Handle portal carousel stages (10 = uniques, 20 = legendaries)
    // Stage 0 is handled in reset() separately
    if (
      PortalCarouselStages.includes(this.state.stageLevel) &&
      this.state.stageLevel > 0
    ) {
      // Agent gets propositions to choose from via PICK actions
      this.state.players.forEach((player) => {
        if (!player.isBot) {
          this.state.shop.assignUniquePropositions(player, this.state, [])
        }
      })
      // Only auto-pick for bots; agent propositions stay pending
      this.autoPickPropositionsForBots()
    }

    // Handle item carousel stages — present 3 random items as propositions
    if (ItemCarouselStages.includes(this.state.stageLevel)) {
      this.state.players.forEach((player) => {
        if (!player.isBot && player.alive) {
          const itemPool =
            this.state.stageLevel >= 20
              ? CraftableItemsNoScarves
              : ItemComponentsNoFossilOrScarf
          const choices = pickNRandomIn(itemPool, 3)
          choices.forEach((item) => player.itemsProposition.push(item))
        }
      })
    }

    // Refresh shop for all RL agents (non-bots), not just a single agent.
    // In self-play mode all 8 players need shops; in single-agent mode only the agent does.
    this.state.players.forEach((player) => {
      if (
        !player.isBot &&
        player.alive &&
        player.pokemonsProposition.length === 0
      ) {
        if (!player.shopLocked) {
          // Full refresh: release all units and assign new shop
          this.state.shop.assignShop(player, false, this.state)
        } else {
          // Locked shop (from REMOVE_SHOP or manual LOCK_SHOP):
          // Refill only the empty (DEFAULT) slots, keep the units the
          // player wanted to save. Then unlock for next round.
          // This matches the real game's game-commands.ts behavior.
          this.state.shop.refillShop(player, this.state)
          player.shopLocked = false
        }
      }
    })

    // Set up PICK phase
    this.state.phase = GamePhaseState.PICK
    this.state.time =
      (StageDuration[this.state.stageLevel] ?? StageDuration.DEFAULT) * 1000

    // Populate additional pick propositions at stages 5, 8, 11
    if (AdditionalPicksStages.includes(this.state.stageLevel)) {
      const pool =
        this.state.stageLevel === AdditionalPicksStages[0]
          ? this.additionalUncommonPool
          : this.state.stageLevel === AdditionalPicksStages[1]
            ? this.additionalRarePool
            : this.additionalEpicPool

      // Agent gets propositions to choose from
      this.state.players.forEach((player) => {
        if (!player.isBot) {
          const items = pickNRandomIn(ItemComponentsNoScarf, 3)
          for (let i = 0; i < 3; i++) {
            const p = pool.pop()
            if (p) {
              player.pokemonsProposition.push(p)
              player.itemsProposition.push(items[i])
            }
          }
        }
      })

      // Remaining picks go into shared pool
      const remainingPicks = 8 - values(this.state.players).filter(
        (p) => !p.isBot
      ).length
      for (let i = 0; i < remainingPicks; i++) {
        const p = pool.pop()
        if (p) {
          this.state.shop.addAdditionalPokemon(p, this.state)
        }
      }
    }

    // Auto-pick for bots only; agent propositions stay pending for PICK actions
    this.autoPickPropositionsForBots()

    // 7.2: Invalidate all caches after phase transition (shops, propositions changed)
    this.state.players.forEach((_player, id) => {
      this.invalidatePlayerCaches(id)
    })

    // Snapshot synergy count at start of pick phase for delta reward (6.2)
    const agentForSynergy = this.state.players.get(this.agentId)
    if (agentForSynergy) {
      this.prevActiveSynergyCount = this.countActiveSynergyThresholds(agentForSynergy)
    }
  }

  /**
   * Auto-pick pokemon and item propositions for bot players only.
   * The RL agent picks via PICK actions instead.
   * Bots pick randomly, creates the pokemon, places on bench, gives item.
   */
  private autoPickPropositionsForBots(): void {
    this.state.players.forEach((player) => {
      // Skip RL agents — they pick via PICK actions.
      // In self-play mode all players are non-bot, so this is a no-op.
      if (!player.isBot) return
      if (player.pokemonsProposition.length > 0) {
        const propositions = values(player.pokemonsProposition)
        const pick = pickRandomIn(propositions) as Pkm
        const selectedIndex = player.pokemonsProposition.indexOf(pick)

        // Create the pokemon and place on bench
        const pokemon = PokemonFactory.createPokemonFromName(pick, player)
        const freeCellX = getFirstAvailablePositionInBench(player.board)
        if (freeCellX !== null) {
          pokemon.positionX = freeCellX
          pokemon.positionY = 0
          player.board.set(pokemon.id, pokemon)
          pokemon.onAcquired(player)
        }

        // Add to shared pool for additional pick stages
        if (AdditionalPicksStages.includes(this.state.stageLevel)) {
          this.state.shop.addAdditionalPokemon(pick, this.state)
        }

        // Give corresponding item
        if (player.itemsProposition.length > 0) {
          const selectedItem = player.itemsProposition[selectedIndex]
          if (selectedItem != null) {
            player.items.push(selectedItem)
            this.autoEquipItems(player)
          }
        }

        player.pokemonsProposition.clear()
        player.itemsProposition.clear()
        this.room.checkEvolutionsAfterPokemonAcquired(player.id)
      } else if (player.itemsProposition.length > 0) {
        // Item-only propositions (PVE rewards, etc.)
        const pick = pickRandomIn(values(player.itemsProposition))
        player.items.push(pick)
        this.autoEquipItems(player)
        player.itemsProposition.clear()
      }
    })
  }

  /**
   * Fallback: auto-pick a random proposition for the agent.
   * Used when turn ends but propositions are still pending (safety cap hit).
   */
  private autoPickForAgent(agent: Player): void {
    if (agent.pokemonsProposition.length > 0) {
      const propositions = values(agent.pokemonsProposition)
      const pick = pickRandomIn(propositions) as Pkm
      const selectedIndex = agent.pokemonsProposition.indexOf(pick)
      this.pickProposition(agent, selectedIndex)
    }
    // If propositions still there (pickProposition failed due to no space), force clear
    if (agent.pokemonsProposition.length > 0) {
      agent.pokemonsProposition.clear()
      agent.itemsProposition.clear()
    }
    // Item-only propositions (item carousel, PVE rewards)
    if (agent.itemsProposition.length > 0) {
      const items = values(agent.itemsProposition)
      const pick = pickRandomIn(items)
      agent.items.push(pick)
      this.autoEquipItems(agent)
      agent.itemsProposition.clear()
    }
  }

  /**
   * Create bot players from cached bot definitions.
   * Bots use the existing BotV2 system with pre-scripted team progressions.
   */
  private createBotPlayers(): void {
    if (this.cachedBots.length > 0) {
      const numBots = Math.min(TRAINING_NUM_OPPONENTS, this.cachedBots.length)
      const selectedBots = shuffleArray([...this.cachedBots]).slice(0, numBots)

      selectedBots.forEach((bot) => {
        const botPlayer = new Player(
          bot.id,
          bot.name,
          bot.elo,
          0,
          bot.avatar,
          true, // isBot
          this.state.players.size + 1,
          new Map(),
          "",
          Role.BOT,
          this.state
        )
        this.state.players.set(bot.id, botPlayer)
        this.state.botManager.addBot(botPlayer)
      })
    } else {
      // No bots from DB — create dummy opponents with random starter teams
      this.createDummyBotPlayers()
    }
  }

  /**
   * Create simple dummy bot players when no MongoDB bots are available.
   * Each bot gets a random common pokemon placed on their board so that
   * combat simulations can run. Bots "level up" by getting additional
   * random pokemon as stages progress (handled in advanceToNextPickPhase).
   */
  private createDummyBotPlayers(): void {
    const commonPool = PRECOMPUTED_POKEMONS_PER_RARITY.COMMON
    for (let i = 0; i < TRAINING_NUM_OPPONENTS; i++) {
      const botId = `dummy-bot-${i}`
      const botPlayer = new Player(
        botId,
        `Bot-${i + 1}`,
        1000,
        0,
        getAvatarString(PkmIndex[Pkm.MAGIKARP], false),
        true,
        this.state.players.size + 1,
        new Map(),
        "",
        Role.BOT,
        this.state
      )
      this.state.players.set(botId, botPlayer)

      // Assign a synergy theme for this bot's team (persists all game)
      this.botSynergies.set(botId, pickRandomIn(SynergyArray))

      // Give the bot a starter pokemon on the board
      const starterPkm = pickRandomIn(commonPool)
      const pokemon = PokemonFactory.createPokemonFromName(starterPkm, botPlayer)
      pokemon.positionX = 3
      pokemon.positionY = 1
      botPlayer.board.set(pokemon.id, pokemon)
      botPlayer.boardSize = 1
    }
  }

  /**
   * Update dummy bot teams to scale with the current stage level.
   * Team sizes and rarity pools mirror real player leveling curves
   * so the agent experiences proper late-game pressure.
   *
   * Each bot is assigned a synergy theme at creation (e.g. WATER, DARK, WILD).
   * All slots prefer pokemon matching that synergy, so bots trigger real
   * synergy bonuses in combat. Falls back to unthemed picks when a
   * rarity+star+synergy combo has no candidates.
   */
  private updateDummyBotTeams(): void {
    const stage = this.state.stageLevel
    const R = PRECOMPUTED_POKEMONS_PER_RARITY

    // Each slot defines a rarity and target star level.
    type Slot = { rarity: keyof typeof R; stars: number }

    // Helper: pick a random pokemon from a rarity pool at the target star level,
    // filtered to the bot's assigned synergy theme. Fallback chain:
    //   1. exact stars + synergy match
    //   2. exact stars (no synergy filter)
    //   3. <= stars + synergy match
    //   4. <= stars (no synergy filter)
    //   5. unfiltered pool (safety net)
    const pickFromPool = (slot: Slot, synergy: Synergy): Pkm => {
      const pool = R[slot.rarity]
      const hasSynergy = (p: Pkm) => getPokemonData(p).types.includes(synergy)

      const exactSyn = pool.filter((p) => getPokemonData(p).stars === slot.stars && hasSynergy(p))
      if (exactSyn.length > 0) return pickRandomIn(exactSyn)

      const exact = pool.filter((p) => getPokemonData(p).stars === slot.stars)
      if (exact.length > 0) return pickRandomIn(exact)

      const fallbackSyn = pool.filter((p) => getPokemonData(p).stars <= slot.stars && hasSynergy(p))
      if (fallbackSyn.length > 0) return pickRandomIn(fallbackSyn)

      const fallback = pool.filter((p) => getPokemonData(p).stars <= slot.stars)
      return pickRandomIn(fallback.length > 0 ? fallback : pool)
    }

    // s(rarity, stars) shorthand for slot definitions
    const s = (rarity: keyof typeof R, stars = 1): Slot => ({ rarity, stars })

    // Per-slot composition by stage bracket.
    // UNIQUE persists from stage 11+, LEGENDARY from stage 21+.
    // Bots carry no items to keep difficulty approachable.
    let slots: Slot[]

    if (stage >= 29) {
      slots = [
        s("UNIQUE",3), s("LEGENDARY",3),
        s("EPIC",2), s("EPIC",2), s("EPIC",2), s("EPIC",2), s("EPIC",2),
        s("ULTRA",2), s("ULTRA",2)
      ]
    } else if (stage >= 26) {
      slots = [
        s("UNIQUE",3), s("LEGENDARY",3),
        s("RARE",3), s("RARE",3), s("EPIC",2), s("EPIC",2), s("ULTRA",2), s("ULTRA",1)
      ]
    } else if (stage >= 23) {
      slots = [
        s("UNIQUE",3), s("LEGENDARY",3),
        s("RARE",2), s("RARE",2), s("EPIC",2), s("EPIC",1), s("EPIC",1), s("ULTRA",1)
      ]
    } else if (stage >= 18) {
      slots = [
        s("UNIQUE",3),
        s("RARE",2), s("RARE",2), s("RARE",2), s("RARE",2), s("EPIC",1), s("EPIC",1)
      ]
    } else if (stage >= 14) {
      slots = [
        s("UNIQUE",3),
        s("UNCOMMON",2), s("UNCOMMON",1), s("RARE",2), s("RARE",1), s("RARE",1), s("EPIC",1)
      ]
    } else if (stage >= 11) {
      slots = [
        s("UNIQUE",3),
        s("COMMON",2), s("COMMON",2), s("UNCOMMON",1), s("RARE",1), s("RARE",1)
      ]
    } else if (stage >= 10) {
      slots = [s("COMMON",2), s("COMMON",2), s("COMMON",1), s("UNCOMMON",1), s("RARE",1)]
    } else if (stage >= 8) {
      slots = [s("COMMON",2), s("COMMON",1), s("UNCOMMON",1), s("RARE",1)]
    } else if (stage >= 5) {
      slots = [s("COMMON",1), s("COMMON",1), s("UNCOMMON",1)]
    } else {
      slots = [s("COMMON",1), s("COMMON",1)]
    }

    this.state.players.forEach((player) => {
      if (!player.isBot || !player.alive) return
      if (!player.id.startsWith("dummy-bot-")) return

      // Each bot uses a different synergy per group of 3 slots for type diversity.
      // The first group uses the bot's assigned theme; subsequent groups pick a
      // new synergy that hasn't been used yet on this board.
      const baseSynergy = this.botSynergies.get(player.id) ?? pickRandomIn(SynergyArray)
      const groupCount = Math.ceil(slots.length / 3)
      const groupSynergies: Synergy[] = [baseSynergy]
      for (let g = 1; g < groupCount; g++) {
        let next: Synergy
        do {
          next = pickRandomIn(SynergyArray)
        } while (groupSynergies.includes(next))
        groupSynergies.push(next)
      }

      // Clear existing board and rebuild
      player.board.forEach((_pokemon, key) => {
        player.board.delete(key)
      })

      // Place units using range-based positioning (same priority as real game):
      //   Range 1 (melee):   front first  → rows [3, 2, 1]
      //   Range 2 (ranged):  middle first  → rows [2, 1, 3]
      //   Range 3+ (casters): back first   → rows [1, 2, 3]
      const occupied = new Set<string>()
      const placeUnit = (pokemon: Pokemon, range: number) => {
        const rowOrder =
          range >= 3 ? [1, 2, 3]
          : range === 2 ? [2, 1, 3]
          : [3, 2, 1]
        for (const y of rowOrder) {
          for (let x = 0; x < 8; x++) {
            const key = `${x},${y}`
            if (!occupied.has(key)) {
              occupied.add(key)
              pokemon.positionX = x
              pokemon.positionY = y
              return
            }
          }
        }
      }

      for (let t = 0; t < slots.length; t++) {
        const synergy = groupSynergies[Math.floor(t / 3)]
        const pkm = pickFromPool(slots[t], synergy)
        const pokemon = PokemonFactory.createPokemonFromName(pkm, player)
        placeUnit(pokemon, getPokemonData(pkm).range)
        player.board.set(pokemon.id, pokemon)
      }
      player.boardSize = slots.length
    })
  }

  /**
   * Extract observation vector for a player (defaults to the primary RL agent).
   * Parameterized to support self-play where each of 8 players needs its own observation.
   */
  getObservation(playerId?: string): number[] {
    const targetId = playerId ?? this.agentId
    const agent = this.state.players.get(targetId)

    if (!agent) {
      return new Array(TOTAL_OBS_SIZE).fill(0)
    }

    // 7.2: Return cached observation if still valid
    if (!this.observationDirty.has(targetId) && this.observationCache.has(targetId)) {
      return this.observationCache.get(targetId)!
    }

    const obs: number[] = []

    const rarityMap: Record<string, number> = {
      [Rarity.COMMON]: 0.1,
      [Rarity.UNCOMMON]: 0.2,
      [Rarity.RARE]: 0.4,
      [Rarity.EPIC]: 0.6,
      [Rarity.ULTRA]: 0.8,
      [Rarity.UNIQUE]: 0.9,
      [Rarity.LEGENDARY]: 1.0,
      [Rarity.HATCH]: 0.3,
      [Rarity.SPECIAL]: 0.5
    }

    // ── Player stats (14) ──────────────────────────────────────────────
    // All values clamped to [0, 1] to prevent silent clipping in SB3
    const clamp01 = (v: number) => Math.min(1, Math.max(0, v))
    obs.push(clamp01(agent.life / 100))
    obs.push(clamp01(agent.money / 300))              // was /100, gold can reach 200+
    obs.push(clamp01(agent.experienceManager.level / 9))
    obs.push(clamp01((agent.streak + 20) / 40))       // was /10, streaks range -20..+20 → normalize to 0..1
    obs.push(clamp01(agent.interest / 5))
    obs.push(agent.alive ? 1 : 0)
    obs.push(clamp01(agent.rank / 8))
    obs.push(clamp01(agent.boardSize / 9))
    obs.push(clamp01((agent.experienceManager.expNeeded ?? 0) / 32))
    obs.push(clamp01((agent.shopFreeRolls ?? 0) / 3))
    obs.push(clamp01((agent.rerollCount ?? 0) / 50))  // was /20, can exceed 20 late game
    obs.push(agent.shopLocked ? 1 : 0)
    obs.push(clamp01((agent.totalMoneyEarned ?? 0) / 500))   // was /200, can exceed 200
    obs.push(clamp01((agent.totalPlayerDamageDealt ?? 0) / 300))  // was /100, can exceed 100

    // ── Shop (6 slots × 9 features = 54) ───────────────────────────────
    for (let i = 0; i < 6; i++) {
      const pkm = agent.shop[i]
      if (pkm && pkm !== Pkm.DEFAULT) {
        const data = getPokemonData(pkm)
        const types = data.types ?? []
        // Count copies on bench/board for evolution check
        let copyCount = 0
        agent.board.forEach((p) => {
          if (p.name === pkm) copyCount++
        })
        obs.push(1) // hasUnit
        obs.push(clamp01(getPkmSpeciesIndex(pkm))) // speciesIndex
        obs.push(clamp01(rarityMap[data.rarity] ?? 0)) // rarity
        obs.push(clamp01(getBuyPrice(pkm, this.state.specialGameRule) / 20)) // cost (was /10, legendary=20)
        obs.push(clamp01(types[0] ? getSynergyIndex(types[0]) : 0)) // type1
        obs.push(clamp01(types[1] ? getSynergyIndex(types[1]) : 0)) // type2
        obs.push(clamp01(types[2] ? getSynergyIndex(types[2]) : 0)) // type3
        obs.push(0) // type4 (always 0, no items on shop pokemon)
        obs.push(copyCount >= 2 ? 1 : 0) // isEvoPossible
      } else {
        obs.push(0, 0, 0, 0, 0, 0, 0, 0, 0) // 9 zeros
      }
    }

    // ── Board (32 cells × 12 features = 384) ──────────────────────────
    // Grid: y=0 bench, y=1-3 board. Cell = y*8+x.
    for (let y = 0; y < GRID_HEIGHT; y++) {
      for (let x = 0; x < GRID_WIDTH; x++) {
        const pokemon = this.findPokemonAt(agent, x, y)
        if (pokemon) {
          const data = getPokemonData(pokemon.name)
          const types = Array.from(pokemon.types.values()) as Synergy[]
          obs.push(1) // hasUnit
          obs.push(clamp01(getPkmSpeciesIndex(pokemon.name))) // speciesIndex
          obs.push(clamp01(pokemon.stars / 3)) // stars
          obs.push(clamp01(rarityMap[data.rarity] ?? 0)) // rarity
          obs.push(clamp01(types[0] ? getSynergyIndex(types[0]) : 0)) // type1
          obs.push(clamp01(types[1] ? getSynergyIndex(types[1]) : 0)) // type2
          obs.push(clamp01(types[2] ? getSynergyIndex(types[2]) : 0)) // type3
          obs.push(0) // type4 (stone types only during simulation)
          obs.push(clamp01(pokemon.atk / 100)) // atk (was /50, can exceed)
          obs.push(clamp01(pokemon.hp / 1000)) // hp (was /500, can exceed)
          obs.push(clamp01(pokemon.range / 5)) // range (was /4)
          obs.push(clamp01(pokemon.items.size / 3)) // numItems
        } else {
          obs.push(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) // 12 zeros
        }
      }
    }

    // ── Held items (10) ────────────────────────────────────────────────
    const heldItems = Array.from(agent.items.values()) as Item[]
    for (let i = 0; i < OBS_HELD_ITEMS; i++) {
      if (i < heldItems.length) {
        obs.push(clamp01(getItemIndex(heldItems[i])))
      } else {
        obs.push(0)
      }
    }

    // ── Synergies (31 values, normalized) ─────────────────────────────
    for (const synergy of SynergyArray) {
      const val = agent.synergies.get(synergy) ?? 0
      obs.push(clamp01(val / 10))
    }

    // ── Game info (7) ──────────────────────────────────────────────────
    obs.push(clamp01(this.state.stageLevel / 50))
    obs.push(clamp01(this.state.phase / 2))
    const playersAlive = values(this.state.players).filter(
      (p) => p.alive
    ).length
    obs.push(clamp01(playersAlive / 8))
    obs.push(
      agent.pokemonsProposition.length > 0 || agent.itemsProposition.length > 0
        ? 1
        : 0
    ) // hasPropositions
    obs.push(
      clamp01(this.state.weather
        ? getWeatherIndex(this.state.weather as Weather)
        : 0)
    ) // weatherIndex
    obs.push(this.state.stageLevel in PVEStages ? 1 : 0) // isPVE
    obs.push(
      clamp01(getMaxTeamSize(
        agent.experienceManager.level,
        this.state.specialGameRule
      ) / 9)
    ) // maxTeamSize

    // ── Opponents (7 × 10 features = 70) ──────────────────────────────
    const opponents = values(this.state.players).filter(
      (p) => p.id !== targetId && p.alive
    )
    for (let i = 0; i < OBS_OPPONENT_COUNT; i++) {
      if (i < opponents.length) {
        const opp = opponents[i]
        obs.push(clamp01(opp.life / 100))
        obs.push(clamp01(opp.rank / 8))
        obs.push(clamp01(opp.experienceManager.level / 9))
        obs.push(clamp01(opp.money / 300)) // was /100, bots can hoard too
        obs.push(clamp01((opp.streak + 20) / 40)) // was /10, match player stat encoding
        obs.push(clamp01(opp.boardSize / 9))
        // Top 2 synergies by count
        let top1Syn: Synergy | null = null
        let top1Count = 0
        let top2Syn: Synergy | null = null
        let top2Count = 0
        opp.synergies.forEach((count, syn) => {
          if (count > top1Count) {
            top2Syn = top1Syn
            top2Count = top1Count
            top1Syn = syn as Synergy
            top1Count = count
          } else if (count > top2Count) {
            top2Syn = syn as Synergy
            top2Count = count
          }
        })
        obs.push(clamp01(top1Syn ? getSynergyIndex(top1Syn) : 0))
        obs.push(clamp01(top1Count / 10))
        obs.push(clamp01(top2Syn ? getSynergyIndex(top2Syn) : 0))
        obs.push(clamp01(top2Count / 10))
      } else {
        obs.push(0, 0, 0, 0, 0, 0, 0, 0, 0, 0) // 10 zeros
      }
    }

    // ── Propositions (6 × 7 features = 42) ─────────────────────────────
    const propositions = values(agent.pokemonsProposition) as Pkm[]
    const itemOnlyPropositions = propositions.length === 0 && agent.itemsProposition.length > 0
    for (let i = 0; i < OBS_PROPOSITION_SLOTS; i++) {
      if (itemOnlyPropositions && i < agent.itemsProposition.length) {
        // Item-only propositions (item carousel, PVE rewards):
        // speciesIndex=0, rarity=0, types=0, last feature = item index
        obs.push(0) // speciesIndex (no pokemon)
        obs.push(0) // rarity
        obs.push(0) // type1
        obs.push(0) // type2
        obs.push(0) // type3
        obs.push(0) // type4
        obs.push(clamp01(getItemIndex(agent.itemsProposition[i]))) // itemIndex
      } else if (i < propositions.length && propositions[i]) {
        const data = getPokemonData(propositions[i])
        const types = data.types ?? []
        obs.push(clamp01(getPkmSpeciesIndex(propositions[i]))) // speciesIndex
        obs.push(clamp01(rarityMap[data.rarity] ?? 0)) // rarity
        obs.push(clamp01(types[0] ? getSynergyIndex(types[0]) : 0)) // type1
        obs.push(clamp01(types[1] ? getSynergyIndex(types[1]) : 0)) // type2
        obs.push(clamp01(types[2] ? getSynergyIndex(types[2]) : 0)) // type3
        obs.push(0) // type4
        obs.push(
          clamp01(agent.itemsProposition.length > i &&
            agent.itemsProposition[i] != null
            ? getItemIndex(agent.itemsProposition[i])
            : 0)
        ) // itemIndex (0 if no paired item)
      } else {
        obs.push(0, 0, 0, 0, 0, 0, 0) // 7 zeros
      }
    }

    // Pad to exact size if needed
    while (obs.length < TOTAL_OBS_SIZE) obs.push(0)

    const result = obs.slice(0, TOTAL_OBS_SIZE)
    // 7.2: Cache the observation
    this.observationCache.set(targetId, result)
    this.observationDirty.delete(targetId)
    return result
  }

  /**
   * Compute valid action mask (1 = valid, 0 = invalid).
   * Parameterized to support self-play where each player needs its own mask.
   */
  getActionMask(playerId?: string): number[] {
    const targetId = playerId ?? this.agentId
    const mask = new Array(TOTAL_ACTIONS).fill(0)
    const agent = this.state.players.get(targetId)

    if (!agent || !agent.alive || this.state.phase !== GamePhaseState.PICK) {
      mask[TrainingAction.END_TURN] = 1
      return mask
    }

    // If agent has pokemon propositions pending, only PICK actions are valid
    if (agent.pokemonsProposition.length > 0) {
      const freeSpace = getFreeSpaceOnBench(agent.board)
      for (let i = 0; i < agent.pokemonsProposition.length; i++) {
        const pkm = agent.pokemonsProposition[i] as PkmProposition
        if (!pkm) continue
        // Check if agent has bench space (or if it could evolve)
        const firstPkm = pkm in PkmDuos ? PkmDuos[pkm as PkmDuo][0] : (pkm as Pkm)
        const pokemon = PokemonFactory.createPokemonFromName(firstPkm, agent)
        const isEvolution =
          pokemon.evolutionRule &&
          pokemon.evolutionRule instanceof CountEvolutionRule &&
          pokemon.evolutionRule.canEvolveIfGettingOne(pokemon, agent)
        const numNeeded =
          pkm in PkmDuos
            ? PkmDuos[pkm as PkmDuo].length
            : 1
        if (freeSpace >= numNeeded || isEvolution) {
          mask[TrainingAction.PICK_0 + i] = 1
        }
      }
      // If no proposition is valid (bench full, no evolution), allow END_TURN as fallback
      if (mask.every((m) => m === 0)) {
        mask[TrainingAction.END_TURN] = 1
      }
      return mask
    }

    // Item-only propositions (item carousel, PVE rewards)
    if (agent.itemsProposition.length > 0) {
      for (let i = 0; i < agent.itemsProposition.length; i++) {
        if (agent.itemsProposition[i] != null) {
          mask[TrainingAction.PICK_0 + i] = 1
        }
      }
      return mask
    }

    // Normal PICK phase: economy actions
    // END_TURN is always valid
    mask[TrainingAction.END_TURN] = 1

    // LOCK_SHOP — valid once per turn (prevents toggle spam)
    if (!this.lockShopUsedThisTurn) {
      mask[TrainingAction.LOCK_SHOP] = 1
    }

    // BUY actions (all 6 shop slots)
    for (let i = 0; i < 6; i++) {
      const pkm = agent.shop[i]
      if (pkm && pkm !== Pkm.DEFAULT) {
        const cost = getBuyPrice(pkm, this.state.specialGameRule)
        const freeSpace = getFreeSpaceOnBench(agent.board)
        if (agent.money >= cost && freeSpace > 0) {
          mask[TrainingAction.BUY_0 + i] = 1
        }
      }
    }

    // SELL actions — all 32 grid cells
    for (let cell = 0; cell < GRID_CELLS; cell++) {
      const [x, y] = cellToXY(cell)
      if (this.findPokemonAt(agent, x, y)) {
        mask[TrainingAction.SELL_0 + cell] = 1
      }
    }

    // MOVE actions — empty cells that a unit could move to
    // Masks out the last move target to prevent back-and-forth oscillation
    const firstUnit = this.findFirstAvailableUnit(agent)
    if (firstUnit) {
      const sourceY = firstUnit.positionY
      const sourceCell = firstUnit.positionY * GRID_WIDTH + firstUnit.positionX
      const maxTeamSize = getMaxTeamSize(
        agent.experienceManager.level,
        this.state.specialGameRule
      )
      for (let cell = 0; cell < GRID_CELLS; cell++) {
        const [x, y] = cellToXY(cell)
        if (this.findPokemonAt(agent, x, y)) continue // occupied
        // Bench → board: only valid if board not full
        if (y >= 1 && sourceY === 0 && agent.boardSize >= maxTeamSize) continue
        // Block the cell the unit just came from (prevents A→B→A oscillation)
        if (cell === this.lastMoveCell && this.consecutiveMoves >= 2) continue
        mask[TrainingAction.MOVE_0 + cell] = 1
      }
    }

    // REFRESH
    const rollCost = agent.shopFreeRolls > 0 ? 0 : 1
    if (agent.money >= rollCost) {
      mask[TrainingAction.REFRESH] = 1
    }

    // LEVEL_UP
    if (agent.money >= 4 && agent.experienceManager.canLevelUp()) {
      mask[TrainingAction.LEVEL_UP] = 1
    }

    // REMOVE_SHOP — valid if slot has a pokemon and player can afford it (gold is gate only)
    for (let i = 0; i < 6; i++) {
      const pkm = agent.shop[i]
      if (pkm && pkm !== Pkm.DEFAULT) {
        const cost = getBuyPrice(pkm, this.state.specialGameRule)
        if (agent.money >= cost) {
          mask[TrainingAction.REMOVE_SHOP_0 + i] = 1
        }
      }
    }

    // COMBINE — valid pairs of held items that form a recipe
    if (agent.items.length >= 2) {
      const items = Array.from(agent.items.values()) as Item[]
      const pairs = this.getCachedItemPairs(targetId, items)
      for (let p = 0; p < pairs.length; p++) {
        const [i, j] = pairs[p]
        if (findRecipeResult(items[i], items[j])) {
          mask[TrainingAction.COMBINE_0 + p] = 1
        }
      }
    }

    return mask
  }

  private findPokemonAt(
    player: Player,
    x: number,
    y: number
  ): Pokemon | null {
    const grid = this.getPositionGrid(player)
    const cellKey = y * GRID_WIDTH + x
    return grid.get(cellKey) ?? null
  }

  /**
   * 7.2: Build or return cached position grid for a player.
   * Maps cell index (y*8+x) → Pokemon for O(1) lookups.
   */
  private getPositionGrid(player: Player): Map<number, Pokemon> {
    if (!this.positionGridDirty.has(player.id) && this.positionGridCache.has(player.id)) {
      return this.positionGridCache.get(player.id)!
    }
    const grid = new Map<number, Pokemon>()
    player.board.forEach((pokemon) => {
      const cellKey = pokemon.positionY * GRID_WIDTH + pokemon.positionX
      grid.set(cellKey, pokemon)
    })
    this.positionGridCache.set(player.id, grid)
    this.positionGridDirty.delete(player.id)
    return grid
  }

  /**
   * 7.2: Invalidate position grid and observation cache for a player.
   * Call after any board mutation (buy, sell, move, evolution, etc.).
   */
  private invalidatePlayerCaches(playerId: string): void {
    this.positionGridDirty.add(playerId)
    this.observationDirty.add(playerId)
    this.itemPairCache.delete(playerId)
  }

  /**
   * 7.2: Return cached item pairs for a player, recomputing only when invalidated.
   */
  private getCachedItemPairs(playerId: string, items: Item[]): [number, number][] {
    const cached = this.itemPairCache.get(playerId)
    if (cached) return cached
    const pairs = enumerateItemPairs(items as string[])
    this.itemPairCache.set(playerId, pairs)
    return pairs
  }

  /**
   * Count how many synergies have reached their first activation threshold.
   * Used for delta-based synergy reward shaping.
   */
  private countActiveSynergyThresholds(player: Player): number {
    let count = 0
    player.synergies.forEach((value, synergy) => {
      const triggers = SynergyTriggers[synergy as Synergy]
      if (triggers && triggers.length > 0 && value >= triggers[0]) {
        count++
      }
    })
    return count
  }

  /**
   * Compute detailed synergy tier info using real SynergyTriggers breakpoints.
   * Returns { totalTiers, qualifyingCount } where:
   *   totalTiers  = sum of tier levels reached across all synergies
   *   qualifyingCount = number of synergies at tier 2+
   */
  private computeSynergyTierInfo(player: Player): { totalTiers: number; qualifyingCount: number } {
    let totalTiers = 0
    let qualifyingCount = 0
    player.synergies.forEach((value, synergy) => {
      const triggers = SynergyTriggers[synergy as Synergy]
      if (!triggers || triggers.length === 0) return
      let tierReached = 0
      for (const threshold of triggers) {
        if (value >= threshold) tierReached++
        else break
      }
      if (tierReached > 0) {
        totalTiers += tierReached
        if (tierReached >= 2) qualifyingCount++
      }
    })
    return { totalTiers, qualifyingCount }
  }

  private computeFinalReward(agent: Player): number {
    // Reward based on final placement: rank 1 = best, rank 8 = worst
    return (9 - agent.rank) * REWARD_PLACEMENT_SCALE - REWARD_PLACEMENT_OFFSET
  }

  private getInfo(playerId?: string): StepResult["info"] {
    const targetId = playerId ?? this.agentId
    const agent = this.state.players.get(targetId)

    // Count active synergies (entries with count > 0)
    let synergyCount = 0
    if (agent) {
      agent.synergies.forEach((count) => {
        if (count > 0) synergyCount++
      })
    }

    // Build detailed board/bench/shop/synergy arrays for replay debugging
    const boardUnits: StepResult["info"]["board"] = []
    const benchUnits: StepResult["info"]["bench"] = []
    const shopSlots: (string | null)[] = []
    const activeSynergies: StepResult["info"]["synergies"] = []
    const heldItems: string[] = []

    if (agent) {
      // Board + bench units
      agent.board.forEach((pokemon) => {
        const items = Array.from(pokemon.items.values()) as string[]
        if (pokemon.positionY > 0) {
          boardUnits.push({ name: pokemon.name, x: pokemon.positionX, y: pokemon.positionY, stars: pokemon.stars, items })
        } else {
          benchUnits.push({ name: pokemon.name, x: pokemon.positionX, stars: pokemon.stars, items })
        }
      })

      // Shop slots
      for (let i = 0; i < 6; i++) {
        const pkm = agent.shop[i]
        shopSlots.push(pkm && pkm !== Pkm.DEFAULT ? pkm : null)
      }

      // Active synergies (with threshold met)
      agent.synergies.forEach((count, synergy) => {
        if (count > 0) {
          const triggers = SynergyTriggers[synergy as Synergy]
          const threshold = (triggers && triggers.length > 0) ? triggers[0] : 0
          activeSynergies.push({ name: synergy as string, count, threshold })
        }
      })
      activeSynergies.sort((a, b) => b.count - a.count)

      // Held items
      for (let i = 0; i < agent.items.length; i++) {
        heldItems.push(agent.items[i] as string)
      }
    }

    const level = agent?.experienceManager?.level ?? 1

    // Opponent info (populated after fights when opponentId is set)
    let opponentInfo: StepResult["info"]["opponent"] = null
    if (agent?.opponentId && agent.opponentId !== "pve") {
      const opp = this.state.players.get(agent.opponentId)
      if (opp) {
        const oppBoard: StepResult["info"]["board"] = []
        opp.board.forEach((pokemon) => {
          if (pokemon.positionY > 0) {
            const items = Array.from(pokemon.items.values()) as string[]
            oppBoard.push({ name: pokemon.name, x: pokemon.positionX, y: pokemon.positionY, stars: pokemon.stars, items })
          }
        })
        opponentInfo = {
          name: agent.opponentName ?? opp.name,
          odLife: opp.life,
          level: opp.experienceManager?.level ?? 1,
          board: oppBoard
        }
      }
    } else if (agent?.opponentId === "pve") {
      opponentInfo = {
        name: agent.opponentName ?? "PVE",
        odLife: 0,
        level: 0,
        board: []
      }
    }

    return {
      stage: this.state.stageLevel,
      phase:
        this.state.phase === GamePhaseState.PICK
          ? "PICK"
          : this.state.phase === GamePhaseState.FIGHT
            ? "FIGHT"
            : "TOWN",
      rank: agent?.rank ?? 8,
      life: agent?.life ?? 0,
      money: agent?.money ?? 0,
      actionsThisTurn: SELF_PLAY
        ? (this.actionsPerPlayer.get(targetId) ?? 0)
        : this.actionsThisTurn,
      actionMask: this.getActionMask(targetId),
      gold: agent?.money ?? 0,
      boardSize: agent?.boardSize ?? 0,
      synergyCount,
      itemsHeld: agent?.items.length ?? 0,
      level,
      maxTeamSize: getMaxTeamSize(level, this.state.specialGameRule),
      exp: agent?.experienceManager?.experience ?? 0,
      expNeeded: agent?.experienceManager?.expNeeded ?? 0,
      shop: shopSlots,
      board: boardUnits,
      bench: benchUnits,
      synergies: activeSynergies,
      items: heldItems,
      opponent: opponentInfo
    }
  }

  // ─── Self-play infrastructure ───────────────────────────────────────────

  /**
   * Reset turn-tracking state at the start of each pick phase.
   * Called after fights resolve and before the next round of actions.
   */
  private resetTurnState(): void {
    this.turnEnded.clear()
    this.actionsPerPlayer.clear()
    this.state.players.forEach((_player, id) => {
      this.turnEnded.set(id, false)
      this.actionsPerPlayer.set(id, 0)
    })
  }

  /**
   * Create 8 RL agent players for self-play mode (no bots).
   */
  private createSelfPlayAgents(): void {
    for (let i = 0; i < 8; i++) {
      const playerId = `rl-agent-${i}-${nanoid(6)}`
      const player = new Player(
        playerId,
        `RL-Agent-${i}`,
        1000,
        0,
        getAvatarString(PkmIndex[Pkm.PIKACHU], false),
        false, // isBot = false (RL agent)
        i + 1,
        new Map(),
        "",
        Role.BASIC,
        this.state
      )
      this.state.players.set(playerId, player)
      this.playerIds.push(playerId)
    }
  }

  /**
   * Self-play batch step: process one action per player simultaneously.
   *
   * This is the core method for the SelfPlayVecEnv. sb3 calls step() once
   * with 8 actions (one per sub-env/player) and expects 8 transitions back.
   *
   * Three critical semantics (flagged during design review):
   *
   * Issue #1 (fight trigger timing): turnEnded[] persists across step calls
   * within a round. Fight triggers ONLY when the LAST alive player ends their
   * turn, not on the first END_TURN. Player 1 might end turn in call N, but
   * the fight doesn't run until player 8 ends in call N+3.
   *
   * Issue #2 (reward attribution): When the fight triggers, ALL 8 players
   * get their combat reward in that response — not just the player whose
   * END_TURN happened to be last.
   *
   * Issue #3 (VecEnv reset semantics): Dead players return done=False with
   * END_TURN-only action mask until the game ACTUALLY ends. Only when
   * gameFinished do ALL 8 players return done=True simultaneously.
   * This prevents sb3 from trying to auto-reset individual sub-envs mid-game.
   */
  stepBatch(actions: number[]): StepResult[] {
    if (!SELF_PLAY) {
      throw new Error("stepBatch is only available in SELF_PLAY mode")
    }

    if (actions.length !== this.playerIds.length) {
      throw new Error(
        `Expected ${this.playerIds.length} actions, got ${actions.length}`
      )
    }

    this.totalSteps++

    // If game already finished, return terminal state for all players
    if (this.state.gameFinished) {
      return this.playerIds.map((id) => ({
        observation: this.getObservation(id),
        reward: 0,
        done: true,
        info: this.getInfo(id)
      }))
    }

    // Track duplicate buy rewards per player
    const dupBuyRewards = new Map<string, number>()

    // ── Phase 1: Process each player's action ──────────────────────────

    for (let i = 0; i < this.playerIds.length; i++) {
      const playerId = this.playerIds[i]
      const player = this.state.players.get(playerId)!
      const action = actions[i]

      // Issue #3: Dead players are no-ops. They send END_TURN (masked to
      // only allow it) and we skip processing entirely.
      if (!player.alive) {
        continue
      }

      // Player already ended turn this round: no-op (padding for sb3)
      if (this.turnEnded.get(playerId)) {
        continue
      }

      // Handle proposition picking (stage 0 starters, uniques, additionals)
      if (player.pokemonsProposition.length > 0) {
        if (
          action >= TrainingAction.PICK_0 &&
          action <= TrainingAction.PICK_0 + 5
        ) {
          this.pickProposition(
            player,
            action - TrainingAction.PICK_0
          )
          // Don't count proposition picks toward turn end — player continues shopping
          continue
        }
        // Non-proposition action during propositions (END_TURN fallback when
        // bench is full and no proposition can be picked). Fall through to
        // normal action processing so the turn-end logic can handle it.
      }

      // Item-only propositions (item carousel, PVE rewards)
      if (player.pokemonsProposition.length === 0 && player.itemsProposition.length > 0) {
        if (
          action >= TrainingAction.PICK_0 &&
          action <= TrainingAction.PICK_0 + 5
        ) {
          this.pickItemProposition(
            player,
            action - TrainingAction.PICK_0
          )
          // Don't count item picks toward turn end — player continues shopping
          continue
        }
        // Shouldn't happen (mask only allows PICK during item propositions),
        // but fall through to normal processing as safety
      }

      // Snapshot existing copy count BEFORE buy (evolution merges copies)
      const isBuyBatch = action >= TrainingAction.BUY_0 && action <= TrainingAction.BUY_5
      let preBuyCopiesBatch = 0
      if (isBuyBatch) {
        const shopIdx = action - TrainingAction.BUY_0
        const shopName = player.shop[shopIdx]
        if (shopName && shopName !== Pkm.DEFAULT) {
          const targetIndex = getPokemonData(shopName)?.index
          if (targetIndex) {
            preBuyCopiesBatch = values(player.board).filter(
              (p) => p.index === targetIndex
            ).length
          }
        }
      }

      // Execute normal PICK phase action
      const actionExecutedBatch = this.executeAction(action, player)

      // Reward for buying duplicates (encourages building toward evolutions)
      // After stage 20, boost rewards to encourage spending gold on upgrades.
      const lateGameBatch = this.state.stageLevel > 20
      if (isBuyBatch && actionExecutedBatch && preBuyCopiesBatch >= 2) {
        const prev = dupBuyRewards.get(playerId) ?? 0
        dupBuyRewards.set(playerId, prev + (lateGameBatch ? REWARD_BUY_EVOLUTION_LATEGAME : REWARD_BUY_EVOLUTION))
      } else if (isBuyBatch && actionExecutedBatch && preBuyCopiesBatch === 1) {
        const prev = dupBuyRewards.get(playerId) ?? 0
        dupBuyRewards.set(playerId, prev + (lateGameBatch ? REWARD_BUY_DUPLICATE_LATEGAME : REWARD_BUY_DUPLICATE))
      }

      const actCount = (this.actionsPerPlayer.get(playerId) ?? 0) + 1
      this.actionsPerPlayer.set(playerId, actCount)

      // Check if this player's turn should end
      if (
        action === TrainingAction.END_TURN ||
        actCount >= TRAINING_MAX_ACTIONS_PER_TURN
      ) {
        this.turnEnded.set(playerId, true)

        // Safety: auto-resolve any pending propositions
        if (player.pokemonsProposition.length > 0 || player.itemsProposition.length > 0) {
          this.autoPickForAgent(player)
        }

        // Auto-place team on board
        if (TRAINING_AUTO_PLACE) {
          this.autoPlaceTeam(player)
        }

        // Auto-equip held items onto board units before combat
        this.autoEquipItems(player)
      }
    }

    // 7.1: Bench penalty for all alive RL players at turn end (batch path)
    const benchPenalties = new Map<string, number>()
    if (!TRAINING_AUTO_PLACE) {
      for (let i = 0; i < this.playerIds.length; i++) {
        const playerId = this.playerIds[i]
        if (!this.turnEnded.get(playerId)) continue
        const player = this.state.players.get(playerId)!
        if (!player.alive) continue
        const maxTeamSize = getMaxTeamSize(
          player.experienceManager.level,
          this.state.specialGameRule
        )
        if (player.boardSize < maxTeamSize) {
          let benchCount = 0
          player.board.forEach((p) => {
            if (isOnBench(p)) benchCount++
          })
          if (benchCount > 0) {
            const openSlots = maxTeamSize - player.boardSize
            const penaltyUnits = Math.min(benchCount, openSlots)
            benchPenalties.set(playerId, penaltyUnits * REWARD_BENCH_PENALTY)
          }
        }
      }
    }

    // ── Phase 2: Check fight trigger ────────────────────────────────────
    // Issue #1: Fight triggers when ALL alive players have ended their turn.
    // turnEnded persists across calls — a player who ended in call N stays
    // ended until the fight resolves and resetTurnState() is called.

    let fightTriggered = false
    let perPlayerRewards = new Map<string, number>()

    // Handle stage 0 specially: advance to stage 1 when all players have
    // picked their propositions (no fight at stage 0)
    if (this.state.stageLevel === 0) {
      const allPicked = this.playerIds.every((id) => {
        const player = this.state.players.get(id)!
        return player.pokemonsProposition.length === 0
      })
      if (allPicked) {
        this.state.stageLevel = 1
        this.state.players.forEach((player) => {
          if (player.alive) {
            this.state.shop.assignShop(player, false, this.state)
          }
        })
        this.resetTurnState()
      }
    } else {
      const allAliveEnded = this.playerIds.every((id) => {
        const player = this.state.players.get(id)!
        return !player.alive || this.turnEnded.get(id)
      })

      if (allAliveEnded) {
        // All alive players have ended turn — run fight
        perPlayerRewards = this.runFightPhase()
        fightTriggered = true

        if (!this.state.gameFinished) {
          this.advanceToNextPickPhase()
          this.resetTurnState()
        }
      }
    }

    // ── Phase 3: Build per-player results ───────────────────────────────

    return this.playerIds.map((id) => {
      const player = this.state.players.get(id)!
      let reward = 0
      let done = false

      // 7.1: Add bench penalty (computed before fight)
      reward += benchPenalties.get(id) ?? 0

      // Add duplicate buy reward
      reward += dupBuyRewards.get(id) ?? 0

      if (fightTriggered) {
        // Issue #2: ALL players get their combat reward
        reward += perPlayerRewards.get(id) ?? 0

        if (this.state.gameFinished) {
          reward += this.computeFinalReward(player)
          // Issue #3: ALL players get done=True simultaneously
          done = true
        }
      }

      // Issue #3: Dead players MUST return done=False until the game
      // actually ends. If we returned done=True for dead players mid-game,
      // sb3's VecEnv would try to auto-reset that sub-env, which would
      // destroy the shared game state for all other players.
      if (!player.alive && !this.state.gameFinished) {
        done = false
      }

      return {
        observation: this.getObservation(id),
        reward,
        done,
        info: this.getInfo(id)
      }
    })
  }
}
