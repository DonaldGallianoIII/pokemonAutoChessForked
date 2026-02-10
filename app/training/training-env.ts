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
  StageDuration
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
  MAX_PROPOSITIONS,
  OBS_PROPOSITION_FEATURES,
  OBS_PROPOSITION_SLOTS,
  REWARD_PER_DRAW,
  REWARD_PER_KILL,
  REWARD_PER_LOSS,
  REWARD_PER_WIN,
  REWARD_PLACEMENT_OFFSET,
  REWARD_PLACEMENT_SCALE,
  SELF_PLAY,
  TOTAL_ACTIONS,
  TOTAL_OBS_SIZE,
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
  }
}

export class TrainingEnv {
  state!: GameState
  room!: HeadlessRoom
  agentId!: string
  actionsThisTurn = 0
  totalSteps = 0
  lastBattleResult: BattleResult | null = null
  cachedBots: IBot[] = []
  additionalUncommonPool: Pkm[] = []
  additionalRarePool: Pkm[] = []
  additionalEpicPool: Pkm[] = []

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
    this.totalSteps = 0
    this.lastBattleResult = null
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
      // Execute the agent's action
      const actionExecuted = this.executeAction(action, agent)
      this.actionsThisTurn++

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
        if (agent.pokemonsProposition.length > 0) {
          this.autoPickForAgent(agent)
        }

        // Auto-place pokemon on board if there's room
        this.autoPlaceTeam(agent)

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
    // If agent has propositions pending, only allow PICK actions
    if (agent.pokemonsProposition.length > 0) {
      if (action >= TrainingAction.PICK_0 && action <= TrainingAction.PICK_0 + 5) {
        return this.pickProposition(agent, action - TrainingAction.PICK_0)
      }
      return false // no other actions allowed during proposition phase
    }

    // BUY_0..BUY_4 (0-4)
    if (action >= TrainingAction.BUY_0 && action <= TrainingAction.BUY_0 + 4) {
      return this.buyPokemon(agent, action - TrainingAction.BUY_0)
    }
    // BUY_5 (5) — no-op until Phase 2.2 (6th shop slot)
    if (action === TrainingAction.BUY_5) return false
    // REFRESH (6)
    if (action === TrainingAction.REFRESH) return this.rerollShop(agent)
    // LEVEL_UP (7)
    if (action === TrainingAction.LEVEL_UP) return this.levelUp(agent)
    // LOCK_SHOP (8) — no-op until Phase 2.1
    if (action === TrainingAction.LOCK_SHOP) return false
    // END_TURN (9)
    if (action === TrainingAction.END_TURN) return true
    // MOVE_0..MOVE_31 (10-41) — no-op until Phase 2.4
    if (action >= TrainingAction.MOVE_0 && action <= TrainingAction.MOVE_0 + 31) return false
    // SELL_0..SELL_31 (42-73): bench sells (y=0) work, board sells (y>0) are no-ops
    if (action >= TrainingAction.SELL_0 && action <= TrainingAction.SELL_0 + 31) {
      const [x, y] = cellToXY(action - TrainingAction.SELL_0)
      if (y === 0) return this.sellPokemonAtBench(agent, x)
      return false // board cell sells — no-op until Phase 2.3
    }
    // REMOVE_SHOP_0..5 (74-79) — no-op until Phase 2.5
    if (action >= TrainingAction.REMOVE_SHOP_0 && action <= TrainingAction.REMOVE_SHOP_0 + 5) return false
    // PICK_0..PICK_5 (80-85)
    if (action >= TrainingAction.PICK_0 && action <= TrainingAction.PICK_0 + 5) {
      return this.pickProposition(agent, action - TrainingAction.PICK_0)
    }
    // COMBINE_0..5 (86-91) — no-op until Phase 2.6
    if (action >= TrainingAction.COMBINE_0 && action <= TrainingAction.COMBINE_0 + 5) return false

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
    return true
  }

  private sellPokemonAtBench(player: Player, benchIndex: number): boolean {
    // Find pokemon at bench position benchIndex
    let targetPokemon: Pokemon | null = null
    let targetId: string | null = null

    player.board.forEach((pokemon, key) => {
      if (pokemon.positionY === 0 && pokemon.positionX === benchIndex) {
        targetPokemon = pokemon
        targetId = key
      }
    })

    if (!targetPokemon || !targetId) return false

    player.board.delete(targetId)
    this.state.shop.releasePokemon(
      (targetPokemon as Pokemon).name,
      player,
      this.state
    )

    const sellPrice = getSellPrice(targetPokemon, this.state.specialGameRule)
    player.addMoney(sellPrice, false, null)
    ;(targetPokemon as Pokemon).items.forEach((it: Item) => {
      player.items.push(it)
    })

    player.updateSynergies()
    player.boardSize = this.room.getTeamSize(player.board)
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
    return true
  }

  private levelUp(player: Player): boolean {
    const cost = 4 // standard level up cost
    if (player.money < cost || !player.experienceManager.canLevelUp())
      return false

    player.addExperience(4)
    player.money -= cost
    return true
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

    // Run all simulations synchronously
    let steps = 0
    let allFinished = false
    while (!allFinished && steps < TRAINING_MAX_FIGHT_STEPS) {
      allFinished = true
      this.state.simulations.forEach((simulation) => {
        if (!simulation.finished) {
          simulation.update(TRAINING_SIMULATION_DT)
          allFinished = false
        }
      })
      steps++
    }

    // Force-finish any remaining simulations
    this.state.simulations.forEach((simulation) => {
      if (!simulation.finished) {
        simulation.onFinish()
      }
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

    // Track agent's last battle result for logging
    const agent = this.state.players.get(this.agentId)
    if (agent) {
      const lastHistory = agent.history.at(-1)
      if (lastHistory) {
        this.lastBattleResult = lastHistory.result as BattleResult
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

    // Handle item carousel stages — give each alive player a random item
    if (ItemCarouselStages.includes(this.state.stageLevel)) {
      this.state.players.forEach((player) => {
        if (!player.isBot && player.alive) {
          const itemPool =
            this.state.stageLevel >= 20
              ? CraftableItemsNoScarves
              : ItemComponentsNoFossilOrScarf
          player.items.push(pickRandomIn(itemPool))
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
          this.state.shop.assignShop(player, false, this.state)
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
          }
        }

        player.pokemonsProposition.clear()
        player.itemsProposition.clear()
        this.room.checkEvolutionsAfterPokemonAcquired(player.id)
      } else if (player.itemsProposition.length > 0) {
        // Item-only propositions (PVE rewards, etc.)
        const pick = pickRandomIn(values(player.itemsProposition))
        player.items.push(pick)
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
   * Bots get more pokemon as the game progresses (roughly matching
   * the expected team size at each stage).
   */
  private updateDummyBotTeams(): void {
    const stage = this.state.stageLevel
    // Target team size scales with stage: 1 at stage 1, up to ~6 by stage 30+
    const targetSize = Math.min(6, Math.max(1, Math.ceil(stage / 5)))

    // Pick rarity pool based on stage
    const pool =
      stage >= 20
        ? PRECOMPUTED_POKEMONS_PER_RARITY.EPIC
        : stage >= 10
          ? PRECOMPUTED_POKEMONS_PER_RARITY.RARE
          : stage >= 5
            ? PRECOMPUTED_POKEMONS_PER_RARITY.UNCOMMON
            : PRECOMPUTED_POKEMONS_PER_RARITY.COMMON

    this.state.players.forEach((player) => {
      if (!player.isBot || !player.alive) return
      if (!player.id.startsWith("dummy-bot-")) return

      // Clear existing board and rebuild
      player.board.forEach((_pokemon, key) => {
        player.board.delete(key)
      })

      for (let t = 0; t < targetSize; t++) {
        const pkm = pickRandomIn(pool)
        const pokemon = PokemonFactory.createPokemonFromName(pkm, player)
        pokemon.positionX = t % 8
        pokemon.positionY = 1 + Math.floor(t / 8)
        player.board.set(pokemon.id, pokemon)
      }
      player.boardSize = targetSize
    })
  }

  /**
   * Extract observation vector for a player (defaults to the primary RL agent).
   * Parameterized to support self-play where each of 8 players needs its own observation.
   */
  getObservation(playerId?: string): number[] {
    const targetId = playerId ?? this.agentId
    const obs: number[] = []
    const agent = this.state.players.get(targetId)

    if (!agent) {
      return new Array(TOTAL_OBS_SIZE).fill(0)
    }

    // Player stats (8)
    obs.push(agent.life / 100) // normalized
    obs.push(agent.money / 100)
    obs.push(agent.experienceManager.level / 9)
    obs.push(agent.streak / 10)
    obs.push(agent.interest / 5)
    obs.push(agent.alive ? 1 : 0)
    obs.push(agent.rank / 8)
    obs.push(agent.boardSize / 9)

    // Shop (5 slots, encoded as pokemon rarity 0-1)
    for (let i = 0; i < 5; i++) {
      const pkm = agent.shop[i]
      if (pkm && pkm !== Pkm.DEFAULT) {
        const data = getPokemonData(pkm)
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
        obs.push(rarityMap[data.rarity] ?? 0)
      } else {
        obs.push(0)
      }
    }

    // Board/bench (40 slots * 3 features = 120)
    // Bench: 8 positions (y=0, x=0..7)
    for (let x = 0; x < 8; x++) {
      const pokemon = this.findPokemonAt(agent, x, 0)
      if (pokemon) {
        const data = getPokemonData(pokemon.name)
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
        obs.push(1) // has pokemon
        obs.push(pokemon.stars / 3)
        obs.push(rarityMap[data.rarity] ?? 0)
      } else {
        obs.push(0, 0, 0)
      }
    }

    // Board: 4x8 = 32 positions (y=1..4, x=0..7)
    for (let y = 1; y <= 4; y++) {
      for (let x = 0; x < 8; x++) {
        const pokemon = this.findPokemonAt(agent, x, y)
        if (pokemon) {
          const data = getPokemonData(pokemon.name)
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
          obs.push(1)
          obs.push(pokemon.stars / 3)
          obs.push(rarityMap[data.rarity] ?? 0)
        } else {
          obs.push(0, 0, 0)
        }
      }
    }

    // Synergies (32 values, normalized)
    for (const synergy of SynergyArray) {
      const val = agent.synergies.get(synergy) ?? 0
      obs.push(val / 10) // normalize
    }

    // Game info (4)
    obs.push(this.state.stageLevel / 50)
    obs.push(this.state.phase / 2)
    const playersAlive = values(this.state.players).filter(
      (p) => p.alive
    ).length
    obs.push(playersAlive / 8)
    obs.push(agent.pokemonsProposition.length > 0 ? 1 : 0) // hasPropositions

    // Opponent stats (16 = 8 opponents * 2 features)
    const opponents = values(this.state.players).filter(
      (p) => p.id !== targetId
    )
    for (let i = 0; i < 8; i++) {
      if (i < opponents.length) {
        obs.push(opponents[i].life / 100)
        obs.push(opponents[i].rank / 8)
      } else {
        obs.push(0, 0)
      }
    }

    // Proposition slots (6 slots × 3 features = 18)
    // Each slot: rarity, numTypes, hasItem
    const propositions = values(agent.pokemonsProposition) as Pkm[]
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
    for (let i = 0; i < MAX_PROPOSITIONS; i++) {
      if (i < propositions.length && propositions[i]) {
        const data = getPokemonData(propositions[i])
        obs.push(rarityMap[data.rarity] ?? 0) // rarity
        obs.push((data.types?.length ?? 0) / 5) // numTypes normalized
        obs.push(
          agent.itemsProposition.length > i &&
            agent.itemsProposition[i] != null
            ? 1
            : 0
        ) // hasItem
      } else {
        obs.push(0, 0, 0)
      }
    }

    // Pad to exact size if needed
    while (obs.length < TOTAL_OBS_SIZE) obs.push(0)

    return obs.slice(0, TOTAL_OBS_SIZE)
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

    // If agent has propositions pending, only PICK actions are valid
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

    // Normal PICK phase: economy actions
    // END_TURN is always valid
    mask[TrainingAction.END_TURN] = 1

    // BUY actions (slots 0-4 only; BUY_5 stays masked until Phase 3.2)
    for (let i = 0; i < 5; i++) {
      const pkm = agent.shop[i]
      if (pkm && pkm !== Pkm.DEFAULT) {
        const cost = getBuyPrice(pkm, this.state.specialGameRule)
        const freeSpace = getFreeSpaceOnBench(agent.board)
        if (agent.money >= cost && freeSpace > 0) {
          mask[TrainingAction.BUY_0 + i] = 1
        }
      }
    }

    // SELL actions (bench positions 0-7, mapped to grid cells 0-7 where y=0)
    for (let x = 0; x < 8; x++) {
      const pokemon = this.findPokemonAt(agent, x, 0)
      if (pokemon) {
        mask[TrainingAction.SELL_0 + x] = 1
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

    return mask
  }

  private findPokemonAt(
    player: Player,
    x: number,
    y: number
  ): Pokemon | null {
    let found: Pokemon | null = null
    player.board.forEach((pokemon) => {
      if (pokemon.positionX === x && pokemon.positionY === y) {
        found = pokemon
      }
    })
    return found
  }

  private computeFinalReward(agent: Player): number {
    // Reward based on final placement: rank 1 = best, rank 8 = worst
    return (9 - agent.rank) * REWARD_PLACEMENT_SCALE - REWARD_PLACEMENT_OFFSET
  }

  private getInfo(playerId?: string): StepResult["info"] {
    const targetId = playerId ?? this.agentId
    const agent = this.state.players.get(targetId)
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
      actionMask: this.getActionMask(targetId)
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

      // Execute normal PICK phase action
      this.executeAction(action, player)
      const actCount = (this.actionsPerPlayer.get(playerId) ?? 0) + 1
      this.actionsPerPlayer.set(playerId, actCount)

      // Check if this player's turn should end
      if (
        action === TrainingAction.END_TURN ||
        actCount >= TRAINING_MAX_ACTIONS_PER_TURN
      ) {
        this.turnEnded.set(playerId, true)

        // Safety: auto-resolve any pending propositions
        if (player.pokemonsProposition.length > 0) {
          this.autoPickForAgent(player)
        }

        // Auto-place team on board
        this.autoPlaceTeam(player)
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

      if (fightTriggered) {
        // Issue #2: ALL players get their combat reward
        reward = perPlayerRewards.get(id) ?? 0

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
