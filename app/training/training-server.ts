/**
 * Training HTTP Server
 *
 * Exposes the TrainingEnv as a simple REST API that the Python PPO
 * trainer connects to. No Colyseus, no WebSocket, no Firebase needed.
 *
 * Endpoints:
 *   POST /reset          -> { observation, reward, done, info }
 *   POST /step           -> { observation, reward, done, info }  body: { action: number }
 *   GET  /observe         -> { observation, info }
 *   GET  /action-space    -> { n: number }
 *   GET  /observation-space -> { n: number }
 *   GET  /health          -> { status: "ok" }
 */
import express from "express"
import { TrainingEnv, StepResult } from "./training-env"
import { NUM_RL_AGENTS, SELF_PLAY, TOTAL_ACTIONS, TOTAL_OBS_SIZE, TRAINING_API_PORT } from "./training-config"
import { logger } from "../utils/logger"

export async function startTrainingServer() {
  const app = express()
  app.use(express.json())

  const env = new TrainingEnv()
  await env.initialize()

  let isInitialized = false

  app.get("/health", (_req, res) => {
    res.json({ status: "ok", initialized: isInitialized })
  })

  app.get("/action-space", (_req, res) => {
    res.json({ n: TOTAL_ACTIONS })
  })

  // Returns the number of RL agents this server expects per step-multi call.
  // Python side queries this to set VecEnv num_envs correctly.
  app.get("/num-agents", (_req, res) => {
    res.json({ n: NUM_RL_AGENTS })
  })

  app.get("/observation-space", (_req, res) => {
    res.json({ n: TOTAL_OBS_SIZE })
  })

  app.post("/reset", (_req, res) => {
    try {
      const result = env.reset()
      isInitialized = true
      res.json(result)
    } catch (error: any) {
      logger.error("Reset error:", error)
      res.status(500).json({ error: error.message })
    }
  })

  app.post("/step", (req, res) => {
    try {
      if (!isInitialized) {
        res.status(400).json({ error: "Environment not initialized. Call /reset first." })
        return
      }
      const action = req.body.action
      if (action === undefined || action < 0 || action >= TOTAL_ACTIONS) {
        res.status(400).json({ error: `Invalid action: ${action}. Must be 0-${TOTAL_ACTIONS - 1}` })
        return
      }
      const result = env.step(action)
      res.json(result)
    } catch (error: any) {
      logger.error("Step error:", error)
      res.status(500).json({ error: error.message })
    }
  })

  app.get("/observe", (_req, res) => {
    try {
      if (!isInitialized) {
        res.status(400).json({ error: "Environment not initialized. Call /reset first." })
        return
      }
      res.json({
        observation: env.getObservation(),
        info: {
          actionMask: env.getActionMask()
        }
      })
    } catch (error: any) {
      logger.error("Observe error:", error)
      res.status(500).json({ error: error.message })
    }
  })

  // Batch step endpoint for efficiency (run N steps at once)
  app.post("/batch-step", (req, res) => {
    try {
      if (!isInitialized) {
        res.status(400).json({ error: "Environment not initialized. Call /reset first." })
        return
      }
      const actions: number[] = req.body.actions
      if (!Array.isArray(actions)) {
        res.status(400).json({ error: "Expected { actions: number[] }" })
        return
      }

      const results: StepResult[] = []
      for (const action of actions) {
        const result = env.step(action)
        results.push(result)
        if (result.done) break
      }
      res.json({ results })
    } catch (error: any) {
      logger.error("Batch step error:", error)
      res.status(500).json({ error: error.message })
    }
  })

  // Batch step: process one action per RL agent.
  // Used by SelfPlayVecEnv in both full self-play (8 agents) and hybrid mode (N agents + bots).
  // Requires NUM_RL_AGENTS >= 2 (set SELF_PLAY=true for 8, or NUM_RL_AGENTS=N for hybrid).
  app.post("/step-multi", (req, res) => {
    try {
      if (NUM_RL_AGENTS < 2) {
        res.status(400).json({
          error: "step-multi requires NUM_RL_AGENTS >= 2. " +
            "Set SELF_PLAY=true (8 agents) or NUM_RL_AGENTS=2..7 (hybrid mode)."
        })
        return
      }
      if (!isInitialized) {
        res.status(400).json({ error: "Environment not initialized. Call /reset first." })
        return
      }
      const actions: number[] = req.body.actions
      if (!Array.isArray(actions) || actions.length !== NUM_RL_AGENTS) {
        res.status(400).json({
          error: `Expected { actions: number[${NUM_RL_AGENTS}] } — one action per RL agent`
        })
        return
      }
      for (const action of actions) {
        if (action < 0 || action >= TOTAL_ACTIONS) {
          res.status(400).json({
            error: `Invalid action: ${action}. Must be 0-${TOTAL_ACTIONS - 1}`
          })
          return
        }
      }

      const results = env.stepBatch(actions)

      // Return parallel arrays for efficient numpy conversion on the Python side
      res.json({
        observations: results.map((r) => r.observation),
        rewards: results.map((r) => r.reward),
        dones: results.map((r) => r.done),
        infos: results.map((r) => r.info)
      })
    } catch (error: any) {
      logger.error("Step-multi error:", error)
      res.status(500).json({ error: error.message })
    }
  })

  // Run full game with random actions (for benchmarking)
  app.post("/benchmark", (_req, res) => {
    try {
      const startTime = Date.now()
      let result = env.reset()
      let steps = 0

      while (!result.done) {
        // Pick random valid action from mask
        const mask = result.info.actionMask
        const validActions: number[] = []
        mask.forEach((v, i) => {
          if (v === 1) validActions.push(i)
        })
        const action =
          validActions[Math.floor(Math.random() * validActions.length)]
        result = env.step(action)
        steps++
      }

      const elapsedMs = Date.now() - startTime
      res.json({
        steps,
        elapsedMs,
        stepsPerSecond: (steps / elapsedMs) * 1000,
        finalRank: result.info.rank,
        finalStage: result.info.stage,
        finalReward: result.reward
      })
    } catch (error: any) {
      logger.error("Benchmark error:", error)
      res.status(500).json({ error: error.message })
    }
  })

  app.listen(TRAINING_API_PORT, () => {
    logger.info(`Training server listening on port ${TRAINING_API_PORT}`)
    const modeLabel = SELF_PLAY
      ? "SELF_PLAY (8 RL agents)"
      : NUM_RL_AGENTS > 1
        ? `HYBRID (${NUM_RL_AGENTS} RL agents + ${8 - NUM_RL_AGENTS} bots)`
        : "SINGLE_AGENT (1 RL + 7 bots)"
    logger.info(`  Mode: ${modeLabel}`)
    logger.info(`  POST /reset          - Reset environment`)
    logger.info(`  POST /step           - Take action { action: 0-${TOTAL_ACTIONS - 1} }`)
    if (NUM_RL_AGENTS >= 2) {
      logger.info(`  POST /step-multi     - Batch step { actions: number[${NUM_RL_AGENTS}] }`)
    }
    logger.info(`  GET  /observe        - Get current observation`)
    logger.info(`  POST /benchmark      - Run full game with random actions`)
    logger.info(`  GET  /health         - Health check`)
  })

  return app
}
