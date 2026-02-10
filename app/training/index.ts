/**
 * Training mode entry point.
 *
 * Usage:
 *   npx ts-node app/training/index.ts
 *
 * This starts a headless training server without any Colyseus/Firebase
 * dependencies. The Python PPO trainer connects to the HTTP API.
 *
 * Environment variables:
 *   TRAINING_PORT - HTTP port (default: 9100)
 *   MONGO_URI     - MongoDB connection string (needed for bot definitions)
 *   USE_MEMORY_DB - Set to "true" to use in-memory MongoDB (no install needed)
 *   SKIP_MONGO    - Set to "true" to skip MongoDB entirely (no bots)
 */
import { Encoder } from "@colyseus/schema"
import { connect } from "mongoose"
import { startTrainingServer } from "./training-server"
import { logger } from "../utils/logger"

// Increase buffer for schema encoding (same as main server)
Encoder.BUFFER_SIZE = 512 * 1024

async function main() {
  const skipMongo = process.env.SKIP_MONGO === "true"
  const useMemoryDb = process.env.USE_MEMORY_DB === "true"
  let mongoUri =
    process.env.MONGO_URI ?? "mongodb://localhost:27017/pokemon-auto-chess"

  if (skipMongo) {
    logger.info("Skipping MongoDB (SKIP_MONGO=true). No bot opponents will be available.")
  } else {
    if (useMemoryDb) {
      logger.info("Starting in-memory MongoDB...")
      const { MongoMemoryServer } = await import("mongodb-memory-server")
      const mongod = await MongoMemoryServer.create()
      mongoUri = mongod.getUri()
      logger.info(`In-memory MongoDB started at ${mongoUri}`)
    }

    logger.info("Connecting to MongoDB...")
    try {
      await connect(mongoUri)
      logger.info("Connected to MongoDB")
    } catch (err) {
      logger.warn(
        "Could not connect to MongoDB. Bot opponents will not be available.",
        err
      )
      logger.warn("Starting with empty bot pool - games may have fewer opponents.")
    }
  }

  logger.info("Starting training server...")
  await startTrainingServer()
}

main().catch((err) => {
  logger.error("Fatal error:", err)
  process.exit(1)
})
