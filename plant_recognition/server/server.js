import express from "express";
import cookieParser from "cookie-parser";
import morgan from "morgan";
import mongoose from "mongoose";
import path from "path";
import cors from "cors";
import { fileURLToPath } from "url";
import ejs from "ejs";

import { CONFIG } from "./utils/config.js";
import { logger } from "./utils/logger.js";
import { notFound, errorHandler } from "./middleware/error.js";
import { requireAuth } from "./middleware/auth.js";

import authRoutes from "./routes/auth.routes.js";
import analyzeRoutes from "./routes/analyze.routes.js";
import sightingsAPIRoutes from "./routes/sightings.routes.js";
import sseRoutes from "./routes/sse.routes.js";
import configRoutes from "./routes/config.routes.js";

import User from "./models/User.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "..");
const PUBLIC_DIR = path.resolve(PROJECT_ROOT, "public");
const UPLOADS_DIR = path.resolve(PROJECT_ROOT, "uploads");
const VIEWS_DIR = path.resolve(PUBLIC_DIR, "views");

const app = express();

app.set("view engine", "ejs");
app.set("views", VIEWS_DIR);

app.use(morgan("dev"));
app.use(cookieParser());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

app.use(
  cors({
    origin: true,
    credentials: true,
  })
);

app.use("/uploads", express.static(UPLOADS_DIR));
app.use(express.static(PUBLIC_DIR));

app.use("/api/auth", authRoutes);
app.use("/api/analyze", analyzeRoutes);
app.use("/api/sightings", sightingsAPIRoutes);
app.use("/api/events", sseRoutes);
app.use("/", configRoutes);

async function attachUser(req, res, next) {
  try {
    const user = await User.findById(req.auth.userId).select("username");
    res.locals.username = user?.username || "User";
    next();
  } catch (e) {
    next(e);
  }
}

app.get("/", (_req, res) => {
  res.render("index");
});

app.get("/app", requireAuth, attachUser, (_req, res) => {
  res.render("app");
});

app.get("/settings", requireAuth, attachUser, (_req, res) => {
  res.render("settings");
});

app.get("/sightings", requireAuth, attachUser, (_req, res) => {
  res.render("sightings");
});

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    env: CONFIG.NODE_ENV,
    db: mongoose.connection.readyState,
    time: new Date().toISOString(),
  });
});

app.use(notFound);
app.use(errorHandler);

(async () => {
  try {
    await mongoose.connect(CONFIG.MONGODB_URI, {
      serverSelectionTimeoutMS: 15000,
    });
    logger.info("MongoDB connected");

    app.listen(CONFIG.PORT, () => {
      logger.info(`Server running on http://localhost:${CONFIG.PORT}`);
      logger.info(`Serving public from: ${PUBLIC_DIR}`);
      logger.info(`Serving views from: ${VIEWS_DIR}`);
      logger.info(`Serving uploads from: ${UPLOADS_DIR}`);
    });
  } catch (err) {
    logger.error("Failed to start server", err);
    process.exit(1);
  }
})();
