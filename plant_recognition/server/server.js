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
import accountRoutes from "./routes/account.routes.js";

import User from "./models/User.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, "..");
const PUBLIC_DIR = path.resolve(PROJECT_ROOT, "public");
const UPLOADS_DIR = path.resolve(PROJECT_ROOT, "uploads");
const VIEWS_DIR = path.resolve(PUBLIC_DIR, "views");

import en from "../public/js/languages/en.json" assert { type: "json" };
import afr from "../public/js/languages/afr.json" assert { type: "json" };
import zulu from "../public/js/languages/zulu.json" assert { type: "json" };

const languages = {
  en,
  afr,
  zulu
};

const app = express();

app.set("view engine", "ejs");
app.set("views", VIEWS_DIR);

app.use(morgan("dev"));
app.use(cookieParser());
app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true }));

app.use("/uploads", express.static(UPLOADS_DIR));
app.use(express.static(PUBLIC_DIR));

function setLanguage(req, res, next) {
  const availableLangs = Object.keys(languages);
  let userLang = 'en'; // Set default language

  // 1. Check for a language parameter in the URL (e.g., /settings?lang=afr)
  // This has the highest priority and is used to CHANGE the language.
  if (req.query.lang && availableLangs.includes(req.query.lang)) {
    userLang = req.query.lang;
    // Set a cookie that expires in 30 days to remember the choice.
    res.cookie('lang', userLang, { maxAge: 1000 * 60 * 60 * 24 * 30, httpOnly: true });
  }
  // 2. If no parameter, check if a language cookie already exists.
  // This is used for all subsequent page loads.
  else if (req.cookies.lang && availableLangs.includes(req.cookies.lang)) {
    userLang = req.cookies.lang;
  }

  // 3. Make the text and the current language code available to all EJS templates.
  res.locals.text = languages[userLang];
  res.locals.currentLang = userLang; // This is needed for Step 2
  next();
}

app.use(setLanguage);

app.use("/api/account", accountRoutes);

app.use(
  cors({
    origin: true,
    credentials: true,
  })
);

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

app.get("/account", requireAuth, attachUser, (_req, res) => {
  res.render("account");
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
