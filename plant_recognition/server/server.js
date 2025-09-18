import express from "express";
import cookieParser from "cookie-parser";
import morgan from "morgan";
import mongoose from "mongoose";
import path from "path";
import cors from "cors";
import { fileURLToPath } from "url";
import ejs from "ejs";
import { createServer } from "http";
// import WebSocket, { WebSocketServer } from "ws";
// this caused an error so I used the following more reliable code
import pkg from "ws";
const WebSocket = pkg;
const WebSocketServer = pkg.WebSocketServer || pkg.Server;

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

import en from "../public/js/languages/en.json" with { type: "json" };
import afr from "../public/js/languages/afr.json" with { type: "json" };
import zulu from "../public/js/languages/zulu.json" with { type: "json" };

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

// Serve mobile companion app as static files
app.use('/mobile', express.static(path.resolve(PROJECT_ROOT, 'mobile_companion')));

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

// // Companion code registration endpoint (must be before notFound middleware)
app.post('/api/companion/register', requireAuth, async (req, res) => {
  try {
    const { companionCode } = req.body;
    const userId = req.auth.userId;

    logger.info(`Registering companion code: ${companionCode} for user: ${userId}`);

    if (!companionCode || !/^\d{6}$/.test(companionCode)) {
      logger.error(`Invalid companion code format: ${companionCode}`);
      return res.status(400).json({ error: 'Invalid companion code' });
    }

    // Register the code for this user
    registeredCodes.set(companionCode, userId);

    logger.info(`Code ${companionCode} registered successfully for user ${userId}`);
    logger.info(`Total registered codes: ${registeredCodes.size}`);

    // Set expiry (24 hours)
    setTimeout(() => {
      registeredCodes.delete(companionCode);
      logger.info(`Code ${companionCode} expired and removed`);
    }, 24 * 60 * 60 * 1000);

    res.json({ success: true });
  } catch (error) {
    logger.error('Companion code registration error:', error);
    res.status(500).json({ error: 'Registration failed' });
  }
});

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

// Mobile Companion WebSocket Management
const companionConnections = new Map(); // companionCode -> { userId, ws }
const registeredCodes = new Map(); // companionCode -> userId

function broadcastToUser(userId, message) {
  // Find companion connection for this user
  for (const [code, connection] of companionConnections.entries()) {
    if (connection.userId === userId && connection.ws.readyState === WebSocket.OPEN) {
      connection.ws.send(JSON.stringify(message));
    }
  }
}

(async () => {
  try {
    await mongoose.connect(CONFIG.MONGODB_URI, {
      serverSelectionTimeoutMS: 15000,
    });
    logger.info("MongoDB connected");

    // Create HTTP server
    const server = createServer(app);

    // Create WebSocket server
    const wss = new WebSocketServer({
      server,
      path: '/mobile-companion'
    });

    wss.on('connection', (ws, req) => {
      logger.info('Mobile companion connection attempt');

      ws.on('message', async (data) => {
        try {
          const message = JSON.parse(data.toString());

          if (message.type === 'connect') {
            await handleCompanionConnect(ws, message.companionCode);
          } else if (message.type === 'image_capture') {
            await handleImageCapture(ws, message);
          }
        } catch (error) {
          logger.error('WebSocket message error:', error);
          ws.send(JSON.stringify({
            type: 'error',
            message: 'Invalid message format'
          }));
        }
      });

      ws.on('close', () => {
        // Remove from connections
        for (const [code, connection] of companionConnections.entries()) {
          if (connection.ws === ws) {
            companionConnections.delete(code);
            logger.info(`Mobile companion disconnected: ${code}`);
            break;
          }
        }
      });
    });

    async function handleCompanionConnect(ws, companionCode) {
      logger.info(`Attempting to connect with companion code: ${companionCode}`);

      // Validate companion code format
      if (!companionCode || !/^\d{6}$/.test(companionCode)) {
        logger.info(`Invalid companion code format: ${companionCode}`);
        ws.send(JSON.stringify({
          type: 'connection_failed',
          message: 'Invalid companion code format'
        }));
        return;
      }

      // Check if code is registered
      const userId = registeredCodes.get(companionCode);
      logger.info(`Looking up companion code ${companionCode}, found userId: ${userId}`);
      logger.info(`Registered codes:`, Array.from(registeredCodes.keys()));

      if (!userId) {
        ws.send(JSON.stringify({
          type: 'connection_failed',
          message: 'Invalid or expired companion code'
        }));
        return;
      }

      try {
        const user = await User.findById(userId);
        if (!user) {
          ws.send(JSON.stringify({
            type: 'connection_failed',
            message: 'User not found'
          }));
          return;
        }

        // Store the connection
        companionConnections.set(companionCode, {
          userId: userId,
          ws: ws
        });

        ws.send(JSON.stringify({
          type: 'connection_confirmed',
          user: { username: user.username }
        }));

        logger.info(`Mobile companion connected: ${companionCode} -> ${user.username}`);
      } catch (error) {
        logger.error('Companion connection error:', error);
        ws.send(JSON.stringify({
          type: 'connection_failed',
          message: 'Connection verification failed'
        }));
      }
    }

    async function handleImageCapture(ws, message) {
      try {
        // Find the user for this WebSocket connection
        let userId = null;
        for (const [code, connection] of companionConnections.entries()) {
          if (connection.ws === ws) {
            userId = connection.userId;
            break;
          }
        }

        if (!userId) {
          ws.send(JSON.stringify({
            type: 'error',
            message: 'Not authenticated'
          }));
          return;
        }

        // Convert base64 image to buffer
        const imageBuffer = Buffer.from(message.image, 'base64');

        // Save image to uploads directory (like multer would do)
        const filename = `mobile_capture_${Date.now()}.jpg`;
        const filePath = path.resolve(UPLOADS_DIR, filename);

        // Write image buffer to disk
        const fs = await import('fs');
        await fs.promises.writeFile(filePath, imageBuffer);

        // Create a proper file object that matches multer's format
        const mockFile = {
          buffer: imageBuffer,
          mimetype: 'image/jpeg',
          originalname: filename,
          fieldname: 'image',
          encoding: '7bit',
          size: imageBuffer.length,
          path: filePath  // This is what storage.service.js expects
        };

        // Import the analyze controller
        const analyzeController = await import('./controllers/analyze.controller.js');

        // Create mock request/response objects that match the real ones
        const mockReq = {
          file: mockFile,
          body: {
            lat: message.lat || 0,
            lng: message.lng || 0,
            fromVideo: false
          },
          auth: { userId: userId }
        };

        const mockRes = {
          json: (result) => {
            // Send result back to mobile AND trigger SSE updates like normal
            ws.send(JSON.stringify({
              type: 'classification_result',
              result: result
            }));

            // The analyze controller should handle SSE updates automatically
            // through the publish() calls in the controller
          },
          status: (code) => mockRes,
          send: (data) => mockRes
        };

        const mockNext = (error) => {
          logger.error('Mobile image analysis error:', error);
          ws.send(JSON.stringify({
            type: 'error',
            message: 'Analysis failed'
          }));
        };

        // Trigger the analysis - this will:
        // 1. Save the image file
        // 2. Run ML classification
        // 3. Create sighting in database
        // 4. Trigger LLM analysis
        // 5. Send SSE updates to main app
        // 6. Update map in real-time
        await analyzeController.analyzeOnce(mockReq, mockRes, mockNext);

      } catch (error) {
        logger.error('Image capture handling error:', error);
        ws.send(JSON.stringify({
          type: 'error',
          message: 'Failed to process image'
        }));
      }
    }

    server.listen(CONFIG.PORT, '0.0.0.0', () => {
      logger.info(`Main app: http://localhost:${CONFIG.PORT}`);
      logger.info(`Mobile companion: http://192.168.101.251:${CONFIG.PORT}/mobile`);
      logger.info(`Serving public from: ${PUBLIC_DIR}`);
      logger.info(`Serving views from: ${VIEWS_DIR}`);
      logger.info(`Serving uploads from: ${UPLOADS_DIR}`);
    });
  } catch (err) {
    logger.error("Failed to start server", err);
    process.exit(1);
  }
})();
