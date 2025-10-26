import express from "express";
import cookieParser from "cookie-parser";
import morgan from "morgan";
import mongoose from "mongoose";
import path from "path";
import cors from "cors";
import { fileURLToPath } from "url";
import ejs from "ejs";
import { createServer as createHttpServer } from "http";
import { createServer as createHttpsServer } from "https";
import fs from "fs";
import WebSocket, { WebSocketServer } from "ws";

import { CONFIG } from "./utils/config.js";
import { logger } from "./utils/logger.js";
import { notFound, errorHandler } from "./middleware/error.js";
import { requireAuth } from "./middleware/auth.js";
import QRCode from "qrcode";

import authRoutes from "./routes/auth.routes.js";
import analyzeRoutes from "./routes/analyze.routes.js";
import sightingsAPIRoutes from "./routes/sightings.routes.js";
import sseRoutes from "./routes/sse.routes.js";
import configRoutes from "./routes/config.routes.js";
import accountRoutes from "./routes/account.routes.js";
import storageRoutes from "./routes/storage.routes.js";
import alertsRoutes from "./routes/alerts.routes.js";
import videoSessionRoutes from "./routes/videoSession.routes.js";


import User from "./models/User.js";
import VideoSession from "./models/VideoSession.js";
import { saveFile } from "./services/storage.service.js";
import { warmLLM } from "./services/llm.service.js";

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
app.use("/api/storage", storageRoutes);
app.use("/api/sightings", sightingsAPIRoutes);
app.use("/api/alerts", alertsRoutes);
app.use("/api/video-sessions", (req, res, next) => {
  console.log('Video sessions route accessed:', req.method, req.url);
  next();
}, videoSessionRoutes);

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
    registeredCodes.set(companionCode, { userId, storagePreference: 'server' });

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

// QR Code endpoint for mobile access
app.get('/qr-mobile', requireAuth, async (req, res) => {
  try {
    const userId = req.auth.userId;
    logger.info(`QR code request from user: ${userId}`);

    // Get user's storage preference
    const user = await User.findById(userId).select('storagePreference');
    const storagePreference = user?.storagePreference || 'server';
    logger.info(`User ${userId} storage preference: ${storagePreference}`);

    const protocol = req.secure || req.get('X-Forwarded-Proto') === 'https' ? 'https' : 'http';

    // Generate a 6-digit companion code
    const companionCode = Math.floor(100000 + Math.random() * 900000).toString();
    logger.info(`Generated companion code: ${companionCode} for user: ${userId}`);

    // Register the code for this user (24 hour expiry) with storage preference
    registeredCodes.set(companionCode, { userId, storagePreference });
    setTimeout(() => {
      registeredCodes.delete(companionCode);
      logger.info(`QR companion code ${companionCode} expired and removed`);
    }, 24 * 60 * 60 * 1000);

    // Include companion code and storage preference in mobile URL for auto-connect
    const mobileUrl = `${protocol}://192.168.101.251:${CONFIG.PORT}/mobile?code=${companionCode}&storage=${storagePreference}`;

    const qrCodeDataURL = await QRCode.toDataURL(mobileUrl, {
      width: 256,
      margin: 2,
      color: {
        dark: '#000000',
        light: '#ffffff'
      }
    });

    const response = {
      success: true,
      qrCode: qrCodeDataURL,
      mobileUrl: mobileUrl,
      companionCode: companionCode,
      storagePreference: storagePreference
    };

    logger.info(`Sending QR response with companionCode: ${companionCode} and storage: ${storagePreference}`);
    res.json(response);
  } catch (error) {
    logger.error('QR code generation failed:', error);
    res.status(500).json({ success: false, error: 'Failed to generate QR code' });
  }
});

app.use(notFound);
app.use(errorHandler);

// Mobile Companion WebSocket Management
const companionConnections = new Map(); // companionCode -> { userId, ws, username, storagePreference, activeSessionId, sessionStartedAt, sessionLocation }
const registeredCodes = new Map(); // companionCode -> userId

function getCompanionContext(ws) {
  for (const [code, connection] of companionConnections.entries()) {
    if (connection.ws === ws) {
      return { code, connection };
    }
  }
  return null;
}

function getCompanionCountForUser(userId) {
  let count = 0;
  for (const [code, connection] of companionConnections.entries()) {
    if (connection.userId === userId && connection.ws.readyState === WebSocket.OPEN) {
      count++;
    }
  }
  return count;
}

function resolveStoragePreference(connectionInfo, requestedPreference) {
  if (requestedPreference && typeof requestedPreference === "string") {
    return requestedPreference;
  }
  return connectionInfo?.storagePreference || "server";
}

function sendRpcSuccess(ws, type, requestId, extra = {}) {
  const payload = {
    type,
    success: true,
    ...extra,
  };
  if (requestId) payload.requestId = requestId;
  ws.send(JSON.stringify(payload));
}

function sendRpcError(ws, type, requestId, message) {
  const payload = {
    type,
    success: false,
    message,
  };
  if (requestId) payload.requestId = requestId;
  ws.send(JSON.stringify(payload));
}

function parseCoordinates(lat, lng) {
  const parsedLat = parseFloat(lat);
  const parsedLng = parseFloat(lng);
  if (
    Number.isFinite(parsedLat) &&
    Number.isFinite(parsedLng) &&
    Math.abs(parsedLat) <= 90 &&
    Math.abs(parsedLng) <= 180
  ) {
    return { lat: parsedLat, lng: parsedLng };
  }
  return null;
}

async function finalizeVideoSession(sessionId, userId) {
  if (!sessionId) return;
  const session = await VideoSession.findOne({ _id: sessionId, owner: userId });
  if (!session) return;
  if (!session.endTime) {
    session.endTime = new Date();
    session.duration = Math.floor(
      (session.endTime - session.startTime) / 1000
    );
    await session.save();
  }
}

async function recordVideoDetection(sessionId, userId, data) {
  if (!sessionId || !mongoose.Types.ObjectId.isValid(sessionId)) return;
  if (!data?.sightingId) return;
  const detectionDoc = {
    timestamp: Number.isFinite(Number(data.timestamp))
      ? Number(data.timestamp)
      : 0,
    frameUrl: data.frameUrl || null,
    sightingId: data.sightingId,
    status: "pending",
  };
  await VideoSession.updateOne(
    { _id: sessionId, owner: userId },
    { $push: { detections: detectionDoc } }
  );

  if (data.frameUrl) {
    await VideoSession.updateOne(
      {
        _id: sessionId,
        owner: userId,
        $or: [
          { thumbnailUrl: { $exists: false } },
          { thumbnailUrl: null },
          { thumbnailUrl: '' }
        ],
      },
      { $set: { thumbnailUrl: data.frameUrl } }
    ).catch(() => {});
  }
}

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

    // Create HTTPS server with self-signed certificates
    let server;
    try {
      const httpsOptions = {
        key: fs.readFileSync(path.resolve(__dirname, 'key.pem')),
        cert: fs.readFileSync(path.resolve(__dirname, 'cert.pem'))
      };
      server = createHttpsServer(httpsOptions, app);
      logger.info("HTTPS server created with SSL certificates");
    } catch (error) {
      logger.warn("SSL certificates not found, falling back to HTTP:", error.message);
      server = createHttpServer(app);
    }

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
          } else if (message.type === 'video_session_start') {
            await handleVideoSessionStart(ws, message);
          } else if (message.type === 'video_session_stop') {
            await handleVideoSessionStop(ws, message);
          } else if (message.type === 'video_upload') {
            await handleVideoUpload(ws, message);
          } else {
            logger.warn('Unknown mobile companion message type:', message.type);
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
            if (connection.activeSessionId) {
              finalizeVideoSession(connection.activeSessionId, connection.userId).catch((err) =>
                logger.warn('Failed to finalize mobile video session on disconnect:', err)
              );
            }
            const userId = connection.userId;
            companionConnections.delete(code);
            logger.info(`Mobile companion disconnected: ${code}`);
            logger.info(`Remaining companions for user ${userId}: ${getCompanionCountForUser(userId)}`);
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
      const codeData = registeredCodes.get(companionCode);
      logger.info(`Looking up companion code ${companionCode}, found:`, codeData);
      logger.info(`Registered codes:`, Array.from(registeredCodes.keys()));

      if (!codeData || !codeData.userId) {
        ws.send(JSON.stringify({
          type: 'connection_failed',
          message: 'Invalid or expired companion code'
        }));
        return;
      }

      const userId = codeData.userId;
      const registeredStoragePreference = codeData.storagePreference;

      try {
        const user = await User.findById(userId);
        if (!user) {
          ws.send(JSON.stringify({
            type: 'connection_failed',
            message: 'User not found'
          }));
          return;
        }

        // Use the storage preference from when the QR code was generated (ensures consistency)
        const storagePreference = registeredStoragePreference || user.storagePreference || 'server';

        // Debug: Show all current connections before disconnection
        logger.info(`=== COMPANION DISCONNECTION DEBUG ===`);
        logger.info(`New companion ${companionCode} connecting for user ${userId}`);
        logger.info(`Current companion connections:`, Array.from(companionConnections.entries()).map(([code, conn]) => ({
          code,
          userId: conn.userId,
          wsState: conn.ws.readyState,
          username: conn.username
        })));

        // Disconnect any existing companions for this user (single companion session)
        let disconnectedCount = 0;
        for (const [existingCode, existingConnection] of companionConnections.entries()) {
          logger.info(`Checking existing companion ${existingCode}: userId=${existingConnection.userId}, wsState=${existingConnection.ws.readyState}`);
          
          if (existingConnection.userId === userId) {
            if (existingConnection.ws.readyState === WebSocket.OPEN) {
              logger.info(`Disconnecting existing companion ${existingCode} for user ${userId} (new companion ${companionCode} connecting)`);
              existingConnection.ws.send(JSON.stringify({
                type: 'session_replaced',
                message: 'A new companion session has been started. This session will be disconnected.'
              }));
              existingConnection.ws.close();
              companionConnections.delete(existingCode);
              disconnectedCount++;
            } else {
              logger.info(`Existing companion ${existingCode} for user ${userId} has closed WebSocket (state: ${existingConnection.ws.readyState}), removing from map`);
              companionConnections.delete(existingCode);
            }
          }
        }
        
        logger.info(`Disconnected ${disconnectedCount} existing companion(s) for user ${userId}`);
        logger.info(`=== END COMPANION DISCONNECTION DEBUG ===`);

        // Store the new connection
        companionConnections.set(companionCode, {
          userId: userId,
          ws: ws,
          username: user.username,
          storagePreference: storagePreference,
          sessionLocation: null,
          lastKnownCoords: null,
        });

        ws.send(JSON.stringify({
          type: 'connection_confirmed',
          user: { username: user.username },
          storagePreference: storagePreference
        }));

        logger.info(`Mobile companion connected: ${companionCode} -> ${user.username} (userId: ${userId})`);
        logger.info(`Active companions for user ${userId}: ${getCompanionCountForUser(userId)}`);
        logger.info(`All active companions:`, Array.from(companionConnections.entries()).map(([code, conn]) => ({
          code,
          userId: conn.userId,
          username: conn.username
        })));
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
        const ctx = getCompanionContext(ws);
        const connectionInfo = ctx?.connection;
        const userId = connectionInfo?.userId;

        if (!userId) {
          ws.send(JSON.stringify({
            type: 'error',
            message: 'Not authenticated'
          }));
          return;
        }

        // Convert base64 image to buffer
        const imageBuffer = Buffer.from(message.image, 'base64');

        // Save image to uploads directory with user ID labeling (like multer would do)
        const timestamp = Date.now();
        const filename = `user-${userId}-mobile_capture-${timestamp}.jpg`;
        const filePath = path.resolve(UPLOADS_DIR, filename);

        // Write image buffer to disk
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

        const storagePreference = resolveStoragePreference(
          connectionInfo,
          message.storagePreference
        );
        const parsedCoords = parseCoordinates(message.lat, message.lng);
        if (parsedCoords && connectionInfo) {
          connectionInfo.lastKnownCoords = parsedCoords;
          connectionInfo.sessionLocation = parsedCoords;
        }
        const coords =
          parsedCoords ||
          connectionInfo?.sessionLocation ||
          connectionInfo?.lastKnownCoords ||
          null;

        const activeSessionId =
          message.videoSessionId || connectionInfo?.activeSessionId || null;
        const derivedTimestamp = connectionInfo?.sessionStartedAt
          ? Math.round((Date.now() - connectionInfo.sessionStartedAt) / 1000)
          : null;
        const frameTimestamp = Number.isFinite(Number(message.videoTimestamp))
          ? Number(message.videoTimestamp)
          : derivedTimestamp;

        // Create mock request/response objects that match the real ones
        const mockReq = {
          file: mockFile,
          body: {
            lat: coords?.lat || 0,
            lng: coords?.lng || 0,
            storageType: storagePreference,
          },
          auth: { userId: userId },
          userStoragePreference: storagePreference
        };

        if (message.localFileId) {
          mockReq.body.localFileId = message.localFileId;
        }

        if (activeSessionId) {
          mockReq.body.videoSessionId = activeSessionId;
          if (frameTimestamp != null) {
            mockReq.body.videoTimestamp = frameTimestamp;
          }
          if (message.isLiveStream) {
            mockReq.body.fromVideo = "true";
          }
        }

        const mockRes = {
          json: (result) => {
            (async () => {
              // Only send classification_result if not filtered out for low confidence
              if (result && (result.success !== false || result.reason !== 'low_confidence')) {
                ws.send(JSON.stringify({
                  type: 'classification_result',
                  result: result
                }));
              }

              if (
                activeSessionId &&
                result &&
                result.sightingId &&
                !result.duplicate
              ) {
                await recordVideoDetection(activeSessionId, userId, {
                  timestamp: frameTimestamp ?? 0,
                  frameUrl: result.imageUrl || null,
                  sightingId: result.sightingId
                });
              }
            })().catch((err) => {
              logger.warn('Mobile classification post-processing failed:', err);
            });
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

    async function handleVideoSessionStart(ws, message) {
      const requestId = message.requestId || null;
      try {
        const ctx = getCompanionContext(ws);
        const connectionInfo = ctx?.connection;
        const userId = connectionInfo?.userId;

        if (!userId) {
          sendRpcError(ws, 'video_session_started', requestId, 'Not authenticated');
          return;
        }

        const coords = parseCoordinates(message.lat, message.lng);
        const session = await VideoSession.create({
          owner: userId,
          sessionType: message.sessionType || 'live_video',
          location: coords
            ? { type: 'Point', coordinates: [coords.lng, coords.lat] }
            : undefined,
          storageType: resolveStoragePreference(connectionInfo, message.storagePreference),
        });

        if (connectionInfo) {
          connectionInfo.activeSessionId = session._id;
          connectionInfo.sessionStartedAt = Date.now();
          if (coords) {
            connectionInfo.sessionLocation = coords;
            connectionInfo.lastKnownCoords = coords;
          } else if (!connectionInfo.sessionLocation && connectionInfo.lastKnownCoords) {
            connectionInfo.sessionLocation = connectionInfo.lastKnownCoords;
          }
        }

        sendRpcSuccess(ws, 'video_session_started', requestId, {
          sessionId: session._id,
          startTime: session.startTime,
        });
      } catch (error) {
        logger.error('Mobile video session start failed:', error);
        sendRpcError(ws, 'video_session_started', requestId, 'Failed to start video session');
      }
    }

    async function handleVideoSessionStop(ws, message) {
      const requestId = message.requestId || null;
      try {
        const ctx = getCompanionContext(ws);
        const connectionInfo = ctx?.connection;
        const userId = connectionInfo?.userId;

        if (!userId) {
          sendRpcError(ws, 'video_session_stopped', requestId, 'Not authenticated');
          return;
        }

        const sessionId = message.sessionId || connectionInfo?.activeSessionId;
        if (!sessionId) {
          sendRpcError(ws, 'video_session_stopped', requestId, 'Session not found');
          return;
        }

        const session = await VideoSession.findOne({ _id: sessionId, owner: userId });
        if (!session) {
          sendRpcError(ws, 'video_session_stopped', requestId, 'Session not found');
          return;
        }

        session.endTime = new Date();
        const providedDuration = Number.isFinite(Number(message.duration))
          ? Number(message.duration)
          : Math.floor((session.endTime - session.startTime) / 1000);
        session.duration = providedDuration;
        await session.save();

        if (connectionInfo) {
          delete connectionInfo.activeSessionId;
          delete connectionInfo.sessionStartedAt;
          if (connectionInfo.sessionLocation) {
            connectionInfo.lastKnownCoords = connectionInfo.sessionLocation;
          }
          delete connectionInfo.sessionLocation;
        }

        sendRpcSuccess(ws, 'video_session_stopped', requestId, {
          sessionId,
          duration: session.duration,
        });
      } catch (error) {
        logger.error('Mobile video session stop failed:', error);
        sendRpcError(ws, 'video_session_stopped', requestId, 'Failed to stop video session');
      }
    }

    async function handleVideoUpload(ws, message) {
      const requestId = message.requestId || null;
      try {
        const ctx = getCompanionContext(ws);
        const connectionInfo = ctx?.connection;
        const userId = connectionInfo?.userId;

        if (!userId) {
          sendRpcError(ws, 'video_uploaded', requestId, 'Not authenticated');
          return;
        }

        const sessionId = message.sessionId || connectionInfo?.activeSessionId;
        if (!sessionId) {
          sendRpcError(ws, 'video_uploaded', requestId, 'Session not found');
          return;
        }

        if (!message.video) {
          sendRpcError(ws, 'video_uploaded', requestId, 'Missing video payload');
          return;
        }

        const videoBuffer = Buffer.from(message.video, 'base64');
        const mimeType = message.mimeType || 'video/webm';
        const extension = mimeType.includes('mp4') ? '.mp4' : '.webm';
        const filename = `user-${userId}-mobile_video-${Date.now()}${extension}`;
        const filePath = path.resolve(UPLOADS_DIR, filename);

        await fs.promises.writeFile(filePath, videoBuffer);

        const storagePreference = resolveStoragePreference(
          connectionInfo,
          message.storagePreference
        );

        const session = await VideoSession.findOne({ _id: sessionId, owner: userId });
        if (!session) {
          await fs.promises.unlink(filePath).catch(() => {});
          sendRpcError(ws, 'video_uploaded', requestId, 'Session not found');
          return;
        }

        const pseudoFile = {
          path: filePath,
          originalname: filename,
          mimetype: mimeType,
        };

        let videoUrl = null;
        try {
          videoUrl = await saveFile(pseudoFile, userId, storagePreference);
        } catch (error) {
          await fs.promises.unlink(filePath).catch(() => {});
          throw error;
        }

        if (videoUrl) {
          session.videoUrl = videoUrl;
          session.localVideoId = null;
        } else {
          session.videoUrl = null;
          session.localVideoId = message.localVideoId || `mobile-${Date.now()}`;
        }
        session.storageType = storagePreference;
        await session.save();

        sendRpcSuccess(ws, 'video_uploaded', requestId, {
          sessionId,
          videoUrl: session.videoUrl,
          localVideoId: session.localVideoId,
          storageType: session.storageType,
        });
      } catch (error) {
        logger.error('Mobile video upload failed:', error);
        sendRpcError(ws, 'video_uploaded', requestId, 'Failed to upload video');
      }
    }

    server.listen(CONFIG.PORT, '0.0.0.0', () => {
      const protocol = server.key ? 'https' : 'http'; // Check if HTTPS
      logger.info(`Main app: ${protocol}://localhost:${CONFIG.PORT}`);
      logger.info(`Mobile companion: ${protocol}://192.168.101.251:${CONFIG.PORT}/mobile`);
      if (protocol === 'https') {
        logger.info(`HTTPS enabled - Mobile permissions will work!`);
        logger.info(`Accept the security warning on mobile devices`);
      }
      logger.info(`Serving public from: ${PUBLIC_DIR}`);
      logger.info(`Serving views from: ${VIEWS_DIR}`);
      logger.info(`Serving uploads from: ${UPLOADS_DIR}`);

      // Best-effort LLM warm-up (non-blocking)
      setTimeout(async () => {
        try {
          const ok = await warmLLM();
          if (ok) logger.info('[LLM Warmup] Completed');
          else logger.warn('[LLM Warmup] Skipped or failed');
        } catch (e) {
          logger.warn('[LLM Warmup] Error during warmup');
        }
      }, 0);
    });
  } catch (err) {
    logger.error("Failed to start server", err);
    process.exit(1);
  }
})();
