// server/server.js
import express from 'express';
import cookieParser from 'cookie-parser';
import morgan from 'morgan';
import mongoose from 'mongoose';
import path from 'path';
import cors from 'cors';
import { fileURLToPath } from 'url';

import { CONFIG } from './utils/config.js';
import { logger } from './utils/logger.js';
import { notFound, errorHandler } from './middleware/error.js';

import authRoutes from './routes/auth.routes.js';
import analyzeRoutes from './routes/analyze.routes.js';
import sightingsRoutes from './routes/sightings.routes.js';
import sseRoutes from './routes/sse.routes.js';
import configRoutes from './routes/config.routes.js';

// ESM-safe __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Resolve project root (.../plant_recognition)
const PROJECT_ROOT = path.resolve(__dirname, '..');
// Normalized, cross-platform locations
const PUBLIC_DIR = path.resolve(PROJECT_ROOT, 'public');
const UPLOADS_DIR = path.resolve(PROJECT_ROOT, 'uploads');

const app = express();

// Basic middleware
app.use(morgan('dev'));
app.use(cookieParser());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// CORS (adjust origin if you deploy frontend separately)
app.use(
  cors({
    origin: true,
    credentials: true,
  })
);

// Static files
app.use('/uploads', express.static(UPLOADS_DIR));
app.use(express.static(PUBLIC_DIR));

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/analyze', analyzeRoutes);
app.use('/api/sightings', sightingsRoutes);
app.use('/api/events', sseRoutes);
app.use('/', configRoutes);

// Frontend routes
app.get('/', (_req, res) => res.sendFile(path.join(PUBLIC_DIR, 'index.html')));
app.get('/app', (_req, res) => res.sendFile(path.join(PUBLIC_DIR, 'app.html')));

// Health endpoint (handy for debugging/deploys)
app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    env: CONFIG.NODE_ENV,
    db: mongoose.connection.readyState, // 1 = connected
    time: new Date().toISOString(),
  });
});

// 404 + error handlers
app.use(notFound);
app.use(errorHandler);

// DB connect + start server
(async () => {
  try {
    await mongoose.connect(CONFIG.MONGODB_URI, {
      // modern mongoose defaults, safe across versions
      serverSelectionTimeoutMS: 15000,
    });
    logger.info('MongoDB connected');

    app.listen(CONFIG.PORT, () => {
      logger.info(`Server running on http://localhost:${CONFIG.PORT}`);
      logger.info(`Serving public from: ${PUBLIC_DIR}`);
      logger.info(`Serving uploads from: ${UPLOADS_DIR}`);
    });
  } catch (err) {
    logger.error('Failed to start server', err);
    process.exit(1);
  }
})();
