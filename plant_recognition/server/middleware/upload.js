// server/middleware/upload.js
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import User from '../models/User.js';

// ESM-safe __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Resolve project root (.../plant_recognition) from /server/middleware
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const UPLOADS_DIR = path.resolve(PROJECT_ROOT, 'uploads');

// Ensure uploads dir exists (Windows-safe)
if (!fs.existsSync(UPLOADS_DIR)) {
  fs.mkdirSync(UPLOADS_DIR, { recursive: true });
}

// Basic filename sanitizer
function safeBaseName(originalName) {
  const base = path.basename(originalName);
  return base.replace(/[^a-zA-Z0-9._-]/g, '_');
}

// Storage-aware multer configuration
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOADS_DIR),
  filename: (req, file, cb) => {
    const ts = Date.now();
    const ext = path.extname(file.originalname) || '.bin';
    const base = safeBaseName(path.basename(file.originalname, ext));
    const userId = req.auth?.userId || 'anonymous';
    // Label files with user ID for proper file server functionality
    cb(null, `user-${userId}-${base}-${ts}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB to allow high-quality images/videos
  },
});

// Middleware to check user storage preference before upload
export const storageAwareUpload = (fieldName) => {
  return async (req, res, next) => {
    try {
      // Get user storage preference
      if (req.auth?.userId) {
        const user = await User.findById(req.auth.userId);
        req.userStoragePreference = user?.storagePreference || 'server';
      } else {
        req.userStoragePreference = 'server'; // Default for non-authenticated requests
      }

      // Apply multer upload
      const multerMiddleware = upload.single(fieldName);
      multerMiddleware(req, res, next);
    } catch (error) {
      console.error('Storage-aware upload error:', error);
      next(error);
    }
  };
};

// Export *default* so routes can `import upload from ...`
export default upload;

// Storage-aware convenience middlewares
export const uploadImage = storageAwareUpload('image');
export const uploadVideo = storageAwareUpload('video');
export const uploadFrame = storageAwareUpload('frame');