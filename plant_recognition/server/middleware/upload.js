// server/middleware/upload.js
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

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

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOADS_DIR),
  filename: (_req, file, cb) => {
    const ts = Date.now();
    const ext = path.extname(file.originalname) || '.bin';
    const base = safeBaseName(path.basename(file.originalname, ext));
    cb(null, `${base}-${ts}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB to allow images/videos if needed
  },
});

// Export *default* so routes can `import upload from ...`
export default upload;

// (Optional) named exports if you want convenience middlewares elsewhere
export const uploadImage = upload.single('image');
export const uploadVideo = upload.single('video');
