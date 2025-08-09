import path from 'path';
import { CONFIG } from '../utils/config.js';

export async function saveFile(file) {
  // Multer already saved to disk; return the URL path
  const rel = '/uploads/' + path.basename(file.path);
  return rel;
}
