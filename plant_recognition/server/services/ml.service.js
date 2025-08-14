// server/services/ml.service.js
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

function urlPathToFs(fileUrlPath) {
  // "/uploads/xyz.jpg" -> "<cwd>/uploads/xyz.jpg"
  const rel = String(fileUrlPath || '').replace(/^\/+/, '');
  const p1 = path.join(process.cwd(), rel);
  if (fs.existsSync(p1)) return p1;

  // legacy/fallback (if someone saved under /public)
  const p2 = path.join(process.cwd(), 'public', rel);
  if (fs.existsSync(p2)) return p2;

  // last resort: just return p1 (better error message from Python)
  return p1;
}

// Calls /python/ml_model.py <imagePathFs>
export function runML(fileUrlPath) {
  const imagePathFs = urlPathToFs(fileUrlPath);

  return new Promise((resolve, reject) => {
    const py = spawn('python', [path.join(process.cwd(), 'python', 'ml_model.py'), imagePathFs], {
      env: process.env,
    });

    let out = '', err = '';
    py.stdout.on('data', d => out += d.toString());
    py.stderr.on('data', d => err += d.toString());
    py.on('close', code => {
      if (code === 0) {
        try {
          const parsed = JSON.parse(out.trim());
          const predicted_species = parsed.predicted_species || 'Unknown';
          const confidence = parsed.confidence ?? 0.5;
          resolve({ predicted_species, confidence, raw: parsed });
        } catch (e) {
          reject(new Error('Bad ML output: ' + e.message + '\n' + out));
        }
      } else {
        reject(new Error('ML failed: ' + err));
      }
    });
  });
}
