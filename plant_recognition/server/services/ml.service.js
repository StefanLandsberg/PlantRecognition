import { spawn } from 'child_process';
import path from 'path';

// Calls /python/ml_model.py <imagePath>
export function runML(fileUrlPath) {
  // In static server, fileUrlPath like "/uploads/xyz.jpg"
  // Construct filesystem path if needed by Python; or pass URL if your Python can read URLs.
  const imagePath = path.join(process.cwd(), 'public', fileUrlPath.replace(/^\/+/, ''));

  return new Promise((resolve, reject) => {
    const py = spawn('python', [path.join(process.cwd(), 'python', 'ml_model.py'), imagePath]);
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
        } catch (e) { reject(new Error('Bad ML output: ' + e.message)); }
      } else {
        reject(new Error('ML failed: ' + err));
      }
    });
  });
}
