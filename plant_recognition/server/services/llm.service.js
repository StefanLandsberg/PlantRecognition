import { spawn } from 'child_process';
import path from 'path';

export async function kickLLM(sightingId, species, confidence) {
  // Fire-and-await pattern (you can convert to queue if needed)
  return new Promise((resolve, reject) => {
    const py = spawn('python', [path.join(process.cwd(), 'python', 'llm_integration.py'), species, String(confidence)]);
    let out = '', err = '';
    py.stdout.on('data', d => out += d.toString());
    py.stderr.on('data', d => err += d.toString());
    py.on('close', code => {
      if (code === 0) {
        try {
          // Expecting JSON with fields you want to store
          const analysis = JSON.parse(out.trim());
          resolve(analysis);
        } catch (e) { reject(new Error('Bad LLM output: ' + e.message)); }
      } else {
        reject(new Error('LLM failed: ' + err));
      }
    });
  });
}
