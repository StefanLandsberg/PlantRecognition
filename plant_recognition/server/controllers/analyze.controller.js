import Sighting from '../models/Sighting.js';
import { saveFile } from '../services/storage.service.js';
import { kickLLM } from '../services/llm.service.js';
import { publish } from '../services/sse.service.js';
import { PythonShell } from 'python-shell';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Use same path resolution as upload middleware
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const UPLOADS_DIR = path.resolve(PROJECT_ROOT, 'uploads');

export async function analyzeOnce(req, res, next) {
  try {
    const { file } = req;
    if (!file) return res.status(400).json({ error: 'Image required' });

    const { lat, lng, fromVideo } = req.body;
    const imageUrlPath = await saveFile(file);
    
    // Convert URL path to full filesystem path for ML model
    const fullImagePath = path.join(UPLOADS_DIR, path.basename(imageUrlPath));
    // Minimal logging for speed

    // 1) ML - Use system Python (no path specified)
    const options = {
      mode: 'text',
      scriptPath: '../python/',
      args: [fullImagePath]
    };

    let predicted_species = 'Unknown species';
    let confidence = 0.0;
    
    try {
      const results = await PythonShell.run('ml_model.py', options);
      
      if (results && results.length > 0) {
        try {
          const mlResult = JSON.parse(results[0]);
          if (!mlResult.error) {
            predicted_species = mlResult.predicted_species || 'Unknown species';
            confidence = parseFloat(mlResult.confidence) || 0.0;
          }
        } catch (parseError) {
          // Use defaults
        }
      }
    } catch (mlError) {
      // Use defaults - don't fail the request
    }

    // 2) Sighting doc (LLM pending)
    const doc = await Sighting.create({
      owner: req.auth.userId,
      imagePath: imageUrlPath,  // Store URL path in DB
      fromVideo: fromVideo === 'true',
      location: { type: 'Point', coordinates: [parseFloat(lng)||0, parseFloat(lat)||0] },
      analysis: {
        predictedSpecies: predicted_species,
        confidence,
        llm: { status: 'pending' }
      }
    });

    // Respond fast
    res.json({
      success: true,
      sightingId: doc._id,
      imageUrl: imageUrlPath,  // Return URL path for frontend
      predictedSpecies: predicted_species,
      confidence
    });

    // 3) Kick LLM and publish when done (async for speed)
    setImmediate(async () => {
      try {
        const llm = await kickLLM(doc._id, predicted_species, confidence);
        await Sighting.updateOne(
          { _id: doc._id },
          { $set: { 'analysis.llm': { summary: llm.summary || '', details: llm, status: 'completed' } } }
        );
        publish(req.auth.userId, { type: 'analysis_done', sightingId: doc._id, llm });
      } catch (e) {
        await Sighting.updateOne(
          { _id: doc._id },
          { $set: { 'analysis.llm.status': 'failed' } }
        );
      }
    });

  } catch (e) { next(e); }
}
