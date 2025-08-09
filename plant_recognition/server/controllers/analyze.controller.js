import Sighting from '../models/Sighting.js';
import { saveFile } from '../services/storage.service.js';
import { runML } from '../services/ml.service.js';
import { kickLLM } from '../services/llm.service.js';
import { publish } from '../services/sse.service.js';

export async function analyzeOnce(req, res, next) {
  try {
    const { file } = req;
    if (!file) return res.status(400).json({ error: 'Image required' });

    const { lat, lng, fromVideo } = req.body;
    const imagePath = await saveFile(file);

    // 1) ML
    const { predicted_species, confidence } = await runML(imagePath);

    // 2) Sighting doc (LLM pending)
    const doc = await Sighting.create({
      owner: req.auth.userId,
      imagePath,
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
      imageUrl: imagePath,
      predictedSpecies: predicted_species,
      confidence
    });

    // 3) Kick LLM and publish when done
    try {
      const llm = await kickLLM(doc._id, predicted_species, confidence);
      await Sighting.updateOne(
        { _id: doc._id },
        { $set: { 'analysis.llm': { summary: llm.summary || '', details: llm, status: 'completed' } } }
      );
      publish(doc.owner.toString(), { type: 'analysis_done', sightingId: doc._id, llm });
    } catch (e) {
      await Sighting.updateOne(
        { _id: doc._id },
        { $set: { 'analysis.llm.status': 'failed' } }
      );
    }

  } catch (e) { next(e); }
}
