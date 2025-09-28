import Sighting from '../models/Sighting.js';
import User from '../models/User.js';
import { saveFile } from '../services/storage.service.js';
import { kickLLM, queueLLMAnalysis } from '../services/llm.service.js';
import { publish } from '../services/sse.service.js';
import { PythonShell } from 'python-shell';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Use same path resolution as upload middleware
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const UPLOADS_DIR = path.resolve(PROJECT_ROOT, 'uploads');

// Duplicate detection configuration
const DUPLICATE_CONFIG = {
  RADIUS_METERS: 15,           // Detection radius in meters
  TIME_WINDOW_DAYS: 30,        // How many days to look back
  MIN_PREVIOUS_CONFIDENCE: 0.7, // Minimum confidence for previous detection
  MIN_CURRENT_CONFIDENCE: 0.6   // Minimum confidence for current detection to trigger check
};

async function checkForDuplicate(species, confidence, lat, lng, userId) {
  // Skip duplicate check if confidence too low or coordinates invalid
  if (!species || confidence < DUPLICATE_CONFIG.MIN_CURRENT_CONFIDENCE || !lat || !lng) {
    return null;
  }

  try {
    const radiusInDegrees = DUPLICATE_CONFIG.RADIUS_METERS / 111000; // Approximate meters to degrees
    const timeThreshold = new Date(Date.now() - (DUPLICATE_CONFIG.TIME_WINDOW_DAYS * 24 * 60 * 60 * 1000));

    // Find nearby detections of the same species within time window
    const nearbyDetection = await Sighting.findOne({
      owner: userId,
      'analysis.predictedSpecies': { $regex: new RegExp(`^${species.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`, 'i') },
      'analysis.confidence': { $gte: DUPLICATE_CONFIG.MIN_PREVIOUS_CONFIDENCE },
      createdAt: { $gte: timeThreshold },
      isRemoved: { $ne: true }, // Don't consider removed sightings as duplicates
      location: {
        $near: {
          $geometry: {
            type: 'Point',
            coordinates: [parseFloat(lng), parseFloat(lat)]
          },
          $maxDistance: DUPLICATE_CONFIG.RADIUS_METERS
        }
      }
    }).sort({ createdAt: -1 }); // Get most recent match

    return nearbyDetection;
  } catch (error) {
    console.error('Error checking for duplicates:', error);
    return null; // On error, allow the detection (fail safe)
  }
}

export async function analyzeOnce(req, res, next) {
  try {
    const { file } = req;
    if (!file) return res.status(400).json({ error: 'Image required' });

    const { lat, lng, fromVideo } = req.body;

    // Get user storage preference
    const user = await User.findById(req.auth.userId);
    const storagePreference = user?.storagePreference || 'server';

    // Handle storage based on user preference
    const imageUrlPath = await saveFile(file, req.auth.userId, storagePreference);

    // Convert URL path to full filesystem path for ML model
    // For local storage users, still use the temp file for ML analysis
    const fullImagePath = imageUrlPath ?
      path.join(UPLOADS_DIR, path.basename(imageUrlPath)) :
      file.path; // Use temp file path for local storage users
    // Minimal logging for speed

    // 1) ML - Ultra-fast classification with 500ms timeout
    const options = {
      mode: 'text',
      scriptPath: '../python/',
      args: [fullImagePath],
      pythonOptions: ['-u', '-O']  // Unbuffered, optimized bytecode
    };

    let predicted_species = 'Unknown species';
    let confidence = 0.0;

    try {
      console.log('Running ML classification with path:', fullImagePath);
      const results = await PythonShell.run('ml_model.py', options);
      console.log('ML raw results:', results);

      if (results && results.length > 0) {
        try {
          const mlResult = JSON.parse(results[0]);
          console.log('ML parsed result:', mlResult);

          if (!mlResult.error) {
            predicted_species = mlResult.predicted_species || 'Unknown species';
            confidence = parseFloat(mlResult.confidence) || 0.0;
            console.log('ML classification successful:', { predicted_species, confidence });

            // Filter out low confidence classifications
            if (confidence < 0.5) {
              console.log('Classification rejected due to low confidence:', confidence);

              // Delete the temporary image file
              try {
                if (fs.existsSync(fullImagePath)) {
                  fs.unlinkSync(fullImagePath);
                  console.log('Deleted low confidence image:', filename);
                }
              } catch (deleteError) {
                console.error('Failed to delete low confidence image:', deleteError);
              }

              // Return early without creating sighting or sending to LLM
              return res.json({
                success: false,
                reason: 'low_confidence',
                message: 'Classification confidence too low',
                confidence: confidence
              });
            }
          } else {
            console.error('ML returned error:', mlResult.error);
          }
        } catch (parseError) {
          console.error('Failed to parse ML result:', parseError, 'Raw result:', results[0]);
          // Use defaults
        }
      } else {
        console.error('No results returned from ML classification');
      }
    } catch (mlError) {
      console.error('ML classification failed:', mlError);
      // Use defaults - don't fail the request
    }

    // 2) Check for duplicate detection (video only)
    if (fromVideo === 'true' && lat && lng) {
      const duplicateDetection = await checkForDuplicate(
        predicted_species,
        confidence,
        parseFloat(lat),
        parseFloat(lng),
        req.auth.userId
      );

      if (duplicateDetection) {
        // Return duplicate response without creating new database entry
        return res.json({
          success: true,
          duplicate: true,
          sightingId: duplicateDetection._id,
          imageUrl: imageUrlPath,
          predictedSpecies: predicted_species,
          confidence: confidence,
          originalDetection: {
            id: duplicateDetection._id,
            species: duplicateDetection.analysis.predictedSpecies,
            confidence: duplicateDetection.analysis.confidence,
            detectedAt: duplicateDetection.createdAt,
            daysAgo: Math.floor((Date.now() - duplicateDetection.createdAt) / (1000 * 60 * 60 * 24))
          }
        });
      }
    }

    // 3) Sighting doc (LLM pending)
    const doc = await Sighting.create({
      owner: req.auth.userId,
      imagePath: imageUrlPath,  // Store URL path in DB (null for local storage)
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
      imageUrl: imageUrlPath,  // Return URL path for frontend (null = use local storage)
      storageType: storagePreference,
      predictedSpecies: predicted_species,
      confidence
    });

    // Immediately publish new sighting for real-time map updates
    publish(req.auth.userId, {
      type: 'new_sighting',
      sighting: {
        _id: doc._id,
        analysis: doc.analysis,
        location: doc.location,
        createdAt: doc.createdAt,
        imagePath: doc.imagePath
      }
    });

    // 3) Queue LLM analysis for parallel processing - analyze everything
    setImmediate(() => {
      // Capture userId in closure before async callback
      const userId = req.auth.userId;
      const sightingId = doc._id;

      const llmCallback = async (llm) => {
        try {
          // Batch database update and SSE publish for efficiency
          const [updateResult] = await Promise.allSettled([
            Sighting.updateOne(
              { _id: sightingId },
              { $set: { 'analysis.llm': { summary: llm.summary || '', details: llm.details, status: 'completed' } } }
            ),
            Promise.resolve().then(() => {
              publish(userId, { type: 'analysis_done', sightingId: sightingId, llm });
              console.log('SSE analysis_done event published for:', sightingId, 'to user:', userId);
            })
          ]);

          if (updateResult.status === 'rejected') {
            console.error('Database update failed:', updateResult.reason);
          }
        } catch (error) {
          console.error('LLM callback execution failed:', error);
          await Sighting.updateOne(
            { _id: sightingId },
            { $set: { 'analysis.llm.status': 'failed' } }
          );
        }
      };

      queueLLMAnalysis(sightingId, predicted_species, confidence, llmCallback)
        .catch(async (e) => {
          console.error('LLM queue processing failed:', e);
          await Sighting.updateOne(
            { _id: sightingId },
            { $set: { 'analysis.llm.status': 'failed' } }
          );
        });

      // Clean up temp file for local storage users
      if (storagePreference === 'local' && file.path !== fullImagePath) {
        try {
          fs.unlinkSync(file.path);
          console.log('Cleaned up temp file for local storage user:', file.path);
        } catch (cleanupError) {
          console.warn('Could not clean up temp file:', cleanupError);
        }
      }
    });

  } catch (e) { next(e); }
}
