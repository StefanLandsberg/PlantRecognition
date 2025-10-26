import VideoSession from '../models/VideoSession.js';
import Sighting from '../models/Sighting.js';
import { logger } from '../utils/logger.js';
import { saveFile } from '../services/storage.service.js';
import path from 'path';
import fs from 'fs/promises';
import { uploadVideo as storageAwareUploadVideo } from '../middleware/upload.js';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const UPLOADS_DIR = path.resolve(PROJECT_ROOT, 'uploads');

function getSightingIdValue(sightingRef) {
  if (!sightingRef) return null;
  if (typeof sightingRef === 'string') return sightingRef;
  if (sightingRef instanceof Buffer) return sightingRef.toString();
  if (typeof sightingRef === 'object') {
    if (sightingRef._id) return sightingRef._id.toString();
    if (typeof sightingRef.toString === 'function') {
      const str = sightingRef.toString();
      // Avoid `[object Object]`
      if (!str.startsWith('[object')) return str;
    }
  }
  return null;
}

function deriveDetectionStatus(analysis) {
  if (!analysis) return 'pending';

  const species = String(analysis.predictedSpecies || '').trim().toLowerCase();
  const confidence = typeof analysis.confidence === 'number' ? analysis.confidence : 0;

  if (!species || species.includes('unknown')) {
    return 'unknown';
  }

  return 'invasive';
}

function getUploadFilePath(fileUrl) {
  if (!fileUrl || typeof fileUrl !== 'string') return null;
  if (!fileUrl.startsWith('/uploads/')) return null;
  return path.join(UPLOADS_DIR, path.basename(fileUrl));
}

async function deleteSessionVideoAsset(fileUrl) {
  const filePath = getUploadFilePath(fileUrl);
  if (!filePath) return;
  try {
    await fs.unlink(filePath);
    logger.info(`[VideoSession] Deleted video asset ${filePath}`);
  } catch (error) {
    if (error.code !== 'ENOENT') {
      logger.warn(`[VideoSession] Failed to delete video asset ${filePath}:`, error);
    }
  }
}

async function maybePruneSession(session) {
  if (!session) {
    console.log('[PRUNE] No session provided');
    return false;
  }
  if (!session.endTime) {
    console.log(`[PRUNE] Session ${session._id} has no endTime, skipping`);
    return false;
  }

  console.log(`[PRUNE] Checking session ${session._id} for pruning...`);

  // Sync detection statuses to ensure all are up to date
  await syncDetectionStatuses(session, { pendingOnly: false, persist: true });
  
  const detections = Array.isArray(session.detections) ? session.detections : [];
  console.log(`[PRUNE] Session ${session._id}: ${detections.length} detections`);
  
  // If no detections at all, prune immediately (all were duplicates)
  if (detections.length === 0) {
    console.log(`[PRUNE] Session ${session._id}: No detections, pruning immediately`);
    return true;
  }

  // If detections are still pending, wait for analysis to complete
  const pendingCount = detections.filter(d => d.status === 'pending').length;
  if (pendingCount > 0) {
    console.log(`[PRUNE] Session ${session._id}: ${pendingCount} pending detections, waiting`);
    return false;
  }

  // Keep the session only when there is at least one invasive detection
  const invasiveCount = detections.filter(d => d.status === 'invasive').length;
  const unknownCount = detections.filter(d => d.status === 'unknown').length;
  const duplicateCount = detections.filter(d => d.status === 'duplicate').length;
  
  console.log(`[PRUNE] Session ${session._id}: invasive=${invasiveCount}, unknown=${unknownCount}, duplicate=${duplicateCount}`);
  
  if (invasiveCount > 0) {
    console.log(`[PRUNE] Session ${session._id}: Has invasive detections, keeping`);
    return false;
  }

  console.log(`[PRUNE] Session ${session._id}: No invasive detections, pruning`);
  
  // Delete video asset and session
  await deleteSessionVideoAsset(session.videoUrl);
  await deleteSessionVideoAsset(session.thumbnailUrl);
  
  await VideoSession.deleteOne({ _id: session._id });
  logger.info(`[VideoSession] Deleted session ${session._id} (no invasive detections - only duplicates/unknown)`);
  return true;
}

async function syncDetectionStatuses(session, { pendingOnly = true, persist = true } = {}) {
  if (!session?.detections?.length) return false;

  const detectionsToUpdate = session.detections
    .map((detection, index) => ({ detection, index }))
    .filter(({ detection }) => detection.sightingId && (!pendingOnly || detection.status === 'pending'));

  console.log(`[syncDetectionStatuses] Found ${detectionsToUpdate.length} detections to update`);

  if (detectionsToUpdate.length === 0) {
    return false;
  }

  const sightingIds = [...new Set(detectionsToUpdate
    .map(({ detection }) => getSightingIdValue(detection.sightingId))
    .filter(Boolean))];

  console.log(`[syncDetectionStatuses] Looking up sightings:`, sightingIds);

  if (sightingIds.length === 0) return false;

  const sightings = await Sighting.find({ _id: { $in: sightingIds } }).select('analysis');
  console.log(`[syncDetectionStatuses] Found ${sightings.length} sightings in DB`);
  
  const sightingMap = new Map(sightings.map((sighting) => [sighting._id.toString(), sighting]));

  let dirty = false;

  for (const { detection, index } of detectionsToUpdate) {
    const sightingId = getSightingIdValue(detection.sightingId);
    if (!sightingId) continue;

    const sighting = sightingMap.get(sightingId);
    if (!sighting) {
      console.log(`[syncDetectionStatuses] Sighting not found: ${sightingId}`);
      continue;
    }

    console.log(`[syncDetectionStatuses] Sighting ${sightingId} analysis:`, sighting.analysis);
    const derivedStatus = deriveDetectionStatus(sighting.analysis);
    console.log(`[syncDetectionStatuses] Derived status: ${derivedStatus}, current: ${detection.status}`);
    
    if (derivedStatus !== detection.status) {
      session.detections[index].status = derivedStatus;
      dirty = true;
      console.log(`[syncDetectionStatuses] Updated detection ${index} status: ${detection.status} -> ${derivedStatus}`);
    }
  }

  if (dirty && persist) {
    await session.save();
    console.log(`[syncDetectionStatuses] Saved session with updated statuses`);
    
    // Publish SSE update for real-time frontend updates
    const { publish } = await import('../services/sse.service.js');
    publish(session.owner.toString(), {
      type: 'video_session_updated',
      sessionId: session._id,
      message: 'Detection statuses updated'
    });
  }

  return dirty;
}

export const startSession = async (req, res) => {
  try {
    const { sessionType, lat, lng } = req.body;
    const userId = req.auth.userId;

    if (!userId) {
      return res.status(401).json({ success: false, error: 'User not authenticated' });
    }

    const session = new VideoSession({
      owner: userId,
      sessionType: sessionType || 'live_video',
      location: {
        type: 'Point',
        coordinates: [parseFloat(lng) || 0, parseFloat(lat) || 0]
      }
    });

    await session.save();
    logger.info(`Video session started: ${session._id} for user: ${userId}`);

    res.json({
      success: true,
      sessionId: session._id,
      startTime: session.startTime
    });
  } catch (error) {
    logger.error('Start session error:', error);
    res.status(500).json({ success: false, error: 'Failed to start session', details: error.message });
  }
};

export const endSession = async (req, res) => {
  try {
    const { sessionId } = req.params;
    const userId = req.auth.userId;

    const session = await VideoSession.findOne({ _id: sessionId, owner: userId });
    if (!session) {
      return res.status(404).json({ success: false, error: 'Session not found' });
    }

    session.endTime = new Date();
    session.duration = Math.floor((session.endTime - session.startTime) / 1000);
    await session.save();

    logger.info(`Video session ended: ${sessionId}, duration: ${session.duration}s`);

    // Sync detection statuses and potentially prune the session
    const pruned = await maybePruneSession(session);

    res.json({
      success: true,
      sessionId: session._id,
      duration: session.duration,
      removed: pruned
    });
  } catch (error) {
    logger.error('End session error:', error);
    res.status(500).json({ success: false, error: 'Failed to end session' });
  }
};

export const addDetection = async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { timestamp, frameUrl, sightingId } = req.body;
    const userId = req.auth.userId;

    console.log(`Adding detection to session ${sessionId}: sightingId=${sightingId}, timestamp=${timestamp}`);

    const session = await VideoSession.findOne({ _id: sessionId, owner: userId });
    if (!session) {
      console.log(`Session not found: ${sessionId} for user ${userId}`);
      return res.status(404).json({ success: false, error: 'Session not found' });
    }

    // Always start with 'pending' - will be updated when ML analysis completes
    const detection = {
      timestamp: Number.isFinite(Number(timestamp)) ? Number(timestamp) : 0,
      frameUrl,
      sightingId,
      status: 'pending'
    };

    session.detections.push(detection);
    console.log(`Detection added to session. Total detections: ${session.detections.length}`);

    // Set thumbnail if this is the first detection
    if (!session.thumbnailUrl && frameUrl) {
      session.thumbnailUrl = frameUrl;
    }

    await session.save();

    // Sync detection statuses based on sighting analysis
    try {
      console.log(`About to sync detection statuses for session: ${session._id}`);
      await syncDetectionStatuses(session);
      console.log(`Completed syncing detection statuses for session: ${session._id}`);
    } catch (error) {
      console.error(`Error syncing detection statuses:`, error);
    }

    const detectionIndex = session.detections.length - 1;
    const finalStatus = session.detections[detectionIndex].status;

    res.json({
      success: true,
      detectionIndex,
      status: finalStatus
    });
  } catch (error) {
    logger.error('Add detection error:', error);
    res.status(500).json({ success: false, error: 'Failed to add detection' });
  }
};

export const updateDetectionStatus = async (req, res) => {
  try {
    const { sessionId, detectionIndex } = req.params;
    const { status } = req.body;
    const userId = req.auth.userId;

    const session = await VideoSession.findOne({ _id: sessionId, owner: userId });
    if (!session) {
      return res.status(404).json({ success: false, error: 'Session not found' });
    }

    if (detectionIndex >= session.detections.length) {
      return res.status(404).json({ success: false, error: 'Detection not found' });
    }

    // Update detection status based on classification results
    session.detections[detectionIndex].status = status;

    // Remove detection from timeline only if it's duplicate
    if (status === 'duplicate') {
      session.detections.splice(detectionIndex, 1);
    }

    await session.save();

    res.json({ success: true });
  } catch (error) {
    logger.error('Update detection status error:', error);
    res.status(500).json({ success: false, error: 'Failed to update detection' });
  }
};

export const listSessions = async (req, res) => {
  try {
    const userId = req.auth.userId;
    const { limit = 20, skip = 0 } = req.query;

    const sessions = await VideoSession.find({ owner: userId })
      .sort({ createdAt: -1 })
      .limit(parseInt(limit))
      .skip(parseInt(skip))
      .select('sessionType startTime endTime duration thumbnailUrl videoUrl detections location owner endTime');

    const visibleSessions = [];

    for (const session of sessions) {
      try {
        await syncDetectionStatuses(session, { pendingOnly: false, persist: true });
      } catch (error) {
        logger.warn(`Failed to sync statuses while listing session ${session._id}:`, error);
      }

      const pruned = await maybePruneSession(session);
      if (pruned) continue;

      const plain = session.toObject();
      visibleSessions.push({
        _id: plain._id,
        sessionType: plain.sessionType,
        startTime: plain.startTime,
        endTime: plain.endTime,
        duration: plain.duration,
        thumbnailUrl: plain.thumbnailUrl,
        videoUrl: plain.videoUrl,
        storageType: plain.storageType,
        localVideoId: plain.localVideoId,
        detectionCount: plain.detections?.length || 0,
        location: plain.location
      });
    }

    res.json({
      success: true,
      sessions: visibleSessions
    });
  } catch (error) {
    logger.error('List sessions error:', error);
    res.status(500).json({ success: false, error: 'Failed to list sessions' });
  }
};

export const uploadVideo = [
  storageAwareUploadVideo,
  async (req, res) => {
    try {
      console.log('=== VIDEO UPLOAD REQUEST ===');
      console.log('Request body:', req.body);
      console.log('File info:', req.file ? {
        filename: req.file.filename,
        size: req.file.size,
        mimetype: req.file.mimetype,
        path: req.file.path
      } : 'No file');
      console.log('User storage preference:', req.userStoragePreference);
      
      const { sessionId, localVideoId, storageType } = req.body;
      const userId = req.auth.userId;

      console.log('Looking for session:', sessionId, 'for user:', userId);
      const session = await VideoSession.findOne({ _id: sessionId, owner: userId });
      if (!session) {
        console.log('ERROR: Session not found');
        return res.status(404).json({ success: false, error: 'Session not found' });
      }

      // Handle local storage video (no file upload, just metadata)
      if (storageType === 'local' && localVideoId) {
        session.localVideoId = localVideoId;
        session.storageType = 'local';
        session.videoUrl = null; // Clear server URL
        await session.save();
        
        console.log('Video session updated with local storage info:', localVideoId);
        logger.info(`Video session ${sessionId} updated with local storage: ${localVideoId}`);

        return res.json({
          success: true,
          localVideoId: localVideoId,
          storageType: 'local'
        });
      }

      // Handle server storage video (existing behavior)
      if (!req.file) {
        console.log('ERROR: No video file provided for server storage');
        return res.status(400).json({ success: false, error: 'No video file provided' });
      }

      // Always store video recordings on the server to power review clips
      const storagePreference = 'server';

      // Handle storage based on enforced preference
      const videoUrlPath = await saveFile(req.file, userId, storagePreference);

      // Update session with video path
      session.videoUrl = videoUrlPath;
      session.storageType = storagePreference;
      session.localVideoId = null; // Clear local ID
      await session.save();
      
      console.log('Video saved with storage preference:', storagePreference);
      console.log('Video URL stored in database:', videoUrlPath);

      logger.info(`Video uploaded for session ${sessionId}: ${videoUrlPath || 'local storage'}`);

      res.json({
        success: true,
        videoUrl: videoUrlPath,
        storageType: storagePreference
      });
    } catch (error) {
      console.error('Upload video error:', error);
      logger.error('Upload video error:', error);
      res.status(500).json({ success: false, error: 'Failed to upload video' });
    }
  }
];

export const getSession = async (req, res) => {
  try {
    const { sessionId } = req.params;
    const userId = req.auth.userId;

    const session = await VideoSession.findOne({ _id: sessionId, owner: userId })
      .populate('detections.sightingId', 'analysis imagePath location');

    if (!session) {
      return res.status(404).json({ success: false, error: 'Session not found' });
    }

    await syncDetectionStatuses(session);

    res.json({
      success: true,
      session
    });
  } catch (error) {
    logger.error('Get session error:', error);
    res.status(500).json({ success: false, error: 'Failed to get session' });
  }
};
