import express from 'express';
import { requireAuth } from '../middleware/auth.js';
import * as videoSessionController from '../controllers/videoSession.controller.js';

const router = express.Router();

router.use(requireAuth);

router.post('/upload-video', (req, res, next) => {
  console.log('=== UPLOAD VIDEO ROUTE HIT ===');
  console.log('Method:', req.method);
  console.log('URL:', req.url);
  console.log('Body keys:', Object.keys(req.body || {}));
  console.log('Has file:', !!req.file);
  console.log('Auth user:', req.auth?.userId);
  next();
}, videoSessionController.uploadVideo);

router.post('/start', videoSessionController.startSession);
router.put('/:sessionId/end', videoSessionController.endSession);
router.post('/:sessionId/detections', videoSessionController.addDetection);
router.put('/:sessionId/detections/:detectionIndex/status', videoSessionController.updateDetectionStatus);
router.get('/', videoSessionController.listSessions);
router.get('/:sessionId', videoSessionController.getSession);

export default router;