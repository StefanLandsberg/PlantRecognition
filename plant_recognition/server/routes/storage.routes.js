import express from 'express';
import { requireAuth } from '../middleware/auth.js';
import { getStorageStatus, cleanOldFiles } from '../services/storage.service.js';

const router = express.Router();

// Get storage status (authenticated users only)
router.get('/status', requireAuth, (req, res) => {
  try {
    const status = getStorageStatus();
    res.json(status);
  } catch (error) {
    console.error('Storage status error:', error);
    res.status(500).json({ error: 'Failed to get storage status' });
  }
});

// Manual cleanup (for admin users - add admin check if needed)
router.post('/cleanup', requireAuth, (req, res) => {
  try {
    const { days = 30 } = req.body;
    const result = cleanOldFiles(days);
    res.json({
      success: true,
      message: `Cleanup completed: removed ${result.removedCount} files, freed ${result.freedFormatted}`,
      ...result
    });
  } catch (error) {
    console.error('Manual cleanup error:', error);
    res.status(500).json({ error: 'Failed to perform cleanup' });
  }
});

export default router;