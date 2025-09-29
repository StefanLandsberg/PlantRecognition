import { Router } from 'express';
import { requireAuth } from '../middleware/auth.js';
import * as C from '../controllers/alerts.controller.js';

const r = Router();

// Get active alerts
r.get('/', requireAuth, C.list);

// Get alert statistics
r.get('/stats', requireAuth, C.stats);

// Refresh/generate alerts
r.post('/refresh', requireAuth, C.refresh);

// Dismiss an alert
r.patch('/:id/dismiss', requireAuth, C.dismiss);

export default r;