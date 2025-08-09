import { Router } from 'express';
import { requireAuth } from '../middleware/auth.js';
import { sseHandler } from '../services/sse.service.js';

const r = Router();
r.get('/', requireAuth, sseHandler);
export default r;
