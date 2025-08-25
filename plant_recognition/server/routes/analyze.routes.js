import { Router } from 'express';
import upload from '../middleware/upload.js';
import { requireAuth } from '../middleware/auth.js';
import * as C from '../controllers/analyze.controller.js';

const r = Router();
r.post('/', requireAuth, upload.single('image'), C.analyzeOnce);
export default r;
