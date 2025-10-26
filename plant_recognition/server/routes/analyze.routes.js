import { Router } from 'express';
import { uploadImage } from '../middleware/upload.js';
import { requireAuth } from '../middleware/auth.js';
import * as C from '../controllers/analyze.controller.js';

const r = Router();
r.post('/', requireAuth, uploadImage, C.analyzeOnce);
export default r;
