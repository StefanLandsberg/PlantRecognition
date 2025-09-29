import { Router } from 'express';
import { requireAuth } from '../middleware/auth.js';
import * as C from '../controllers/sightings.controller.js';

const r = Router();
r.get('/', requireAuth, C.list);
r.patch('/:id/remove', requireAuth, C.markRemoved);
export default r;
