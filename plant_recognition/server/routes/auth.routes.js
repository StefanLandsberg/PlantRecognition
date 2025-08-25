import { Router } from 'express';
import * as C from '../controllers/auth.controller.js';
import { requireAuth } from '../middleware/auth.js';

const r = Router();
r.post('/register', C.register);
r.post('/login', C.login);
r.post('/logout', C.logout);
r.get('/me', requireAuth, C.me);
export default r;
