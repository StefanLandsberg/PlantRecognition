import jwt from 'jsonwebtoken';
import { CONFIG } from '../utils/config.js';

export function requireAuth(req, res, next) {
  try {
    const token = req.cookies?.token;
    if (!token) return res.status(401).json({ error: 'Auth required' });
    const payload = jwt.verify(token, CONFIG.JWT_SECRET);
    req.auth = { userId: payload.sub, role: payload.role };
    next();
  } catch {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
}
