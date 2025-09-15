import jwt from 'jsonwebtoken';
import { CONFIG } from '../utils/config.js';

export function requireAuth(req, res, next) {
  try {
    const token = req.cookies?.token;
    if (!token) {
      if (req.path.startsWith('/api/')) {
        return res.status(401).json({ error: 'Auth required' });
      }
      return res.redirect('/');
    }
    const payload = jwt.verify(token, CONFIG.JWT_SECRET);
    req.auth = { userId: payload.sub, role: payload.role };
    next();
  } catch {
    if (req.path.startsWith('/api/')) {
      return res.status(401).json({ error: 'Invalid or expired token' });
    }
    return res.redirect('/');
  }
}
