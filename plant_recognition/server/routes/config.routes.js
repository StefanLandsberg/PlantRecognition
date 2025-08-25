import { Router } from 'express';
import { CONFIG } from '../utils/config.js';

const r = Router();
r.get('/config.js', (req, res) => {
  res.type('application/javascript').send(
    `window.APP_CONFIG = ${JSON.stringify({
      GOOGLE_MAPS_API_KEY: CONFIG.GOOGLE_MAPS_API_KEY
    })};`
  );
});
export default r;
