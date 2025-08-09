# plant_recognition (Vanilla JS + Node + MongoDB Atlas)

## Quick start
1. Create `.env` from sample in this repo. Fill:
   - `MONGODB_URI` (Atlas)
   - `JWT_SECRET`
   - `GOOGLE_MAPS_API_KEY`
2. `npm i`
3. `npm run dev`
4. Visit `http://localhost:3000`

## Notes
- Login-first flow using JWT (httpOnly cookies).
- One-pager after login:
  - Google Maps hero
  - Bottom bar: Live Video (frame every 3s) & Upload
  - Detections drawer + SSE for LLM completion.
- Integrate your existing Python `ml_model.py` and `llm_integration.py` in `/python`.
