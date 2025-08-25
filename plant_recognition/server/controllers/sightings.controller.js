import Sighting from '../models/Sighting.js';

export async function list(req, res, next) {
  try {
    const { bbox } = req.query; // "minLng,minLat,maxLng,maxLat"
    const filter = { owner: req.auth.userId };
    if (bbox) {
      const [minLng, minLat, maxLng, maxLat] = bbox.split(',').map(Number);
      filter.location = { $geoWithin: { $box: [[minLng, minLat],[maxLng, maxLat]] } };
    }
    const docs = await Sighting.find(filter).sort({ createdAt: -1 }).limit(200);
    res.json({ success: true, data: docs });
  } catch (e) { next(e); }
}
