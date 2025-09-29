import Sighting from '../models/Sighting.js';

export async function list(req, res, next) {
  try {
    const { bbox, includeRemoved = 'false' } = req.query; // "minLng,minLat,maxLng,maxLat"
    const filter = { owner: req.auth.userId };

    // By default, exclude removed sightings unless specifically requested
    if (includeRemoved !== 'true') {
      filter.isRemoved = { $ne: true };
    }

    if (bbox) {
      const [minLng, minLat, maxLng, maxLat] = bbox.split(',').map(Number);
      filter.location = { $geoWithin: { $box: [[minLng, minLat],[maxLng, maxLat]] } };
    }
    const docs = await Sighting.find(filter).sort({ createdAt: -1 }).limit(200);
    res.json({ success: true, data: docs });
  } catch (e) { next(e); }
}

export async function markRemoved(req, res, next) {
  try {
    const { id } = req.params;
    const { removedAt, removedBy } = req.body;

    const doc = await Sighting.findOneAndUpdate(
      { _id: id, owner: req.auth.userId },
      {
        isRemoved: true,
        removedAt: removedAt || new Date(),
        removedBy: removedBy || 'user'
      },
      { new: true }
    );

    if (!doc) {
      return res.status(404).json({ success: false, error: 'Sighting not found' });
    }

    res.json({ success: true, data: doc });
  } catch (e) { next(e); }
}
