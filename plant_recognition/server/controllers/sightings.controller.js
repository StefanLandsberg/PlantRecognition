import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import Sighting from '../models/Sighting.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const UPLOADS_DIR = path.resolve(PROJECT_ROOT, 'uploads');

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
    
    // Validate and fix image paths on-the-fly
    const validatedDocs = await Promise.all(docs.map(async (doc) => {
      if (!doc.imagePath) return doc;
      
      // Check if image exists
      const relativePath = doc.imagePath.replace(/^\/+/, '');
      const imageFs = path.resolve(PROJECT_ROOT, relativePath);
      
      if (!fs.existsSync(imageFs)) {
        // Try to find the image in the uploads directory
        const filename = path.basename(doc.imagePath);
        const expectedPath = path.join(UPLOADS_DIR, filename);
        
        if (fs.existsSync(expectedPath)) {
          const newPath = `/uploads/${filename}`;
          // Update in database
          await Sighting.updateOne(
            { _id: doc._id },
            { $set: { imagePath: newPath } }
          );
          // Return updated doc
          return { ...doc.toObject(), imagePath: newPath };
        } else {
          // Clear invalid path
          await Sighting.updateOne(
            { _id: doc._id },
            { $set: { imagePath: null } }
          );
          return { ...doc.toObject(), imagePath: null };
        }
      }
      
      return doc;
    }));
    
    res.json({ success: true, data: validatedDocs });
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
