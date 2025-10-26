import mongoose from 'mongoose';
import { CONFIG } from '../utils/config.js';

async function cleanupVideoSessions() {
  try {
    await mongoose.connect(CONFIG.MONGODB_URI);
    console.log('Connected to MongoDB');

    const db = mongoose.connection.db;
    
    // Check if videosessions collection exists
    const collections = await db.listCollections({ name: 'videosessions' }).toArray();
    
    if (collections.length > 0) {
      console.log('VideoSessions collection exists, dropping it...');
      await db.dropCollection('videosessions');
      console.log('VideoSessions collection dropped successfully');
    } else {
      console.log('VideoSessions collection does not exist');
    }

    console.log('Cleanup completed');
    process.exit(0);
  } catch (error) {
    console.error('Cleanup failed:', error);
    process.exit(1);
  }
}

cleanupVideoSessions();