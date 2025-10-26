import mongoose from 'mongoose';

const detectionSchema = new mongoose.Schema({
  timestamp: { type: Number, required: true }, // seconds from session start
  frameUrl: String, // path to captured frame
  sightingId: { type: mongoose.Schema.Types.ObjectId, ref: 'Sighting' },
  status: { type: String, enum: ['pending', 'invasive', 'duplicate', 'unknown'], default: 'pending' }
}, { _id: false });

const schema = new mongoose.Schema({
  owner: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  sessionType: { type: String, enum: ['live_video', 'uploaded_video'], required: true },
  startTime: { type: Date, default: Date.now },
  endTime: Date,
  duration: Number, // seconds
  videoUrl: String, // path to recorded video file (server storage)
  localVideoId: String, // local storage video ID
  thumbnailUrl: String, // first frame as thumbnail (server storage)
  localThumbnailId: String, // local storage thumbnail ID
  storageType: { type: String, enum: ['server', 'local'], default: 'server' },
  detections: [detectionSchema],
  location: { 
    type: { type: String, enum: ['Point'], default: 'Point' }, 
    coordinates: { type: [Number], default: [0,0] } 
  }
}, { timestamps: true });

schema.index({ owner: 1, createdAt: -1 });
schema.index({ location: '2dsphere' });

// Use a specific collection name to avoid conflicts with old schema
const VideoSession = mongoose.model('VideoSession', schema, 'video_sessions_v2');

export default VideoSession;