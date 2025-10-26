import mongoose from 'mongoose';

const llmSchema = new mongoose.Schema({
  summary: String,
  details: mongoose.Schema.Types.Mixed,
  status: { type: String, enum: ['pending','completed','failed'], default: 'pending' }
}, { _id: false });

const analysisSchema = new mongoose.Schema({
  predictedSpecies: String,
  confidence: Number,
  llm: llmSchema
}, { _id: false });

const schema = new mongoose.Schema({
  owner: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
  imagePath: String,
  localFileId: String,  // For local storage files
  storageType: { type: String, enum: ['server', 'local'], default: 'server' },
  videoSessionId: { type: mongoose.Schema.Types.ObjectId, ref: 'VideoSession', default: null },
  videoTimestamp: { type: Number, default: null },

  location: { type: { type: String, enum: ['Point'], default: 'Point' }, coordinates: { type: [Number], default: [0,0] } }, // [lng, lat]
  analysis: analysisSchema,
  capturedAt: { type: Date, default: Date.now },
  // Removal tracking
  removedAt: { type: Date, default: null },
  removedBy: { type: String, default: null },
  isRemoved: { type: Boolean, default: false }
}, { timestamps: true });

schema.index({ location: '2dsphere' });

export default mongoose.model('Sighting', schema);
