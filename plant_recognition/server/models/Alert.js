import mongoose from 'mongoose';

const alertSchema = new mongoose.Schema({
  owner: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  type: {
    type: String,
    enum: ['risk', 'weather', 'general'],
    required: true,
    index: true
  },
  level: {
    type: String,
    enum: ['info', 'warning', 'critical'],
    required: true,
    index: true
  },
  title: {
    type: String,
    required: true,
    trim: true,
    maxlength: 200
  },
  description: {
    type: String,
    required: true,
    trim: true,
    maxlength: 1000
  },
  action: {
    type: String,
    trim: true,
    maxlength: 200
  },
  isDismissed: {
    type: Boolean,
    default: false,
    index: true
  },
  dismissedAt: {
    type: Date,
    index: true
  },
  // Optional metadata for alert generation logic
  metadata: {
    alertKey: String, // Unique key to prevent duplicates (e.g., "invasive_hotspot_2024-01-15")
    dataSnapshot: Object, // Snapshot of data used to generate this alert
    expiresAt: Date // Optional expiration date for alerts
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Compound indexes for efficient queries
alertSchema.index({ owner: 1, isDismissed: 1, createdAt: -1 });
alertSchema.index({ owner: 1, type: 1, isDismissed: 1 });
alertSchema.index({ 'metadata.alertKey': 1, owner: 1 }, { unique: true, sparse: true });

// Virtual for checking if alert is active
alertSchema.virtual('isActive').get(function() {
  if (this.isDismissed) return false;
  if (this.metadata?.expiresAt && this.metadata.expiresAt < new Date()) return false;
  return true;
});

// Instance method to dismiss alert
alertSchema.methods.dismiss = function() {
  this.isDismissed = true;
  this.dismissedAt = new Date();
  return this.save();
};

// Static method to find active alerts
alertSchema.statics.findActive = function(owner, type = null) {
  const filter = {
    owner,
    isDismissed: false,
    $or: [
      { 'metadata.expiresAt': { $exists: false } },
      { 'metadata.expiresAt': { $gt: new Date() } }
    ]
  };

  if (type) {
    filter.type = type;
  }

  return this.find(filter).sort({ level: 1, createdAt: -1 }); // Critical first, then by newest
};

// Static method to create or update alert
alertSchema.statics.createOrUpdate = async function(alertData) {
  const { owner, metadata, ...alertFields } = alertData;

  // If alertKey is provided, check for existing alert
  if (metadata?.alertKey) {
    const existing = await this.findOne({
      owner,
      'metadata.alertKey': metadata.alertKey,
      isDismissed: false
    });

    if (existing) {
      // Update existing alert with new data
      Object.assign(existing, alertFields);
      existing.metadata = { ...existing.metadata, ...metadata };
      return existing.save();
    }
  }

  // Create new alert
  return this.create({ owner, ...alertFields, metadata });
};

export default mongoose.model('Alert', alertSchema);