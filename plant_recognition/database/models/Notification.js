const mongoose = require('mongoose');

const notificationSchema = new mongoose.Schema({
    // Notification Information
    type: {
        type: String,
        enum: ['high_risk_detection', 'removal_completed', 'spread_alert', 'weather_advisory', 'system_alert'],
        required: true
    },
    
    title: { type: String, required: true },
    message: { type: String, required: true },
    
    // Priority and Status
    priority: {
        type: String,
        enum: ['low', 'medium', 'high', 'critical'],
        default: 'medium'
    },
    
    status: {
        type: String,
        enum: ['unread', 'read', 'archived'],
        default: 'unread'
    },
    
    // Related Data
    relatedSighting: { type: mongoose.Schema.Types.ObjectId, ref: 'Sighting' },
    relatedSpecies: String,
    location: {
        coordinates: {
            latitude: Number,
            longitude: Number
        },
        region: String
    },
    
    // Metadata
    createdBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
    readBy: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
    readAt: Date,
    expiresAt: Date,
    isActive: { type: Boolean, default: true }
}, {
    timestamps: true
});

// Indexes
notificationSchema.index({ status: 1, createdAt: -1 });
notificationSchema.index({ type: 1, createdAt: -1 });
notificationSchema.index({ priority: 1, createdAt: -1 });
notificationSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });

// Static method to create high-risk detection notification
notificationSchema.statics.createHighRiskDetection = async function(sighting) {
    const speciesName = sighting.species.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    const confidencePercent = (sighting.confidence * 100).toFixed(1);
    const region = sighting.location?.region || 'unknown region';
    const severityLevel = sighting.getSeverityLevel ? sighting.getSeverityLevel() : 'high';
    
    return await this.create({
        type: 'high_risk_detection',
        title: 'High-Risk Plant Detected',
        message: `New sighting of ${speciesName} detected in ${region}. Confidence: ${confidencePercent}%. Requires immediate attention.`,
        priority: severityLevel,
        relatedSighting: sighting._id,
        relatedSpecies: speciesName,
        location: {
            coordinates: {
                latitude: sighting.latitude,
                longitude: sighting.longitude
            },
            region: region
        }
    });
};

// Static method to create removal completed notification
notificationSchema.statics.createRemovalCompleted = async function(sighting, user) {
    const speciesName = sighting.species.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    const address = sighting.formattedAddress || `${sighting.latitude}, ${sighting.longitude}`;
    
    return await this.create({
        type: 'removal_completed',
        title: 'Removal Task Completed',
        message: `${speciesName} removal completed at ${address}. Field worker: ${user.username}.`,
        priority: 'medium',
        relatedSighting: sighting._id,
        relatedSpecies: speciesName,
        location: {
            coordinates: {
                latitude: sighting.latitude,
                longitude: sighting.longitude
            }
        },
        createdBy: user._id
    });
};

// Static method to create spread alert
notificationSchema.statics.createSpreadAlert = async function(species, count, region) {
    const speciesName = species.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    
    return await this.create({
        type: 'spread_alert',
        title: 'Spread Alert',
        message: `${speciesName} showing rapid spread in ${region}. ${count} new sightings in 24 hours.`,
        priority: 'high',
        relatedSpecies: speciesName,
        location: { region }
    });
};

// Static method to create weather advisory
notificationSchema.statics.createWeatherAdvisory = async function(weatherData) {
    const { type, severity, location, description } = weatherData;
    
    return await this.create({
        type: 'weather_advisory',
        title: `${severity} Weather Advisory`,
        message: `${type} weather conditions detected in ${location}. ${description}`,
        priority: severity === 'severe' ? 'critical' : 'high',
        location: { region: location }
    });
};

// Static method to get unread notifications
notificationSchema.statics.getUnreadNotifications = async function(userId) {
    return await this.find({
        status: 'unread',
        isActive: true,
        $or: [
            { readBy: { $ne: userId } },
            { readBy: { $exists: false } }
        ]
    }).sort({ createdAt: -1 }).limit(50);
};

// Static method to mark as read
notificationSchema.statics.markAsRead = async function(notificationId, userId) {
    return await this.findByIdAndUpdate(notificationId, {
        $addToSet: { readBy: userId },
        readAt: new Date(),
        status: 'read'
    });
};

module.exports = mongoose.model('Notification', notificationSchema); 