const mongoose = require('mongoose');

const sightingSchema = new mongoose.Schema({
    // Plant Information
    species: { type: String, required: true },
    confidence: { type: Number, required: true, min: 0, max: 1 },
    
    // Location Information
    latitude: { type: Number, required: true },
    longitude: { type: Number, required: true },
    location: {
        type: {
            type: String,
            enum: ['Point'],
            default: 'Point'
        },
        coordinates: {
            type: [Number],
            required: true
        }
    },
    
    // Detection Information
    timestamp: { type: Date, default: Date.now },
    description: { type: String, default: '' },
    imagePath: { type: String },
    imageUrl: { type: String }, // URL for frontend access
    
    // Detection Method
    detection: {
        method: { type: String, enum: ['streaming', 'image_upload'], default: 'image_upload' },
        confidence: { type: Number, required: true },
        processingTime: { type: Number, default: 0 }
    },
    
    // Management Information
    management: {
        status: {
            type: String,
            enum: ['pending', 'in_progress', 'completed', 'escalated'],
            default: 'pending'
        },
        priority: {
            type: String,
            enum: ['low', 'medium', 'high', 'critical'],
            default: 'medium'
        },
        llmAnalysis: { type: mongoose.Schema.Types.Mixed, default: null }
    },
    
    // LLM Analysis
    llmAnalysis: { type: mongoose.Schema.Types.Mixed, default: null },
    
    // User Reference
    createdBy: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    
    // Metadata
    isActive: { type: Boolean, default: true }
}, {
    timestamps: true
});

// Indexes for efficient querying
sightingSchema.index({ location: '2dsphere' });
sightingSchema.index({ timestamp: -1 });
sightingSchema.index({ species: 1 });
sightingSchema.index({ 'management.status': 1 });
sightingSchema.index({ confidence: -1 });

// Virtual for formatted address
sightingSchema.virtual('formattedAddress').get(function() {
    return `${this.latitude}, ${this.longitude}`;
});

// Method to get severity level based on confidence
sightingSchema.methods.getSeverityLevel = function() {
    if (this.confidence > 0.9) return 'critical';
    if (this.confidence > 0.7) return 'high';
    if (this.confidence > 0.5) return 'medium';
    return 'low';
};

// Static method to get statistics
sightingSchema.statics.getStats = async function() {
    const stats = await this.aggregate([
        { $match: { isActive: true } },
        {
            $group: {
                _id: null,
                totalSightings: { $sum: 1 },
                pendingAnalysis: {
                    $sum: {
                        $cond: [
                            { $eq: ['$management.status', 'pending'] },
                            1,
                            0
                        ]
                    }
                },
                removalSuccess: {
                    $avg: {
                        $cond: [
                            { $eq: ['$management.status', 'completed'] },
                            1,
                            0
                        ]
                    }
                },
                highRiskDetections: {
                    $sum: {
                        $cond: [
                            { $gt: ['$confidence', 0.9] },
                            1,
                            0
                        ]
                    }
                }
            }
        }
    ]);
    
    return stats[0] || { totalSightings: 0, pendingAnalysis: 0, removalSuccess: 0, highRiskDetections: 0 };
};

// Static method to get species breakdown
sightingSchema.statics.getSpeciesBreakdown = async function() {
    return await this.aggregate([
        { $match: { isActive: true } },
        {
            $group: {
                _id: '$species',
                count: { $sum: 1 },
                avgConfidence: { $avg: '$confidence' }
            }
        },
        { $sort: { count: -1 } }
    ]);
};

// Static method to get spread trends
sightingSchema.statics.getSpreadTrends = async function(days = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    
    return await this.aggregate([
        {
            $match: {
                isActive: true,
                timestamp: { $gte: startDate }
            }
        },
        {
            $group: {
                _id: {
                    date: { $dateToString: { format: '%Y-%m-%d', date: '$timestamp' } },
                    species: '$species'
                },
                count: { $sum: 1 }
            }
        },
        { $sort: { '_id.date': 1 } }
    ]);
};

module.exports = mongoose.model('Detection', sightingSchema, 'detection'); 