const Sighting = require('../models/Sighting');
const Notification = require('../models/Notification');
const User = require('../models/User');

// Create new sighting from model detection
exports.createSighting = async (req, res) => {
    try {
        const {
            species,
            confidence,
            latitude,
            longitude,
            location,
            timestamp,
            description,
            frameData,
            imagePath,
            detection,
            management,
            llmAnalysis
        } = req.body;

        // Input validation
        if (!species || typeof species !== 'string') {
            return res.status(400).json({
                success: false,
                message: 'Species is required and must be a string'
            });
        }

        if (typeof confidence !== 'number' || confidence < 0 || confidence > 1) {
            return res.status(400).json({
                success: false,
                message: 'Confidence must be a number between 0 and 1'
            });
        }

        if (typeof latitude !== 'number' || typeof longitude !== 'number') {
            return res.status(400).json({
                success: false,
                message: 'Latitude and longitude must be valid numbers'
            });
        }

        if (latitude < -90 || latitude > 90) {
            return res.status(400).json({
                success: false,
                message: 'Latitude must be between -90 and 90'
            });
        }

        if (longitude < -180 || longitude > 180) {
            return res.status(400).json({
                success: false,
                message: 'Longitude must be between -180 and 180'
            });
        }

        // Get admin user ID for the detection
        const adminUser = await User.findOne({ username: process.env.ADMIN_USERNAME || 'admin' });
        if (!adminUser) {
            return res.status(500).json({
                success: false,
                message: 'Admin user not found'
            });
        }

        // Create new sighting with the updated schema
        const sighting = new Sighting({
            species: species,
            confidence: confidence,
            latitude: latitude,
            longitude: longitude,
            location: location || {
                type: 'Point',
                coordinates: [longitude, latitude]
            },
            timestamp: timestamp || new Date(),
            description: description || '',
            imagePath: imagePath,
            imageUrl: req.body.imageUrl,
            detection: detection || {
                method: 'image_upload',
                confidence: confidence,
                processingTime: 0
            },
            management: management || {
                status: 'pending',
                priority: 'medium'
            },
            llmAnalysis: llmAnalysis,
            createdBy: adminUser._id
        });

        await sighting.save();

        // Create notification for high-risk detections (high confidence)
        const highRiskThreshold = process.env.HIGH_RISK_THRESHOLD || 0.8;
        if (sighting.confidence > highRiskThreshold) {
            try {
                await Notification.createHighRiskDetection(sighting);
            } catch (notificationError) {
                console.error('Error creating notification:', notificationError);
                // Don't fail the sighting creation if notification fails
            }
        }

        res.status(201).json({
            success: true,
            data: sighting,
            message: 'Sighting created successfully'
        });
    } catch (error) {
        console.error('Error creating sighting:', error);
        
        // Handle mongoose validation errors
        if (error.name === 'ValidationError') {
            const validationErrors = Object.values(error.errors).map(err => err.message);
            return res.status(400).json({
                success: false,
                message: 'Validation failed',
                errors: validationErrors
            });
        }

        res.status(500).json({
            success: false,
            message: 'Error creating sighting',
            error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
        });
    }
};

// Get dashboard statistics
exports.getDashboardStats = async (req, res) => {
    try {
        const stats = await Sighting.getStats();
        
        // Format removal success as percentage
        stats.removalSuccess = Math.round(stats.removalSuccess * 100);
        
        res.json({
            success: true,
            data: stats
        });
    } catch (error) {
        console.error('Error getting dashboard stats:', error);
        res.status(500).json({
            success: false,
            message: 'Error retrieving dashboard statistics',
            error: error.message
        });
    }
};

// Get species breakdown for analytics
exports.getSpeciesBreakdown = async (req, res) => {
    try {
        const breakdown = await Sighting.getSpeciesBreakdown();
        
        res.json({
            success: true,
            data: breakdown
        });
    } catch (error) {
        console.error('Error getting species breakdown:', error);
        res.status(500).json({
            success: false,
            message: 'Error retrieving species breakdown',
            error: error.message
        });
    }
};

// Update LLM analysis for a detection
exports.updateLLMAnalysis = async (req, res) => {
    try {
        const { species, confidence, llmAnalysis } = req.body;
        
        // Find the most recent detection for this species and confidence
        const detection = await Sighting.findOne({
            species: species,
            confidence: confidence
        }).sort({ timestamp: -1 });
        
        if (!detection) {
            return res.status(404).json({
                success: false,
                message: 'Detection not found'
            });
        }
        
        // Update the detection with LLM analysis
        detection.llmAnalysis = llmAnalysis;
        detection.management.llmAnalysis = llmAnalysis;
        
        await detection.save();
        
        console.log('LLM analysis updated for detection:', detection._id);
        
        res.json({
            success: true,
            message: 'LLM analysis updated successfully',
            data: detection
        });
    } catch (error) {
        console.error('Error updating LLM analysis:', error);
        res.status(500).json({
            success: false,
            message: 'Error updating LLM analysis',
            error: error.message
        });
    }
};

// Get spread trends for analytics
exports.getSpreadTrends = async (req, res) => {
    try {
        const { days = 30 } = req.query;
        const trends = await Sighting.getSpreadTrends(parseInt(days));
        
        res.json({
            success: true,
            data: trends
        });
    } catch (error) {
        console.error('Error getting spread trends:', error);
        res.status(500).json({
            success: false,
            message: 'Error retrieving spread trends',
            error: error.message
        });
    }
};

// Get all sightings with filters
exports.getSightings = async (req, res) => {
    try {
        const {
            page = 1,
            limit = 20,
            status,
            species,
            region,
            startDate,
            endDate,
            isInvasive
        } = req.query;

        // Build filter object - don't filter by user for now to see all data
        const filter = { isActive: true };
        
        if (status) filter['management.status'] = status;
        if (species) filter['species'] = new RegExp(species, 'i');
        if (region) filter['location.region'] = new RegExp(region, 'i');
        if (isInvasive !== undefined) filter['isInvasive'] = isInvasive === 'true';
        
        if (startDate || endDate) {
            filter['timestamp'] = {};
            if (startDate) filter['timestamp'].$gte = new Date(startDate);
            if (endDate) filter['timestamp'].$lte = new Date(endDate);
        }

        const sightings = await Sighting.find(filter)
            .sort({ timestamp: -1 })
            .limit(limit * 1)
            .skip((page - 1) * limit);

        const total = await Sighting.countDocuments(filter);

        res.json({
            success: true,
            data: sightings,
            pagination: {
                current: page,
                pages: Math.ceil(total / limit),
                total
            }
        });
    } catch (error) {
        console.error('Error getting sightings:', error);
        res.status(500).json({
            success: false,
            message: 'Error retrieving sightings',
            error: error.message
        });
    }
};

// Get single sighting by ID
exports.getSighting = async (req, res) => {
    try {
        const sighting = await Sighting.findById(req.params.id);

        if (!sighting) {
            return res.status(404).json({
                success: false,
                message: 'Sighting not found'
            });
        }

        res.json({
            success: true,
            data: sighting
        });
    } catch (error) {
        console.error('Error getting sighting:', error);
        res.status(500).json({
            success: false,
            message: 'Error retrieving sighting',
            error: error.message
        });
    }
};

// Update sighting management status
exports.updateSighting = async (req, res) => {
    try {
        const { management, llmAnalysis, updatedBy } = req.body;
        
        // Build update object based on what's provided
        const updateData = {};
        
        if (management) {
            updateData.management = {
                ...management,
                completedDate: management.status === 'completed' ? new Date() : undefined
            };
        }
        
        if (llmAnalysis) {
            updateData.llmAnalysis = llmAnalysis;
        }
        
        if (updatedBy) {
            updateData.updatedBy = updatedBy;
        }
        
        const sighting = await Sighting.findByIdAndUpdate(
            req.params.id,
            updateData,
            { new: true }
        );

        if (!sighting) {
            return res.status(404).json({
                success: false,
                message: 'Sighting not found'
            });
        }

        // Create notification for completed removals
        if (management && management.status === 'completed') {
            const user = await User.findById(updatedBy);
            if (user) {
                await Notification.createRemovalCompleted(sighting, user);
            }
        }

        res.json({
            success: true,
            data: sighting,
            message: 'Sighting updated successfully'
        });
    } catch (error) {
        console.error('Error updating sighting:', error);
        res.status(500).json({
            success: false,
            message: 'Error updating sighting',
            error: error.message
        });
    }
};

// Get map data for interactive map
exports.getMapData = async (req, res) => {
    try {
        const { viewType = 'pins', species, startDate, endDate } = req.query;
        
        const filter = { isActive: true };
        
        if (species) filter['species'] = new RegExp(species, 'i');
        if (startDate || endDate) {
            filter['timestamp'] = {};
            if (startDate) filter['timestamp'].$gte = new Date(startDate);
            if (endDate) filter['timestamp'].$lte = new Date(endDate);
        }

        const sightings = await Sighting.find(filter)
            .select('species location timestamp management confidence')
            .sort({ timestamp: -1 });

        // Format data for map
        const mapData = sightings.map(sighting => ({
            id: sighting._id,
            coordinates: sighting.location.coordinates,
            species: sighting.species,
            confidence: sighting.confidence,
            status: sighting.management.status,
            priority: sighting.management.priority,
            timestamp: sighting.timestamp,
            severity: sighting.getSeverityLevel()
        }));

        res.json({
            success: true,
            data: mapData
        });
    } catch (error) {
        console.error('Error getting map data:', error);
        res.status(500).json({
            success: false,
            message: 'Error retrieving map data',
            error: error.message
        });
    }
};

// Export sightings as CSV
exports.exportSightings = async (req, res) => {
    try {
        const { format = 'csv', ...filters } = req.query;
        
        // Apply same filters as getSightings
        const filter = { isActive: true };
        
        if (filters.status) filter['management.status'] = filters.status;
        if (filters.species) filter['species'] = new RegExp(filters.species, 'i');
        if (filters.region) filter['location.region'] = new RegExp(filters.region, 'i');
        if (filters.isInvasive !== undefined) filter['isInvasive'] = filters.isInvasive === 'true';
        
        if (filters.startDate || filters.endDate) {
            filter['timestamp'] = {};
            if (filters.startDate) filter['timestamp'].$gte = new Date(filters.startDate);
            if (filters.endDate) filter['timestamp'].$lte = new Date(filters.endDate);
        }

        const sightings = await Sighting.find(filter)
            .sort({ timestamp: -1 });

        if (format === 'csv') {
            const csvData = sightings.map(sighting => ({
                'Species': sighting.species,
                'Confidence': sighting.confidence,
                'Latitude': sighting.latitude,
                'Longitude': sighting.longitude,
                'Detection Date': sighting.timestamp,
                'Status': sighting.management.status,
                'Priority': sighting.management.priority,
                'Created By': sighting.createdBy?.username || '',
                'Notes': sighting.management.notes || ''
            }));

            res.setHeader('Content-Type', 'text/csv');
            res.setHeader('Content-Disposition', 'attachment; filename=sightings.csv');
            
            // Convert to CSV
            const csv = [
                Object.keys(csvData[0]).join(','),
                ...csvData.map(row => Object.values(row).map(value => `"${value}"`).join(','))
            ].join('\n');
            
            res.send(csv);
        } else {
            res.json({
                success: true,
                data: sightings
            });
        }
    } catch (error) {
        console.error('Error exporting sightings:', error);
        res.status(500).json({
            success: false,
            message: 'Error exporting sightings',
            error: error.message
        });
    }
}; 

// Get frequency data for analytics
exports.getFrequencyData = async (req, res) => {
    try {
        const { days = 30 } = req.query;
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - days);
        
        const frequencyData = await Sighting.aggregate([
            {
                $match: {
                    isActive: true,
                    timestamp: { $gte: startDate }
                }
            },
            {
                $group: {
                    _id: {
                        date: { $dateToString: { format: '%Y-%m-%d', date: '$timestamp' } }
                    },
                    count: { $sum: 1 }
                }
            },
            { $sort: { '_id.date': 1 } }
        ]);
        
        res.json({
            success: true,
            data: frequencyData
        });
    } catch (error) {
        console.error('Error getting frequency data:', error);
        res.status(500).json({
            success: false,
            message: 'Error retrieving frequency data',
            error: error.message
        });
    }
};

// Get growth data for analytics
exports.getGrowthData = async (req, res) => {
    try {
        const { days = 30 } = req.query;
        const startDate = new Date();
        startDate.setDate(startDate.getDate() - days);
        
        const growthData = await Sighting.aggregate([
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
            {
                $group: {
                    _id: '$_id.species',
                    dailyCounts: {
                        $push: {
                            date: '$_id.date',
                            count: '$count'
                        }
                    },
                    totalCount: { $sum: '$count' }
                }
            },
            { $sort: { totalCount: -1 } }
        ]);
        
        res.json({
            success: true,
            data: growthData
        });
    } catch (error) {
        console.error('Error getting growth data:', error);
        res.status(500).json({
            success: false,
            message: 'Error retrieving growth data',
            error: error.message
        });
    }
}; 