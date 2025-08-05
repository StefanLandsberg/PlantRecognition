const Notification = require('../models/Notification');
const Sighting = require('../models/Sighting');

// Get all notifications
exports.getNotifications = async (req, res) => {
    try {
        const {
            page = 1,
            limit = 20,
            status,
            type,
            priority
        } = req.query;

        const filter = { isActive: true };
        
        if (status) filter.status = status;
        if (type) filter.type = type;
        if (priority) filter.priority = priority;

        const notifications = await Notification.find(filter)
            .sort({ createdAt: -1 })
            .limit(limit * 1)
            .skip((page - 1) * limit);

        const total = await Notification.countDocuments(filter);

        res.json({
            success: true,
            data: notifications,
            pagination: {
                current: page,
                pages: Math.ceil(total / limit),
                total
            }
        });
    } catch (error) {
        console.error('Error getting notifications:', error);
        res.status(500).json({
            success: false,
            message: 'Error retrieving notifications',
            error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
        });
    }
};

// Get unread notifications count
exports.getUnreadCount = async (req, res) => {
    try {
        const userId = req.user?.id;
        
        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User authentication required'
            });
        }
        
        const count = await Notification.countDocuments({
            status: 'unread',
            isActive: true,
            $or: [
                { readBy: { $ne: userId } },
                { readBy: { $exists: false } }
            ]
        });

        res.json({
            success: true,
            data: { count }
        });
    } catch (error) {
        console.error('Error getting unread count:', error);
        res.status(500).json({
            success: false,
            message: 'Error retrieving unread count',
            error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
        });
    }
};

// Mark notification as read
exports.markAsRead = async (req, res) => {
    try {
        const userId = req.user?.id;
        const notificationId = req.params.id;

        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User authentication required'
            });
        }

        const notification = await Notification.markAsRead(notificationId, userId);

        if (!notification) {
            return res.status(404).json({
                success: false,
                message: 'Notification not found'
            });
        }

        res.json({
            success: true,
            data: notification,
            message: 'Notification marked as read'
        });
    } catch (error) {
        console.error('Error marking notification as read:', error);
        res.status(500).json({
            success: false,
            message: 'Error marking notification as read',
            error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
        });
    }
};

// Mark all notifications as read
exports.markAllAsRead = async (req, res) => {
    try {
        const userId = req.user?.id;

        if (!userId) {
            return res.status(401).json({
                success: false,
                message: 'User authentication required'
            });
        }

        const result = await Notification.updateMany(
            {
                status: 'unread',
                isActive: true,
                $or: [
                    { readBy: { $ne: userId } },
                    { readBy: { $exists: false } }
                ]
            },
            {
                $addToSet: { readBy: userId },
                readAt: new Date(),
                status: 'read'
            }
        );

        res.json({
            success: true,
            data: { modifiedCount: result.modifiedCount },
            message: `${result.modifiedCount} notifications marked as read`
        });
    } catch (error) {
        console.error('Error marking all notifications as read:', error);
        res.status(500).json({
            success: false,
            message: 'Error marking notifications as read',
            error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
        });
    }
};

// Create manual notification
exports.createNotification = async (req, res) => {
    try {
        const {
            type,
            title,
            message,
            priority,
            relatedSighting,
            relatedSpecies,
            location
        } = req.body;

        // Input validation
        if (!type || !title || !message) {
            return res.status(400).json({
                success: false,
                message: 'Type, title, and message are required'
            });
        }

        const notification = new Notification({
            type,
            title,
            message,
            priority,
            relatedSighting,
            relatedSpecies,
            location,
            createdBy: req.user?.id
        });

        await notification.save();

        res.status(201).json({
            success: true,
            data: notification,
            message: 'Notification created successfully'
        });
    } catch (error) {
        console.error('Error creating notification:', error);
        
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
            message: 'Error creating notification',
            error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
        });
    }
};

// Delete notification
exports.deleteNotification = async (req, res) => {
    try {
        const notification = await Notification.findByIdAndUpdate(
            req.params.id,
            { isActive: false },
            { new: true }
        );

        if (!notification) {
            return res.status(404).json({
                success: false,
                message: 'Notification not found'
            });
        }

        res.json({
            success: true,
            message: 'Notification deleted successfully'
        });
    } catch (error) {
        console.error('Error deleting notification:', error);
        res.status(500).json({
            success: false,
            message: 'Error deleting notification',
            error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
        });
    }
};

// Get notification statistics
exports.getNotificationStats = async (req, res) => {
    try {
        const stats = await Notification.aggregate([
            { $match: { isActive: true } },
            {
                $group: {
                    _id: null,
                    total: { $sum: 1 },
                    unread: {
                        $sum: {
                            $cond: [
                                { $eq: ['$status', 'unread'] },
                                1,
                                0
                            ]
                        }
                    },
                    byType: {
                        $push: {
                            type: '$type',
                            priority: '$priority'
                        }
                    }
                }
            }
        ]);

        const typeBreakdown = await Notification.aggregate([
            { $match: { isActive: true } },
            {
                $group: {
                    _id: '$type',
                    count: { $sum: 1 }
                }
            },
            { $sort: { count: -1 } }
        ]);

        const priorityBreakdown = await Notification.aggregate([
            { $match: { isActive: true } },
            {
                $group: {
                    _id: '$priority',
                    count: { $sum: 1 }
                }
            },
            { $sort: { count: -1 } }
        ]);

        res.json({
            success: true,
            data: {
                total: stats[0]?.total || 0,
                unread: stats[0]?.unread || 0,
                typeBreakdown,
                priorityBreakdown
            }
        });
    } catch (error) {
        console.error('Error getting notification stats:', error);
        res.status(500).json({
            success: false,
            message: 'Error retrieving notification statistics',
            error: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
        });
    }
}; 