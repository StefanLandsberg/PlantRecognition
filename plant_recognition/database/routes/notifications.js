const express = require('express');
const router = express.Router();
const notificationController = require('../controllers/notificationController');

// Get notifications
router.get('/', notificationController.getNotifications);
router.get('/stats', notificationController.getNotificationStats);
router.get('/unread-count', notificationController.getUnreadCount);

// Mark as read - fixed routes to match frontend expectations
router.post('/:id/read', notificationController.markAsRead);
router.post('/read-all', notificationController.markAllAsRead);

// Create and delete
router.post('/', notificationController.createNotification);
router.delete('/:id', notificationController.deleteNotification);

module.exports = router; 