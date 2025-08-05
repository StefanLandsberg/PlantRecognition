const express = require('express');
const cors = require('cors');
const path = require('path');
const mongoose = require('mongoose');

// Try to load environment variables from env.example
try {
    require('dotenv').config({ path: path.join(__dirname, '..', 'env.example') });
} catch (error) {
    console.log('env.example not found, using default values');
}

// Set default JWT_SECRET if not provided
if (!process.env.JWT_SECRET) {
    process.env.JWT_SECRET = 'your-super-secret-jwt-key-change-this-in-production';
    console.log('JWT_SECRET not found, using default value');
}

const connectDB = require('./config/database');
const User = require('./models/User');

// Import routes
const authRoutes = require('./routes/auth');
const sightingRoutes = require('./routes/sightings');
const notificationRoutes = require('./routes/notifications');

const app = express();

// Health check endpoint
app.get('/health', (req, res) => {
    const dbStatus = mongoose.connection.readyState === 1;
    res.json({
        success: true,
        status: 'ok',
        database: dbStatus ? 'connected' : 'disconnected',
        timestamp: new Date().toISOString()
    });
});

// Connect to MongoDB and create admin user
connectDB().then(async (connected) => {
    if (connected !== false) {
        try {
            // Create admin user if it doesn't exist
            await User.createAdminUser();
            console.log('Admin user setup completed');
        } catch (error) {
            console.error('Error setting up admin user:', error);
        }
    } else {
        console.log('Database functionality disabled - MongoDB not available');
    }
});

// Middleware
app.use(cors({
    origin: process.env.FRONTEND_URL ? [process.env.FRONTEND_URL] : ['http://localhost:3000', 'http://127.0.0.1:3000'],
    credentials: true,
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/sightings', sightingRoutes);
app.use('/api/notifications', notificationRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({
        success: false,
        message: 'Something went wrong!',
        error: process.env.NODE_ENV === 'development' ? err.message : 'Internal server error'
    });
});

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({
        success: false,
        message: 'Route not found'
    });
});

const PORT = process.env.DB_PORT || 3001;

app.listen(PORT, () => {
    console.log(`Database API server running on port ${PORT}`);
    console.log(`Health check: http://localhost:${PORT}/health`);
    console.log(`Auth endpoints: http://localhost:${PORT}/api/auth`);
    console.log(`Sighting endpoints: http://localhost:${PORT}/api/sightings`);
    console.log(`Notification endpoints: http://localhost:${PORT}/api/notifications`);
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down database API server...');
    process.exit(0);
});

module.exports = app; 