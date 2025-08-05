const mongoose = require('mongoose');
const path = require('path');

// Try to load environment variables from env.example
try {
    require('dotenv').config({ path: path.join(__dirname, '..', '..', 'env.example') });
} catch (error) {
    console.log('env.example not found, using default values');
}

const connectDB = async () => {
    try {
        const mongoURI = process.env.MONGODB_URI || 'mongodb://localhost:27017/plant_recognition';
        
        await mongoose.connect(mongoURI, {
            useNewUrlParser: true,
            useUnifiedTopology: true,
            serverSelectionTimeoutMS: 10000, // 10 second timeout for Atlas
            socketTimeoutMS: 45000, // 45 second timeout
            maxPoolSize: 10, // Connection pool size
            minPoolSize: 1, // Minimum connections
            maxIdleTimeMS: 30000, // Max idle time
            retryWrites: true, // Enable retry writes
            w: 'majority' // Write concern
        });
        
        console.log('MongoDB connected successfully');
        
        // Handle connection events
        mongoose.connection.on('error', (err) => {
            console.error('MongoDB connection error:', err);
        });
        
        mongoose.connection.on('disconnected', () => {
            console.log('MongoDB disconnected');
        });
        
        // Graceful shutdown
        process.on('SIGINT', async () => {
            await mongoose.connection.close();
            console.log('MongoDB connection closed through app termination');
            process.exit(0);
        });
        
    } catch (error) {
        console.error('Failed to connect to MongoDB Atlas:', error.message);
        console.log('To fix this issue:');
        console.log('   1. Check your MongoDB Atlas connection string in env.example');
        console.log('   2. Verify your Atlas cluster is running and accessible');
        console.log('   3. Check your IP whitelist in Atlas (add 0.0.0.0/0 for all IPs)');
        console.log('   4. Verify your username and password are correct');
        console.log('   5. Check if your Atlas cluster has the required database');
        console.log('');
        console.log('Quick fix for Atlas:');
        console.log('   - Go to MongoDB Atlas dashboard');
        console.log('   - Check Network Access and add your IP or 0.0.0.0/0');
        console.log('   - Verify Database Access has the correct user');
        console.log('   - Check if the cluster is active');
        console.log('');
        console.log('For now, the application will continue without database functionality');
        console.log('   You can still test the frontend and camera features');
        
        // Don't exit, let the app continue without database
        return false;
    }
};

module.exports = connectDB; 