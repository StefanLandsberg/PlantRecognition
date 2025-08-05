# Plant Recognition Application

A comprehensive plant recognition system with AI-powered detection, real-time analytics, and interactive mapping capabilities.

## Quick Start

### Prerequisites
- **Node.js** (v16 or higher)
- **MongoDB Atlas** account
- **MongoDB Compass** (optional but recommended)

## Setup Instructions

### 1. Install Node.js
1. Download Node.js from [nodejs.org](https://nodejs.org/)
2. Install with default settings
3. Verify installation:
   ```bash
   node --version
   npm --version
   ```

### 2. MongoDB Atlas Setup

#### Step 1: Create MongoDB Atlas Account
1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas)
2. Click "Try Free" and create an account
3. Choose "Free" tier (M0) for development

#### Step 2: Create a Cluster
1. Click "Build a Database"
2. Choose "FREE" tier (M0)
3. Select your preferred cloud provider (AWS, Google Cloud, or Azure)
4. Choose a region close to you
5. Click "Create"

#### Step 3: Set Up Database Access
1. In the left sidebar, click "Database Access"
2. Click "Add New Database User"
3. Choose "Password" authentication
4. Create a username and password (save these!)
5. Select "Read and write to any database"
6. Click "Add User"

#### Step 4: Set Up Network Access
1. In the left sidebar, click "Network Access"
2. Click "Add IP Address"
3. For development: Click "Allow Access from Anywhere" (0.0.0.0/0)
4. For production: Add your specific IP addresses
5. Click "Confirm"

#### Step 5: Get Connection String
1. Click "Database" in the left sidebar
2. Click "Connect" on your cluster
3. Choose "Connect your application"
4. Copy the connection string (it looks like: `mongodb+srv://username:password@cluster.mongodb.net/`)

### 3. MongoDB Compass Setup (Optional)

#### Install MongoDB Compass
1. Download from [MongoDB Compass](https://www.mongodb.com/products/compass)
2. Install with default settings

#### Connect to Atlas
1. Open MongoDB Compass
2. Paste your Atlas connection string
3. Replace `<password>` with your actual password
4. Click "Connect"
5. You should see your cluster and databases

### 4. Environment Configuration

#### Step 1: Copy Environment Template
```bash
cp env.example .env
```

#### Step 2: Configure Environment Variables
Edit the `.env` file with your settings:

```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://your_username:your_password@your_cluster.mongodb.net/plant_recognition?retryWrites=true&w=majority

# Replace with your actual values:
# - your_username: The username you created in Atlas
# - your_password: The password you created in Atlas
# - your_cluster: Your cluster name from Atlas

# Application Settings
PORT=3000
NODE_ENV=development

# File Upload Settings
MAX_FILE_SIZE=10485760
UPLOAD_PATH=./uploads

# JWT Settings
JWT_SECRET=your_jwt_secret_key_here
JWT_EXPIRES_IN=24h

# Optional: Weather API (for weather notifications)
WEATHER_API_KEY=your_weather_api_key_here
```

### 5. Install Dependencies

#### Install Frontend Dependencies
```bash
cd plant_recognition/frontend
npm install
```

#### Install Database Dependencies
```bash
cd plant_recognition/database
npm install
```

### 6. Start the Application

#### Option 1: Start Both Services (Recommended)
From the root directory:
```bash
# Start database service
cd plant_recognition/database
npm start

# In a new terminal, start frontend service
cd plant_recognition/frontend
npm start
```

#### Option 2: Start Individually
```bash
# Terminal 1 - Database
cd plant_recognition/database
npm start

# Terminal 2 - Frontend  
cd plant_recognition/frontend
npm start
```

### 7. Access the Application
- **Frontend**: http://localhost:3000
- **Database API**: http://localhost:3001 (if running separately)

## Project Structure

```
PlantRecognition/
├── plant_recognition/
│   ├── frontend/           # React/Node.js frontend
│   │   ├── public/         # Static files
│   │   ├── package.json    # Frontend dependencies
│   │   └── server.js       # Frontend server
│   ├── database/           # MongoDB backend
│   │   ├── config/         # Database configuration
│   │   ├── controllers/    # API controllers
│   │   ├── models/         # MongoDB models
│   │   ├── routes/         # API routes
│   │   └── package.json    # Database dependencies
│   ├── models/             # ML models
│   └── .env               # Environment variables
├── data_preprocessing/     # Data preparation scripts
├── LLM/                   # Language model integration
├── modeltraining/         # Model training scripts
└── README.md             # This file
```

## Troubleshooting

### Common Issues

#### 1. MongoDB Connection Failed
- **Error**: "MongoServerSelectionError"
- **Solution**: 
  - Check your connection string in `.env`
  - Verify username/password are correct
  - Ensure your IP is whitelisted in Atlas
  - Check if cluster is running

#### 2. Port Already in Use
- **Error**: "EADDRINUSE"
- **Solution**:
  ```bash
  # Find process using port 3000
  netstat -ano | findstr :3000
  
  # Kill the process
  taskkill /PID <process_id> /F
  ```

#### 3. Node Modules Missing
- **Error**: "Cannot find module"
- **Solution**:
  ```bash
  cd plant_recognition/frontend
  rm -rf node_modules package-lock.json
  npm install
  ```

#### 4. Environment Variables Not Loading
- **Error**: "undefined" values
- **Solution**:
  - Ensure `.env` file is in the correct location
  - Check for typos in variable names
  - Restart the application after changes

### Database Verification
To verify your MongoDB connection is working:

1. Open MongoDB Compass
2. Connect to your Atlas cluster
3. You should see a `plant_recognition` database created automatically
4. Collections will be created as you use the application

## Features

- **AI Plant Recognition**: Upload images for instant plant identification
- **Interactive Map**: View detection locations with custom pins
- **Real-time Analytics**: Track detection trends and species distribution
- **Species Encyclopedia**: Browse and search plant information
- **Weather Integration**: Current weather data for detection context
- **Admin Dashboard**: Comprehensive management interface

## Security Notes

- Never commit `.env` files to version control
- Use strong passwords for database access
- Regularly update dependencies
- For production, use environment-specific configurations

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure MongoDB Atlas cluster is running
4. Check console logs for error messages

## Deployment

For production deployment:
1. Set `NODE_ENV=production` in `.env`
2. Use a production MongoDB Atlas cluster
3. Configure proper security groups and IP whitelisting
4. Set up SSL certificates
5. Use a process manager like PM2

---

**Happy Plant Recognition!** 