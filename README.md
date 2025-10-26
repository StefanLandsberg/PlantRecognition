# Plant Recognition System

A comprehensive plant species identification platform with real-time analysis, invasive species detection, and mobile companion support.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Mobile Companion](#mobile-companion)
- [API Reference](#api-reference)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Features

### Core Functionality
- **Real-time Plant Identification**: Advanced machine learning classification with confidence scoring
- **AI-Powered Analysis**: Comprehensive species information using LLM integration
- **Invasive Species Detection**: Specialised risk assessment for environmental protection
- **Geospatial Mapping**: Interactive maps with clustering and location-based insights
- **Dual Storage System**: Flexible server (2GB, 90-day retention) or unlimited local storage

### Advanced Analytics
- **Temporal Analysis**: Time-based pattern recognition with calculated insights
- **Species Analytics**: Comprehensive statistics and distribution analysis
- **Risk Assessment**: Automated alerts for high-risk invasive species
- **Geographic Insights**: Spatial distribution and clustering analysis

### User Experience
- **Mobile Companion App**: Live video capture with QR code pairing
- **Multi-language Support**: English, Afrikaans, and Zulu localisation
- **Accessibility**: Colour-blind support with multiple vision modes
- **Responsive Design**: Optimised for desktop and mobile devices

## Architecture

### Technology Stack
- **Frontend**: Vanilla JavaScript (ES6+), CSS Custom Properties, WebRTC
- **Backend**: Node.js, Express.js, WebSocket for real-time communication
- **Database**: MongoDB Atlas with Mongoose ODM
- **Authentication**: JWT with httpOnly cookies for security
- **File Storage**: Multer with configurable local/server storage
- **Maps**: Google Maps API integration

### Project Structure
```
plant_recognition/
├── server/                 # Backend application
│   ├── models/            # MongoDB data models
│   ├── routes/            # API endpoint definitions
│   ├── services/          # Business logic services
│   ├── middleware/        # Authentication & file handling
│   └── utils/             # Configuration & utilities
├── public/                # Frontend application
│   ├── js/               # JavaScript modules
│   ├── css/              # Stylesheets with theming
│   └── views/            # EJS templates
├── mobile_companion/      # Mobile companion app
├── python/               # ML model integration
└── uploads/              # File storage directory
```

## Quick Start

### Prerequisites
- Node.js 16+ and npm
- MongoDB Atlas account
- Google Maps API key
- Python 3.8+ (for ML model integration)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd plant_recognition
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Environment configuration**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your credentials:
   ```env
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/plantdb
   JWT_SECRET=your-secure-jwt-secret-key
   GOOGLE_MAPS_API_KEY=your-google-maps-api-key
   PORT=3000
   ```

4. **Start the application**
   ```bash
   npm run dev
   ```

5. **Access the application**
   - Main app: `http://localhost:3000`
   - Mobile companion: `http://localhost:3000/mobile-companion`

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGODB_URI` | MongoDB Atlas connection string | Yes |
| `JWT_SECRET` | Secret key for JWT token generation | Yes |
| `GOOGLE_MAPS_API_KEY` | Google Maps API key for mapping | Yes |
| `PORT` | Server port (default: 3000) | No |
| `NODE_ENV` | Environment mode (development/production) | No |

### Storage Configuration

The system supports two storage modes:

- **Server Storage**: 2GB total capacity, 90-day retention, automatic backups
- **Local Storage**: Unlimited capacity, device-only storage, no backups

Configure via user settings or programmatically through the storage service.

## Mobile Companion

### Features
- **Live Video Streaming**: Continuous video frame streaming to main app for real-time classification
- **Manual Image Capture**: Single image capture for individual analysis
- **QR Code Pairing**: Instant connection to main application via same network
- **Resolution Control**: Configurable quality from 512p to 1080p
- **Storage Preferences**: Choose between server and local storage
- **Multi-camera Support**: Front and rear camera switching

### Network Requirements
**IMPORTANT**: The mobile companion and main app must be on the **same network** to function:
- **Local Network**: Both devices connected to same Wi-Fi network
- **Field Usage**: Requires mobile data hotspot on tablet/edge device running main app
- **No Internet**: Connection lost if network is interrupted

### Setup Instructions

1. **Access the companion app**
   ```
   http://localhost:3000/mobile-companion
   ```

2. **Connection Methods**
   - **Manual**: Enter 6-digit code from main app
   - **QR Code**: Scan QR code for instant pairing

3. **Permissions**
   - Camera access for live video capture
   - Location services for GPS tagging (optional)

### Live Video Streaming

The mobile companion app provides **two modes of operation**:

#### 1. Live Video Streaming Mode
- **Continuous classification**: Streams video frames every 3 seconds to main app
- **Real-time analysis**: Results appear on main app as mobile moves around
- **Hands-free operation**: No need to manually capture individual images
- **Field survey optimised**: Perfect for continuous monitoring while walking

#### 2. Manual Capture Mode
- **Single image capture**: Tap to capture individual images
- **Gallery upload**: Upload existing photos from device
- **Precise targeting**: For detailed analysis of specific plants

#### Technical Implementation
- `getUserMedia()` API for camera access
- WebSocket communication for frame streaming
- Canvas-based frame extraction every 3 seconds
- Automatic quality adjustment based on network conditions

### Field Usage Setup

For **field work** without Wi-Fi access:

1. **Equipment needed:**
   - Tablet or edge device (runs main app)
   - Mobile phone (runs companion app)
   - Mobile data plan on tablet

2. **Setup process:**
   - Enable mobile hotspot on tablet/edge device
   - Connect mobile phone to tablet's hotspot
   - Run main app on tablet: `http://localhost:3000`
   - Access companion on phone: `http://[tablet-ip]:3000/mobile-companion`
   - Pair devices using QR code or 6-digit code

3. **Operation:**
   - Tablet displays map and receives classifications
   - Mobile phone streams live video for continuous analysis
   - All data syncs to main app for analysis and storage

## API Reference

### Authentication Endpoints

#### `POST /auth/register`
Register a new user account.

**Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "password": "string"
}
```

#### `POST /auth/login`
Authenticate user and receive JWT token.

**Request Body:**
```json
{
  "email": "string",
  "password": "string"
}
```

### Sightings API

#### `GET /api/sightings`
Retrieve user's plant sightings with filtering options.

**Query Parameters:**
- `species`: Filter by species name
- `startDate`: Filter from date (ISO format)
- `endDate`: Filter to date (ISO format)
- `invasive`: Boolean filter for invasive species

#### `POST /api/sightings`
Create new plant sighting from image upload.

**Request Body:** `multipart/form-data`
- `image`: Image file (JPEG, PNG, WebP, GIF)
- `lat`: Latitude (optional)
- `lng`: Longitude (optional)
- `storagePreference`: "server" or "local"

### Analytics Endpoints

#### `GET /api/analytics/species`
Retrieve species distribution analytics.

#### `GET /api/analytics/temporal`
Get temporal analysis with trend calculations.

#### `GET /api/analytics/risk-assessment`
Invasive species risk assessment data.

### Storage Management

#### `GET /api/storage/status`
Retrieve current storage usage and limits.

#### `POST /api/storage/cleanup`
Trigger manual storage cleanup process.

## Development

### Local Development Setup

1. **Install development dependencies**
   ```bash
   npm install --include=dev
   ```

2. **Start with automatic reload**
   ```bash
   npm run dev
   ```

3. **Run tests**
   ```bash
   npm test
   ```

### Code Style Guidelines

- **JavaScript**: ES6+ modules, async/await patterns
- **CSS**: Custom properties for theming, BEM methodology
- **File Naming**: kebab-case for files, camelCase for variables
- **Comments**: JSDoc format for functions, inline for complex logic

### Testing

```bash
# Run all tests
npm test

# Run specific test suite
npm run test:api
npm run test:frontend

# Coverage report
npm run test:coverage
```

## Deployment

### Production Deployment

1. **Environment preparation**
   ```bash
   NODE_ENV=production
   PORT=80
   ```

2. **Build optimisation**
   ```bash
   npm run build
   ```

3. **Start production server**
   ```bash
   npm start
   ```

### Docker Deployment

```dockerfile
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

### MongoDB Atlas Configuration

Ensure your MongoDB Atlas cluster:
- Allows connections from your deployment IP
- Has appropriate user permissions
- Includes necessary database indexes for performance

## Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes with descriptive messages
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Code Standards

- Follow existing code style and conventions
- Add tests for new functionality
- Update documentation for API changes
- Ensure all tests pass before submitting

### Bug Reports

Include the following information:
- Steps to reproduce the issue
- Expected vs actual behaviour
- Browser/device information
- Console error messages
- Screenshots if applicable

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

For technical support or questions:
- Create an issue on GitHub
- Check existing documentation
- Review the API reference

---

**Version:** 2.0.0
**Last Updated:** January 2025
**Compatibility:** Node.js 16+, MongoDB 4.4+, Modern browsers with WebRTC support