// Configuration module - centralizes all hardcoded values
export const CONFIG = {
    // API Configuration
    DB_API_BASE: (() => {
        try {
            return window.location.hostname === 'localhost' ? 
                'http://localhost:3001' : 
                `${window.location.protocol}//${window.location.hostname}:3001`;
        } catch (error) {
            console.warn('Failed to determine API base URL, using localhost fallback:', error);
            return 'http://localhost:3001';
        }
    })(),
    FRONTEND_PORT: 3000,
    
    // LLM Configuration
    CONFIDENCE_THRESHOLD: 0.8,
    LLM_TIMEOUT: 30000,
    LLM_CHECK_INTERVAL: 2000,
    
    // File Upload Configuration
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_FILE_TYPES: ['image/jpeg', 'image/png', 'image/jpg'],
    
    // Weather Configuration
    WEATHER_CHECK_INTERVAL: 30 * 60 * 1000, // 30 minutes
    WEATHER_API_URL: 'https://api.open-meteo.com/v1/forecast',
    
    // UI Configuration
    MAP_DEFAULT_ZOOM: 13,
    MAP_DEFAULT_CENTER: { lat: -25.8408448, lng: 28.2394624 }, // Pierre van Ryneveld
    
    // Animation Configuration
    FADE_DURATION: 300,
    SLIDE_DURATION: 500,
    
    // Notification Configuration
    NOTIFICATION_TIMEOUT: 5000,
    MAX_NOTIFICATIONS: 50,
    
    // Database Configuration
    PAGINATION_LIMIT: 20,
    CACHE_DURATION: 5 * 60 * 1000, // 5 minutes
    
    // Security Configuration
    SESSION_TIMEOUT: 24 * 60 * 60 * 1000, // 24 hours
    MAX_LOGIN_ATTEMPTS: 5,
    
    // Development Configuration
    DEBUG_MODE: false,
    LOG_LEVEL: 'info'
}; 