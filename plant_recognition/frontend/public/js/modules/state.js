// State management module - centralizes all global state
import { CONFIG } from './config.js';

class AppState {
    constructor() {
        this.state = {
            // Authentication State
            isLoggedIn: false,
            currentUser: null,
            authToken: null,
            
            // Application State
            isVideoMode: false,
            isAdminMode: false,
            currentLocation: null,
            
            // Data State
            detectionHistoryData: [],
            adminData: {
                sightings: [],
                species: [],
                notifications: [],
                analytics: {
                    stats: {},
                    speciesBreakdown: [],
                    spreadTrends: [],
                    frequencyData: [],
                    growthData: []
                }
            },
            
            // UI State
            notifications: [],
            weatherData: null,
            llmStatus: 'initializing',
            
            // Map State
            map: null,
            mapPins: [],
            
            // Charts State
            charts: {
                speciesChart: null,
                trendsChart: null,
                frequencyChart: null,
                growthChart: null
            }
        };
        
        this.listeners = new Map();
        this.subscribers = [];
    }
    
    // Get state
    get(key) {
        return key ? this.state[key] : this.state;
    }
    
    // Set state
    set(key, value) {
        const oldValue = this.state[key];
        this.state[key] = value;
        
        // Notify listeners
        this.notifyListeners(key, value, oldValue);
        
        // Log in debug mode
        if (CONFIG.DEBUG_MODE) {
            console.log(`State changed: ${key}`, { old: oldValue, new: value });
        }
    }
    
    // Update multiple state properties
    update(updates) {
        const oldState = { ...this.state };
        
        Object.assign(this.state, updates);
        
        // Notify for each changed property
        Object.keys(updates).forEach(key => {
            this.notifyListeners(key, updates[key], oldState[key]);
        });
    }
    
    // Subscribe to state changes
    subscribe(callback) {
        this.subscribers.push(callback);
        return () => {
            const index = this.subscribers.indexOf(callback);
            if (index > -1) {
                this.subscribers.splice(index, 1);
            }
        };
    }
    
    // Listen to specific state changes
    listen(key, callback) {
        if (!this.listeners.has(key)) {
            this.listeners.set(key, []);
        }
        this.listeners.get(key).push(callback);
        
        return () => {
            const callbacks = this.listeners.get(key);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        };
    }
    
    // Notify listeners
    notifyListeners(key, newValue, oldValue) {
        // Notify specific listeners
        if (this.listeners.has(key)) {
            const callbacks = this.listeners.get(key);
            if (Array.isArray(callbacks)) {
                callbacks.forEach(callback => {
                    if (typeof callback === 'function') {
                        try {
                            callback(newValue, oldValue);
                        } catch (error) {
                            console.error(`Error in state listener for ${key}:`, error);
                        }
                    }
                });
            }
        }
        
        // Notify general subscribers
        this.subscribers.forEach(callback => {
            if (typeof callback === 'function') {
                try {
                    callback(key, newValue, oldValue);
                } catch (error) {
                    console.error('Error in state subscriber:', error);
                }
            }
        });
    }
    
    // Reset state
    reset() {
        this.state = {
            isLoggedIn: false,
            currentUser: null,
            authToken: null,
            isVideoMode: false,
            isAdminMode: false,
            currentLocation: null,
            detectionHistoryData: [],
            adminData: {
                sightings: [],
                species: [],
                notifications: [],
                analytics: {
                    stats: {},
                    speciesBreakdown: [],
                    spreadTrends: [],
                    frequencyData: [],
                    growthData: []
                }
            },
            notifications: [],
            weatherData: null,
            llmStatus: 'initializing',
            map: null,
            mapPins: [],
            charts: {
                speciesChart: null,
                trendsChart: null,
                frequencyChart: null,
                growthChart: null
            }
        };
        
        // Clear all listeners
        this.listeners.clear();
        this.subscribers = [];
    }
    
    // Get authentication state
    isAuthenticated() {
        return this.state.isLoggedIn && this.state.authToken;
    }
    
    // Get user info
    getUser() {
        return this.state.currentUser;
    }
    
    // Get auth token
    getAuthToken() {
        return this.state.authToken;
    }
    
    // Set authentication
    setAuthentication(user, token) {
        this.update({
            isLoggedIn: true,
            currentUser: user,
            authToken: token
        });
        
        // Store in localStorage
        if (token) {
            localStorage.setItem('authToken', token);
            localStorage.setItem('user', JSON.stringify(user));
        }
    }
    
    // Clear authentication
    clearAuthentication() {
        this.update({
            isLoggedIn: false,
            currentUser: null,
            authToken: null
        });
        
        // Clear localStorage
        localStorage.removeItem('authToken');
        localStorage.removeItem('user');
    }
    
    // Add detection to history
    addDetection(detection) {
        if (!detection || typeof detection !== 'object') {
            console.warn('Invalid detection object provided to addDetection');
            return;
        }
        
        const history = [...this.state.detectionHistoryData];
        history.unshift(detection);
        
        // Limit history size
        if (history.length > 100) {
            history.splice(100);
        }
        
        this.set('detectionHistoryData', history);
    }
    
    // Update detection in history
    updateDetection(detectionId, updates) {
        if (!detectionId) {
            console.warn('No detection ID provided to updateDetection');
            return;
        }
        
        if (!updates || typeof updates !== 'object') {
            console.warn('Invalid updates object provided to updateDetection');
            return;
        }
        
        const history = [...this.state.detectionHistoryData];
        const index = history.findIndex(d => d.id === detectionId || d.databaseId === detectionId);
        
        if (index !== -1) {
            history[index] = { ...history[index], ...updates };
            this.set('detectionHistoryData', history);
        } else {
            console.warn(`Detection with ID ${detectionId} not found in history`);
        }
    }
    
    // Remove detection from history
    removeDetection(detectionId) {
        const history = this.state.detectionHistoryData.filter(d => 
            d.id !== detectionId && d.databaseId !== detectionId
        );
        this.set('detectionHistoryData', history);
    }
    
    // Clear detection history
    clearDetectionHistory() {
        this.set('detectionHistoryData', []);
    }
}

// Create singleton instance
export const appState = new AppState();

// Export for backward compatibility
export const getState = (key) => appState.get(key);
export const setState = (key, value) => appState.set(key, value);
export const updateState = (updates) => appState.update(updates); 