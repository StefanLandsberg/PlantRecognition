// API module - centralizes all API calls and removes duplicates
import { CONFIG } from './config.js';
import { appState } from './state.js';
import { logger, handleError, retry } from './utils.js';

class API {
    constructor() {
        this.baseURL = CONFIG.DB_API_BASE;
        this.cache = new Map();
        this.pendingRequests = new Map();
    }
    
    // Create headers with authentication
    getHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };
        
        const token = appState.getAuthToken();
        if (token && typeof token === 'string' && token.trim().length > 0) {
            headers['Authorization'] = `Bearer ${token.trim()}`;
        }
        
        return headers;
    }
    
    // Make API request with error handling and retry
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: this.getHeaders(),
            ...options
        };
        
        // Check cache for GET requests
        if (options.method === 'GET' || !options.method) {
            const cached = this.cache.get(url);
            if (cached && 
                typeof cached === 'object' && 
                cached.timestamp && 
                cached.data !== undefined &&
                Date.now() - cached.timestamp < CONFIG.CACHE_DURATION) {
                return cached.data;
            }
        }
        
        // Check for pending requests
        if (this.pendingRequests.has(url)) {
            return this.pendingRequests.get(url);
        }
        
        // Create request promise
        const requestPromise = retry(async () => {
            try {
                // Add timeout to fetch
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), CONFIG.LLM_TIMEOUT);
                
                const response = await fetch(url, {
                    ...config,
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                // Cache successful GET requests
                if (options.method === 'GET' || !options.method) {
                    this.cache.set(url, {
                        data,
                        timestamp: Date.now()
                    });
                }
                
                return data;
            } catch (error) {
                if (error.name === 'AbortError') {
                    throw new Error(`Request timeout after ${CONFIG.LLM_TIMEOUT}ms`);
                }
                logger.error(`API request failed: ${url}`, error);
                throw error;
            }
        });
        
        // Store pending request
        this.pendingRequests.set(url, requestPromise);
        
        try {
            const result = await requestPromise;
            this.pendingRequests.delete(url);
            return result;
        } catch (error) {
            this.pendingRequests.delete(url);
            throw error;
        }
    }
    
    // Authentication API
    async login(username, password) {
        try {
            const response = await this.request('/api/auth/login', {
                method: 'POST',
                body: JSON.stringify({ username, password })
            });
            
            if (response.success) {
                appState.setAuthentication(response.data.user, response.data.token);
            }
            
            return response;
        } catch (error) {
            return handleError(error, 'login');
        }
    }
    
    async register(username, email, password) {
        try {
            const response = await this.request('/api/auth/register', {
                method: 'POST',
                body: JSON.stringify({ username, email, password })
            });
            
            if (response.success) {
                appState.setAuthentication(response.data.user, response.data.token);
            }
            
            return response;
        } catch (error) {
            return handleError(error, 'register');
        }
    }
    
    async logout() {
        try {
            await this.request('/api/auth/logout', { method: 'POST' });
            appState.clearAuthentication();
            return { success: true };
        } catch (error) {
            appState.clearAuthentication();
            return handleError(error, 'logout');
        }
    }
    
    async verifyToken() {
        try {
            const response = await this.request('/api/auth/verify');
            return response;
        } catch (error) {
            appState.clearAuthentication();
            return handleError(error, 'verifyToken');
        }
    }
    
    // Sightings API
    async getSightings(params = {}) {
        try {
            const queryString = new URLSearchParams(params).toString();
            const endpoint = `/api/sightings${queryString ? `?${queryString}` : ''}`;
            return await this.request(endpoint);
        } catch (error) {
            return handleError(error, 'getSightings');
        }
    }
    
    async createSighting(sightingData) {
        try {
            const response = await this.request('/api/sightings', {
                method: 'POST',
                body: JSON.stringify(sightingData)
            });
            
            // Clear cache for sightings
            this.clearCache('/api/sightings');
            
            return response;
        } catch (error) {
            return handleError(error, 'createSighting');
        }
    }
    
    async updateSighting(sightingId, updates) {
        try {
            const response = await this.request(`/api/sightings/${sightingId}`, {
                method: 'PUT',
                body: JSON.stringify(updates)
            });
            
            // Clear cache for sightings
            this.clearCache('/api/sightings');
            
            return response;
        } catch (error) {
            return handleError(error, 'updateSighting');
        }
    }
    
    async deleteSighting(sightingId) {
        try {
            const response = await this.request(`/api/sightings/${sightingId}`, {
                method: 'DELETE'
            });
            
            // Clear cache for sightings
            this.clearCache('/api/sightings');
            
            return response;
        } catch (error) {
            return handleError(error, 'deleteSighting');
        }
    }
    
    // Analytics API
    async getDashboardStats() {
        try {
            return await this.request('/api/sightings/stats');
        } catch (error) {
            return handleError(error, 'getDashboardStats');
        }
    }
    
    async getSpeciesBreakdown() {
        try {
            return await this.request('/api/sightings/analytics/species-breakdown');
        } catch (error) {
            return handleError(error, 'getSpeciesBreakdown');
        }
    }
    
    async getSpreadTrends(days = 30) {
        try {
            return await this.request(`/api/sightings/analytics/spread-trends?days=${days}`);
        } catch (error) {
            return handleError(error, 'getSpreadTrends');
        }
    }
    
    async getFrequencyData() {
        try {
            return await this.request('/api/sightings/analytics/frequency-data');
        } catch (error) {
            return handleError(error, 'getFrequencyData');
        }
    }
    
    async getGrowthData() {
        try {
            return await this.request('/api/sightings/analytics/growth-data');
        } catch (error) {
            return handleError(error, 'getGrowthData');
        }
    }
    
    // Map API
    async getMapData(params = {}) {
        try {
            const queryString = new URLSearchParams(params).toString();
            const endpoint = `/api/sightings/map${queryString ? `?${queryString}` : ''}`;
            return await this.request(endpoint);
        } catch (error) {
            return handleError(error, 'getMapData');
        }
    }
    
    // Notifications API
    async getNotifications() {
        try {
            return await this.request('/api/notifications');
        } catch (error) {
            return handleError(error, 'getNotifications');
        }
    }
    
    async markNotificationAsRead(notificationId) {
        try {
            const response = await this.request(`/api/notifications/${notificationId}/read`, {
                method: 'PUT'
            });
            
            // Clear cache for notifications
            this.clearCache('/api/notifications');
            
            return response;
        } catch (error) {
            return handleError(error, 'markNotificationAsRead');
        }
    }
    
    async markAllNotificationsAsRead() {
        try {
            const response = await this.request('/api/notifications/read-all', {
                method: 'PUT'
            });
            
            // Clear cache for notifications
            this.clearCache('/api/notifications');
            
            return response;
        } catch (error) {
            return handleError(error, 'markAllNotificationsAsRead');
        }
    }
    
    // Notification Stats API
    async getNotificationStats() {
        try {
            return await this.request('/api/notifications/stats');
        } catch (error) {
            return handleError(error, 'getNotificationStats');
        }
    }
    
    // Export API
    async exportSightings(params = {}) {
        try {
            const queryString = new URLSearchParams(params).toString();
            const endpoint = `/api/sightings/export${queryString ? `?${queryString}` : ''}`;
            return await this.request(endpoint);
        } catch (error) {
            return handleError(error, 'exportSightings');
        }
    }
    
    // LLM API
    async getLLMAnalysis(species, confidence) {
        try {
            // LLM endpoint is on the frontend server, not the database server
            const response = await fetch(`http://localhost:3000/llm-analysis/${species}?confidence=${confidence}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            return handleError(error, 'getLLMAnalysis');
        }
    }
    
    // Health check
    async healthCheck() {
        try {
            return await this.request('/health');
        } catch (error) {
            return handleError(error, 'healthCheck');
        }
    }
    
    // Cache management
    clearCache(pattern = null) {
        if (pattern && typeof pattern === 'string') {
            // Clear specific cache entries
            const keysToDelete = [];
            for (const [key] of this.cache) {
                if (key.includes(pattern)) {
                    keysToDelete.push(key);
                }
            }
            keysToDelete.forEach(key => this.cache.delete(key));
        } else {
            // Clear all cache
            this.cache.clear();
        }
    }
    
    // Get cache size
    getCacheSize() {
        return this.cache.size;
    }
    
    // Get pending requests count
    getPendingRequestsCount() {
        return this.pendingRequests.size;
    }
}

// Create singleton instance
export const api = new API();

// Export for backward compatibility
export const login = (username, password) => api.login(username, password);
export const register = (username, email, password) => api.register(username, email, password);
export const logout = () => api.logout();
export const getSightings = (params) => api.getSightings(params);
export const createSighting = (data) => api.createSighting(data);
export const updateSighting = (id, updates) => api.updateSighting(id, updates);
export const getDashboardStats = () => api.getDashboardStats();
export const getSpeciesBreakdown = () => api.getSpeciesBreakdown();
export const getLLMAnalysis = (species, confidence) => api.getLLMAnalysis(species, confidence); 