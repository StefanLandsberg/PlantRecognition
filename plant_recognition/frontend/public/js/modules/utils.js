// Utilities module - centralizes common functions and removes duplicates
import { CONFIG } from './config.js';

// Date utilities
export const formatDate = (date) => {
    if (!date) return 'Unknown';
    
    try {
        const d = new Date(date);
        if (isNaN(d.getTime())) {
            return 'Invalid Date';
        }
        
        return d.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (error) {
        console.warn('Error formatting date:', error);
        return 'Invalid Date';
    }
};

export const getTimeAgo = (date) => {
    if (!date) return 'Unknown';
    
    try {
        const now = new Date();
        const past = new Date(date);
        
        if (isNaN(past.getTime())) {
            return 'Invalid Date';
        }
        
        const diffMs = now - past;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMins / 60);
        const diffDays = Math.floor(diffHours / 24);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
        if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
        if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
        
        return formatDate(date);
    } catch (error) {
        console.warn('Error calculating time ago:', error);
        return 'Unknown';
    }
};

// String utilities
export const sanitizeHtml = (str) => {
    if (!str) return '';
    
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
};

export const truncateText = (text, maxLength = 100) => {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
};

export const capitalizeFirst = (str) => {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

// Validation utilities
export const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
};

export const validatePassword = (password) => {
    return password && password.length >= 6;
};

export const validateFile = (file) => {
    if (!file) return { valid: false, error: 'No file selected' };
    
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        return { valid: false, error: 'File too large' };
    }
    
    if (!CONFIG.ALLOWED_FILE_TYPES || !Array.isArray(CONFIG.ALLOWED_FILE_TYPES)) {
        return { valid: false, error: 'Invalid configuration: allowed file types not defined' };
    }
    
    if (!CONFIG.ALLOWED_FILE_TYPES.includes(file.type)) {
        return { valid: false, error: 'Invalid file type' };
    }
    
    return { valid: true };
};

// Number utilities
export const formatNumber = (num) => {
    if (num === null || num === undefined) return '0';
    return num.toLocaleString();
};

export const formatPercentage = (num) => {
    if (num === null || num === undefined) return '0%';
    return `${(num * 100).toFixed(1)}%`;
};

// Color utilities
export const getSpeciesColor = (species) => {
    const colors = {
        'Acacia_mearnsii': '#ff6b6b',
        'Agave': '#4ecdc4',
        'Opuntia': '#45b7d1',
        'Lantana': '#96ceb4',
        'default': '#95a5a6'
    };
    
    return colors[species] || colors.default;
};

export const getRiskColor = (riskLevel) => {
    const colors = {
        'High': '#e74c3c',
        'Medium': '#f39c12',
        'Low': '#27ae60',
        'Unknown': '#95a5a6'
    };
    
    return colors[riskLevel] || colors.Unknown;
};

// Array utilities
export const groupBy = (array, key) => {
    return array.reduce((groups, item) => {
        const group = item[key];
        if (!groups[group]) {
            groups[group] = [];
        }
        groups[group].push(item);
        return groups;
    }, {});
};

export const uniqueBy = (array, key) => {
    const seen = new Set();
    return array.filter(item => {
        const value = item[key];
        if (seen.has(value)) {
            return false;
        }
        seen.add(value);
        return true;
    });
};

// Object utilities
export const deepClone = (obj) => {
    try {
        if (obj === null || typeof obj !== 'object') return obj;
        if (obj instanceof Date) return new Date(obj.getTime());
        if (obj instanceof Array) return obj.map(item => deepClone(item));
        if (typeof obj === 'object') {
            const cloned = {};
            for (const key in obj) {
                if (obj.hasOwnProperty(key)) {
                    cloned[key] = deepClone(obj[key]);
                }
            }
            return cloned;
        }
    } catch (error) {
        console.warn('Error in deepClone:', error);
        return obj; // Return original object if cloning fails
    }
};

export const pick = (obj, keys) => {
    const result = {};
    keys.forEach(key => {
        if (obj.hasOwnProperty(key)) {
            result[key] = obj[key];
        }
    });
    return result;
};

export const omit = (obj, keys) => {
    const result = {};
    Object.keys(obj).forEach(key => {
        if (!keys.includes(key)) {
            result[key] = obj[key];
        }
    });
    return result;
};

// DOM utilities
export const createElement = (tag, className, innerHTML) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (innerHTML) element.innerHTML = sanitizeHtml(innerHTML);
    return element;
};

export const removeElement = (element) => {
    if (element && element.parentNode) {
        element.parentNode.removeChild(element);
    }
};

export const showElement = (element) => {
    if (element) {
        element.style.display = '';
        element.classList.remove('hidden');
    }
};

export const hideElement = (element) => {
    if (element) {
        element.style.display = 'none';
        element.classList.add('hidden');
    }
};

// Async utilities
export const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

export const retry = async (fn, maxAttempts = 3, delayMs = 1000) => {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await fn();
        } catch (error) {
            if (attempt === maxAttempts) throw error;
            await delay(delayMs * attempt);
        }
    }
};

// Error utilities
export const handleError = (error, context = '') => {
    const errorMessage = error.message || error.toString();
    
    if (CONFIG.DEBUG_MODE) {
        console.error(`Error in ${context}:`, error);
    }
    
    return {
        success: false,
        error: errorMessage,
        context
    };
};

// Logging utilities
export const logger = {
    info: (message, data) => {
        if (CONFIG.DEBUG_MODE) {
            console.log(`[INFO] ${message}`, data || '');
        }
    },
    
    warn: (message, data) => {
        if (CONFIG.LOG_LEVEL !== 'error') {
            console.warn(`[WARN] ${message}`, data || '');
        }
    },
    
    error: (message, error) => {
        console.error(`[ERROR] ${message}`, error || '');
    },
    
    debug: (message, data) => {
        if (CONFIG.DEBUG_MODE) {
            console.log(`[DEBUG] ${message}`, data || '');
        }
    }
};

// Cache utilities
export class Cache {
    constructor(ttl = CONFIG.CACHE_DURATION) {
        this.cache = new Map();
        this.ttl = ttl;
    }
    
    set(key, value) {
        this.cache.set(key, {
            value,
            timestamp: Date.now()
        });
    }
    
    get(key) {
        const item = this.cache.get(key);
        if (!item) return null;
        
        if (Date.now() - item.timestamp > this.ttl) {
            this.cache.delete(key);
            return null;
        }
        
        return item.value;
    }
    
    clear() {
        this.cache.clear();
    }
    
    size() {
        return this.cache.size;
    }
}

// Export cache instance
export const cache = new Cache(); 