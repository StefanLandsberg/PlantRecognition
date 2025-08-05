// Authentication module - handles all auth-related functionality
import { CONFIG } from './config.js';
import { appState } from './state.js';
import { api } from './api.js';
import { logger, handleError, validateEmail, validatePassword } from './utils.js';

class AuthManager {
    constructor() {
        this.isInitialized = false;
        this.authCheckInterval = null;
    }
    
    // Initialize authentication
    async initialize() {
        if (this.isInitialized) return;
        
        try {
            // Check for stored authentication
            await this.checkExistingAuth();
            
            // Set up periodic auth check
            this.startAuthCheck();
            
            this.isInitialized = true;
            logger.info('Authentication manager initialized');
        } catch (error) {
            logger.error('Failed to initialize authentication', error);
        }
    }
    
    // Check for existing authentication
    async checkExistingAuth() {
        try {
            const storedToken = localStorage.getItem('authToken');
            const storedUser = localStorage.getItem('user');
            
            if (storedToken && storedUser) {
                try {
                    // Validate stored user JSON
                    const user = JSON.parse(storedUser);
                    if (!user || typeof user !== 'object' || !user.username) {
                        throw new Error('Invalid user data in localStorage');
                    }
                    
                    // Verify token is still valid
                    const verification = await api.verifyToken();
                    
                    if (verification.success) {
                        appState.setAuthentication(user, storedToken);
                        logger.info('Restored authentication from localStorage');
                    } else {
                        // Clear invalid stored auth
                        localStorage.removeItem('authToken');
                        localStorage.removeItem('user');
                        appState.clearAuthentication();
                        logger.warn('Cleared invalid stored authentication');
                    }
                } catch (parseError) {
                    logger.error('Error parsing stored user data', parseError);
                    localStorage.removeItem('authToken');
                    localStorage.removeItem('user');
                    appState.clearAuthentication();
                }
            } else {
                // Ensure clean state
                appState.clearAuthentication();
                logger.info('No stored authentication found');
            }
        } catch (error) {
            logger.error('Error checking existing auth', error);
            appState.clearAuthentication();
        }
    }
    
    // Login user
    async login(username, password) {
        try {
            // Validate input
            if (!username || !password) {
                return { success: false, message: 'Username and password are required' };
            }
            
            // Perform login
            const response = await api.login(username, password);
            
            if (response.success) {
                logger.info('Login successful', { username });
                this.updateAuthUI();
            } else {
                logger.warn('Login failed', { username, error: response.message });
            }
            
            return response;
        } catch (error) {
            const errorResponse = handleError(error, 'login');
            logger.error('Login error', error);
            return errorResponse;
        }
    }
    
    // Register user
    async register(username, email, password, confirmPassword) {
        try {
            // Validate input
            if (!username || !email || !password) {
                return { success: false, message: 'All fields are required' };
            }
            
            if (password !== confirmPassword) {
                return { success: false, message: 'Passwords do not match' };
            }
            
            if (password.length < 6) {
                return { success: false, message: 'Password must be at least 6 characters' };
            }
            
            if (!validateEmail(email)) {
                return { success: false, message: 'Invalid email format' };
            }
            
            // Perform registration
            const response = await api.register(username, email, password);
            
            if (response.success) {
                logger.info('Registration successful', { username, email });
                this.updateAuthUI();
            } else {
                logger.warn('Registration failed', { username, email, error: response.message });
            }
            
            return response;
        } catch (error) {
            const errorResponse = handleError(error, 'register');
            logger.error('Registration error', error);
            return errorResponse;
        }
    }
    
    // Logout user
    async logout() {
        try {
            const response = await api.logout();
            
            // Clear UI state
            this.updateAuthUI();
            
            logger.info('Logout successful');
            return response;
        } catch (error) {
            const errorResponse = handleError(error, 'logout');
            logger.error('Logout error', error);
            
            // Force logout even if API call fails
            appState.clearAuthentication();
            this.updateAuthUI();
            
            return errorResponse;
        }
    }
    
    // Update authentication UI
    updateAuthUI() {
        try {
            const isLoggedIn = appState.isAuthenticated();
            const user = appState.getUser();
            
            // Get UI elements
            const authButtons = document.getElementById('auth-buttons');
            const userInfo = document.getElementById('user-info');
            const userName = document.getElementById('user-name');
            const loginBtn = document.getElementById('login-btn');
            const signupBtn = document.getElementById('signup-btn');
            const logoutBtn = document.getElementById('logout-btn');
            
            // Check if required elements exist
            const requiredElements = [authButtons, userInfo, userName, loginBtn, signupBtn, logoutBtn];
            const missingElements = requiredElements.filter(el => !el);
            
            if (missingElements.length > 0) {
                logger.warn('Some auth UI elements not found', { 
                    missing: missingElements.length,
                    total: requiredElements.length 
                });
                return;
            }
            
            if (isLoggedIn && user) {
                // Show logged-in state
                authButtons.style.display = 'none';
                userInfo.style.display = 'flex';
                userName.textContent = user.username || 'User';
                
                logger.debug('Updated UI to logged-in state', { username: user.username });
            } else {
                // Show logged-out state
                authButtons.style.display = 'flex';
                userInfo.style.display = 'none';
                userName.textContent = '';
                
                logger.debug('Updated UI to logged-out state');
            }
        } catch (error) {
            logger.error('Error updating auth UI', error);
        }
    }
    
    // Set loading state for auth forms
    setAuthLoadingState(isLoading, type = 'login') {
        try {
            if (typeof isLoading !== 'boolean') {
                logger.warn('Invalid loading state provided', { isLoading, type });
                return;
            }
            
            const loginBtn = document.querySelector('#login-form button[type="submit"]');
            const signupBtn = document.querySelector('#signup-form button[type="submit"]');
            
            if (type === 'login' && loginBtn) {
                loginBtn.disabled = isLoading;
                loginBtn.innerHTML = isLoading ? '<i class="fas fa-spinner fa-spin"></i> Loading...' : 'Login';
            }
            
            if (type === 'signup' && signupBtn) {
                signupBtn.disabled = isLoading;
                signupBtn.innerHTML = isLoading ? '<i class="fas fa-spinner fa-spin"></i> Loading...' : 'Sign Up';
            }
            
            if (type !== 'login' && type !== 'signup') {
                logger.warn('Invalid auth type provided', { type });
            }
        } catch (error) {
            logger.error('Error setting auth loading state', error);
        }
    }
    
    // Close modal
    closeModal(modal) {
        if (modal) {
            modal.style.display = 'none';
            modal.classList.remove('show');
        }
    }
    
    // Clear form
    clearForm(form) {
        if (form) {
            form.reset();
            const errorElement = form.querySelector('.error-message');
            if (errorElement) {
                errorElement.textContent = '';
            }
        }
    }
    
    // Start periodic auth check
    startAuthCheck() {
        if (this.authCheckInterval) {
            clearInterval(this.authCheckInterval);
        }
        
        this.authCheckInterval = setInterval(async () => {
            if (appState.isAuthenticated()) {
                try {
                    await api.verifyToken();
                } catch (error) {
                    logger.warn('Token verification failed, logging out');
                    await this.logout();
                }
            }
        }, 5 * 60 * 1000); // Check every 5 minutes
    }
    
    // Stop periodic auth check
    stopAuthCheck() {
        if (this.authCheckInterval) {
            clearInterval(this.authCheckInterval);
            this.authCheckInterval = null;
        }
    }
    
    // Get current user
    getCurrentUser() {
        return appState.getUser();
    }
    
    // Check if user is authenticated
    isAuthenticated() {
        return appState.isAuthenticated();
    }
    
    // Get auth token
    getAuthToken() {
        return appState.getAuthToken();
    }
    
    // Check if user is admin
    isAdmin() {
        const user = appState.getUser();
        return user && user.role === 'admin';
    }
    
    // Cleanup
    cleanup() {
        this.stopAuthCheck();
        this.isInitialized = false;
    }
}

// Create singleton instance
export const authManager = new AuthManager();

// Export for backward compatibility
export const login = (username, password) => authManager.login(username, password);
export const register = (username, email, password, confirmPassword) => 
    authManager.register(username, email, password, confirmPassword);
export const logout = () => authManager.logout();
export const isAuthenticated = () => authManager.isAuthenticated();
export const getCurrentUser = () => authManager.getCurrentUser();
export const isAdmin = () => authManager.isAdmin(); 