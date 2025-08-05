// UI module - handles all UI-related functionality and removes duplicates
import { CONFIG } from './config.js';
import { appState } from './state.js';
import { logger, sanitizeHtml, createElement, showElement, hideElement } from './utils.js';

class UIManager {
    constructor() {
        this.isInitialized = false;
        this.currentTab = 'dashboard';
        this.modals = new Map();
        this.notifications = [];
    }
    
    // Initialize UI manager
    initialize() {
        if (this.isInitialized) return;
        
        try {
            this.setupEventListeners();
            this.setupModals();
            this.isInitialized = true;
            logger.info('UI manager initialized');
        } catch (error) {
            logger.error('Failed to initialize UI manager', error);
        }
    }
    
    // Setup event listeners
    setupEventListeners() {
        try {
            // Tab switching
            const adminTabs = document.querySelectorAll('.admin-tab');
            adminTabs.forEach(tab => {
                tab.addEventListener('click', (e) => {
                    e.preventDefault();
                    const tabName = tab.getAttribute('data-tab');
                    if (tabName && typeof tabName === 'string') {
                        this.switchTab(tabName);
                    } else {
                        logger.warn('Invalid tab data attribute', { tabName });
                    }
                });
            });
            
            // Modal close buttons
            const closeButtons = document.querySelectorAll('.modal-close');
            closeButtons.forEach(button => {
                button.addEventListener('click', (e) => {
                    e.preventDefault();
                    const modal = button.closest('.modal');
                    this.closeModal(modal);
                });
            });
            
            // Modal backdrop clicks
            const modals = document.querySelectorAll('.modal');
            modals.forEach(modal => {
                modal.addEventListener('click', (e) => {
                    if (e.target === modal) {
                        this.closeModal(modal);
                    }
                });
            });
            
            // Escape key to close modals
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    this.closeAllModals();
                }
            });
        } catch (error) {
            logger.error('Error setting up event listeners', error);
        }
    }
    
    // Setup modals
    setupModals() {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            const modalId = modal.id;
            this.modals.set(modalId, modal);
        });
    }
    
    // Switch admin tab
    switchTab(tabName) {
        try {
            if (!tabName || typeof tabName !== 'string') {
                logger.warn('Invalid tab name provided', { tabName });
                return;
            }
            
            // Update tab buttons
            const adminTabs = document.querySelectorAll('.admin-tab');
            adminTabs.forEach(tab => {
                tab.classList.remove('active');
                if (tab.getAttribute('data-tab') === tabName) {
                    tab.classList.add('active');
                }
            });
            
            // Update tab panels
            const adminPanels = document.querySelectorAll('.admin-panel');
            adminPanels.forEach(panel => {
                panel.style.display = 'none';
                if (panel.id === `${tabName}-panel`) {
                    panel.style.display = 'block';
                }
            });
            
            this.currentTab = tabName;
            logger.debug('Switched to tab', { tabName });
            
            // Trigger tab-specific updates
            this.updateTabContent(tabName);
        } catch (error) {
            logger.error('Error switching tab', { tabName, error });
        }
    }
    
    // Update tab content
    updateTabContent(tabName) {
        switch (tabName) {
            case 'dashboard':
                this.updateDashboard();
                break;
            case 'map':
                this.updateMap();
                break;
            case 'sightings':
                this.updateSightings();
                break;
            case 'analytics':
                this.updateAnalytics();
                break;
            case 'notifications':
                this.updateNotifications();
                break;
            case 'reports':
                this.updateReports();
                break;
            case 'videos':
                this.updateVideos();
                break;
            default:
                logger.warn('Unknown tab', { tabName });
        }
    }
    
    // Update dashboard
    updateDashboard() {
        // This will be implemented by the dashboard module
        logger.debug('Updating dashboard');
    }
    
    // Update map
    updateMap() {
        // This will be implemented by the map module
        logger.debug('Updating map');
    }
    
    // Update sightings
    updateSightings() {
        // This will be implemented by the sightings module
        logger.debug('Updating sightings');
    }
    
    // Update analytics
    updateAnalytics() {
        // This will be implemented by the analytics module
        logger.debug('Updating analytics');
    }
    
    // Update notifications
    updateNotifications() {
        // This will be implemented by the notifications module
        logger.debug('Updating notifications');
    }
    
    // Update reports
    updateReports() {
        // This will be implemented by the reports module
        logger.debug('Updating reports');
    }
    
    // Update videos
    updateVideos() {
        // This will be implemented by the videos module
        logger.debug('Updating videos');
    }
    
    // Show modal
    showModal(modalId) {
        const modal = this.modals.get(modalId);
        if (modal) {
            modal.style.display = 'flex';
            modal.classList.add('show');
            document.body.style.overflow = 'hidden';
            logger.debug('Showed modal', { modalId });
        } else {
            logger.warn('Modal not found', { modalId });
        }
    }
    
    // Close modal
    closeModal(modal) {
        if (modal) {
            modal.style.display = 'none';
            modal.classList.remove('show');
            document.body.style.overflow = '';
            logger.debug('Closed modal', { modalId: modal.id });
        }
    }
    
    // Close all modals
    closeAllModals() {
        this.modals.forEach(modal => {
            this.closeModal(modal);
        });
    }
    
    // Show notification
    showNotification(message, type = 'info', duration = 5000) {
        try {
            if (!message || typeof message !== 'string') {
                logger.warn('Invalid notification message', { message });
                return;
            }
            
            if (!['info', 'success', 'warning', 'error'].includes(type)) {
                logger.warn('Invalid notification type', { type });
                type = 'info';
            }
            
            if (typeof duration !== 'number' || duration < 0) {
                logger.warn('Invalid notification duration', { duration });
                duration = 5000;
            }
            
            const notification = createElement('div', `notification notification-${type}`);
            notification.innerHTML = sanitizeHtml(message);
            
            // Add close button
            const closeBtn = createElement('button', 'notification-close');
            closeBtn.innerHTML = '&times;';
            closeBtn.addEventListener('click', () => this.removeNotification(notification));
            notification.appendChild(closeBtn);
            
            // Add to page
            document.body.appendChild(notification);
            
            // Auto-remove after duration
            setTimeout(() => {
                this.removeNotification(notification);
            }, duration);
            
            // Store reference
            this.notifications.push(notification);
            
            logger.debug('Showed notification', { message, type, duration });
        } catch (error) {
            logger.error('Error showing notification', error);
        }
    }
    
    // Remove notification
    removeNotification(notification) {
        if (notification && notification.parentNode) {
            notification.parentNode.removeChild(notification);
            const index = this.notifications.indexOf(notification);
            if (index > -1) {
                this.notifications.splice(index, 1);
            }
        }
    }
    
    // Show loading spinner
    showLoading(container, message = 'Loading...') {
        const loadingDiv = createElement('div', 'loading-spinner-container');
        loadingDiv.innerHTML = `
            <div class="loading-spinner"></div>
            <p>${sanitizeHtml(message)}</p>
        `;
        
        if (container) {
            container.appendChild(loadingDiv);
        }
        
        return loadingDiv;
    }
    
    // Hide loading spinner
    hideLoading(loadingDiv) {
        if (loadingDiv && loadingDiv.parentNode) {
            loadingDiv.parentNode.removeChild(loadingDiv);
        }
    }
    
    // Update status indicator
    updateStatus(message, type = 'info') {
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.querySelector('.status-text');
        
        if (statusIndicator && statusText) {
            statusIndicator.className = `status-indicator status-${type}`;
            statusText.textContent = message;
            
            logger.debug('Updated status', { message, type });
        }
    }
    
    // Toggle mode (classify/admin)
    toggleMode() {
        try {
            const classifySection = document.getElementById('classify-section');
            const adminSection = document.getElementById('admin-section');
            const toggleLabel = document.querySelector('.toggle-label');
            
            if (!classifySection || !adminSection || !toggleLabel) {
                logger.warn('Mode toggle elements not found');
                return;
            }
            
            // Use computed style to check display property
            const classifyDisplay = window.getComputedStyle(classifySection).display;
            const isAdminMode = classifyDisplay === 'none';
            
            if (isAdminMode) {
                // Switch to classify mode
                classifySection.style.display = 'block';
                adminSection.style.display = 'none';
                toggleLabel.textContent = 'Switch to Admin';
                appState.set('isAdminMode', false);
                logger.debug('Switched to classify mode');
            } else {
                // Switch to admin mode
                classifySection.style.display = 'none';
                adminSection.style.display = 'block';
                toggleLabel.textContent = 'Switch to Classify';
                appState.set('isAdminMode', true);
                logger.debug('Switched to admin mode');
            }
        } catch (error) {
            logger.error('Error toggling mode', error);
        }
    }
    
    // Toggle video/upload mode
    toggleVideoMode() {
        const videoContent = document.getElementById('video-content');
        const uploadContent = document.getElementById('upload-content');
        const modeIcon = document.getElementById('mode-icon');
        
        if (!videoContent || !uploadContent || !modeIcon) {
            logger.warn('Video mode toggle elements not found');
            return;
        }
        
        const isVideoMode = videoContent.style.display !== 'none';
        
        if (isVideoMode) {
            // Switch to upload mode
            videoContent.style.display = 'none';
            uploadContent.style.display = 'block';
            modeIcon.className = 'fas fa-video';
            appState.set('isVideoMode', false);
            logger.debug('Switched to upload mode');
        } else {
            // Switch to video mode
            videoContent.style.display = 'block';
            uploadContent.style.display = 'none';
            modeIcon.className = 'fas fa-upload';
            appState.set('isVideoMode', true);
            logger.debug('Switched to video mode');
        }
    }
    
    // Update element visibility
    showElement(element) {
        showElement(element);
    }
    
    hideElement(element) {
        hideElement(element);
    }
    
    // Get current tab
    getCurrentTab() {
        return this.currentTab;
    }
    
    // Get modal by ID
    getModal(modalId) {
        return this.modals.get(modalId);
    }
    
    // Cleanup
    cleanup() {
        this.closeAllModals();
        this.notifications.forEach(notification => {
            this.removeNotification(notification);
        });
        this.notifications = [];
        this.isInitialized = false;
    }
}

// Create singleton instance
export const uiManager = new UIManager();

// Export for backward compatibility
export const switchTab = (tabName) => uiManager.switchTab(tabName);
export const showModal = (modalId) => uiManager.showModal(modalId);
export const closeModal = (modal) => uiManager.closeModal(modal);
export const showNotification = (message, type, duration) => 
    uiManager.showNotification(message, type, duration);
export const updateStatus = (message, type) => uiManager.updateStatus(message, type);
export const toggleMode = () => uiManager.toggleMode();
export const toggleVideoMode = () => uiManager.toggleVideoMode(); 