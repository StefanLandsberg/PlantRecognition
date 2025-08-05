// Admin Portal Module - Main coordinator
import { api } from './api.js';
import { appState } from './state.js';
import { handleError } from './utils.js';

// Import tab modules
import { mapManager } from './admin/map.js';
import { analyticsManager } from './admin/analytics.js';
import { speciesManager } from './admin/species.js';
import { reportsManager } from './admin/reports.js';
import { notificationsManager } from './admin/notifications.js';
import { videoManager } from './admin/video.js';

class AdminManager {
    constructor() {
        this.currentTab = 'map';
        this.initialized = false;
        this.managers = {
            map: mapManager,
            analytics: analyticsManager,
            species: speciesManager,
            reports: reportsManager,
            notifications: notificationsManager,
            video: videoManager
        };
    }

    async initialize() {
        if (this.initialized) return;
        
        try {
            console.log('Initializing Admin Manager...');
            
            // Initialize all tab managers
            for (const [tabName, manager] of Object.entries(this.managers)) {
                await manager.initialize();
                console.log(`Initialized ${tabName} manager`);
            }
            
            // Setup tab switching
            this.setupTabSwitching();
            
            // Load initial data and update stats
            await this.refreshData();
            
            this.initialized = true;
            console.log('Admin Manager initialized successfully');
        } catch (error) {
            console.error('Error initializing Admin Manager:', error);
            handleError(error, 'admin_initialization');
        }
    }

    setupTabSwitching() {
        // Add click listeners to all admin tabs
        document.querySelectorAll('.admin-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                e.preventDefault();
                const tabName = tab.getAttribute('data-tab');
                if (tabName) {
                    this.switchTab(tabName);
                }
            });
        });
    }

    async refreshData() {
        try {
            // Refresh data in all managers
            for (const manager of Object.values(this.managers)) {
                if (manager.refreshData) {
                    await manager.refreshData();
                }
            }
            
            // Update admin stats
            this.updateAdminStats();
        } catch (error) {
            console.error('Error refreshing admin data:', error);
        }
    }

    updateAdminStats() {
        try {
            const totalSightings = mapManager.getSightingsCount();
            const pendingAnalysis = mapManager.getPendingCount();
            const highRiskDetections = mapManager.getHighRiskCount();
            const removalSuccess = mapManager.getRemovalSuccessCount();
            const successRate = totalSightings > 0 ? Math.round((removalSuccess / totalSightings) * 100) : 0;
            
            // Update DOM
            const totalElement = document.getElementById('total-sightings');
            const pendingElement = document.getElementById('active-infestations');
            const successElement = document.getElementById('removal-success');
            const riskElement = document.getElementById('high-risk-detections');
            
            if (totalElement) totalElement.textContent = totalSightings;
            if (pendingElement) pendingElement.textContent = pendingAnalysis;
            if (successElement) successElement.textContent = `${successRate}%`;
            if (riskElement) riskElement.textContent = highRiskDetections;
        } catch (error) {
            console.error('Error updating admin stats:', error);
        }
    }

    switchTab(tabName) {
        if (!this.managers[tabName]) {
            console.error(`Unknown tab: ${tabName}`);
            return;
        }

        // Hide all panels
        document.querySelectorAll('.admin-panel').forEach(panel => {
            panel.style.display = 'none';
        });
        
        // Remove active class from all tabs
        document.querySelectorAll('.admin-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Show selected panel and activate tab
        const selectedPanel = document.getElementById(`${tabName}-panel`);
        const selectedTab = document.querySelector(`[data-tab="${tabName}"]`);
        
        if (selectedPanel) selectedPanel.style.display = 'block';
        if (selectedTab) selectedTab.classList.add('active');
        
        // Update the current tab
        this.currentTab = tabName;
        
        // Initialize tab-specific content
        const manager = this.managers[tabName];
        if (manager && manager.onTabActivated) {
            manager.onTabActivated();
        }
    }

    getCurrentTab() {
        return this.currentTab;
    }

    getManager(tabName) {
        return this.managers[tabName];
    }
}

// Create singleton instance
export const adminManager = new AdminManager();

// Export for backward compatibility
export default adminManager; 