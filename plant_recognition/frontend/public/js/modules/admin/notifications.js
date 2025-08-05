// Notifications Tab Manager
import { api } from '../api.js';
import { sanitizeHtml } from '../../utils/sanitize.js';
import { handleError } from '../utils.js';

class NotificationsManager {
    constructor() {
        this.notifications = [];
        this.weatherData = null;
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        try {
            console.log('Initializing Notifications Manager...');
            await this.loadNotifications();
            await this.loadWeatherData();
            this.initialized = true;
            console.log('Notifications Manager initialized successfully');
        } catch (error) {
            console.error('Error initializing Notifications Manager:', error);
            handleError(error, 'notifications_initialization');
        }
    }

    async loadNotifications() {
        try {
            const response = await api.getNotifications();
            if (response.success && response.data) {
                this.notifications = response.data;
            } else {
                // Generate notifications from sightings data
                this.generateNotificationsFromSightings();
            }
        } catch (error) {
            console.error('Error loading notifications:', error);
            this.generateNotificationsFromSightings();
        }
    }

    async loadWeatherData() {
        try {
            // Get current location (default to Pretoria if not available)
            const location = this.getCurrentLocation();
            
            // Use Open-Meteo API (no key required)
            const response = await fetch(
                `https://api.open-meteo.com/v1/forecast?latitude=${location.lat}&longitude=${location.lng}&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m&timezone=auto`
            );
            
            if (response.ok) {
                this.weatherData = await response.json();
                console.log('Weather data loaded:', this.weatherData);
            } else {
                console.error('Failed to load weather data');
            }
        } catch (error) {
            console.error('Error loading weather data:', error);
        }
    }

    getCurrentLocation() {
        // Try to get from app state, fallback to Pretoria
        try {
            const location = window.appState?.get('currentLocation');
            if (location && location.lat && location.lng) {
                return location;
            }
        } catch (error) {
            console.log('Could not get current location from app state');
        }
        
        // Default to Pretoria, South Africa
        return { lat: -25.8408448, lng: 28.2394624 };
    }

    generateNotificationsFromSightings() {
        this.notifications = [];
        
        // This would be populated from actual sightings data
        // For now, create some sample notifications
        this.notifications = [
            {
                id: 'high-risk-1',
                type: 'high_risk',
                title: 'High Risk Detection',
                message: 'High-risk invasive species detected in Pretoria area',
                timestamp: new Date().toISOString(),
                read: false,
                priority: 'high'
            },
            {
                id: 'new-species-1',
                type: 'new_species',
                title: 'New Species Detected',
                message: 'New invasive species identified in the database',
                timestamp: new Date(Date.now() - 86400000).toISOString(),
                read: false,
                priority: 'medium'
            }
        ];
    }

    async refreshData() {
        await this.loadNotifications();
        await this.loadWeatherData();
    }

    onTabActivated() {
        this.updateNotifications();
    }

    updateNotifications() {
        const notificationsList = document.getElementById('notifications-list');
        if (!notificationsList) return;
        
        const unreadCount = this.notifications.filter(n => !n.read).length;
        const indicator = document.getElementById('notification-indicator');
        
        if (indicator) {
            indicator.style.display = unreadCount > 0 ? 'inline' : 'none';
            indicator.textContent = unreadCount;
        }
        
        // Start with weather card (no mark read button)
        let notificationsHTML = '';
        
        if (this.weatherData) {
            notificationsHTML += this.createWeatherCard();
        }
        
        // Add regular notifications
        if (this.notifications.length === 0) {
            notificationsHTML += `
                <div class="notifications-empty">
                    <i class="fas fa-bell-slash"></i>
                    <h3>No Notifications</h3>
                    <p>You're all caught up!</p>
                </div>
            `;
        } else {
            const regularNotificationsHTML = this.notifications
                .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
                .map(notification => this.createNotificationItem(notification))
                .join('');
            
            notificationsHTML += regularNotificationsHTML;
        }
        
        notificationsList.innerHTML = notificationsHTML;
        
        // Setup mark as read functionality (only for regular notifications)
        this.setupNotificationActions();
    }

    createWeatherCard() {
        if (!this.weatherData || !this.weatherData.current) {
            return '';
        }

        const current = this.weatherData.current;
        const weatherCode = current.weather_code;
        const weatherDescription = this.getWeatherDescription(weatherCode);
        const weatherIcon = this.getWeatherIcon(weatherCode);
        
        return `
            <div class="notification-item weather-card" data-type="weather">
                <div class="notification-header">
                    <div class="notification-title">Current Weather</div>
                    <div class="notification-time">Updated just now</div>
                </div>
                <div class="notification-message">
                    <div class="weather-info">
                        <div class="weather-main">
                            <i class="fas ${weatherIcon}"></i>
                            <span>${current.temperature_2m}°C (feels like ${current.apparent_temperature}°C)</span>
                        </div>
                        <div class="weather-desc">${weatherDescription}</div>
                        <div class="weather-details">
                            <span><i class="fas fa-tint"></i> ${current.relative_humidity_2m}%</span>
                            <span><i class="fas fa-wind"></i> ${current.wind_speed_10m} km/h</span>
                            ${current.precipitation > 0 ? `<span><i class="fas fa-cloud-rain"></i> ${current.precipitation} mm</span>` : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    getWeatherDescription(code) {
        const descriptions = {
            0: 'Clear sky',
            1: 'Mainly clear',
            2: 'Partly cloudy',
            3: 'Overcast',
            45: 'Foggy',
            48: 'Depositing rime fog',
            51: 'Light drizzle',
            53: 'Moderate drizzle',
            55: 'Dense drizzle',
            61: 'Slight rain',
            63: 'Moderate rain',
            65: 'Heavy rain',
            71: 'Slight snow',
            73: 'Moderate snow',
            75: 'Heavy snow',
            77: 'Snow grains',
            80: 'Slight rain showers',
            81: 'Moderate rain showers',
            82: 'Violent rain showers',
            85: 'Slight snow showers',
            86: 'Heavy snow showers',
            95: 'Thunderstorm',
            96: 'Thunderstorm with slight hail',
            99: 'Thunderstorm with heavy hail'
        };
        return descriptions[code] || 'Unknown';
    }

    getWeatherIcon(code) {
        const icons = {
            0: 'fa-sun',
            1: 'fa-sun',
            2: 'fa-cloud-sun',
            3: 'fa-cloud',
            45: 'fa-smog',
            48: 'fa-smog',
            51: 'fa-cloud-rain',
            53: 'fa-cloud-rain',
            55: 'fa-cloud-showers-heavy',
            61: 'fa-cloud-rain',
            63: 'fa-cloud-showers-heavy',
            65: 'fa-cloud-showers-heavy',
            71: 'fa-snowflake',
            73: 'fa-snowflake',
            75: 'fa-snowflake',
            77: 'fa-snowflake',
            80: 'fa-cloud-rain',
            81: 'fa-cloud-showers-heavy',
            82: 'fa-cloud-showers-heavy',
            85: 'fa-snowflake',
            86: 'fa-snowflake',
            95: 'fa-bolt',
            96: 'fa-bolt',
            99: 'fa-bolt'
        };
        return icons[code] || 'fa-cloud';
    }

    createNotificationItem(notification) {
        const readClass = notification.read ? 'read' : '';
        const priorityClass = notification.priority;
        
        return `
            <div class="notification-item ${readClass} ${priorityClass}" data-id="${notification.id}">
                <div class="notification-header">
                    <div class="notification-title">${sanitizeHtml(notification.title)}</div>
                    <div class="notification-time">${this.formatDate(notification.timestamp)}</div>
                </div>
                <div class="notification-message">${sanitizeHtml(notification.message)}</div>
                <div class="notification-actions">
                    ${!notification.read ? `
                        <button class="mark-read-btn" data-id="${notification.id}">
                            <i class="fas fa-check"></i>
                            Mark Read
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }

    getNotificationIcon(type) {
        const icons = {
            high_risk: 'fa-exclamation-triangle',
            new_species: 'fa-seedling',
            management: 'fa-tools',
            system: 'fa-cog'
        };
        return icons[type] || 'fa-bell';
    }

    setupNotificationActions() {
        // Mark individual as read (only for regular notifications, not weather)
        document.querySelectorAll('.mark-read-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const notificationId = e.target.closest('.mark-read-btn').dataset.id;
                this.markNotificationAsRead(notificationId);
            });
        });
        
        // Mark all as read
        const markAllBtn = document.getElementById('mark-all-read-btn');
        if (markAllBtn) {
            markAllBtn.addEventListener('click', () => {
                this.markAllNotificationsAsRead();
            });
        }
    }

    async markNotificationAsRead(notificationId) {
        try {
            await api.markNotificationAsRead(notificationId);
            const notification = this.notifications.find(n => n.id === notificationId);
            if (notification) {
                notification.read = true;
                this.updateNotifications();
            }
        } catch (error) {
            console.error('Error marking notification as read:', error);
        }
    }

    async markAllNotificationsAsRead() {
        try {
            await api.markAllNotificationsAsRead();
            this.notifications.forEach(n => n.read = true);
            this.updateNotifications();
        } catch (error) {
            console.error('Error marking all notifications as read:', error);
        }
    }

    formatDate(dateString) {
        if (!dateString) return 'Unknown';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }
}

// Create singleton instance
export const notificationsManager = new NotificationsManager(); 