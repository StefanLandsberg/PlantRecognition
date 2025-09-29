// js/sightings/notifications.js

let alertsCache = {
  risk: [],
  weather: [],
  general: []
};

// API endpoints
const ALERTS_API = '/api/alerts';

export const updateNotificationBadges = () => {
  const riskCount = getActiveNotifications('risk').length;
  const weatherCount = getActiveNotifications('weather').length;
  const totalAlerts = riskCount + weatherCount;

  const riskBadge = document.getElementById('risk-notification-badge');
  if (riskBadge) {
    if (totalAlerts > 0) {
      riskBadge.textContent = totalAlerts;
      riskBadge.style.display = 'flex';
      riskBadge.className = 'notification-badge';
      if (riskCount > 0) {
        riskBadge.classList.add('danger');
      } else if (weatherCount > 0) {
        riskBadge.classList.add('info');
      }
    } else {
      riskBadge.style.display = 'none';
    }
  }
};

// Fetch alerts from server
export const fetchAlerts = async (type = null) => {
  try {
    const url = type ? `${ALERTS_API}?type=${type}` : ALERTS_API;
    const response = await fetch(url, {
      credentials: 'include'
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch alerts: ${response.status}`);
    }

    const result = await response.json();
    if (result.success) {
      // Update cache
      if (type) {
        alertsCache[type] = result.data.filter(alert => alert.type === type);
      } else {
        // Group alerts by type
        alertsCache = { risk: [], weather: [], general: [] };
        result.data.forEach(alert => {
          if (alertsCache[alert.type]) {
            alertsCache[alert.type].push(alert);
          }
        });
      }

      updateNotificationBadges();
      return result.data;
    }
  } catch (error) {
    console.error('Error fetching alerts:', error);
    return [];
  }
};

// Refresh alerts (trigger server-side generation)
export const refreshAlerts = async () => {
  try {
    const response = await fetch(`${ALERTS_API}/refresh`, {
      method: 'POST',
      credentials: 'include'
    });

    if (!response.ok) {
      throw new Error(`Failed to refresh alerts: ${response.status}`);
    }

    const result = await response.json();
    if (result.success) {
      // Update cache
      alertsCache = { risk: [], weather: [], general: [] };
      result.data.forEach(alert => {
        if (alertsCache[alert.type]) {
          alertsCache[alert.type].push(alert);
        }
      });

      updateNotificationBadges();
      return result.data;
    }
  } catch (error) {
    console.error('Error refreshing alerts:', error);
    return [];
  }
};

// Dismiss an alert
export const dismissNotification = async (type, notificationId) => {
  try {
    const response = await fetch(`${ALERTS_API}/${notificationId}/dismiss`, {
      method: 'PATCH',
      credentials: 'include'
    });

    if (!response.ok) {
      throw new Error(`Failed to dismiss alert: ${response.status}`);
    }

    const result = await response.json();
    if (result.success) {
      // Remove from cache
      alertsCache[type] = alertsCache[type].filter(alert => alert.id !== notificationId);
      updateNotificationBadges();
      return true;
    }
  } catch (error) {
    console.error('Error dismissing alert:', error);
    return false;
  }
};

// Initialize notifications system
export const initializeNotifications = async () => {
  await refreshAlerts();
};

// Get active notifications from cache
export const getActiveNotifications = (type) => {
  return alertsCache[type] || [];
};

// Legacy compatibility - these methods now work with server-side alerts
export const addNotification = async (type, notification) => {
  // This is now handled server-side through refreshAlerts()
  console.warn('addNotification called - alerts are now server-side managed');
  return refreshAlerts();
};

export const removeNotification = async (type, notificationId) => {
  return dismissNotification(type, notificationId);
};

// Expose cache for direct access (read-only)
export const notificationState = {
  get risk() { return alertsCache.risk; },
  get weather() { return alertsCache.weather; },
  get general() { return alertsCache.general; }
};