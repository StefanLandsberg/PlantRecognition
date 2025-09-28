// js/sightings/notifications.js

let notifications = {
  risk: [],
  weather: [],
  general: [],
  completedAlerts: [] // Track completed alert IDs
};

const saveNotifications = () => {
  localStorage.setItem('plantNotifications', JSON.stringify(notifications));
};

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

export const addNotification = (type, notification) => {
  notifications[type].push({
    ...notification,
    id: Date.now() + Math.random(),
    timestamp: new Date().toISOString(),
    dismissed: false
  });
  saveNotifications();
  updateNotificationBadges();
};

const checkDailyWeatherAlert = () => {
  const today = new Date().toDateString();
  const hasWeatherToday = notifications.weather.some(n =>
    new Date(n.timestamp).toDateString() === today && !n.dismissed
  );

  if (!hasWeatherToday) {
    const weatherConditions = ['Sunny', 'Partly Cloudy', 'Overcast', 'Light Rain', 'Heavy Rain', 'Windy'];
    const condition = weatherConditions[Math.floor(Math.random() * weatherConditions.length)];
    const temp = Math.floor(Math.random() * 20) + 15;

    let alertLevel = 'info';
    let alertTitle = 'Daily Weather Update';
    let alertAction = 'Check detailed forecast';

    if (condition === 'Heavy Rain') {
      alertLevel = 'warning';
      alertTitle = 'Weather Alert: Heavy Rain';
      alertAction = 'Review safety protocols';
    } else if (condition === 'Windy') {
      alertLevel = 'warning';
      alertTitle = 'Weather Alert: High Winds';
      alertAction = 'Adjust monitoring plans';
    }

    addNotification('weather', {
      title: alertTitle,
      description: `Today's conditions: ${condition}, ${temp}°C. ${condition === 'Heavy Rain' || condition === 'Windy' ? 'Take precautions during field work.' : 'Optimal conditions for species monitoring.'}`,
      level: alertLevel,
      action: alertAction
    });
  }
};

export const initializeNotifications = () => {
  const savedNotifications = localStorage.getItem('plantNotifications');
  if (savedNotifications) {
    notifications = JSON.parse(savedNotifications);
  }
  checkDailyWeatherAlert();
  updateNotificationBadges();
};

export const dismissNotification = (type, notificationId) => {
  const notification = notifications[type].find(n => n.id === notificationId);
  if (notification) {
    notification.dismissed = true;
    saveNotifications();
    updateNotificationBadges();
  }
};

export const removeNotification = (type, notificationId) => {
  notifications[type] = notifications[type].filter(n => n.id !== notificationId);
  saveNotifications();
  updateNotificationBadges();
};

export const getActiveNotifications = (type) => {
  return notifications[type].filter(n => !n.dismissed);
};

// Expose notifications for direct access if needed, e.g., for completedAlerts
export const notificationState = notifications;