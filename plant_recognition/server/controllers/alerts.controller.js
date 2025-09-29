import Alert from '../models/Alert.js';
import Sighting from '../models/Sighting.js';
import { logger } from '../utils/logger.js';

// Helper function to identify invasive sightings
const isSightingInvasive = (sighting) => {
  return sighting.analysis?.predictedSpecies &&
         sighting.analysis.predictedSpecies !== 'Unknown' &&
         !sighting.isRemoved;
};

// Helper function to identify hotspots
const identifyHotspots = (sightings) => {
  const locationGroups = {};
  const threshold = 0.01; // ~1km grouping

  sightings.forEach(sighting => {
    if (!sighting.location?.coordinates) return;

    const [lng, lat] = sighting.location.coordinates;
    const key = `${Math.round(lat / threshold) * threshold},${Math.round(lng / threshold) * threshold}`;

    if (!locationGroups[key]) {
      locationGroups[key] = [];
    }
    locationGroups[key].push(sighting);
  });

  return Object.values(locationGroups).filter(group => group.length >= 3);
};

// Generate risk alerts based on sighting data
const generateRiskAlerts = async (userId, sightings) => {
  const alerts = [];
  const activeSightings = sightings.filter(s => !s.isRemoved);
  const today = new Date();
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

  // 1. High-risk invasive species detection
  const highRiskInvasives = activeSightings.filter(s => {
    const risk = s.analysis?.llm?.details?.risk_level?.toLowerCase() || '';
    return risk.includes('high') || risk.includes('severe');
  });

  if (highRiskInvasives.length > 0) {
    const recentInvasive = highRiskInvasives.filter(s => new Date(s.createdAt) >= weekAgo);
    if (recentInvasive.length > 0) {
      alerts.push({
        owner: userId,
        type: 'risk',
        level: 'critical',
        title: 'New Invasive Species Detected',
        description: `${recentInvasive.length} high-risk invasive sightings in the past week. Immediate containment required.`,
        action: 'Deploy containment teams',
        metadata: {
          alertKey: `invasive_critical_${today.toISOString().split('T')[0]}`,
          dataSnapshot: {
            highRiskCount: recentInvasive.length,
            species: recentInvasive.map(s => s.analysis?.predictedSpecies).filter(Boolean)
          }
        }
      });
    }
  }

  // 2. Species concentration hotspots
  const hotspots = identifyHotspots(activeSightings);
  if (hotspots.length > 0) {
    alerts.push({
      owner: userId,
      type: 'risk',
      level: 'warning',
      title: 'Species Concentration Detected',
      description: `${hotspots.length} hotspots identified. Monitor for potential spreading.`,
      action: 'Increase surveillance',
      metadata: {
        alertKey: `hotspots_${today.toISOString().split('T')[0]}`,
        dataSnapshot: {
          hotspotCount: hotspots.length,
          locations: hotspots.map(h => h[0]?.location?.coordinates).filter(Boolean)
        }
      }
    });
  }

  // 3. Low detection activity warning
  const recentSightings = activeSightings.filter(s => new Date(s.createdAt) >= weekAgo);
  if (recentSightings.length < 5 && activeSightings.length > 0) {
    alerts.push({
      owner: userId,
      type: 'risk',
      level: 'info',
      title: 'Low Detection Activity',
      description: 'Detection activity is below normal levels. Consider increasing monitoring.',
      action: 'Schedule additional surveys',
      metadata: {
        alertKey: `low_activity_${today.toISOString().split('T')[0]}`,
        dataSnapshot: {
          recentCount: recentSightings.length,
          totalActive: activeSightings.length
        }
      }
    });
  }

  return alerts;
};

// Generate daily weather alert (placeholder - can integrate with real weather API)
const generateWeatherAlert = async (userId) => {
  const today = new Date();
  const todayKey = today.toISOString().split('T')[0];

  // Check if we already have a weather alert for today
  const existingWeatherAlert = await Alert.findOne({
    owner: userId,
    type: 'weather',
    'metadata.alertKey': `weather_${todayKey}`,
    isDismissed: false
  });

  if (existingWeatherAlert) {
    return null; // Already have weather alert for today
  }

  // Generate random weather conditions for demo
  const conditions = [
    { condition: 'Sunny', temp: 22, level: 'info', action: 'Optimal for surveys' },
    { condition: 'Partly Cloudy', temp: 18, level: 'info', action: 'Good conditions for fieldwork' },
    { condition: 'Light Rain', temp: 16, level: 'warning', action: 'Use waterproof equipment' },
    { condition: 'Heavy Rain', temp: 14, level: 'warning', action: 'Review safety protocols' },
    { condition: 'High Winds', temp: 20, level: 'warning', action: 'Adjust monitoring plans' },
    { condition: 'Severe Storm', temp: 12, level: 'critical', action: 'Suspend operations' }
  ];

  const weather = conditions[Math.floor(Math.random() * conditions.length)];

  return {
    owner: userId,
    type: 'weather',
    level: weather.level,
    title: `Weather: ${weather.condition}`,
    description: `Current: ${weather.temp}°C. ${weather.condition === 'Heavy Rain' || weather.condition === 'High Winds' || weather.condition === 'Severe Storm' ? 'Take precautions during field work.' : 'Good conditions for species monitoring.'}`,
    action: weather.action,
    metadata: {
      alertKey: `weather_${todayKey}`,
      expiresAt: new Date(today.getTime() + 24 * 60 * 60 * 1000), // Expires in 24 hours
      dataSnapshot: {
        condition: weather.condition,
        temperature: weather.temp
      }
    }
  };
};

// List active alerts for user
export async function list(req, res, next) {
  try {
    const { type } = req.query;
    const userId = req.auth.userId;

    const alerts = await Alert.findActive(userId, type);

    res.json({
      success: true,
      data: alerts.map(alert => ({
        id: alert._id,
        type: alert.type,
        level: alert.level,
        title: alert.title,
        description: alert.description,
        action: alert.action,
        timestamp: alert.createdAt,
        isActive: alert.isActive
      }))
    });
  } catch (e) {
    next(e);
  }
}

// Dismiss an alert
export async function dismiss(req, res, next) {
  try {
    const { id } = req.params;
    const userId = req.auth.userId;

    const alert = await Alert.findOne({ _id: id, owner: userId });

    if (!alert) {
      return res.status(404).json({ success: false, error: 'Alert not found' });
    }

    await alert.dismiss();

    res.json({ success: true, data: { id: alert._id, dismissed: true } });
  } catch (e) {
    next(e);
  }
}

// Generate and refresh alerts for user
export async function refresh(req, res, next) {
  try {
    const userId = req.auth.userId;

    // Get user's sightings
    const sightings = await Sighting.find({ owner: userId }).sort({ createdAt: -1 }).limit(200);

    // Generate risk alerts
    const riskAlerts = await generateRiskAlerts(userId, sightings);

    // Create or update risk alerts
    for (const alertData of riskAlerts) {
      try {
        await Alert.createOrUpdate(alertData);
      } catch (error) {
        // Handle duplicate key errors gracefully
        if (error.code !== 11000) {
          logger.error('Error creating alert:', error);
        }
      }
    }

    // Generate weather alert if needed
    const weatherAlert = await generateWeatherAlert(userId);
    if (weatherAlert) {
      try {
        await Alert.createOrUpdate(weatherAlert);
      } catch (error) {
        if (error.code !== 11000) {
          logger.error('Error creating weather alert:', error);
        }
      }
    }

    // Return updated alerts
    const alerts = await Alert.findActive(userId);

    res.json({
      success: true,
      data: alerts.map(alert => ({
        id: alert._id,
        type: alert.type,
        level: alert.level,
        title: alert.title,
        description: alert.description,
        action: alert.action,
        timestamp: alert.createdAt,
        isActive: alert.isActive
      })),
      generated: riskAlerts.length + (weatherAlert ? 1 : 0)
    });
  } catch (e) {
    next(e);
  }
}

// Get alert statistics
export async function stats(req, res, next) {
  try {
    const userId = req.auth.userId;

    const stats = await Alert.aggregate([
      { $match: { owner: userId } },
      {
        $group: {
          _id: '$level',
          count: { $sum: 1 },
          active: {
            $sum: {
              $cond: [
                { $and: [
                  { $eq: ['$isDismissed', false] },
                  { $or: [
                    { $not: { $ifNull: ['$metadata.expiresAt', false] } },
                    { $gt: ['$metadata.expiresAt', new Date()] }
                  ]}
                ]},
                1,
                0
              ]
            }
          }
        }
      }
    ]);

    const result = {
      critical: { total: 0, active: 0 },
      warning: { total: 0, active: 0 },
      info: { total: 0, active: 0 }
    };

    stats.forEach(stat => {
      result[stat._id] = {
        total: stat.count,
        active: stat.active
      };
    });

    res.json({ success: true, data: result });
  } catch (e) {
    next(e);
  }
}