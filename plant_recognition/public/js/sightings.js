import { SightingsAPI } from "./api.js";
import * as analytics from "./sightings/analytics.js";
import { calculateDistance } from "./sightings/utils.js";


function fmtDate(s) {
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s || "";
  }
}

// Make fmtDate globally available
window.fmtDate = fmtDate;

function fmtLatLng(coords) {
  if (!Array.isArray(coords) || coords.length < 2) return "";
  const [lng, lat] = coords;
  return `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
}

function pct(n) {
  if (n == null || isNaN(n)) return "Unknown";
  return (Number(n) * 100).toFixed(1) + "%";
}

function formatLLMSection(llmData, section) {
  if (!llmData || !llmData.details) return '<p style="color: var(--text);">No analysis data available.</p>';

  const analysisData = llmData.details;

  switch (section) {
    case 'species':
      const speciesInfo = analysisData.advisory_content?.species_identification;
      return `
        <h4 style="color: var(--accent);">Species Information</h4>
        <p style="color: var(--text);"><strong>Scientific Name:</strong> ${speciesInfo?.scientific_name || analysisData.species || 'Unknown'}</p>
        <p style="color: var(--text);"><strong>Common Names:</strong> ${speciesInfo?.common_names || analysisData.common_name || 'Unknown'}</p>
        <p style="color: var(--text);"><strong>Family:</strong> ${speciesInfo?.family || analysisData.family || 'Unknown'}</p>
      `;

    case 'legal':
      const legalInfo = analysisData.advisory_content?.legal_status;
      return `
        <h4 style="color: var(--accent);">Legal Status</h4>
        <p style="color: var(--text);"><strong>NEMBA Category:</strong> ${legalInfo?.nemba_category || 'Unknown'}</p>
        <p style="color: var(--text);"><strong>Legal Requirements:</strong> ${legalInfo?.legal_requirements || 'Unknown'}</p>
        <p style="color: var(--text);"><strong>Risk Level:</strong> ${analysisData.risk_level || 'Unknown'}</p>
        <p style="color: var(--text);"><strong>Invasive Status:</strong> ${analysisData.invasive_status ? 'Yes' : 'No'}</p>
      `;

    case 'description':
      const physicalDesc = analysisData.advisory_content?.physical_description;
      return `
        <h4 style="color: var(--accent);">Physical Description</h4>
        <p style="color: var(--text);">${physicalDesc || analysisData.description || 'No description available.'}</p>
        <p style="color: var(--text);"><strong>Origin:</strong> ${analysisData.origin || 'Unknown'}</p>
      `;

    case 'distribution':
      const distributionInfo = analysisData.advisory_content?.distribution || analysisData.where_found || analysisData.distribution;
      if (!distributionInfo || distributionInfo === 'Not found') {
        return '<p style="color: var(--text);">No distribution information available.</p>';
      }
      return `
        <h4 style="color: var(--accent);">Where Found</h4>
        <p style="color: var(--text);">${distributionInfo}</p>
      `;

    case 'control':
      const controlInfo = analysisData.advisory_content?.control_methods || analysisData.treatment || analysisData.control_methods;
      if (!controlInfo || controlInfo === 'Not found') {
        return '<p style="color: var(--text);">No control methods available.</p>';
      }
      return `
        <h4 style="color: var(--accent);">Control Methods</h4>
        <p style="color: var(--text);">${controlInfo}</p>
      `;

    case 'action':
      const actionInfo = analysisData.action_required || analysisData.advisory_content?.action_required;
      if (!actionInfo || actionInfo === 'Not found') {
        return '<p style="color: var(--text);">No action required.</p>';
      }
      return `
        <h4 style="color: var(--accent);">Action Required</h4>
        <p style="color: var(--text);">${actionInfo}</p>
      `;

    default:
      return '<p style="color: var(--text);">Select a section to view details.</p>';
  }
}

function createSightingLLMDropdown(sighting) {
  // Fix the data path - LLM data is in sighting.analysis.llm, not sighting.llm
  const hasLLM = sighting.analysis?.llm && sighting.analysis.llm.details;

  if (!hasLLM) {
    // Check if LLM processing is pending
    const llmStatus = sighting.analysis?.llm?.status;
    const statusText = llmStatus === 'pending' ? 'Processing...' :
                     llmStatus === 'failed' ? 'Analysis failed' :
                     'No analysis available';

    return `
      <div class="llm-dropdown">
        <div class="llm-dropdown-header">
          <span>AI Analysis</span>
          <span>${statusText}</span>
        </div>
      </div>
    `;
  }

  return `
    <div class="llm-dropdown">
      <div class="llm-dropdown-header" onclick="toggleLLMDropdown('${sighting._id}')">
        <span>AI Analysis</span>
        <span class="llm-dropdown-arrow">▼</span>
      </div>
      <div class="llm-dropdown-content llm-dropdown-content-hidden">
        <div class="llm-section-selector">
          <button class="llm-section-btn active" onclick="showLLMSection('${sighting._id}', 'species')">Species Info</button>
          <button class="llm-section-btn" onclick="showLLMSection('${sighting._id}', 'legal')">Legal Status</button>
          <button class="llm-section-btn" onclick="showLLMSection('${sighting._id}', 'description')">Description</button>
          <button class="llm-section-btn" onclick="showLLMSection('${sighting._id}', 'distribution')">Distribution</button>
          <button class="llm-section-btn" onclick="showLLMSection('${sighting._id}', 'control')">Control</button>
          <button class="llm-section-btn" onclick="showLLMSection('${sighting._id}', 'action')">Action</button>
        </div>
        <div class="llm-section-content" id="llm-content-${sighting._id}">
          ${formatLLMSection(sighting.analysis.llm, 'species')}
        </div>
      </div>
    </div>
  `;
}

function showImageModal(imageUrl) {
  if (!imageUrl) return; // Don't show modal for null images

  // Validate URL format
  if (typeof imageUrl !== 'string' || (!imageUrl.startsWith('http') && !imageUrl.startsWith('/') && !imageUrl.startsWith('data:'))) {
    console.warn('Invalid image URL format:', imageUrl);
    return;
  }

  const modal = document.createElement('div');
  modal.className = 'image-modal';
  modal.innerHTML = `<img src="${imageUrl}" alt="Plant sighting image" />`;

  document.body.appendChild(modal);

  requestAnimationFrame(() => {
    modal.classList.add('show');
  });

  modal.addEventListener('click', () => {
    modal.classList.remove('show');
    setTimeout(() => modal.remove(), 300);
  });
}

// === NOTIFICATION SYSTEM ===
let notifications = {
  risk: [],
  weather: [],
  general: [],
  completedAlerts: [] // Track completed alert IDs
};

function initializeNotifications() {
  // Load notifications from localStorage
  const savedNotifications = localStorage.getItem('plantNotifications');
  if (savedNotifications) {
    try {
      const parsed = JSON.parse(savedNotifications);
      notifications.risk = Array.isArray(parsed?.risk) ? parsed.risk : [];
      notifications.weather = Array.isArray(parsed?.weather) ? parsed.weather : [];
      notifications.general = Array.isArray(parsed?.general) ? parsed.general : [];
      notifications.completedAlerts = Array.isArray(parsed?.completedAlerts) ? parsed.completedAlerts : [];
    } catch (error) {
      console.warn('Failed to parse stored notifications:', error);
      notifications = {
        risk: [],
        weather: [],
        general: [],
        completedAlerts: []
      };
    }
  }

  // Update notification badges
  updateNotificationBadges();
}

function saveNotifications() {
  localStorage.setItem('plantNotifications', JSON.stringify(notifications));
}

function addNotification(type, notification) {
  notifications[type].push({
    ...notification,
    id: Date.now() + Math.random(),
    timestamp: new Date().toISOString(),
    dismissed: false
  });
  saveNotifications();
  updateNotificationBadges();
}

function dismissNotification(type, notificationId) {
  const notification = notifications[type].find(n => n.id === notificationId);
  if (notification) {
    notification.dismissed = true;
    saveNotifications();
    updateNotificationBadges();
  }
}

function removeNotification(type, notificationId) {
  notifications[type] = notifications[type].filter(n => n.id !== notificationId);
  saveNotifications();
  updateNotificationBadges();
}

function getActiveNotifications(type) {
  return notifications[type].filter(n => !n.dismissed);
}

function updateDailyActivityAlert(sightings) {
  const todayKey = new Date().toDateString();

  notifications.weather = notifications.weather.filter((n) => n.meta?.type !== "daily-activity");

  const todaySightings = sightings.filter((s) => {
    const created = new Date(s.createdAt || s.capturedAt);
    return !Number.isNaN(created.getTime()) && created.toDateString() === todayKey;
  });

  const removalsToday = sightings.filter((s) => {
    if (!s.removedAt) return false;
    const removed = new Date(s.removedAt);
    return !Number.isNaN(removed.getTime()) && removed.toDateString() === todayKey;
  });

  const highRiskToday = todaySightings.filter((s) => {
    const risk = s.analysis?.llm?.details?.risk_level || "";
    return typeof risk === "string" && risk.toLowerCase().includes("high");
  });

  const sevenDaysAgo = new Date();
  sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

  const recentSightings = sightings.filter((s) => {
    const created = new Date(s.createdAt || s.capturedAt);
    return !Number.isNaN(created.getTime()) && created >= sevenDaysAgo;
  });

  const recentDayKeys = new Set(
    recentSightings.map((s) => new Date(s.createdAt || s.capturedAt).toDateString())
  );

  const avgDailyActivity = recentDayKeys.size > 0
    ? recentSightings.length / recentDayKeys.size
    : todaySightings.length;

  const differenceFromAverage = avgDailyActivity
    ? todaySightings.length - avgDailyActivity
    : 0;

  let level = "info";
  if (todaySightings.length === 0) {
    level = "info";
  } else if (highRiskToday.length >= 3 || todaySightings.length >= avgDailyActivity * 2) {
    level = "critical";
  } else if (highRiskToday.length > 0 || todaySightings.length >= avgDailyActivity * 1.25) {
    level = "warning";
  }

  const trendDescription = avgDailyActivity
    ? `${differenceFromAverage >= 0 ? "+" : ""}${differenceFromAverage.toFixed(1)} vs 7-day avg`
    : "No 7-day baseline";

  const description = todaySightings.length === 0
    ? (removalsToday.length > 0
        ? `No new sightings logged today. ${removalsToday.length} removal${removalsToday.length === 1 ? "" : "s"} recorded.`
        : "No new sightings logged today. Use the time to inspect known hotspots.")
    : `${todaySightings.length} detection${todaySightings.length === 1 ? "" : "s"} today (${trendDescription}). ${highRiskToday.length} high-risk detection${highRiskToday.length === 1 ? "" : "s"} flagged. Removals logged: ${removalsToday.length}.`;

  let action = "Maintain routine monitoring.";
  if (level === "critical") {
    action = "Deploy crews to hotspots immediately.";
  } else if (level === "warning") {
    action = "Prioritize verification of high-risk detections.";
  } else if (todaySightings.length === 0 && removalsToday.length === 0) {
    action = "Schedule proactive site inspections.";
  } else if (removalsToday.length > 0 && todaySightings.length === 0) {
    action = "Document removal outcomes and monitor treated areas.";
  }

  addNotification("weather", {
    title: "Daily Activity Summary",
    description,
    level,
    action,
    meta: {
      type: "daily-activity",
      date: todayKey,
      metrics: {
        detections: todaySightings.length,
        highRisk: highRiskToday.length,
        removals: removalsToday.length,
        averageDaily: Number(avgDailyActivity.toFixed(2)),
        variance: Number(differenceFromAverage.toFixed(2))
      }
    }
  });
}

// Global function to manually trigger weather alerts for testing
window.triggerWeatherAlert = function() {
  updateDailyActivityAlert(window.sightingsData || []);
  showNotificationToast('Daily activity summary refreshed', 'info');
};

function updateNotificationBadges() {
  const riskCount = getActiveNotifications('risk').length;
  const weatherCount = getActiveNotifications('weather').length;
  const totalAlerts = riskCount + weatherCount;

  // Update risk assessment badge
  const riskBadge = document.getElementById('risk-notification-badge');
  if (riskBadge) {
    if (totalAlerts > 0) {
      riskBadge.textContent = totalAlerts;
      riskBadge.style.display = 'flex';

      // Set badge color - RED OVERRIDES BLUE (risk takes priority over weather)
      riskBadge.className = 'notification-badge';
      if (riskCount > 0) {
        riskBadge.classList.add('danger'); // Red for risk alerts
      } else if (weatherCount > 0) {
        riskBadge.classList.add('info'); // Blue for weather alerts only
      }
    } else {
      riskBadge.style.display = 'none';
    }
  }
}


// Global functions for dropdown interactions
window.toggleLLMDropdown = function(sightingId) {
  const dropdown = document.querySelector(`[data-sighting-id="${sightingId}"] .llm-dropdown`);
  const content = document.querySelector(`[data-sighting-id="${sightingId}"] .llm-dropdown-content`);
  const arrow = document.querySelector(`[data-sighting-id="${sightingId}"] .llm-dropdown-arrow`);

  if (dropdown && content) {
    const isHidden = content.classList.contains('llm-dropdown-content-hidden');

    if (isHidden) {
      content.classList.remove('llm-dropdown-content-hidden');
      if (arrow) {
        arrow.textContent = '▲';
      }
      dropdown.classList.add('open');
    } else {
      content.classList.add('llm-dropdown-content-hidden');
      if (arrow) {
        arrow.textContent = '▼';
      }
      dropdown.classList.remove('open');
    }
  }
};

window.showLLMSection = function(sightingId, section) {
  const container = document.querySelector(`[data-sighting-id="${sightingId}"]`);
  if (!container) return;

  // Update active button
  container.querySelectorAll('.llm-section-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  container.querySelector(`[onclick*="'${section}'"]`).classList.add('active');

  // Find sighting data and update content
  const sightings = window.sightingsData || [];
  const sighting = sightings.find(s => s._id === sightingId);
  if (sighting) {
    const content = container.querySelector(`#llm-content-${sightingId}`);
    if (content) {
      content.innerHTML = formatLLMSection(sighting.analysis?.llm, section);
    }
  }
};

// Dropdown and Tab switching
window.toggleDropdown = function() {
  const dropdown = document.querySelector('.dropdown-container');
  const menu = document.getElementById('analytics-dropdown');

  dropdown.classList.toggle('open');
  menu.classList.toggle('show');

  // Add click-outside listener when dropdown is opened
  if (dropdown.classList.contains('open')) {
    setTimeout(() => {
      const handleClickOutside = (e) => {
        if (!dropdown.contains(e.target)) {
          dropdown.classList.remove('open');
          menu.classList.remove('show');
          document.removeEventListener('click', handleClickOutside);
        }
      };
      document.addEventListener('click', handleClickOutside);
    }, 10);
  }
};

// Toggle LLM dropdown
window.toggleLLMDropdown = function(sightingId) {
  const dropdown = document.querySelector(`[data-sighting-id="${sightingId}"] .llm-dropdown`);
  if (!dropdown) return;

  dropdown.classList.toggle('open');

  const content = dropdown.querySelector('.llm-dropdown-content');
  if (content) {
    content.classList.toggle('llm-dropdown-content-hidden');
  }
};


window.switchTab = function(tabName) {
  // Close dropdown
  const dropdown = document.querySelector('.dropdown-container');
  const menu = document.getElementById('analytics-dropdown');
  dropdown.classList.remove('open');
  menu.classList.remove('show');

  // Update tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
  });

  if (tabName === 'sightings') {
    document.querySelector(`[onclick*="'${tabName}'"]`).classList.add('active');
  } else {
    // Highlight dropdown button for analytics tabs
    document.querySelector('.dropdown-btn').classList.add('active');
  }

  // Update tab content
  document.querySelectorAll('.tab-content').forEach(content => {
    content.classList.remove('active');
  });
  document.getElementById(`${tabName}-tab`).classList.add('active');

  // Load appropriate analytics
  // Note: Tab switching is now handled by ui.js module
};



// === GEOGRAPHIC INSIGHTS ===
function loadGeographicInsights(sightings) {
  const container = document.getElementById('geographic-insights-container');

  // Filter out removed sightings for geographic analysis
  const activeSightings = sightings.filter(s => !s.isRemoved);

  const locationClusters = analytics.analyzeLocationClusters(activeSightings);
  const densityMap = analytics.createDensityAnalysis(activeSightings);
  const coverageStats = analytics.calculateCoverageStats(activeSightings);

  container.innerHTML = `
    <div class="dashboard-header">
      <h2 class="dashboard-title">Geographic Insights</h2>
      <p class="dashboard-subtitle">Spatial distribution patterns and geographic coverage analysis</p>
    </div>

    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value">${coverageStats.totalArea.toFixed(1)}km²</div>
        <div class="metric-label">Coverage Area</div>
        <div class="metric-change neutral">Geographic footprint</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${locationClusters.length}</div>
        <div class="metric-label">Location Clusters</div>
        <div class="metric-change neutral">Distinct geographic groups</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${densityMap.hotspots}</div>
        <div class="metric-label">High Density Areas</div>
        <div class="metric-change ${densityMap.hotspots > 0 ? 'positive' : 'neutral'}">
          Concentration zones
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${coverageStats.avgDistance.toFixed(1)}km</div>
        <div class="metric-label">Avg Distance</div>
        <div class="metric-change neutral">Between sightings</div>
      </div>
    </div>

    <div class="geo-grid">
      <div class="chart-container">
        <div class="chart-header">
          <h3 class="chart-title">Species Density Distribution</h3>
        </div>
        <div class="chart-content">
          <div class="simple-chart" id="density-distribution-chart"></div>
        </div>
      </div>

      <div class="chart-container">
        <div class="chart-header">
          <h3 class="chart-title">Geographic Coverage Analysis</h3>
        </div>
        <div class="chart-content">
          <div class="simple-chart" id="coverage-analysis-chart"></div>
        </div>
      </div>
    </div>

    <div class="geo-grid">
      <div class="chart-container">
        <div class="chart-header">
          <h3 class="chart-title">Distance Between Sightings</h3>
        </div>
        <div class="chart-content">
          <div class="simple-chart" id="distance-distribution-chart"></div>
        </div>
      </div>

      <div class="chart-container">
        <div class="chart-header">
          <h3 class="chart-title">Location Cluster Analysis</h3>
        </div>
        <div class="chart-content">
          <div class="simple-chart" id="cluster-size-chart"></div>
        </div>
      </div>
    </div>

  `;

  // Generate meaningful geographic charts
  generateDensityDistributionChart(sightings, 'density-distribution-chart');
  generateCoverageAnalysisChart(coverageStats, 'coverage-analysis-chart');
  generateDistanceDistributionChart(sightings, 'distance-distribution-chart');
  generateClusterSizeChart(locationClusters, 'cluster-size-chart');
}

// === TEMPORAL ANALYSIS ===
function loadTemporalAnalysis(sightings) {
  const container = document.getElementById('temporal-analysis-container');

  const timePatterns = analytics.analyzeTimePatterns(sightings);
  const trends = analytics.calculateTrends(sightings);
  const seasonalData = analytics.analyzeSeasonalPatterns(sightings);

  // CSS styles now in sightings.css

  const invasiveAnalytics = analytics.generateInvasiveAnalytics(sightings);

  container.innerHTML = `
    <div class="dashboard-header">
      <h2 class="dashboard-title">Invasive Species Intelligence Dashboard</h2>
      <p class="dashboard-subtitle">Comprehensive threat assessment and management analytics</p>
    </div>

    <div class="farmer-alert-section">
      <div class="alert-banner ${invasiveAnalytics.todayThreatLevel}">
        <h3>Today's Threat Level: ${invasiveAnalytics.todayThreatLevel.toUpperCase()}</h3>
        <p>${invasiveAnalytics.todayRecommendation}</p>
      </div>

      <div class="immediate-actions">
        <h4>Immediate Actions Required:</h4>
        <ul>
          ${invasiveAnalytics.immediateActions.map(action => `<li>${action}</li>`).join('')}
        </ul>
      </div>
    </div>

    <div class="farmer-metrics-grid">
      <div class="metric-card">
        <div class="metric-value invasive-count">${invasiveAnalytics.newInvasivesToday}</div>
        <div class="metric-label">New Invasives Today</div>
        <div class="metric-change">${invasiveAnalytics.invasiveTrend}</div>
      </div>
      <div class="metric-card">
        <div class="metric-value hotspot-count">${invasiveAnalytics.hotspotCount}</div>
        <div class="metric-label">Active Hotspots</div>
        <div class="metric-description">Areas needing attention</div>
      </div>
      <div class="metric-card">
        <div class="metric-value spread-rate">${invasiveAnalytics.spreadRate.toFixed(1)}m/day</div>
        <div class="metric-label">Avg Spread Rate</div>
        <div class="metric-change">${invasiveAnalytics.spreadTrend}</div>
      </div>
      <div class="metric-card">
        <div class="metric-value environment-impact ${invasiveAnalytics.environmentalImpact.severity}">${invasiveAnalytics.environmentalImpact.severity}</div>
        <div class="metric-label">Environmental Impact</div>
        <div class="metric-description">${invasiveAnalytics.environmentalImpact.description}</div>
        <div class="environmental-details">
          <strong>Active Threats:</strong> ${invasiveAnalytics.environmentalImpact.totalActiveInvasives} invasive species<br>
          <strong>Risk Score:</strong> ${invasiveAnalytics.environmentalImpact.riskScore.toFixed(0)}/100<br>
          <strong>Urgency:</strong> ${invasiveAnalytics.environmentalImpact.urgency}
        </div>
      </div>
    </div>

    <div class="chart-container">
      <div class="chart-header">
        <h3 class="chart-title">Invasion Intelligence Timeline</h3>
        <div class="chart-controls">
          <button class="chart-control-btn active" data-timeline="daily" onclick="updateTimelineView('daily', this)">Daily Threats</button>
          <button class="chart-control-btn" data-timeline="weekly" onclick="updateTimelineView('weekly', this)">Weekly Trends</button>
          <button class="chart-control-btn" data-timeline="monthly" onclick="updateTimelineView('monthly', this)">Monthly Strategy</button>
          <button class="chart-control-btn" data-timeline="yearly" onclick="updateTimelineView('yearly', this)">Annual Patterns</button>
        </div>
      </div>
      <div class="chart-content">
        <div class="simple-chart" id="timeline-chart"></div>
      </div>
    </div>

    <div class="timeline-container" id="compact-timeline-container">
      <div class="timeline-line"></div>
      <div id="timeline-events" class="timeline-events">${generateInvasiveTimelineEvents(sightings, 'daily')}</div>
    </div>

    <div class="farmer-insights-grid">
      <div class="insight-card weather">
        <h4>Environmental Impact</h4>
        <p>${invasiveAnalytics.weatherImpact}</p>
      </div>
      <div class="insight-card control">
        <h4>Management Effectiveness</h4>
        <p>${invasiveAnalytics.controlEffectiveness}% removal success rate this week</p>
      </div>
    </div>
  `;

  // Generate charts
  generateTimelineChart(timePatterns, 'timeline-chart');
  generateHourlyChart(timePatterns, 'hourly-chart');
  generateSeasonalChart(seasonalData, 'seasonal-chart');
}

// Timeline view update function
window.updateTimelineView = function(period, buttonElement) {
  // Update active button
  document.querySelectorAll('.chart-control-btn').forEach(btn => btn.classList.remove('active'));
  buttonElement.classList.add('active');

  // Get current sightings data
  const sightings = window.sightingsData || [];

  // Update timeline events based on period
  const timelineContainer = document.getElementById('timeline-events');
  if (timelineContainer) {
    timelineContainer.innerHTML = generateInvasiveTimelineEvents(sightings, period);
  }

  // Update the chart
  const timePatterns = analytics.analyzeTimePatterns(sightings);
  generateTimelineChart(timePatterns, 'timeline-chart', period);
};

// === RISK ASSESSMENT & ALERTS ===
function loadRiskAssessment(sightings) {
  const container = document.getElementById('risk-assessment-container');

  const alerts = analytics.generateRiskAlerts(sightings);
  const priorities = analytics.calculatePriorities(sightings);
  const recommendations = analytics.generateRecommendations(sightings);

  // Get stored notifications
  const riskNotifications = getActiveNotifications('risk');
  const weatherNotifications = getActiveNotifications('weather');
  const allNotifications = [...riskNotifications, ...weatherNotifications];

  // Combine current alerts with stored notifications, excluding completed ones
  const activeStoredNotifications = allNotifications.filter(n =>
    n.dismissed !== true &&
    !notifications.completedAlerts?.includes(n.id) &&
    !alerts.critical.concat(alerts.warning, alerts.info).some(a =>
      a.title === n.title && a.description === n.description
    )
  );

  // Sort alerts: Info first, then Warning, then Critical
  const sortedAlerts = [
    ...alerts.info,
    ...activeStoredNotifications.filter(n => n.level === 'info'),
    ...alerts.warning,
    ...activeStoredNotifications.filter(n => n.level === 'warning'),
    ...alerts.critical,
    ...activeStoredNotifications.filter(n => n.level === 'critical')
  ];

  container.innerHTML = `
    <div class="dashboard-header">
      <h2 class="dashboard-title">Risk Assessment & Alerts</h2>
      <p class="dashboard-subtitle">Priority threats and recommended actions for conservation management</p>
    </div>

    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value">${alerts.critical.length}</div>
        <div class="metric-label">Critical Alerts</div>
        <div class="metric-change ${alerts.critical.length > 0 ? 'negative' : 'positive'}">
          Immediate action required
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${alerts.warning.length}</div>
        <div class="metric-label">Warning Alerts</div>
        <div class="metric-change ${alerts.warning.length > 0 ? 'negative' : 'neutral'}">
          Monitor closely
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${allNotifications.length}</div>
        <div class="metric-label">Active Notifications</div>
        <div class="metric-change ${allNotifications.length > 0 ? 'negative' : 'positive'}">
          Including weather alerts
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${recommendations.length}</div>
        <div class="metric-label">Active Recommendations</div>
        <div class="metric-change positive">Ready for implementation</div>
      </div>
    </div>

    <div class="alert-grid">
      ${sortedAlerts.map(alert => `
        <div class="alert-card ${alert.level}" data-alert-id="${alert.id || 'temp-' + Date.now()}">
          <div class="alert-header">
            <div class="alert-level ${alert.level}">${alert.level}</div>
            <div class="alert-timestamp">${fmtDate(alert.timestamp)}</div>
          </div>
          <div class="alert-title">${alert.title}</div>
          <div class="alert-description">${alert.description}</div>
          ${alert.level !== 'info' ? `
            <div class="alert-actions">
              <button class="alert-action-btn secondary" onclick="handleAlertDismiss('${alert.id || 'temp'}', '${alert.level}')">
                Dismiss
              </button>
            </div>
          ` : ''}
        </div>
      `).join('')}
    </div>

    <div class="chart-container">
      <div class="chart-header">
        <h3 class="chart-title">Management Recommendations</h3>
      </div>
      <div class="alert-recommendations">
        ${recommendations.map(rec => `
          <div class="recommendation-card">
            <h4 class="recommendation-title">${rec.title}</h4>
            <p class="recommendation-description">${rec.description}</p>
            <div class="recommendation-meta">
              <span class="recommendation-meta-text">Priority: ${rec.priority} | Estimated Impact: ${rec.impact}</span>
            </div>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

// Global alert action handlers
window.handleAlertAction = function(alertId, action, level) {
  // Show action confirmation
  const actionModal = document.createElement('div');
  actionModal.className = 'action-modal';
  actionModal.innerHTML = `
    <div class="action-modal-content">
      <h3>Action Confirmation</h3>
      <p>Confirm action: <strong>${action}</strong></p>
      <p>This will mark the alert as handled and may trigger automated processes.</p>
      <div class="action-modal-buttons">
        <button class="alert-action-btn" onclick="confirmAlertAction('${alertId}', '${action}')">Confirm</button>
        <button class="alert-action-btn secondary" onclick="closeActionModal()">Cancel</button>
      </div>
    </div>
  `;

  document.body.appendChild(actionModal);
  requestAnimationFrame(() => {
    actionModal.classList.add('show');
  });
};

window.handleAlertDismiss = function(alertId, level) {
  // Find the alert card
  const alertCard = document.querySelector(`[data-alert-id="${alertId}"]`);
  const alertTitle = alertCard?.querySelector('.alert-title')?.textContent || '';

  // Prevent dismissal of weather alerts
  if (alertTitle.includes('Weather') || alertTitle.includes('Daily Weather')) {
    showNotificationToast('Weather alerts cannot be dismissed', 'warning');
    return;
  }

  // Allow dismissal of risk alerts only
  if (alertCard) {
    alertCard.style.opacity = '0.5';
    alertCard.style.transform = 'scale(0.95)';
  }

  // Dismiss from notification system (only risk alerts)
  dismissNotification('risk', alertId);

  // Remove card after animation
  setTimeout(() => {
    if (alertCard) {
      alertCard.remove();
    }
  }, 300);

  // Show dismissal feedback
  showNotificationToast('Risk alert dismissed', 'info');
};

window.confirmAlertAction = function(alertId, action) {
  const alertCard = document.querySelector(`[data-alert-id="${alertId}"]`);
  const alertTitle = alertCard?.querySelector('.alert-title')?.textContent || '';

  // Handle weather alerts differently - they cannot be completed
  if (alertTitle.includes('Weather') || alertTitle.includes('Daily Weather')) {
    closeActionModal();
    showNotificationToast('Weather alert acknowledged', 'info');
    return;
  }

  // Handle risk alerts - mark as completed
  if (alertCard) {
    alertCard.style.background = 'rgba(103, 212, 167, 0.1)';
    alertCard.style.borderColor = 'var(--accent)';
  }

  // Track as completed and remove from notifications
  if (!notifications.completedAlerts) notifications.completedAlerts = [];
  notifications.completedAlerts.push(alertId);
  removeNotification('risk', alertId);

  closeActionModal();

  // Show success feedback
  showNotificationToast(`Action executed: ${action}`, 'success');

  // Remove card after showing success
  setTimeout(() => {
    if (alertCard) {
      alertCard.remove();
    }
  }, 2000);
};

window.closeActionModal = function() {
  const modal = document.querySelector('.action-modal');
  if (modal) {
    modal.classList.remove('show');
    setTimeout(() => modal.remove(), 300);
  }
};

// Notification function removed - use showNotification from map.js

async function load() {
  const container = document.getElementById("sightings-container");
  const empty = document.getElementById("sightings-empty");

  container.innerHTML = "";

  // Initialize notification system
  initializeNotifications();

  try {
    const { data } = await SightingsAPI.list("", false); // Exclude removed sightings from main view

    if (!data || data.length === 0) {
      empty.style.display = "block";
      return;
    }

    empty.style.display = "none";
    window.sightingsData = data; // Store for global access
    updateDailyActivityAlert(data);

    // Separate invasive and non-invasive species
    const invasiveSightings = data.filter(s => {
      const species = s.analysis?.predictedSpecies || 'Unknown';
      return species !== 'Unknown' && species !== 'Unknown species' && !species.includes('Unknown');
    }).sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

    const nonInvasiveSightings = data.filter(s => {
      const species = s.analysis?.predictedSpecies || 'Unknown';
      return species === 'Unknown' || species === 'Unknown species' || species.includes('Unknown');
    }).sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

    // Add invasive species section
    if (invasiveSightings.length > 0) {
      const invasiveHeader = document.createElement("div");
      invasiveHeader.className = "section-header";
      invasiveHeader.innerHTML = `
        <h2 class="invasive-species-header">
          Invasive Species Detected (${invasiveSightings.length})
        </h2>
      `;
      container.appendChild(invasiveHeader);
    }

    // Render invasive species cards
    for (const sighting of invasiveSightings) {
      const card = document.createElement("div");
      card.className = "sighting-card invasive-sighting";
      card.setAttribute('data-sighting-id', sighting._id);
      card.style.border = "2px solid #ef4444";
      card.style.background = "rgba(239, 68, 68, 0.05)";

      // Get risk level from LLM analysis
      const riskLevel = sighting.analysis?.llm?.details?.risk_level || 'Medium';
      const riskColor = riskLevel.toLowerCase().includes('high') ? '#dc2626' :
                       riskLevel.toLowerCase().includes('medium') ? '#d97706' : '#10b981';

      card.innerHTML = `
        <button class="remove-btn" onclick="removeSighting('${sighting._id}', '${sighting.analysis?.predictedSpecies || 'Unknown Species'}')">×</button>
        <div class="sighting-header">
${(sighting.imagePath || sighting.imageUrl) && typeof (sighting.imagePath || sighting.imageUrl) === 'string' && ((sighting.imagePath || sighting.imageUrl).startsWith('/') || (sighting.imagePath || sighting.imageUrl).startsWith('http')) ?
            `<img src="${sighting.imagePath || sighting.imageUrl}" alt="Plant" class="sighting-thumbnail" onclick="showImageModal('${sighting.imagePath || sighting.imageUrl}')" onerror="this.style.display='none'" />` :
            `<div class="no-image-placeholder sighting-thumbnail"></div>`}

          <div class="sighting-info">
            <h3 class="sighting-species">
              <span class="species-label invasive-label">[INVASIVE]</span>
              ${sighting.analysis?.predictedSpecies || 'Unknown Species'}
            </h3>

            <div class="sighting-meta">
              <span>Date: ${fmtDate(sighting.capturedAt || sighting.createdAt)}</span>
              ${sighting.location?.coordinates ? `<span>Location: ${fmtLatLng(sighting.location.coordinates)}</span>` : ''}
            </div>

            <div class="sighting-badges">
              <span class="sighting-badge confidence">${pct(sighting.analysis?.confidence)} confidence</span>
              <span class="sighting-badge source">${sighting.fromVideo ? 'Live Video' : 'Upload'}</span>
              <span class="sighting-badge risk-level" style="background-color: ${riskColor}; color: var(--background);">
                Risk: ${riskLevel}
              </span>
            </div>
          </div>
        </div>

        ${createSightingLLMDropdown(sighting)}
      `;

      container.appendChild(card);
    }

    // Add non-invasive species section
    if (nonInvasiveSightings.length > 0) {
      const nonInvasiveHeader = document.createElement("div");
      nonInvasiveHeader.className = "section-header non-invasive-header";
      nonInvasiveHeader.innerHTML = `
        <h2 class="non-invasive-species-header">
          Non-Invasive Species (${nonInvasiveSightings.length})
        </h2>
      `;
      container.appendChild(nonInvasiveHeader);

      // Render non-invasive species cards (smaller format)
      for (const sighting of nonInvasiveSightings) {
        const card = document.createElement("div");
        card.className = "sighting-card non-invasive-sighting";
        card.setAttribute('data-sighting-id', sighting._id);
        card.style.border = "1px solid #10b981";
        card.style.background = "rgba(16, 185, 129, 0.03)";
        card.style.transform = "scale(0.9)";
        card.style.margin = "0.5rem 0";

        card.innerHTML = `
          <button class="remove-btn" onclick="removeSighting('${sighting._id}', '${sighting.analysis?.predictedSpecies || 'Unknown Species'}')">×</button>
          <div class="sighting-header">
${(sighting.imagePath || sighting.imageUrl) && typeof (sighting.imagePath || sighting.imageUrl) === 'string' && ((sighting.imagePath || sighting.imageUrl).startsWith('/') || (sighting.imagePath || sighting.imageUrl).startsWith('http')) ?
            `<img src="${sighting.imagePath || sighting.imageUrl}" alt="Plant" class="sighting-thumbnail" onclick="showImageModal('${sighting.imagePath || sighting.imageUrl}')" onerror="this.style.display='none'" />` :
            `<div class="no-image-placeholder sighting-thumbnail"></div>`}

            <div class="sighting-info">
              <h3 class="sighting-species">
                <span class="species-label non-invasive-label">[NON-INVASIVE]</span>
                ${sighting.analysis?.predictedSpecies || 'Unknown Species'}
              </h3>

              <div class="sighting-meta">
                <span>Date: ${fmtDate(sighting.capturedAt || sighting.createdAt)}</span>
                ${sighting.location?.coordinates ? `<span>Location: ${fmtLatLng(sighting.location.coordinates)}</span>` : ''}
              </div>

              <div class="sighting-badges">
                <span class="sighting-badge confidence">${pct(sighting.analysis?.confidence)} confidence</span>
                <span class="sighting-badge source">Upload</span>
              </div>
            </div>
          </div>

          ${createSightingLLMDropdown(sighting)}
        `;

        container.appendChild(card);
      }
    }
  } catch (e) {
    console.error("Failed to load sightings", e);
    empty.textContent = "Failed to load sightings.";
    empty.style.display = "block";
  }
}

// === ANALYSIS HELPER FUNCTIONS ===

// Invasive Species Analytics Generator
function renderPeriodSpecificData(item, period) {
  // Calculate actual data-driven insights based on the sightings in this time period
  const sightings = item.sightings || [];
  const invasiveSightings = sightings.filter(s => isInvasiveSpeciesFromAnalysis(s));
  const removedSightings = sightings.filter(s => s.isRemoved);

  switch (period) {
    case 'weekly':
      const avgConfidence = sightings.length > 0 ?
        (sightings.reduce((sum, s) => sum + (s.analysis?.confidence || 0), 0) / sightings.length * 100).toFixed(1) : 0;
      const invasivePercentage = sightings.length > 0 ?
        (invasiveSightings.length / sightings.length * 100).toFixed(1) : 0;

      return `
        <div class="weekly-insights">
          <div class="insight-detail-text"><strong>Detection Rate:</strong> ${sightings.length} sightings (${avgConfidence}% avg confidence)</div>
          <div class="insight-detail-text"><strong>Invasive Rate:</strong> ${invasivePercentage}% of detections this week</div>
        </div>
      `;
    case 'monthly':
      const uniqueSpecies = [...new Set(sightings.map(s => s.analysis?.predictedSpecies).filter(Boolean))];
      const removalEffectiveness = sightings.length > 0 ?
        (removedSightings.length / sightings.length * 100).toFixed(1) : 0;
      const diversityIndex = uniqueSpecies.length;

      return `
        <div class="monthly-insights">
          <div class="insight-detail-text"><strong>Species Diversity:</strong> ${diversityIndex} distinct species detected</div>
          <div class="insight-detail-text"><strong>Management Success:</strong> ${removalEffectiveness}% removal rate</div>
          <div class="insight-detail-text"><strong>Risk Assessment:</strong> ${invasiveSightings.length} invasive detections</div>
        </div>
      `;
    case 'yearly':
      const monthSpread = [...new Set(sightings.map(s => new Date(s.createdAt).getMonth()))].length;
      const locationSpread = sightings.filter(s => s.location?.coordinates).length;
      const trendDirection = sightings.length > 12 ?
        (sightings.slice(0, 6).length > sightings.slice(-6).length ? 'decreasing' : 'increasing') : 'stable';

      return `
        <div class="yearly-insights">
          <div class="insight-detail-text"><strong>Seasonal Activity:</strong> Active in ${monthSpread}/12 months</div>
          <div class="insight-detail-text"><strong>Geographic Spread:</strong> ${locationSpread} tracked locations</div>
          <div class="insight-detail-text"><strong>Annual Trend:</strong> ${trendDirection} detection pattern</div>
        </div>
      `;
    default:
      return '';
  }
}

// Helper function to determine if a species is invasive based on analysis
function isInvasiveSpeciesFromAnalysis(sighting) {
  return sighting.analysis?.llm?.details?.invasive_status ||
         sighting.analysis?.llm?.details?.risk_level?.toLowerCase().includes('high') ||
         sighting.analysis?.llm?.details?.risk_level?.toLowerCase().includes('severe');
}

// Invasive Species Analysis
function generateDetectionTrendsChart(sightings, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  // Group sightings by week for the last 8 weeks
  const now = new Date();
  const weeklyData = {};
  
  for (let i = 7; i >= 0; i--) {
    const weekStart = new Date(now.getTime() - (i * 7 * 24 * 60 * 60 * 1000));
    const weekEnd = new Date(weekStart.getTime() + (7 * 24 * 60 * 60 * 1000));
    const weekKey = `Week ${8-i}`;
    
    const weekSightings = sightings.filter(s => {
      const sightingDate = new Date(s.createdAt);
      return sightingDate >= weekStart && sightingDate < weekEnd;
    });
    
    weeklyData[weekKey] = weekSightings.length;
  }
  
  const data = Object.entries(weeklyData).map(([week, count]) => ({
    label: week,
    value: count,
    color: count > 5 ? '#ef4444' : count > 2 ? '#f59e0b' : '#10b981'
  }));
  
  generateSimpleBarChart(data, container);
}

function generateRiskBreakdownChart(sightings, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const riskCounts = { high: 0, medium: 0, low: 0, unknown: 0 };

  sightings.forEach(s => {
    const risk = s.llm?.details?.risk_level?.toLowerCase() || 'unknown';
    if (risk.includes('high') || risk.includes('severe')) riskCounts.high++;
    else if (risk.includes('medium') || risk.includes('moderate')) riskCounts.medium++;
    else if (risk.includes('low') || risk.includes('minimal')) riskCounts.low++;
    else riskCounts.unknown++;
  });

  const data = [
    { label: 'High', value: riskCounts.high, color: '#ef4444' },
    { label: 'Medium', value: riskCounts.medium, color: '#f59e0b' },
    { label: 'Low', value: riskCounts.low, color: '#10b981' },
    { label: 'Unknown', value: riskCounts.unknown, color: '#6b7280' }
  ];

  generateSimpleBarChart(data, container);
}

// === NEW MEANINGFUL GEOGRAPHIC CHARTS ===

function generateDensityDistributionChart(sightings, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  // Calculate species density per km²
  const speciesData = {};
  sightings.forEach(s => {
    const species = s.analysis?.predictedSpecies || 'Unknown';
    if (!speciesData[species]) speciesData[species] = 0;
    speciesData[species]++;
  });

  const data = Object.entries(speciesData)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 8)
    .map(([species, count]) => ({
      label: species.substring(0, 12) + (species.length > 12 ? '...' : ''),
      value: count,
      color: '#67d4a7'
    }));

  generateSimpleBarChart(data, container);
}

function generateCoverageAnalysisChart(coverageStats, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const data = [
    { label: 'Total Area', value: Math.round(coverageStats.totalArea), color: '#3b82f6' },
    { label: 'Avg Distance', value: Math.round(coverageStats.avgDistance), color: '#10b981' },
    { label: 'Coverage Score', value: Math.round(coverageStats.totalArea / Math.max(coverageStats.avgDistance, 1)), color: '#f59e0b' }
  ];

  generateSimpleBarChart(data, container);
}

function generateDistanceDistributionChart(sightings, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const validSightings = sightings.filter(s => s.location?.coordinates);
  const distances = [];

  for (let i = 0; i < validSightings.length; i++) {
    for (let j = i + 1; j < validSightings.length && j < i + 5; j++) { // Limit comparisons
      const dist = calculateDistance(
        validSightings[i].location.coordinates,
        validSightings[j].location.coordinates
      );
      distances.push(Math.round(dist));
    }
  }

  // Group into distance ranges
  const ranges = { '0-1km': 0, '1-5km': 0, '5-10km': 0, '10-20km': 0, '20km+': 0 };
  distances.forEach(dist => {
    if (dist < 1) ranges['0-1km']++;
    else if (dist < 5) ranges['1-5km']++;
    else if (dist < 10) ranges['5-10km']++;
    else if (dist < 20) ranges['10-20km']++;
    else ranges['20km+']++;
  });

  const data = Object.entries(ranges).map(([range, count]) => ({
    label: range,
    value: count,
    color: '#67d4a7'
  }));

  generateSimpleBarChart(data, container);
}

function generateClusterSizeChart(locationClusters, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const data = locationClusters
    .slice(0, 8)
    .map((cluster, index) => ({
      label: `Cluster ${index + 1}`,
      value: cluster.count,
      color: cluster.count > 5 ? '#ef4444' : cluster.count > 3 ? '#f59e0b' : '#10b981'
    }));

  generateSimpleBarChart(data, container);
}

function generateSpeciesRangeChart(sightings, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const speciesRanges = {};
  const speciesSightings = {};

  // Group sightings by species
  sightings.forEach(s => {
    const species = s.analysis?.predictedSpecies || 'Unknown';
    if (!speciesSightings[species]) speciesSightings[species] = [];
    if (s.location?.coordinates) {
      speciesSightings[species].push(s.location.coordinates);
    }
  });

  // Calculate range for each species
  Object.entries(speciesSightings).forEach(([species, locations]) => {
    if (locations.length < 2) {
      speciesRanges[species] = 0;
      return;
    }

    let maxDistance = 0;
    for (let i = 0; i < locations.length; i++) {
      for (let j = i + 1; j < locations.length; j++) {
        const dist = calculateDistance(locations[i], locations[j]);
        maxDistance = Math.max(maxDistance, dist);
      }
    }
    speciesRanges[species] = Math.round(maxDistance);
  });

  const data = Object.entries(speciesRanges)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 8)
    .map(([species, range]) => ({
      label: species.substring(0, 12) + (species.length > 12 ? '...' : ''),
      value: range,
      color: range > 10 ? '#ef4444' : range > 5 ? '#f59e0b' : '#10b981'
    }));

  generateSimpleBarChart(data, container);
}

function generateTimelineChart(timePatterns, containerId, period = 'daily') {
  const container = document.getElementById(containerId);
  if (!container) return;

  let data = [];

  switch (period) {
    case 'daily':
      const days = Object.keys(timePatterns.dayCounts).slice(-7);
      data = days.map(day => ({
        label: new Date(day).toLocaleDateString('en', { weekday: 'short' }),
        value: timePatterns.dayCounts[day] || 0,
        color: '#67d4a7'
      }));
      break;

    case 'weekly':
      const weeks = Object.keys(timePatterns.weekCounts || {}).slice(-6);
      data = weeks.map(week => ({
        label: `Week ${new Date(week).getMonth() + 1}/${new Date(week).getDate()}`,
        value: timePatterns.weekCounts[week] || 0,
        color: '#67d4a7'
      }));
      break;

    case 'monthly':
      const months = Object.keys(timePatterns.monthCounts || {}).slice(-6);
      data = months.map(month => ({
        label: new Date(month + '-01').toLocaleDateString('en', { month: 'short' }),
        value: timePatterns.monthCounts[month] || 0,
        color: '#67d4a7'
      }));
      break;

    case 'yearly':
      const years = Object.keys(timePatterns.yearCounts || {}).slice(-3);
      data = years.map(year => ({
        label: year,
        value: timePatterns.yearCounts[year] || 0,
        color: '#67d4a7'
      }));
      break;
  }

  // Create a more compact chart
  container.style.height = '200px';
  generateSimpleBarChart(data, container);
}

function generateHourlyChart(timePatterns, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const data = timePatterns.hourCounts.map((count, hour) => ({
    label: `${hour}:00`,
    value: count,
    color: hour === timePatterns.peakHour ? '#ef4444' : '#67d4a7'
  })).filter((_, hour) => hour % 3 === 0); // Show every 3rd hour

  generateSimpleBarChart(data, container);
}

function generateSeasonalChart(seasonalData, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const data = Object.keys(seasonalData.seasonCounts).map(season => ({
    label: season,
    value: seasonalData.seasonCounts[season],
    color: season === seasonalData.dominantSeason ? '#ef4444' : '#67d4a7'
  }));

  generateSimpleBarChart(data, container);
}

function generateInvasiveTimelineEvents(sightings, period = 'daily') {
  // Filter invasive species for farmer focus
  const invasiveSightings = sightings.filter(s => {
    const species = s.analysis?.predictedSpecies || 'Unknown';
    return species !== 'Unknown' && species !== 'Unknown species' && !species.includes('Unknown');
  });

  const sortedSightings = invasiveSightings.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

  let groupedData = [];
  let limit = 10;

  switch (period) {
    case 'daily':
      // Focus on immediate threats for farmers
      groupedData = sortedSightings.slice(0, 12);
      return groupedData.map((sighting, index) => {
        const riskLevel = sighting.analysis?.llm?.details?.risk_level || 'Medium';
        const threatUrgency = analytics.getThreatUrgency(sighting);

        return `
          <div class="timeline-item farmer-threat timeline-threat-item">
            <div class="timeline-marker timeline-threat-marker" style="background: ${analytics.getThreatColor(riskLevel)}; box-shadow-color: ${analytics.getThreatColor(riskLevel)};"></div>
            <div class="timeline-content timeline-threat-content" style="background: linear-gradient(135deg, ${analytics.getThreatColorAlpha(riskLevel, 0.1)} 0%, ${analytics.getThreatColorAlpha(riskLevel, 0.02)} 100%); border-left-color: ${analytics.getThreatColor(riskLevel)};">
              <div class="timeline-header timeline-threat-header">
                <div class="timeline-date timeline-threat-date">${fmtDate(sighting.createdAt)}</div>
                <div class="threat-badge" style="background: ${analytics.getThreatColor(riskLevel)};">${riskLevel.toUpperCase()} RISK</div>
              </div>
              <div class="timeline-title timeline-threat-title">${sighting.analysis?.predictedSpecies || 'Unknown Species'}</div>
              <div class="farmer-action">
                ⚠ ${threatUrgency}
              </div>
              <div class="timeline-details timeline-threat-details">
                ${((sighting.analysis?.confidence || 0) * 100).toFixed(1)}% confidence
                ${sighting.location?.coordinates ? ` • ${fmtLatLng(sighting.location.coordinates)}` : ''}
              </div>
            </div>
          </div>
        `;
      }).join('');

    case 'weekly':
      // Weekly spread analysis for tactical planning
      const weeklyGroups = {};
      sortedSightings.forEach(sighting => {
        const weekStart = new Date(sighting.createdAt);
        weekStart.setDate(weekStart.getDate() - weekStart.getDay());
        const weekKey = weekStart.toISOString().split('T')[0];
        if (!weeklyGroups[weekKey]) weeklyGroups[weekKey] = [];
        weeklyGroups[weekKey].push(sighting);
      });

      groupedData = Object.entries(weeklyGroups).slice(0, 6).map(([week, sightings]) => ({
        period: `Week of ${new Date(week).toLocaleDateString()}`,
        count: sightings.length,
        species: [...new Set(sightings.map(s => s.analysis?.predictedSpecies || 'Unknown'))],
        avgRisk: analytics.calculateAvgRisk(sightings),
        spreadRate: analytics.calculateWeeklySpread(sightings),
        actionItems: analytics.generateWeeklyActions(sightings),
        sightings: sightings, // Add actual sightings data
        type: 'weekly'
      }));
      break;

    case 'monthly':
      // Monthly strategic overview
      const monthlyGroups = {};
      sortedSightings.forEach(sighting => {
        const month = new Date(sighting.createdAt).toISOString().slice(0, 7);
        if (!monthlyGroups[month]) monthlyGroups[month] = [];
        monthlyGroups[month].push(sighting);
      });

      groupedData = Object.entries(monthlyGroups).slice(0, 4).map(([month, sightings]) => ({
        period: new Date(month + '-01').toLocaleDateString('en-US', { month: 'long', year: 'numeric' }),
        count: sightings.length,
        species: [...new Set(sightings.map(s => s.analysis?.predictedSpecies || 'Unknown'))],
        environmentalSeverity: analytics.calculateEnvironmentalSeverity(sightings),
        controlEffectiveness: analytics.calculateControlEffectiveness(sightings),
        seasonalFactors: analytics.getSeasonalFactors(month),
        sightings: sightings, // Add actual sightings data
        type: 'monthly'
      }));
      break;

    case 'yearly':
      // Annual cycles and long-term trends
      const yearlyGroups = {};
      sortedSightings.forEach(sighting => {
        const year = new Date(sighting.createdAt).getFullYear().toString();
        if (!yearlyGroups[year]) yearlyGroups[year] = [];
        yearlyGroups[year].push(sighting);
      });

      groupedData = Object.entries(yearlyGroups).slice(0, 3).map(([year, sightings]) => ({
        period: year,
        count: sightings.length,
        species: [...new Set(sightings.map(s => s.analysis?.predictedSpecies || 'Unknown'))],
        climateCorrelation: analytics.getClimateCorrelation(sightings),
        predictiveInsights: analytics.generatePredictiveInsights(sightings),
        ecosystemRecovery: analytics.calculateEcosystemRecovery(sightings),
        sightings: sightings, // Add actual sightings data
        type: 'yearly'
      }));
      break;
  }

  // Return formatted grouped data for weekly/monthly/yearly
  return groupedData.map((item, index) => {
    const riskColor = item.avgRisk ? analytics.getThreatColor(item.avgRisk) : '#67d4a7';

    return `
      <div class="timeline-item farmer-strategic timeline-strategic-item">
        <div class="timeline-marker timeline-strategic-marker" style="background: ${riskColor};"></div>
        <div class="timeline-content timeline-strategic-content" style="border-left-color: ${riskColor};">
          <div class="timeline-header timeline-strategic-header">
            <div class="timeline-date timeline-strategic-date">${item.period}</div>
            <div class="timeline-count" style="background: ${riskColor};">${item.count} Invasive${item.count !== 1 ? 's' : ''}</div>
          </div>

          ${renderPeriodSpecificData(item, period)}

          <div class="timeline-species timeline-strategic-species">
            <strong>Top Species:</strong> ${item.species.slice(0, 3).join(', ')}${item.species.length > 3 ? ` +${item.species.length - 3} more` : ''}
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function generateTimelineEvents(sightings, period = 'daily') {
  const sortedSightings = sightings.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

  let groupedData = [];
  let limit = 10;

  switch (period) {
    case 'daily':
      // Show recent daily detections
      groupedData = sortedSightings.slice(0, 15);
      break;
    case 'weekly':
      // Group by week, show weekly summaries
      const weeklyGroups = {};
      sortedSightings.forEach(sighting => {
        const weekStart = new Date(sighting.createdAt);
        weekStart.setDate(weekStart.getDate() - weekStart.getDay());
        const weekKey = weekStart.toISOString().split('T')[0];
        if (!weeklyGroups[weekKey]) weeklyGroups[weekKey] = [];
        weeklyGroups[weekKey].push(sighting);
      });
      groupedData = Object.entries(weeklyGroups).slice(0, 8).map(([week, sightings]) => ({
        period: `Week of ${new Date(week).toLocaleDateString()}`,
        count: sightings.length,
        species: [...new Set(sightings.map(s => s.analysis?.predictedSpecies || 'Unknown'))].slice(0, 3).join(', '),
        type: 'weekly'
      }));
      break;
    case 'monthly':
      // Group by month
      const monthlyGroups = {};
      sortedSightings.forEach(sighting => {
        const month = new Date(sighting.createdAt).toISOString().slice(0, 7);
        if (!monthlyGroups[month]) monthlyGroups[month] = [];
        monthlyGroups[month].push(sighting);
      });
      groupedData = Object.entries(monthlyGroups).slice(0, 6).map(([month, sightings]) => ({
        period: new Date(month + '-01').toLocaleDateString('en-US', { month: 'long', year: 'numeric' }),
        count: sightings.length,
        species: [...new Set(sightings.map(s => s.analysis?.predictedSpecies || 'Unknown'))].slice(0, 2).join(', '),
        type: 'monthly'
      }));
      break;
    case 'yearly':
      // Group by year
      const yearlyGroups = {};
      sortedSightings.forEach(sighting => {
        const year = new Date(sighting.createdAt).getFullYear();
        if (!yearlyGroups[year]) yearlyGroups[year] = [];
        yearlyGroups[year].push(sighting);
      });
      groupedData = Object.entries(yearlyGroups).slice(0, 5).map(([year, sightings]) => ({
        period: year,
        count: sightings.length,
        species: `${[...new Set(sightings.map(s => s.analysis?.predictedSpecies || 'Unknown'))].length} unique species`,
        type: 'yearly'
      }));
      break;
  }

  if (period === 'daily') {
    return groupedData.map((sighting, index) => `
      <div class="timeline-item timeline-simple-item">
        <div class="timeline-marker timeline-simple-marker"></div>
        <div class="timeline-content timeline-simple-content">
          <div class="timeline-header timeline-simple-header">
            <div class="timeline-date timeline-simple-date">${fmtDate(sighting.createdAt)}</div>
            <div class="timeline-confidence">${((sighting.analysis?.confidence || 0) * 100).toFixed(1)}%</div>
          </div>
          <div class="timeline-title timeline-simple-title">${sighting.analysis?.predictedSpecies || 'Unknown Species'}</div>
          ${sighting.location?.coordinates ? `<div class="timeline-location">${fmtLatLng(sighting.location.coordinates)}</div>` : ''}
        </div>
      </div>
    `).join('');
  } else {
    return groupedData.map((item, index) => `
      <div class="timeline-item timeline-summary-item">
        <div class="timeline-marker timeline-summary-marker"></div>
        <div class="timeline-content timeline-summary-content">
          <div class="timeline-header timeline-summary-header">
            <div class="timeline-date timeline-summary-date">${item.period}</div>
            <div class="timeline-count timeline-summary-count">${item.count} Detection${item.count !== 1 ? 's' : ''}</div>
          </div>
          <div class="timeline-species timeline-summary-species">
            <strong>Species:</strong> ${item.species || 'Various species detected'}
          </div>
        </div>
      </div>
    `).join('');
  }
}

function generateSimpleBarChart(data, container) {
  container.innerHTML = '';

  const maxValue = Math.max(...data.map(d => d.value));
  if (maxValue === 0) {
    container.innerHTML = '<div class="no-data-display">No data available</div>';
    return;
  }

  data.forEach(item => {
    const bar = document.createElement('div');
    bar.className = 'chart-bar';
    bar.style.height = `${(item.value / maxValue) * 200}px`;
    bar.style.backgroundColor = item.color;

    const label = document.createElement('div');
    label.className = 'chart-bar-label';
    label.textContent = item.label;

    const value = document.createElement('div');
    value.className = 'chart-bar-value';
    value.textContent = item.value;

    bar.appendChild(label);
    bar.appendChild(value);
    container.appendChild(bar);
  });
}

// Global chart control functions - removed duplicate definition

// Removal Functionality
window.removeSighting = function(sightingId, speciesName) {
  // Show confirmation modal
  const modal = document.createElement('div');
  modal.className = 'removal-modal';
  modal.innerHTML = `
    <div class="removal-modal-content">
      <h3>Confirm Removal</h3>
      <p>Are you sure you want to mark this <strong>${speciesName}</strong> sighting as removed?</p>
      <p><em>This indicates the plant has been physically removed from the location.</em></p>
      <div class="removal-modal-buttons">
        <button class="btn-confirm" onclick="confirmRemoval('${sightingId}')">Mark as Removed</button>
        <button class="btn-cancel" onclick="closeRemovalModal()">Cancel</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  window.currentRemovalModal = modal;
};

window.confirmRemoval = async function(sightingId) {
  try {
    const response = await fetch(`/api/sightings/${sightingId}/remove`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        removedAt: new Date().toISOString(),
        removedBy: 'user' // This could be enhanced with actual user info
      }),
      credentials: 'include' // Include cookies for authentication
    });

    // Handle both successful responses and redirects that indicate success
    if (response.ok || response.status === 200) {
      // Remove the card from DOM with smooth animation
      const card = document.querySelector(`[data-sighting-id="${sightingId}"]`);
      if (card) {
        card.style.transition = 'all 0.3s ease';
        card.style.opacity = '0';
        card.style.transform = 'translateX(-100%)';
        setTimeout(() => card.remove(), 300);
      }

      // Show success message
      showNotification('Plant removal recorded successfully!', 'success');

      // Refresh analytics immediately to show updated environmental impact
      setTimeout(() => {
        // Save current scroll position before reload
        const scrollPosition = window.pageYOffset || document.documentElement.scrollTop;

        load().then(() => {
          // Restore scroll position after reload
          setTimeout(() => {
            window.scrollTo(0, scrollPosition);
          }, 100); // Small delay to ensure content is loaded
        }).catch(() => {
          // If load() doesn't return a promise, use alternative approach
          setTimeout(() => {
            window.scrollTo(0, scrollPosition);
          }, 1000);
        });
      }, 500);

    } else {
      // Try to parse response to get more details
      let errorMessage = 'Failed to record removal';
      try {
        const errorData = await response.json();
        errorMessage = errorData.error || errorMessage;
      } catch (e) {
        // If JSON parsing fails, use default message
      }

      // Check if it's an authentication issue
      if (response.status === 401 || response.status === 302) {
        errorMessage = 'Authentication required. Please refresh the page and try again.';
      }

      throw new Error(errorMessage);
    }
  } catch (error) {
    console.error('Error removing sighting:', error);

    // For network errors, still try to refresh to see if removal actually worked
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      showNotification('Network error, but removal may have succeeded. Refreshing...', 'warning');
      // Wait a moment then refresh to see if it worked
      setTimeout(() => {
        // Save current scroll position before reload
        const scrollPosition = window.pageYOffset || document.documentElement.scrollTop;

        load().then(() => {
          // Restore scroll position after reload
          setTimeout(() => {
            window.scrollTo(0, scrollPosition);
          }, 100);
        }).catch(() => {
          // If load() doesn't return a promise, use alternative approach
          setTimeout(() => {
            window.scrollTo(0, scrollPosition);
          }, 1000);
        });
      }, 1000);
    } else {
      showNotification(`Failed to record removal: ${error.message}`, 'error');
    }
  }

  closeRemovalModal();
};

window.closeRemovalModal = function() {
  if (window.currentRemovalModal) {
    window.currentRemovalModal.remove();
    window.currentRemovalModal = null;
  }
};

// Duplicate showNotification function removed - use the one from map.js

load();
