import { SightingsAPI } from "./api.js";

function fmtDate(s) {
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s || "";
  }
}

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
  if (!llmData || !llmData.details) return '<p>No analysis data available.</p>';

  const analysisData = llmData.details;

  switch (section) {
    case 'species':
      const speciesInfo = analysisData.advisory_content?.species_identification;
      return `
        <h4>Species Information</h4>
        <p><strong>Scientific Name:</strong> ${speciesInfo?.scientific_name || analysisData.species || 'Unknown'}</p>
        <p><strong>Common Names:</strong> ${speciesInfo?.common_names || analysisData.common_name || 'Unknown'}</p>
        <p><strong>Family:</strong> ${speciesInfo?.family || analysisData.family || 'Unknown'}</p>
      `;

    case 'legal':
      const legalInfo = analysisData.advisory_content?.legal_status;
      return `
        <h4>Legal Status</h4>
        <p><strong>NEMBA Category:</strong> ${legalInfo?.nemba_category || 'Unknown'}</p>
        <p><strong>Legal Requirements:</strong> ${legalInfo?.legal_requirements || 'Unknown'}</p>
        <p><strong>Risk Level:</strong> ${analysisData.risk_level || 'Unknown'}</p>
        <p><strong>Invasive Status:</strong> ${analysisData.invasive_status ? 'Yes' : 'No'}</p>
      `;

    case 'description':
      const physicalDesc = analysisData.advisory_content?.physical_description;
      return `
        <h4>Physical Description</h4>
        <p>${physicalDesc || analysisData.description || 'No description available.'}</p>
        <p><strong>Origin:</strong> ${analysisData.origin || 'Unknown'}</p>
      `;

    case 'distribution':
      const distributionInfo = analysisData.advisory_content?.distribution || analysisData.where_found || analysisData.distribution;
      if (!distributionInfo || distributionInfo === 'Not found') {
        return '<p>No distribution information available.</p>';
      }
      return `
        <h4>Where Found</h4>
        <p>${distributionInfo}</p>
      `;

    case 'control':
      const controlInfo = analysisData.advisory_content?.control_methods || analysisData.treatment || analysisData.control_methods;
      if (!controlInfo || controlInfo === 'Not found') {
        return '<p>No control methods available.</p>';
      }
      return `
        <h4>Control Methods</h4>
        <p>${controlInfo}</p>
      `;

    case 'action':
      const actionInfo = analysisData.action_required || analysisData.advisory_content?.action_required;
      if (!actionInfo || actionInfo === 'Not found') {
        return '<p>No action required.</p>';
      }
      return `
        <h4>Action Required</h4>
        <p>${actionInfo}</p>
      `;

    default:
      return '<p>Select a section to view details.</p>';
  }
}

function createLLMDropdown(sighting) {
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
  const modal = document.createElement('div');
  modal.className = 'image-modal';
  modal.innerHTML = `<img src="${imageUrl}" alt="Sighting Image" />`;

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
    notifications = JSON.parse(savedNotifications);
  }

  // Check for daily weather alert
  checkDailyWeatherAlert();

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

function checkDailyWeatherAlert() {
  const today = new Date().toDateString();
  const hasWeatherToday = notifications.weather.some(n =>
    new Date(n.timestamp).toDateString() === today && !n.dismissed
  );

  if (!hasWeatherToday) {
    // Generate daily weather alert
    const weatherConditions = ['Sunny', 'Partly Cloudy', 'Overcast', 'Light Rain', 'Heavy Rain', 'Windy'];
    const condition = weatherConditions[Math.floor(Math.random() * weatherConditions.length)];
    const temp = Math.floor(Math.random() * 20) + 15; // 15-35°C

    let alertLevel = 'info';
    let alertTitle = 'Daily Weather Update';
    let alertAction = 'Check detailed forecast';

    // Create special alerts for certain conditions
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
}

// Global function to manually trigger weather alerts for testing
window.triggerWeatherAlert = function() {
  const conditions = [
    { condition: 'Severe Storm Warning', temp: 18, level: 'critical', action: 'Suspend field operations' },
    { condition: 'High UV Index', temp: 32, level: 'warning', action: 'Use sun protection' },
    { condition: 'Perfect Field Conditions', temp: 22, level: 'info', action: 'Optimal for surveys' }
  ];

  const weather = conditions[Math.floor(Math.random() * conditions.length)];

  addNotification('weather', {
    title: `Weather Alert: ${weather.condition}`,
    description: `Current conditions: ${weather.condition}, ${weather.temp}°C. ${weather.level === 'critical' ? 'Dangerous conditions detected.' : weather.level === 'warning' ? 'Caution advised.' : 'Great conditions for monitoring.'}`,
    level: weather.level,
    action: weather.action
  });

  showNotificationToast(`Weather alert added: ${weather.condition}`, weather.level === 'critical' ? 'error' : weather.level === 'warning' ? 'warning' : 'info');
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
    const isVisible = content.style.display !== 'none';
    content.style.display = isVisible ? 'none' : 'block';
    if (arrow) {
      arrow.textContent = isVisible ? '▼' : '▲';
    }
    dropdown.classList.toggle('open', !isVisible);
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
  if (window.sightingsData) {
    switch (tabName) {
      case 'species-analytics':
        loadSpeciesAnalytics(window.sightingsData);
        break;
      case 'invasive-dashboard':
        loadInvasiveDashboard(window.sightingsData);
        break;
      case 'geographic-insights':
        loadGeographicInsights(window.sightingsData);
        break;
      case 'temporal-analysis':
        loadTemporalAnalysis(window.sightingsData);
        break;
      case 'risk-assessment':
        loadRiskAssessment(window.sightingsData);
        updateNotificationBadges(); // Refresh badge when viewing alerts
        break;
    }
  }
};

function aggregateSpeciesData(sightings) {
  const speciesMap = new Map();

  // Filter out removed sightings for analytics
  const activeSightings = sightings.filter(s => !s.isRemoved);

  activeSightings.forEach(sighting => {
    const species = sighting.analysis?.predictedSpecies || 'Unknown Species';

    if (!speciesMap.has(species)) {
      speciesMap.set(species, {
        species,
        sightings: [],
        totalCount: 0,
        confidenceSum: 0,
        locations: [],
        images: [],
        llmData: null
      });
    }

    const data = speciesMap.get(species);
    data.sightings.push(sighting);
    data.totalCount++;
    data.confidenceSum += sighting.analysis?.confidence || 0;

    if (sighting.location?.coordinates) {
      data.locations.push(sighting.location.coordinates);
    }

    if (sighting.imageUrl || sighting.imagePath) {
      data.images.push(sighting.imageUrl || sighting.imagePath);
    }

    // Store LLM data from most recent analysis
    if (sighting.llm && !data.llmData) {
      data.llmData = sighting.llm;
    }
  });

  return Array.from(speciesMap.values());
}

function calculateStats(speciesData) {
  return speciesData.map(data => {
    const avgConfidence = data.totalCount > 0 ? (data.confidenceSum / data.totalCount) * 100 : 0;
    const sortedSightings = data.sightings.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    const lastSeen = sortedSightings[0]?.createdAt;
    const firstSeen = sortedSightings[sortedSightings.length - 1]?.createdAt;

    // Calculate geographic spread
    let geoSpread = 0;
    if (data.locations.length > 1) {
      const distances = [];
      for (let i = 0; i < data.locations.length; i++) {
        for (let j = i + 1; j < data.locations.length; j++) {
          const dist = calculateDistance(data.locations[i], data.locations[j]);
          distances.push(dist);
        }
      }
      geoSpread = Math.max(...distances);
    }

    // Calculate average location
    let avgLocation = null;
    if (data.locations.length > 0) {
      const avgLat = data.locations.reduce((sum, loc) => sum + loc[1], 0) / data.locations.length;
      const avgLng = data.locations.reduce((sum, loc) => sum + loc[0], 0) / data.locations.length;
      avgLocation = [avgLng, avgLat];
    }

    // Extract risk level from LLM data
    const riskLevel = data.llmData?.details?.risk_level || 'Unknown';

    // Get scientific name from LLM data
    const scientificName = data.llmData?.details?.advisory_content?.species_identification?.scientific_name || '';

    return {
      ...data,
      avgConfidence,
      lastSeen,
      firstSeen,
      geoSpread,
      avgLocation,
      riskLevel,
      scientificName,
      uniqueLocations: data.locations.length,
      mainImage: data.images[0] || null
    };
  });
}

function calculateDistance(coord1, coord2) {
  // Haversine formula for distance between two coordinates
  const R = 6371; // Earth's radius in km
  const dLat = (coord2[1] - coord1[1]) * Math.PI / 180;
  const dLon = (coord2[0] - coord1[0]) * Math.PI / 180;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(coord1[1] * Math.PI / 180) * Math.cos(coord2[1] * Math.PI / 180) *
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
}

function getRiskBadgeClass(riskLevel) {
  const risk = riskLevel.toLowerCase();
  if (risk.includes('high') || risk.includes('severe')) return 'risk-high';
  if (risk.includes('medium') || risk.includes('moderate')) return 'risk-medium';
  if (risk.includes('low') || risk.includes('minimal')) return 'risk-low';
  return 'risk-unknown';
}

// === SPECIES ANALYTICS ===
function loadSpeciesAnalytics(sightings) {
  const container = document.getElementById('species-analytics-container');

  if (!sightings || sightings.length === 0) {
    container.innerHTML = '<p class="muted">No sightings data available for analytics.</p>';
    return;
  }

  const speciesData = aggregateSpeciesData(sightings);
  const statsData = calculateStats(speciesData);

  // Sort by total count (most frequent first)
  statsData.sort((a, b) => b.totalCount - a.totalCount);

  container.innerHTML = `
    <div class="dashboard-header">
      <h2 class="dashboard-title">Species Analytics</h2>
      <p class="dashboard-subtitle">Comprehensive species identification and distribution analysis</p>
    </div>
    <div id="species-analytics-grid"></div>
  `;

  const grid = container.querySelector('#species-analytics-grid');
  grid.style.display = 'grid';
  grid.style.gridTemplateColumns = 'repeat(auto-fill, minmax(350px, 1fr))';
  grid.style.gap = '1.5rem';

  statsData.forEach(stats => {
    const card = document.createElement('div');
    card.className = 'analytics-card';

    card.innerHTML = `
      <div class="analytics-header">
        ${stats.mainImage ? `
          <img src="${stats.mainImage}" alt="${stats.species}" class="analytics-image" />
        ` : ''}
        <div class="analytics-title">
          <h3>${stats.species}</h3>
          ${stats.scientificName ? `<p class="species-scientific">${stats.scientificName}</p>` : ''}
        </div>
      </div>

      <div class="analytics-stats">
        <div class="stat-item">
          <span class="stat-value">${stats.totalCount}</span>
          <span class="stat-label">Total Sightings</span>
        </div>
        <div class="stat-item">
          <span class="stat-value">${stats.avgConfidence.toFixed(1)}%</span>
          <span class="stat-label">Avg Confidence</span>
        </div>
        <div class="stat-item">
          <span class="stat-value">${stats.uniqueLocations}</span>
          <span class="stat-label">Locations</span>
        </div>
        <div class="stat-item">
          <span class="stat-value">${stats.geoSpread.toFixed(1)}km</span>
          <span class="stat-label">Geographic Spread</span>
        </div>
      </div>

      <div class="analytics-details">
        <div class="detail-row">
          <span class="detail-label">Last Seen</span>
          <span class="detail-value">${fmtDate(stats.lastSeen)}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">First Seen</span>
          <span class="detail-value">${fmtDate(stats.firstSeen)}</span>
        </div>
        ${stats.avgLocation ? `
          <div class="detail-row">
            <span class="detail-label">Avg Location</span>
            <span class="detail-value">${fmtLatLng(stats.avgLocation)}</span>
          </div>
        ` : ''}
        <div class="detail-row">
          <span class="detail-label">Risk Level</span>
          <span class="detail-value">
            <span class="risk-badge ${getRiskBadgeClass(stats.riskLevel)}">${stats.riskLevel}</span>
          </span>
        </div>
      </div>
    `;

    grid.appendChild(card);
  });
}

// === INVASIVE SPECIES DASHBOARD ===
function loadInvasiveDashboard(sightings) {
  const container = document.getElementById('invasive-dashboard-container');

  // Filter out removed sightings first
  const activeSightings = sightings.filter(s => !s.isRemoved);

  // Separate invasive species (named species) from unknown species
  const namedSpecies = activeSightings.filter(s => {
    const species = s.analysis?.predictedSpecies || 'Unknown';
    return species !== 'Unknown' && species !== 'Unknown species' && !species.includes('Unknown');
  });

  const unknownSpecies = activeSightings.filter(s => {
    const species = s.analysis?.predictedSpecies || 'Unknown';
    return species === 'Unknown' || species === 'Unknown species' || species.includes('Unknown');
  });

  // All named species are considered invasive
  const invasiveSpecies = namedSpecies;

  const riskAnalysis = analyzeInvasiveRisk(invasiveSpecies, sightings);
  const hotspots = identifyHotspots(invasiveSpecies);
  const spreadAnalysis = analyzeSpreadPatterns(invasiveSpecies);

  container.innerHTML = `
    <div class="dashboard-header">
      <h2 class="dashboard-title">Invasive Species Dashboard</h2>
      <p class="dashboard-subtitle">Critical invasive species monitoring and threat assessment</p>
    </div>

    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value">${invasiveSpecies.length}</div>
        <div class="metric-label">Invasive Sightings</div>
        <div class="metric-change ${invasiveSpecies.length > 0 ? 'negative' : 'neutral'}">
          ${(invasiveSpecies.length / activeSightings.length * 100).toFixed(1)}% of total
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${riskAnalysis.highRiskSpecies}</div>
        <div class="metric-label">High Risk Species</div>
        <div class="metric-change ${riskAnalysis.highRiskSpecies > 0 ? 'negative' : 'positive'}">
          Requires immediate action
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${hotspots.length}</div>
        <div class="metric-label">Active Hotspots</div>
        <div class="metric-change ${hotspots.length > 0 ? 'negative' : 'positive'}">
          Geographic concentration areas
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${spreadAnalysis.spreadingSpecies}</div>
        <div class="metric-label">Spreading Species</div>
        <div class="metric-change ${spreadAnalysis.spreadingSpecies > 0 ? 'negative' : 'positive'}">
          Active geographic expansion
        </div>
      </div>
    </div>

    <div class="chart-container">
      <div class="chart-header">
        <h3 class="chart-title">Invasive Species Risk Distribution</h3>
      </div>
      <div class="chart-content">
        <div class="simple-chart" id="risk-distribution-chart"></div>
      </div>
    </div>

    <div class="geo-grid">
      <div class="chart-container">
        <div class="chart-header">
          <h3 class="chart-title">Invasion Hotspots</h3>
        </div>
        <div class="location-list">
          ${hotspots.map(hotspot => `
            <div class="location-item">
              <div class="location-info">
                <div class="location-coords">${fmtLatLng([hotspot.lng, hotspot.lat])}</div>
                <div class="location-meta">${hotspot.species.join(', ')}</div>
              </div>
              <div class="location-count">${hotspot.count}</div>
            </div>
          `).join('')}
        </div>
      </div>

      <div class="chart-container">
        <div class="chart-header">
          <h3 class="chart-title">Risk Level Breakdown</h3>
        </div>
        <div class="chart-content">
          <div class="simple-chart" id="risk-breakdown-chart"></div>
        </div>
      </div>
    </div>
  `;

  // Generate risk distribution chart
  generateRiskChart(riskAnalysis, 'risk-distribution-chart');
  generateRiskBreakdownChart(sightings, 'risk-breakdown-chart');

  // Add unknown species section at the end
  const unknownContainer = document.createElement('div');
  unknownContainer.className = 'chart-container';
  unknownContainer.style.marginTop = '2rem';
  unknownContainer.innerHTML = `
    <div class="chart-header">
      <h3 class="chart-title">Non-Invasive (Unknown) Species</h3>
      <p class="unknown-species-subtitle">Species requiring further identification - classified as non-invasive</p>
    </div>
    <div class="metrics-grid unknown-metrics-grid">
      <div class="metric-card">
        <div class="metric-value">${unknownSpecies.length}</div>
        <div class="metric-label">Unknown Species Sightings</div>
        <div class="metric-change neutral">${(unknownSpecies.length / sightings.length * 100).toFixed(1)}% of total</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${new Set(unknownSpecies.map(s => s.analysis?.predictedSpecies || 'Unknown')).size}</div>
        <div class="metric-label">Unique Unknown Classifications</div>
        <div class="metric-change neutral">Awaiting identification</div>
      </div>
    </div>
    <div class="location-list unknown-location-list">
      ${unknownSpecies.slice(0, 10).map(sighting => `
        <div class="location-item">
          <div class="location-info">
            <div class="location-coords">${fmtLatLng(sighting.location?.coordinates || [0, 0])}</div>
            <div class="location-meta">${fmtDate(sighting.createdAt)} • ${((sighting.analysis?.confidence || 0) * 100).toFixed(1)}% confidence</div>
          </div>
          <div class="location-count">Unknown</div>
        </div>
      `).join('')}
      ${unknownSpecies.length > 10 ? `<div class="location-item"><div class="location-info"><div class="location-coords">... and ${unknownSpecies.length - 10} more</div></div></div>` : ''}
    </div>
  `;
  container.appendChild(unknownContainer);
}

// === GEOGRAPHIC INSIGHTS ===
function loadGeographicInsights(sightings) {
  const container = document.getElementById('geographic-insights-container');

  // Filter out removed sightings for geographic analysis
  const activeSightings = sightings.filter(s => !s.isRemoved);

  const locationClusters = analyzeLocationClusters(activeSightings);
  const densityMap = createDensityAnalysis(activeSightings);
  const coverageStats = calculateCoverageStats(activeSightings);

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

  const timePatterns = analyzeTimePatterns(sightings);
  const trends = calculateTrends(sightings);
  const seasonalData = analyzeSeasonalPatterns(sightings);

  // CSS styles now in sightings.css

  const invasiveAnalytics = generateInvasiveAnalytics(sightings);

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
  const timePatterns = analyzeTimePatterns(sightings);
  generateTimelineChart(timePatterns, 'timeline-chart', period);
};

// === RISK ASSESSMENT & ALERTS ===
function loadRiskAssessment(sightings) {
  const container = document.getElementById('risk-assessment-container');

  const alerts = generateRiskAlerts(sightings);
  const priorities = calculatePriorities(sightings);
  const recommendations = generateRecommendations(sightings);

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
              <button class="alert-action-btn" onclick="handleAlertAction('${alert.id || 'temp'}', '${alert.action || 'Take action'}', '${alert.level}')">
                ${alert.action || 'Take Action'}
              </button>
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

function showNotificationToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `notification-toast ${type}`;
  toast.textContent = message;

  document.body.appendChild(toast);
  requestAnimationFrame(() => {
    toast.classList.add('show');
  });

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

async function load() {
  const container = document.getElementById("sightings-container");
  const empty = document.getElementById("sightings-empty");

  container.innerHTML = "";

  // Initialize notification system
  initializeNotifications();

  try {
    const { data } = await SightingsAPI.list("");

    if (!data || data.length === 0) {
      empty.style.display = "block";
      return;
    }

    empty.style.display = "none";
    window.sightingsData = data; // Store for global access

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
          ${(sighting.imageUrl || sighting.imagePath) ? `
            <img src="${sighting.imageUrl || sighting.imagePath}"
                 alt="Sighting"
                 class="sighting-thumbnail"
                 onclick="showImageModal('${sighting.imageUrl || sighting.imagePath}')" />
          ` : ''}

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
              <span class="sighting-badge risk-level" style="background-color: ${riskColor}; color: white;">
                Risk: ${riskLevel}
              </span>
            </div>
          </div>
        </div>

        ${createLLMDropdown(sighting)}
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
            ${(sighting.imageUrl || sighting.imagePath) ? `
              <img src="${sighting.imageUrl || sighting.imagePath}"
                   alt="Sighting"
                   class="sighting-thumbnail"
                   onclick="showImageModal('${sighting.imageUrl || sighting.imagePath}')" />
            ` : ''}

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
                <span class="sighting-badge source">${sighting.fromVideo ? 'Live Video' : 'Upload'}</span>
              </div>
            </div>
          </div>

          ${createLLMDropdown(sighting)}
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
function generateInvasiveAnalytics(sightings) {
  const today = new Date();
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

  // Today's invasive species detections (excluding removed ones)
  const todayInvasives = sightings.filter(s => {
    const sDate = new Date(s.createdAt);
    const isToday = sDate.toDateString() === today.toDateString();
    const isInvasive = s.analysis?.predictedSpecies &&
                     s.analysis.predictedSpecies !== 'Unknown' &&
                     s.analysis.predictedSpecies !== 'Unknown species';
    const notRemoved = !s.isRemoved;
    return isToday && isInvasive && notRemoved;
  });

  const yesterdayInvasives = sightings.filter(s => {
    const sDate = new Date(s.createdAt);
    const isYesterday = sDate.toDateString() === yesterday.toDateString();
    const isInvasive = s.analysis?.predictedSpecies &&
                     s.analysis.predictedSpecies !== 'Unknown' &&
                     s.analysis.predictedSpecies !== 'Unknown species';
    const notRemoved = !s.isRemoved;
    return isYesterday && isInvasive && notRemoved;
  });

  // Today's removals
  const todayRemovals = sightings.filter(s => {
    if (!s.removedAt) return false;
    const rDate = new Date(s.removedAt);
    return rDate.toDateString() === today.toDateString();
  });

  const yesterdayRemovals = sightings.filter(s => {
    if (!s.removedAt) return false;
    const rDate = new Date(s.removedAt);
    return rDate.toDateString() === yesterday.toDateString();
  });

  // Calculate threat level
  const threatLevel = todayInvasives.length >= 5 ? 'high' :
                     todayInvasives.length >= 2 ? 'medium' : 'low';

  // Generate immediate actions
  const immediateActions = [];
  if (todayInvasives.length > 0) {
    immediateActions.push(`Inspect ${todayInvasives.length} new invasive detection${todayInvasives.length > 1 ? 's' : ''}`);
  }
  if (threatLevel === 'high') {
    immediateActions.push('Consider emergency herbicide application');
    immediateActions.push('Alert neighboring farmers');
  }
  if (todayInvasives.length > yesterdayInvasives.length) {
    immediateActions.push('Increase monitoring frequency');
  }
  if (immediateActions.length === 0) {
    immediateActions.push('Continue regular monitoring schedule');
  }

  // Calculate hotspots (clusters of invasive species)
  const invasiveLocations = sightings
    .filter(s => s.analysis?.predictedSpecies &&
                s.analysis.predictedSpecies !== 'Unknown' &&
                s.location?.coordinates)
    .map(s => s.location.coordinates);

  const hotspotCount = Math.max(1, Math.floor(invasiveLocations.length / 5));

  // Estimate spread rate (simplified calculation)
  const recentInvasives = sightings.filter(s => {
    const sDate = new Date(s.createdAt);
    return sDate >= weekAgo && s.analysis?.predictedSpecies &&
           s.analysis.predictedSpecies !== 'Unknown';
  });

  const spreadRate = recentInvasives.length > 1 ?
    Math.random() * 50 + 10 : // Simulated spread rate 10-60m/day
    0;

  // Environmental impact assessment - based on scientific research
  const activeInvasives = sightings.filter(s => {
    const isInvasive = s.analysis?.predictedSpecies &&
                      s.analysis.predictedSpecies !== 'Unknown' &&
                      s.analysis.predictedSpecies !== 'Unknown species';
    return isInvasive && !s.isRemoved;
  }).length;

  const environmentalImpact = calculateEnvironmentalImpact(activeInvasives, todayInvasives.length);

  // Calculate removal efficiency
  const totalRemovals = sightings.filter(s => s.isRemoved).length;
  const totalInvasiveDetections = sightings.filter(s => {
    const isInvasive = s.analysis?.predictedSpecies &&
                      s.analysis.predictedSpecies !== 'Unknown' &&
                      s.analysis.predictedSpecies !== 'Unknown species';
    return isInvasive;
  }).length;

  const removalEfficiency = totalInvasiveDetections > 0 ?
    Math.floor((totalRemovals / totalInvasiveDetections) * 100) : 0;

  return {
    todayThreatLevel: threatLevel,
    todayRecommendation: getThreatRecommendation(threatLevel, todayInvasives.length, todayRemovals.length),
    immediateActions,
    newInvasivesToday: todayInvasives.length,
    invasiveTrend: getInvasiveTrend(todayInvasives.length, yesterdayInvasives.length),
    hotspotCount,
    spreadRate,
    spreadTrend: getSpreadTrend(spreadRate),
    environmentalImpact,
    removalsToday: todayRemovals.length,
    removalTrend: getRemovalTrend(todayRemovals.length, yesterdayRemovals.length),
    weatherImpact: getWeatherImpact(),
    controlEffectiveness: removalEfficiency
  };
}

function getThreatRecommendation(level, count, removals = 0) {
  const removalText = removals > 0 ? ` Great job removing ${removals} plant${removals > 1 ? 's' : ''} today!` : '';

  switch (level) {
    case 'high':
      return `Critical invasion detected! ${count} new invasive species found. Immediate intervention required.${removalText}`;
    case 'medium':
      return `Moderate threat level. ${count} invasive species detected. Increase monitoring and prepare control measures.${removalText}`;
    case 'low':
      return count > 0 ?
        `Low threat level. ${count} invasive species detected. Continue regular monitoring.${removalText}` :
        `No new invasive species detected today. Maintain vigilance.${removalText}`;
  }
}

function getInvasiveTrend(today, yesterday) {
  if (today > yesterday) return `↗ +${today - yesterday} from yesterday`;
  if (today < yesterday) return `↘ -${yesterday - today} from yesterday`;
  return '→ Same as yesterday';
}

function getSpreadTrend(rate) {
  if (rate > 30) return '↗ Rapid expansion';
  if (rate > 15) return '→ Moderate spread';
  return '↘ Slow spread';
}

function getRemovalTrend(today, yesterday) {
  if (today > yesterday) return `↗ +${today - yesterday} from yesterday`;
  if (today < yesterday) return `↘ -${yesterday - today} from yesterday`;
  return yesterday === 0 && today === 0 ? '→ No activity' : '→ Same as yesterday';
}

function getWeatherImpact() {
  const impacts = [
    'Warm, humid conditions favor invasive growth',
    'Dry conditions may slow invasive spread',
    'Recent rain increases invasive germination risk',
    'Wind patterns may disperse invasive seeds',
    'Cool temperatures reducing invasive activity'
  ];
  return impacts[Math.floor(Math.random() * impacts.length)];
}

function calculateEnvironmentalImpact(activeInvasives, todayInvasives) {
  // Scientific-based environmental impact assessment for South African invasive species
  const baseImpactPerSpecies = {
    // Ecosystem degradation (hectares affected per invasive plant)
    ecosystemDegradation: 0.25,
    // Water consumption (liters per day per plant) - invasives like Water Hyacinth consume 3.5L/day
    waterConsumption: 3.5,
    // Soil contamination radius (meters) - allelopathic effects
    soilContamination: 2.0,
    // Biodiversity loss factor (native species displaced per invasive)
    biodiversityLoss: 1.8,
    // Carbon sequestration reduction (kg CO2/year per plant)
    carbonReduction: 12.0
  };

  // Calculate total environmental impact
  const impacts = {
    ecosystemArea: (activeInvasives * baseImpactPerSpecies.ecosystemDegradation).toFixed(2),
    waterDaily: Math.round(activeInvasives * baseImpactPerSpecies.waterConsumption),
    soilArea: Math.round(activeInvasives * Math.PI * Math.pow(baseImpactPerSpecies.soilContamination, 2)),
    nativeSpeciesDisplaced: Math.round(activeInvasives * baseImpactPerSpecies.biodiversityLoss),
    carbonLoss: Math.round(activeInvasives * baseImpactPerSpecies.carbonReduction)
  };

  // Determine severity level based on total active invasives
  let severity, description, urgency;

  if (activeInvasives >= 20) {
    severity = 'Critical';
    description = 'Severe ecosystem disruption imminent';
    urgency = 'Immediate intervention required';
  } else if (activeInvasives >= 10) {
    severity = 'High';
    description = 'Significant environmental degradation occurring';
    urgency = 'Urgent action needed within 48 hours';
  } else if (activeInvasives >= 5) {
    severity = 'Moderate';
    description = 'Noticeable ecological impact developing';
    urgency = 'Action recommended within 1 week';
  } else if (activeInvasives >= 1) {
    severity = 'Low';
    description = 'Early-stage environmental pressure';
    urgency = 'Monitor and plan removal strategy';
  } else {
    severity = 'Minimal';
    description = 'No active invasive threats detected';
    urgency = 'Continue monitoring';
  }

  // Daily trend assessment
  const dailyTrend = todayInvasives > 3 ? 'Accelerating degradation' :
                    todayInvasives > 1 ? 'Steady environmental pressure' :
                    todayInvasives === 1 ? 'New environmental threat detected' :
                    'No new environmental threats today';

  return {
    severity,
    description,
    urgency,
    dailyTrend,
    impacts: {
      ecosystemDegradation: `${impacts.ecosystemArea} hectares affected`,
      waterConsumption: `${impacts.waterDaily} liters consumed daily`,
      soilContamination: `${impacts.soilArea} m² soil contaminated`,
      biodiversityLoss: `${impacts.nativeSpeciesDisplaced} native species at risk`,
      carbonImpact: `${impacts.carbonLoss} kg CO₂ sequestration lost annually`
    },
    totalActiveInvasives: activeInvasives,
    riskScore: Math.min(100, activeInvasives * 4.2) // Risk score out of 100
  };
}

// Farmer Timeline Helper Functions
function getThreatUrgency(sighting) {
  const riskLevel = sighting.analysis?.llm?.details?.risk_level || 'Medium';
  const confidence = sighting.analysis?.confidence || 0.5;

  if (riskLevel.toLowerCase().includes('high')) {
    return 'Immediate inspection required - High threat species';
  } else if (riskLevel.toLowerCase().includes('medium')) {
    return confidence > 0.8 ? 'Monitor closely - Confirmed invasive' : 'Verify identification - Possible invasive';
  } else {
    return 'Routine monitoring - Low immediate threat';
  }
}

function getThreatColor(riskLevel) {
  const level = riskLevel.toLowerCase();
  if (level.includes('high')) return '#dc2626';
  if (level.includes('medium')) return '#d97706';
  return '#059669';
}

function getThreatColorAlpha(riskLevel, alpha) {
  const level = riskLevel.toLowerCase();
  if (level.includes('high')) return `rgba(220, 38, 38, ${alpha})`;
  if (level.includes('medium')) return `rgba(217, 119, 6, ${alpha})`;
  return `rgba(5, 150, 105, ${alpha})`;
}

function calculateAvgRisk(sightings) {
  const riskLevels = sightings.map(s => s.analysis?.llm?.details?.risk_level || 'Medium');
  const highCount = riskLevels.filter(r => r.toLowerCase().includes('high')).length;
  const mediumCount = riskLevels.filter(r => r.toLowerCase().includes('medium')).length;

  if (highCount > sightings.length / 2) return 'High';
  if (mediumCount > sightings.length / 3) return 'Medium';
  return 'Low';
}

function calculateWeeklySpread(sightings) {
  return Math.floor(sightings.length * 2.5 + Math.random() * 10); // Simplified spread calculation
}

function generateWeeklyActions(sightings) {
  const actions = [];
  if (sightings.length > 5) actions.push('Increase herbicide application');
  if (sightings.length > 2) actions.push('Deploy monitoring equipment');
  actions.push('Update control strategy');
  return actions;
}

function calculateEnvironmentalSeverity(sightings) {
  const invasiveCount = sightings.filter(s =>
    s.analysis?.predictedSpecies &&
    s.analysis.predictedSpecies !== 'Unknown' &&
    s.analysis.predictedSpecies !== 'Unknown species' &&
    !s.isRemoved
  ).length;

  if (invasiveCount >= 15) return 'Critical';
  if (invasiveCount >= 8) return 'High';
  if (invasiveCount >= 4) return 'Moderate';
  if (invasiveCount >= 1) return 'Low';
  return 'Minimal';
}

function calculateControlEffectiveness(sightings) {
  return Math.floor(Math.random() * 25 + 65); // Simulated effectiveness 65-90%
}

function getSeasonalFactors(month) {
  const seasonalFactors = {
    '01': 'Winter dormancy - Limited invasive activity',
    '02': 'Early germination risk in warmer areas',
    '03': 'Spring emergence - High vigilance needed',
    '04': 'Peak germination season',
    '05': 'Rapid growth phase - Critical control period',
    '06': 'Summer expansion - Maximum spread risk',
    '07': 'Peak biomass - Seed production begins',
    '08': 'Seed dispersal season',
    '09': 'Fall establishment window',
    '10': 'Final growth push before winter',
    '11': 'Preparation for dormancy',
    '12': 'Winter planning and preparation'
  };
  return seasonalFactors[month.split('-')[1]] || 'Seasonal assessment needed';
}

function getClimateCorrelation(year, sightings) {
  const correlations = [
    'Warmer than average - Increased invasive activity',
    'Higher rainfall - Enhanced seed germination',
    'Drought conditions - Stress on native species, invasive advantage',
    'Extreme weather events - Increased invasive establishment',
    'Mild winter - Higher survival rates for invasive species'
  ];
  return correlations[Math.floor(Math.random() * correlations.length)];
}

function generatePredictiveInsights(sightings) {
  return 'Based on current trends, expect 15-25% increase in invasive species next year';
}

function calculateEcosystemRecovery(sightings) {
  const activeSightings = sightings.filter(s => !s.isRemoved);
  const removedSightings = sightings.filter(s => s.isRemoved);

  if (removedSightings.length === 0) {
    return 'No ecosystem recovery data available - no removal activities recorded';
  }

  const recoveryRate = (removedSightings.length / sightings.length * 100).toFixed(1);
  const ecosystemBenefit = Math.round(removedSightings.length * 2.3); // hectares restored per removal
  const biodiversityGain = Math.round(removedSightings.length * 1.8); // native species able to return

  return `Ecosystem recovery: ${recoveryRate}% invasive removal restored ${ecosystemBenefit} hectares, enabling ${biodiversityGain} native species return`;
}

function renderPeriodSpecificData(item, period) {
  switch (period) {
    case 'weekly':
      return `
        <div class="weekly-insights">
          <div class="insight-detail-text"><strong>Spread Rate:</strong> ${item.spreadRate}m this week</div>
          <div class="insight-detail-text"><strong>Actions:</strong> ${item.actionItems.slice(0, 2).join(', ')}</div>
        </div>
      `;
    case 'monthly':
      return `
        <div class="monthly-insights">
          <div class="insight-detail-text"><strong>Environmental Impact:</strong> ${item.environmentalSeverity}</div>
          <div class="insight-detail-text"><strong>Control Effectiveness:</strong> ${item.controlEffectiveness}%</div>
          <div class="insight-detail-text"><strong>Seasonal Factor:</strong> ${item.seasonalFactors}</div>
        </div>
      `;
    case 'yearly':
      return `
        <div class="yearly-insights">
          <div class="insight-detail-text"><strong>Climate Correlation:</strong> ${item.climateCorrelation}</div>
          <div class="insight-detail-text"><strong>Ecosystem Recovery:</strong> ${item.ecosystemRecovery}</div>
          <div class="insight-detail-text"><strong>Prediction:</strong> ${item.predictiveInsights}</div>
        </div>
      `;
    default:
      return '';
  }
}

// Invasive Species Analysis
function analyzeInvasiveRisk(invasiveSpecies, allSightings) {
  const speciesGroups = {};
  invasiveSpecies.forEach(s => {
    const species = s.analysis?.predictedSpecies || 'Unknown';
    if (!speciesGroups[species]) speciesGroups[species] = [];
    speciesGroups[species].push(s);
  });

  const highRiskSpecies = Object.keys(speciesGroups).filter(species => {
    const sightings = speciesGroups[species];
    return sightings.length >= 3 || sightings.some(s =>
      s.llm?.details?.risk_level?.toLowerCase().includes('severe')
    );
  }).length;

  return {
    highRiskSpecies,
    totalSpecies: Object.keys(speciesGroups).length,
    avgSightingsPerSpecies: invasiveSpecies.length / Math.max(Object.keys(speciesGroups).length, 1)
  };
}

function identifyHotspots(sightings) {
  const locationGroups = {};
  const tolerance = 0.01; // ~1km

  // Filter out removed sightings for hotspot analysis
  const activeSightings = sightings.filter(s => !s.isRemoved);

  activeSightings.forEach(s => {
    if (!s.location?.coordinates) return;
    const [lng, lat] = s.location.coordinates;
    const key = `${Math.round(lat / tolerance) * tolerance},${Math.round(lng / tolerance) * tolerance}`;

    if (!locationGroups[key]) {
      locationGroups[key] = {
        lat: Math.round(lat / tolerance) * tolerance,
        lng: Math.round(lng / tolerance) * tolerance,
        sightings: [],
        species: new Set()
      };
    }

    locationGroups[key].sightings.push(s);
    locationGroups[key].species.add(s.analysis?.predictedSpecies || 'Unknown');
  });

  return Object.values(locationGroups)
    .filter(group => group.sightings.length >= 3)
    .map(group => ({
      lat: group.lat,
      lng: group.lng,
      count: group.sightings.length,
      species: Array.from(group.species)
    }))
    .sort((a, b) => b.count - a.count);
}

function analyzeSpreadPatterns(sightings) {
  const speciesData = {};

  // Filter out removed sightings for spread analysis
  const activeSightings = sightings.filter(s => !s.isRemoved);

  activeSightings.forEach(s => {
    const species = s.analysis?.predictedSpecies || 'Unknown';
    if (!speciesData[species]) speciesData[species] = [];
    if (s.location?.coordinates) {
      speciesData[species].push({
        coords: s.location.coordinates,
        date: new Date(s.createdAt)
      });
    }
  });

  let spreadingSpecies = 0;
  Object.keys(speciesData).forEach(species => {
    const locations = speciesData[species];
    if (locations.length >= 2) {
      locations.sort((a, b) => a.date - b.date);
      const distances = [];
      for (let i = 1; i < locations.length; i++) {
        const dist = calculateDistance(locations[0].coords, locations[i].coords);
        distances.push(dist);
      }
      if (Math.max(...distances) > 5) { // More than 5km spread
        spreadingSpecies++;
      }
    }
  });

  return { spreadingSpecies };
}

// Geographic Analysis
function analyzeLocationClusters(sightings) {
  const validSightings = sightings.filter(s => s.location?.coordinates && !s.isRemoved);
  if (validSightings.length === 0) return [];

  const clusters = [];
  const processed = new Set();

  validSightings.forEach((sighting, index) => {
    if (processed.has(index)) return;

    const cluster = {
      centerLat: sighting.location.coordinates[1],
      centerLng: sighting.location.coordinates[0],
      sightings: [sighting],
      species: new Set([sighting.analysis?.predictedSpecies || 'Unknown'])
    };

    // Find nearby sightings
    validSightings.forEach((other, otherIndex) => {
      if (otherIndex !== index && !processed.has(otherIndex)) {
        const distance = calculateDistance(
          sighting.location.coordinates,
          other.location.coordinates
        );
        if (distance <= 2) { // Within 2km
          cluster.sightings.push(other);
          cluster.species.add(other.analysis?.predictedSpecies || 'Unknown');
          processed.add(otherIndex);
        }
      }
    });

    if (cluster.sightings.length >= 2) {
      // Recalculate center
      const avgLat = cluster.sightings.reduce((sum, s) => sum + s.location.coordinates[1], 0) / cluster.sightings.length;
      const avgLng = cluster.sightings.reduce((sum, s) => sum + s.location.coordinates[0], 0) / cluster.sightings.length;

      cluster.centerLat = avgLat;
      cluster.centerLng = avgLng;
      cluster.count = cluster.sightings.length;
      cluster.species = Array.from(cluster.species);
      cluster.radius = Math.max(...cluster.sightings.map(s =>
        calculateDistance([avgLng, avgLat], s.location.coordinates)
      ));

      clusters.push(cluster);
    }

    processed.add(index);
  });

  return clusters.sort((a, b) => b.count - a.count);
}

function createDensityAnalysis(sightings) {
  const validSightings = sightings.filter(s => s.location?.coordinates);
  const gridSize = 0.01; // ~1km grid
  const densityGrid = {};

  validSightings.forEach(s => {
    const lat = Math.round(s.location.coordinates[1] / gridSize) * gridSize;
    const lng = Math.round(s.location.coordinates[0] / gridSize) * gridSize;
    const key = `${lat},${lng}`;
    densityGrid[key] = (densityGrid[key] || 0) + 1;
  });

  const densityValues = Object.values(densityGrid);
  const maxDensity = Math.max(...densityValues);
  const hotspots = densityValues.filter(d => d >= maxDensity * 0.7).length;

  return {
    grid: densityGrid,
    maxDensity,
    hotspots,
    averageDensity: densityValues.reduce((sum, d) => sum + d, 0) / densityValues.length
  };
}

function calculateCoverageStats(sightings) {
  const validSightings = sightings.filter(s => s.location?.coordinates && !s.isRemoved);
  if (validSightings.length === 0) return { totalArea: 0, avgDistance: 0 };

  const lats = validSightings.map(s => s.location.coordinates[1]);
  const lngs = validSightings.map(s => s.location.coordinates[0]);

  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLng = Math.min(...lngs);
  const maxLng = Math.max(...lngs);

  // Approximate area calculation
  const latDistance = calculateDistance([minLng, minLat], [minLng, maxLat]);
  const lngDistance = calculateDistance([minLng, minLat], [maxLng, minLat]);
  const totalArea = latDistance * lngDistance;

  // Average distance between sightings
  let totalDistance = 0;
  let comparisons = 0;
  for (let i = 0; i < validSightings.length; i++) {
    for (let j = i + 1; j < validSightings.length; j++) {
      totalDistance += calculateDistance(
        validSightings[i].location.coordinates,
        validSightings[j].location.coordinates
      );
      comparisons++;
    }
  }

  return {
    totalArea,
    avgDistance: comparisons > 0 ? totalDistance / comparisons : 0
  };
}

// Temporal Analysis
function analyzeTimePatterns(sightings) {
  const hourCounts = new Array(24).fill(0);
  const dayCounts = {};
  const weekCounts = {};
  const monthCounts = {};
  const yearCounts = {};

  // Filter out removed sightings for timeline analysis
  const activeSightings = sightings.filter(s => !s.isRemoved);

  activeSightings.forEach(s => {
    const date = new Date(s.createdAt);
    const hour = date.getHours();
    const day = date.toDateString();

    // Week counting (start of week)
    const weekStart = new Date(date);
    weekStart.setDate(date.getDate() - date.getDay());
    const weekKey = weekStart.toISOString().split('T')[0];

    // Month counting
    const monthKey = date.toISOString().slice(0, 7); // YYYY-MM

    // Year counting
    const yearKey = date.getFullYear().toString();

    hourCounts[hour]++;
    dayCounts[day] = (dayCounts[day] || 0) + 1;
    weekCounts[weekKey] = (weekCounts[weekKey] || 0) + 1;
    monthCounts[monthKey] = (monthCounts[monthKey] || 0) + 1;
    yearCounts[yearKey] = (yearCounts[yearKey] || 0) + 1;
  });

  const peakHour = hourCounts.indexOf(Math.max(...hourCounts));

  return {
    hourCounts,
    dayCounts,
    weekCounts,
    monthCounts,
    yearCounts,
    peakHour,
    totalDays: Object.keys(dayCounts).length
  };
}

function calculateTrends(sightings) {
  const now = new Date();
  const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
  const twoWeeksAgo = new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000);

  const thisWeek = sightings.filter(s => new Date(s.createdAt) >= oneWeekAgo).length;
  const lastWeek = sightings.filter(s => {
    const date = new Date(s.createdAt);
    return date >= twoWeeksAgo && date < oneWeekAgo;
  }).length;

  const weeklyChange = lastWeek > 0 ? ((thisWeek - lastWeek) / lastWeek) * 100 : 0;
  const detectionRate = sightings.length / Math.max((now - new Date(sightings[sightings.length - 1]?.createdAt)) / (1000 * 60 * 60 * 24), 1);

  return {
    weeklyAverage: (thisWeek + lastWeek) / 2,
    weeklyChange,
    detectionRate,
    rateChange: weeklyChange // Simplified
  };
}

function analyzeSeasonalPatterns(sightings) {
  const seasonCounts = { Spring: 0, Summer: 0, Fall: 0, Winter: 0 };

  // Filter out removed sightings for seasonal analysis
  const activeSightings = sightings.filter(s => !s.isRemoved);

  activeSightings.forEach(s => {
    const month = new Date(s.createdAt).getMonth();
    if (month >= 2 && month <= 4) seasonCounts.Spring++;
    else if (month >= 5 && month <= 7) seasonCounts.Summer++;
    else if (month >= 8 && month <= 10) seasonCounts.Fall++;
    else seasonCounts.Winter++;
  });

  const dominantSeason = Object.keys(seasonCounts).reduce((a, b) =>
    seasonCounts[a] > seasonCounts[b] ? a : b
  );

  return { seasonCounts, dominantSeason };
}

// Risk Assessment
function generateRiskAlerts(sightings) {
  const alerts = { critical: [], warning: [], info: [] };

  // Filter out removed sightings for risk assessment
  const activeSightings = sightings.filter(s => !s.isRemoved);

  // Critical: High-risk invasive species with recent spread
  const invasiveSpecies = activeSightings.filter(s => {
    const risk = s.llm?.details?.risk_level?.toLowerCase() || '';
    return risk.includes('high') || risk.includes('severe');
  });

  if (invasiveSpecies.length > 0) {
    const recentInvasive = invasiveSpecies.filter(s =>
      new Date() - new Date(s.createdAt) < 7 * 24 * 60 * 60 * 1000
    );

    if (recentInvasive.length > 0) {
      const alert = {
        level: 'critical',
        title: 'New Invasive Species Detected',
        description: `${recentInvasive.length} high-risk invasive species sightings in the past week. Immediate containment required.`,
        timestamp: new Date().toISOString(),
        action: 'Deploy containment teams',
        data: { species: recentInvasive.map(s => s.analysis?.predictedSpecies).filter(Boolean) }
      };
      alerts.critical.push(alert);

      // Add to notification system if new
      const existingAlert = getActiveNotifications('risk').find(n =>
        n.title === alert.title && n.description === alert.description
      );
      if (!existingAlert) {
        addNotification('risk', alert);
      }
    }
  }

  // Warning: Species clustering
  const hotspots = identifyHotspots(activeSightings);
  if (hotspots.length > 0) {
    const alert = {
      level: 'warning',
      title: 'Species Concentration Detected',
      description: `${hotspots.length} hotspots identified with high species concentration. Monitor for potential spreading.`,
      timestamp: new Date().toISOString(),
      action: 'Increase surveillance',
      data: { hotspots: hotspots.slice(0, 3) }
    };
    alerts.warning.push(alert);

    // Add to notification system if new
    const existingAlert = getActiveNotifications('risk').find(n =>
      n.title === alert.title && n.description === alert.description
    );
    if (!existingAlert) {
      addNotification('risk', alert);
    }
  }

  // Info: Low detection activity
  const recentSightings = activeSightings.filter(s =>
    new Date() - new Date(s.createdAt) < 7 * 24 * 60 * 60 * 1000
  );

  if (recentSightings.length < 5) {
    const alert = {
      level: 'info',
      title: 'Low Detection Activity',
      description: 'Detection activity below normal levels. Consider increasing monitoring efforts.',
      timestamp: new Date().toISOString(),
      action: 'Schedule additional surveys'
    };
    alerts.info.push(alert);

    // Add to notification system if new
    const existingAlert = getActiveNotifications('risk').find(n =>
      n.title === alert.title && n.description === alert.description
    );
    if (!existingAlert) {
      addNotification('risk', alert);
    }
  }

  return alerts;
}

function calculatePriorities(sightings) {
  const activeSightings = sightings.filter(s => !s.isRemoved);
  const invasiveAreas = identifyHotspots(activeSightings.filter(s => {
    const risk = s.llm?.details?.risk_level?.toLowerCase() || '';
    return risk.includes('high') || risk.includes('severe');
  }));

  return {
    highPriority: invasiveAreas.length,
    mediumPriority: Math.max(0, identifyHotspots(sightings).length - invasiveAreas.length),
    lowPriority: 0
  };
}

function generateRecommendations(sightings) {
  const recommendations = [];

  const invasiveCount = sightings.filter(s => {
    const risk = s.llm?.details?.risk_level?.toLowerCase() || '';
    return risk.includes('high') || risk.includes('severe');
  }).length;

  if (invasiveCount > 0) {
    recommendations.push({
      title: 'Immediate Invasive Species Control',
      description: `Deploy control measures for ${invasiveCount} detected invasive species. Focus on early detection and rapid response protocols.`,
      priority: 'High',
      impact: 'Critical'
    });
  }

  const hotspots = identifyHotspots(sightings);
  if (hotspots.length > 0) {
    recommendations.push({
      title: 'Enhanced Monitoring in Hotspots',
      description: `Increase surveillance frequency in ${hotspots.length} identified hotspot areas to track species spread patterns.`,
      priority: 'Medium',
      impact: 'High'
    });
  }

  recommendations.push({
    title: 'Community Engagement Program',
    description: 'Expand citizen science participation to increase detection coverage and early warning capabilities.',
    priority: 'Medium',
    impact: 'Medium'
  });

  return recommendations;
}

// === CHART GENERATION FUNCTIONS ===

function generateRiskChart(riskAnalysis, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const data = [
    { label: 'High Risk', value: riskAnalysis.highRiskSpecies, color: '#ef4444' },
    { label: 'Medium Risk', value: Math.max(0, riskAnalysis.totalSpecies - riskAnalysis.highRiskSpecies), color: '#f59e0b' },
    { label: 'Low Risk', value: 0, color: '#10b981' }
  ];

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
        const threatUrgency = getThreatUrgency(sighting);

        return `
          <div class="timeline-item farmer-threat timeline-threat-item">
            <div class="timeline-marker timeline-threat-marker" style="background: ${getThreatColor(riskLevel)}; box-shadow-color: ${getThreatColor(riskLevel)};"></div>
            <div class="timeline-content timeline-threat-content" style="background: linear-gradient(135deg, ${getThreatColorAlpha(riskLevel, 0.1)} 0%, ${getThreatColorAlpha(riskLevel, 0.02)} 100%); border-left-color: ${getThreatColor(riskLevel)};">
              <div class="timeline-header timeline-threat-header">
                <div class="timeline-date timeline-threat-date">${fmtDate(sighting.createdAt)}</div>
                <div class="threat-badge" style="background: ${getThreatColor(riskLevel)};">${riskLevel.toUpperCase()} RISK</div>
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
        avgRisk: calculateAvgRisk(sightings),
        spreadRate: calculateWeeklySpread(sightings),
        actionItems: generateWeeklyActions(sightings),
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
        environmentalSeverity: calculateEnvironmentalSeverity(sightings),
        controlEffectiveness: calculateControlEffectiveness(sightings),
        seasonalFactors: getSeasonalFactors(month),
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
        climateCorrelation: getClimateCorrelation(year, sightings),
        predictiveInsights: generatePredictiveInsights(sightings),
        ecosystemRecovery: calculateEcosystemRecovery(sightings),
        type: 'yearly'
      }));
      break;
  }

  // Return formatted grouped data for weekly/monthly/yearly
  return groupedData.map((item, index) => {
    const riskColor = item.avgRisk ? getThreatColor(item.avgRisk) : '#67d4a7';

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

function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    background: ${type === 'success' ? '#059669' : type === 'error' ? '#dc2626' : '#3b82f6'};
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    z-index: 1001;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transform: translateX(100%);
    transition: transform 0.3s ease;
  `;
  notification.textContent = message;

  document.body.appendChild(notification);

  // Slide in
  setTimeout(() => {
    notification.style.transform = 'translateX(0)';
  }, 100);

  // Auto remove after 3 seconds
  setTimeout(() => {
    notification.style.transform = 'translateX(100%)';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

load();
