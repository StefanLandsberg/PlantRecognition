// js/sightings/ui.js

// API and module imports
import { SightingsAPI } from "../api.js";
import * as utils from "./utils.js";
import * as llmRenderer from "./llmRenderer.js";
import * as notifications from "./notifications.js";
import * as analytics from "./analytics.js";
import * as charts from "./charts.js";

// === UI HELPERS (Modals, Toasts) ===

export const showNotification = (message, type = "info") => {
  const el = document.createElement("div");
  el.className = `notification ${type}`;
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.classList.add("show"), 10);
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 500);
  }, 3000);
};

export const showNotificationToast = (message, type = "info") => {
  const toast = document.createElement("div");
  toast.className = `notification-toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("show"));
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.remove(), 300);
  }, 3000);
};

const showImageModal = (imageUrl) => {
  const modal = document.createElement("div");
  modal.className = "image-modal";
  modal.innerHTML = `<img src="${imageUrl}" alt="Sighting Image" />`;
  document.body.appendChild(modal);
  requestAnimationFrame(() => modal.classList.add("show"));
  modal.addEventListener("click", () => {
    modal.classList.remove("show");
    setTimeout(() => modal.remove(), 300);
  });
};

// === TAB SWITCHING ===

export const switchTab = (tabName) => {
  // Handle dropdown menu visibility
  const dropdownContainer = document.querySelector(".dropdown-container");
  const dropdown = document.getElementById("analytics-dropdown");
  if (dropdownContainer) dropdownContainer.classList.remove("open");
  if (dropdown) dropdown.classList.remove("show");

  // Remove active class from all tab buttons
  document
    .querySelectorAll(".tab-btn")
    .forEach((btn) => btn.classList.remove("active"));

  // Set active button based on tab type
  if (tabName === "sightings" || tabName === "video-review") {
    const targetBtn = document.querySelector(`[onclick*="'${tabName}'"]`);
    if (targetBtn) targetBtn.classList.add("active");
  } else {
    // Analytics tabs - highlight the dropdown button
    const dropdownBtn = document.querySelector(".dropdown-btn");
    if (dropdownBtn) dropdownBtn.classList.add("active");
  }

  // Hide all tab contents and show selected tab
  document
    .querySelectorAll(".tab-content")
    .forEach((content) => content.classList.remove("active"));
  const selectedTab = document.getElementById(`${tabName}-tab`);
  if (selectedTab) {
    selectedTab.classList.add("active");
  }

  // Load content based on tab
  loadTabContent(tabName);
};

export const toggleDropdown = () => {
  const dropdownContainer = document.querySelector(".dropdown-container");
  const dropdown = document.getElementById("analytics-dropdown");
  if (dropdownContainer) dropdownContainer.classList.toggle("open");
  if (dropdown) dropdown.classList.toggle("show");
};

const loadTabContent = async (tabName) => {
  try {
    // For sightings tab, use existing load function
    if (tabName === "sightings") {
      await loadSightings();
      return;
    }

    // For analytics tabs, get sightings data
    let sightings = window.sightingsData;
    if (!sightings) {
      const response = await SightingsAPI.list();
      sightings = response.data || response;
    }

    const actions = {
      "video-review": async () => {
        const { loadVideoSessions } = await import("./videoReview.js");
        await loadVideoSessions();
      },
      "species-analytics": loadSpeciesAnalytics,
      "invasive-dashboard": loadInvasiveDashboard,
      "geographic-insights": loadGeographicInsights,
      "cluster-map": loadClusterMap,
      "temporal-analysis": loadTemporalAnalysis,
      "risk-assessment": async (data) => {
        await loadRiskAssessment(data);
        if (notifications.updateNotificationBadges) {
          notifications.updateNotificationBadges();
        }
      },
    };

    const action = actions[tabName];
    if (action) {
      await action(sightings);
    }
  } catch (error) {
    console.error("Error loading tab content:", error);
    showNotification("Error loading content", "error");
  }
};

const renderSighting = (sighting) => {
  const imageUrl = sighting.imageUrl || sighting.imagePath;
  const species = sighting.analysis?.predictedSpecies || "Unknown Species";
  const confidence = sighting.analysis?.confidence
    ? `${(sighting.analysis.confidence * 100).toFixed(1)}%`
    : "Unknown";
  const date = sighting.createdAt
    ? utils.fmtDate(sighting.createdAt)
    : "Unknown Date";
  const location = sighting.location?.coordinates
    ? utils.fmtLatLng(sighting.location.coordinates)
    : "Unknown Location";

  return `
    <div class="sighting-card" data-sighting-id="${sighting._id}">
      <div class="sighting-image">
        ${
          imageUrl
            ? `<img src="${imageUrl}" alt="${species}" onclick="showImageModal('${imageUrl}')" />`
            : '<div class="no-image-placeholder"></div>'
        }
      </div>
      <div class="sighting-content">
        <div class="sighting-header">
          <h3 class="sighting-species">${species}</h3>
          <div class="sighting-confidence">${confidence} confidence</div>
        </div>
        <div class="sighting-details">
          <div class="sighting-date">${date}</div>
          <div class="sighting-location">${location}</div>
        </div>
        ${
          sighting.analysis?.llm
            ? `
          <div class="llm-dropdown">
            <button class="llm-dropdown-btn" onclick="toggleLLMDropdown('${
              sighting._id
            }')">
              Analysis Details <span class="llm-dropdown-arrow">▼</span>
            </button>
            <div class="llm-dropdown-content llm-dropdown-content-hidden">
              <div class="llm-section-buttons">
                <button class="llm-section-btn active" onclick="showLLMSection('${
                  sighting._id
                }', 'overview')">Overview</button>
                <button class="llm-section-btn" onclick="showLLMSection('${
                  sighting._id
                }', 'identification')">ID</button>
                <button class="llm-section-btn" onclick="showLLMSection('${
                  sighting._id
                }', 'risk')">Risk</button>
                <button class="llm-section-btn" onclick="showLLMSection('${
                  sighting._id
                }', 'action')">Action</button>
              </div>
              <div id="llm-content-${sighting._id}" class="llm-content">
                ${llmRenderer.formatLLMSection(
                  sighting.analysis.llm,
                  "overview"
                )}
              </div>
            </div>
          </div>
        `
            : ""
        }
        <div class="sighting-actions">
          <button class="btn-secondary" onclick="removeSighting('${
            sighting._id
          }')">Mark as Removed</button>
        </div>
      </div>
    </div>`;
};

// === ANALYTICS TAB RENDERERS ===

const loadSpeciesAnalytics = (sightings) => {
  const container = document.getElementById("species-analytics-container");
  if (!sightings || sightings.length === 0) {
    container.innerHTML =
      '<p class="muted">No sightings data available for analytics.</p>';
    return;
  }

  const speciesData = analytics.aggregateSpeciesData(sightings);
  const statsData = analytics
    .calculateStats(speciesData)
    .sort((a, b) => b.totalCount - a.totalCount);

  container.innerHTML = `
        <div class="dashboard-header">
            <h2 class="dashboard-title">Species Analytics</h2>
            <p class="dashboard-subtitle">Comprehensive species identification and distribution analysis</p>
        </div>
        <div id="species-analytics-grid"></div>`;

  const grid = container.querySelector("#species-analytics-grid");
  statsData.forEach((stats) => {
    const card = document.createElement("div");
    card.className = "analytics-card";
    card.innerHTML = `
            <div class="analytics-header">
                ${
                  stats.mainImage &&
                  typeof stats.mainImage === "string" &&
                  (stats.mainImage.startsWith("/") ||
                    stats.mainImage.startsWith("http"))
                    ? `<img src="${stats.mainImage}" alt="${stats.species}" class="analytics-image" onerror="this.style.display='none'" />`
                    : `<div class="no-image-placeholder analytics-image"></div>`
                }
                <div class="analytics-title">
                    <h3>${stats.species}</h3>
                    ${
                      stats.scientificName
                        ? `<p class="species-scientific">${stats.scientificName}</p>`
                        : ""
                    }
                </div>
            </div>
            <div class="analytics-stats">
                <div class="stat-item"><span class="stat-value">${
                  stats.totalCount
                }</span><span class="stat-label">Sightings</span></div>
                <div class="stat-item"><span class="stat-value">${stats.avgConfidence.toFixed(
                  1
                )}%</span><span class="stat-label">Confidence</span></div>
                <div class="stat-item"><span class="stat-value">${
                  stats.uniqueLocations
                }</span><span class="stat-label">Locations</span></div>
                <div class="stat-item"><span class="stat-value">${stats.geoSpread.toFixed(
                  1
                )}km</span><span class="stat-label">Spread</span></div>
            </div>
            <div class="analytics-details">
                <div class="detail-row"><span class="detail-label">Last Seen</span><span class="detail-value">${utils.fmtDate(
                  stats.lastSeen
                )}</span></div>
                <div class="detail-row"><span class="detail-label">First Seen</span><span class="detail-value">${utils.fmtDate(
                  stats.firstSeen
                )}</span></div>
                ${
                  stats.avgLocation
                    ? `<div class="detail-row"><span class="detail-label">Avg Location</span><span class="detail-value">${utils.fmtLatLng(
                        stats.avgLocation
                      )}</span></div>`
                    : ""
                }
                <div class="detail-row"><span class="detail-label">Risk Level</span><span class="detail-value"><span class="risk-badge ${utils.getRiskBadgeClass(
                  stats.riskLevel
                )}">${stats.riskLevel}</span></span></div>
            </div>`;
    grid.appendChild(card);
  });
};

const loadInvasiveDashboard = (sightings) => {
  const container = document.getElementById("invasive-dashboard-container");
  const activeSightings = sightings.filter((s) => !s.isRemoved);
  const isNamedSpecies = (s) => {
    const species = s.analysis?.predictedSpecies || "Unknown";
    return species !== "Unknown" && !species.includes("Unknown");
  };
  const namedSpecies = activeSightings.filter(isNamedSpecies);
  const unknownSpecies = activeSightings.filter((s) => !isNamedSpecies(s));
  const invasiveSpecies = namedSpecies;
  const riskAnalysis = analytics.analyzeInvasiveRisk(invasiveSpecies);
  const hotspots = analytics.identifyHotspots(invasiveSpecies);
  const spreadAnalysis = analytics.analyzeSpreadPatterns(invasiveSpecies);

  container.innerHTML = `
    <div class="dashboard-header">
      <h2 class="dashboard-title">Invasive Species Dashboard</h2>
      <p class="dashboard-subtitle">Critical invasive species monitoring and threat assessment</p>
    </div>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-value">${invasiveSpecies.length}</div>
        <div class="metric-label">Invasive Sightings</div>
        <div class="metric-change negative">${(
          (invasiveSpecies.length / activeSightings.length) *
          100
        ).toFixed(1)}% of total</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${riskAnalysis.highRiskSpecies}</div>
        <div class="metric-label">High Risk Species</div>
        <div class="metric-change negative">Requires immediate action</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${hotspots.length}</div>
        <div class="metric-label">Active Hotspots</div>
        <div class="metric-change negative">Geographic concentration</div>
      </div>
      <div class="metric-card">
        <div class="metric-value">${spreadAnalysis.spreadingSpecies}</div>
        <div class="metric-label">Spreading Species</div>
        <div class="metric-change negative">Active geographic expansion</div>
      </div>
    </div>
    <div class="chart-container"><div class="chart-header"><h3 class="chart-title">Species Detection Trends</h3></div><div class="chart-content"><div class="simple-chart" id="detection-trends-chart"></div></div></div>
    <div class="geo-grid">
      <div class="chart-container"><div class="chart-header"><h3 class="chart-title">Invasion Hotspots</h3></div><div class="location-list">${hotspots
        .map(
          (h) =>
            `<div class="location-item"><div class="location-info"><div class="location-coords">${utils.fmtLatLng(
              [h.lng, h.lat]
            )}</div><div class="location-meta">${h.species.join(
              ", "
            )}</div></div><div class="location-count">${h.count}</div></div>`
        )
        .join("")}</div></div>
      <div class="chart-container"><div class="chart-header"><h3 class="chart-title">Risk Level Breakdown</h3></div><div class="chart-content"><div class="simple-chart" id="risk-breakdown-chart"></div></div></div>
    </div>`;

  charts.generateDetectionTrendsChart(
    invasiveSpecies,
    "detection-trends-chart"
  );
  charts.generateRiskBreakdownChart(sightings, "risk-breakdown-chart");

  const unknownContainer = document.createElement("div");
  unknownContainer.className = "chart-container";
  unknownContainer.style.marginTop = "2rem";
  unknownContainer.innerHTML = `
      <div class="chart-header"><h3 class="chart-title">Non-Invasive (Unknown) Species</h3><p class="unknown-species-subtitle">Species requiring further identification</p></div>
      <div class="metrics-grid unknown-metrics-grid"><div class="metric-card"><div class="metric-value">${
        unknownSpecies.length
      }</div><div class="metric-label">Unknown Sightings</div><div class="metric-change neutral">${(
    (unknownSpecies.length / sightings.length) *
    100
  ).toFixed(1)}% of total</div></div></div>`;
  container.appendChild(unknownContainer);
};

const loadGeographicInsights = (sightings) => {
  const container = document.getElementById("geographic-insights-container");
  const activeSightings = sightings.filter((s) => !s.isRemoved);
  const locationClusters = analytics.analyzeLocationClusters(activeSightings);
  const densityMap = analytics.createDensityAnalysis(activeSightings);
  const coverageStats = analytics.calculateCoverageStats(activeSightings);

  container.innerHTML = `
    <div class="dashboard-header"><h2 class="dashboard-title">Geographic Insights</h2><p class="dashboard-subtitle">Spatial distribution patterns and coverage analysis</p></div>
    <div class="metrics-grid">
      <div class="metric-card"><div class="metric-value">${coverageStats.totalArea.toFixed(
        1
      )}km²</div><div class="metric-label">Coverage Area</div><div class="metric-change neutral">Footprint</div></div>
      <div class="metric-card"><div class="metric-value">${
        locationClusters.length
      }</div><div class="metric-label">Location Clusters</div><div class="metric-change neutral">Distinct groups</div></div>
      <div class="metric-card"><div class="metric-value">${
        densityMap.hotspots
      }</div><div class="metric-label">High Density Areas</div><div class="metric-change positive">Concentration zones</div></div>
      <div class="metric-card"><div class="metric-value">${coverageStats.avgDistance.toFixed(
        1
      )}km</div><div class="metric-label">Avg Distance</div><div class="metric-change neutral">Between sightings</div></div>
    </div>
    <div class="geo-grid">
      <div class="chart-container"><div class="chart-header"><h3 class="chart-title">Species Density Distribution</h3></div><div class="chart-content"><div class="simple-chart" id="density-distribution-chart"></div></div></div>
      <div class="chart-container"><div class="chart-header"><h3 class="chart-title">Geographic Coverage Analysis</h3></div><div class="chart-content"><div class="simple-chart" id="coverage-analysis-chart"></div></div></div>
    </div>
    <div class="geo-grid">
        <div class="chart-container"><div class="chart-header"><h3 class="chart-title">Distance Between Sightings</h3></div><div class="chart-content"><div class="simple-chart" id="distance-distribution-chart"></div></div></div>
        <div class="chart-container"><div class="chart-header"><h3 class="chart-title">Location Cluster Analysis</h3></div><div class="chart-content"><div class="simple-chart" id="cluster-size-chart"></div></div></div>
    </div>`;

  charts.generateDensityDistributionChart(
    sightings,
    "density-distribution-chart"
  );
  charts.generateCoverageAnalysisChart(
    coverageStats,
    "coverage-analysis-chart"
  );
  charts.generateDistanceDistributionChart(
    sightings,
    "distance-distribution-chart",
    utils.calculateDistance
  );
  charts.generateClusterSizeChart(locationClusters, "cluster-size-chart");
};

const loadTemporalAnalysis = (sightings) => {
  const container = document.getElementById("temporal-analysis-container");
  const timePatterns = analytics.analyzeTimePatterns(sightings);
  const invasiveAnalytics = analytics.generateInvasiveAnalytics(sightings);

  container.innerHTML = `
    <div class="dashboard-header"><h2 class="dashboard-title">Invasive Species Intelligence Dashboard</h2><p class="dashboard-subtitle">Comprehensive threat assessment and management analytics</p></div>
    <div class="farmer-alert-section">
      <div class="alert-banner ${
        invasiveAnalytics.todayThreatLevel
      }"><h3>Today's Threat Level: ${invasiveAnalytics.todayThreatLevel.toUpperCase()}</h3><p>${
    invasiveAnalytics.todayRecommendation
  }</p></div>
      <div class="immediate-actions"><h4>Immediate Actions Required:</h4><ul>${invasiveAnalytics.immediateActions
        .map((action) => `<li>${action}</li>`)
        .join("")}</ul></div>
    </div>
    <div class="farmer-metrics-grid">
      <div class="metric-card"><div class="metric-value invasive-count">${
        invasiveAnalytics.newInvasivesToday
      }</div><div class="metric-label">New Invasives Today</div><div class="metric-change">${
    invasiveAnalytics.invasiveTrend
  }</div></div>
      <div class="metric-card"><div class="metric-value hotspot-count">${
        invasiveAnalytics.hotspotCount
      }</div><div class="metric-label">Active Hotspots</div><div class="metric-description">Areas needing attention</div></div>
      <div class="metric-card"><div class="metric-value spread-rate">${invasiveAnalytics.spreadRate.toFixed(
        1
      )}m/day</div><div class="metric-label">Avg Spread Rate</div><div class="metric-change">${
    invasiveAnalytics.spreadTrend
  }</div></div>
      <div class="metric-card"><div class="metric-value environment-impact ${
        invasiveAnalytics.environmentalImpact.severity
      }">${
    invasiveAnalytics.environmentalImpact.severity
  }</div><div class="metric-label">Environmental Impact</div><div class="metric-description">${
    invasiveAnalytics.environmentalImpact.description
  }</div><div class="environmental-details"><strong>Active Threats:</strong> ${
    invasiveAnalytics.environmentalImpact.totalActiveInvasives
  }<br><strong>Risk Score:</strong> ${invasiveAnalytics.environmentalImpact.riskScore.toFixed(
    0
  )}/100<br><strong>Urgency:</strong> ${
    invasiveAnalytics.environmentalImpact.urgency
  }</div></div>
    </div>
    <div class="chart-container">
      <div class="chart-header">
        <h3 class="chart-title">Invasion Intelligence Timeline</h3>
        <div class="chart-controls">
          <button class="chart-control-btn active" data-timeline="daily" onclick="updateTimelineView('daily', this)">Daily</button>
          <button class="chart-control-btn" data-timeline="weekly" onclick="updateTimelineView('weekly', this)">Weekly</button>
          <button class="chart-control-btn" data-timeline="monthly" onclick="updateTimelineView('monthly', this)">Monthly</button>
          <button class="chart-control-btn" data-timeline="yearly" onclick="updateTimelineView('yearly', this)">Annual</button>
        </div>
      </div>
      <div class="chart-content"><div class="simple-chart" id="timeline-chart"></div></div>
    </div>
    <div class="timeline-container" id="compact-timeline-container"><div class="timeline-line"></div><div id="timeline-events" class="timeline-events">${generateInvasiveTimelineEvents(
      sightings,
      "daily"
    )}</div></div>
    <div class="farmer-insights-grid">
      <div class="insight-card weather"><h4>Environmental Impact</h4><p>${
        invasiveAnalytics.weatherImpact
      }</p></div>
      <div class="insight-card control"><h4>Management Effectiveness</h4><p>${
        invasiveAnalytics.controlEffectiveness
      }% removal success rate this week</p></div>
    </div>`;

  charts.generateTimelineChart(timePatterns, "timeline-chart");
};

const loadRiskAssessment = async (sightings) => {
  const container = document.getElementById("risk-assessment-container");
  const recommendations = analytics.generateRecommendations(sightings);

  // Refresh server-side alerts but keep existing analytics display
  await notifications.refreshAlerts();

  // Get server-managed alerts
  const riskAlerts = notifications.getActiveNotifications("risk");
  const weatherAlerts = notifications.getActiveNotifications("weather");
  const allNotifications = [...riskAlerts, ...weatherAlerts];

  // Sort alerts by priority
  const sortedAlerts = allNotifications.sort((a, b) => {
    const order = { critical: 0, warning: 1, info: 2 };
    return order[a.level] - order[b.level];
  });

  container.innerHTML = `
    <div class="dashboard-header"><h2 class="dashboard-title">Risk Assessment & Alerts</h2><p class="dashboard-subtitle">Priority threats and recommended actions</p></div>
    <div class="metrics-grid">
      <div class="metric-card"><div class="metric-value">${
        sortedAlerts.filter((a) => a.level === "critical").length
      }</div><div class="metric-label">Critical Alerts</div><div class="metric-change negative">Immediate action</div></div>
      <div class="metric-card"><div class="metric-value">${
        sortedAlerts.filter((a) => a.level === "warning").length
      }</div><div class="metric-label">Warning Alerts</div><div class="metric-change negative">Monitor closely</div></div>
      <div class="metric-card"><div class="metric-value">${
        allNotifications.length
      }</div><div class="metric-label">Active Notifications</div><div class="metric-change positive">Server-managed</div></div>
      <div class="metric-card"><div class="metric-value">${
        recommendations.length
      }</div><div class="metric-label">Recommendations</div><div class="metric-change positive">For implementation</div></div>
    </div>
    <div class="alert-grid">
      ${sortedAlerts
        .map(
          (alert) => `
        <div class="alert-card ${alert.level}" data-alert-id="${alert.id}">
          <div class="alert-header"><div class="alert-level ${alert.level}">${
            alert.level
          }</div><div class="alert-timestamp">${utils.fmtDate(
            alert.timestamp
          )}</div></div>
          <div class="alert-title">${
            alert.title
          }</div><div class="alert-description">${alert.description}</div>
          <div class="alert-actions">${
            alert.title.includes("Weather")
              ? ""
              : '<button class="alert-action-btn secondary" onclick="handleAlertDismiss(\'' +
                alert.id +
                "')\">Dismiss</button>"
          }</div>
        </div>`
        )
        .join("")}
    </div>
    <div class="chart-container"><div class="chart-header"><h3 class="chart-title">Management Recommendations</h3></div><div class="alert-recommendations">${recommendations
      .map(
        (rec) =>
          `<div class="recommendation-card"><h4 class="recommendation-title">${rec.title}</h4><p class="recommendation-description">${rec.description}</p><div class="recommendation-meta"><span>Priority: ${rec.priority} | Impact: ${rec.impact}</span></div></div>`
      )
      .join("")}</div></div>`;
};

const loadClusterMap = async (sightings) => {
  const container = document.getElementById("cluster-map-container");

  // Filter out sightings without location data
  const validSightings = sightings.filter(
    (s) => s.location?.coordinates && !s.isRemoved
  );

  if (validSightings.length === 0) {
    container.innerHTML = `
      <div class="dashboard-header">
        <h2 class="dashboard-title">Interactive Cluster Map</h2>
        <p class="dashboard-subtitle">No sightings with location data available</p>
      </div>
      <div class="empty-state">
        <p class="muted">Add some sightings with location data to see them on the map.</p>
      </div>`;
    return;
  }

  // Get unique species for filtering
  const allSpecies = [
    ...new Set(
      validSightings.map((s) => s.analysis?.predictedSpecies || "Unknown")
    ),
  ].sort();
  const riskLevels = ["All", "High", "Medium", "Low"];

  container.innerHTML = `
    <div class="dashboard-header">
      <h2 class="dashboard-title">Interactive Cluster Map</h2>
      <p class="dashboard-subtitle">Filter and explore sightings by species and risk level</p>
    </div>
    
    <div class="map-controls">
      <div class="filter-group">
        <label for="species-filter">Filter by Species:</label>
        <select id="species-filter" onchange="filterMapSightings()">
          <option value="all">All Species (${validSightings.length})</option>
          ${allSpecies
            .map((species) => {
              const count = validSightings.filter(
                (s) => (s.analysis?.predictedSpecies || "Unknown") === species
              ).length;
              return `<option value="${species}">${species} (${count})</option>`;
            })
            .join("")}
        </select>
      </div>
      
      <div class="filter-group">
        <label for="risk-filter">Filter by Risk Level:</label>
        <select id="risk-filter" onchange="filterMapSightings()">
          ${riskLevels
            .map((level) => {
              if (level === "All") {
                return `<option value="all">All Risk Levels (${validSightings.length})</option>`;
              }
              const count = validSightings.filter((s) => {
                const risk = s.analysis?.llm?.details?.risk_level || "Medium";
                return risk.toLowerCase().includes(level.toLowerCase());
              }).length;
              return `<option value="${level.toLowerCase()}">${level} Risk (${count})</option>`;
            })
            .join("")}
        </select>
      </div>
      
      <div class="filter-group">
        <button onclick="resetMapFilters()" class="filter-reset-btn">Reset Filters</button>
      </div>
    </div>
    
    <div class="map-stats">
      <div class="stat-item">
        <span class="stat-value" id="visible-count">${
          validSightings.length
        }</span>
        <span class="stat-label">Visible Sightings</span>
      </div>
      <div class="stat-item">
        <span class="stat-value" id="species-count">${allSpecies.length}</span>
        <span class="stat-label">Species Types</span>
      </div>
      <div class="stat-item">
        <span class="stat-value" id="invasive-count">${
          validSightings.filter(
            (s) =>
              s.analysis?.predictedSpecies &&
              s.analysis.predictedSpecies !== "Unknown"
          ).length
        }</span>
        <span class="stat-label">Invasive Species</span>
      </div>
    </div>
    
    <div class="map-container">
      <div id="cluster-map" style="height: 500px; width: 100%; border-radius: 8px;"></div>
    </div>
    
    <div class="map-legend">
      <h4>Map Legend</h4>
      <div class="legend-items">
        <div class="legend-item">
          <div class="legend-color" style="background: #dc2626;"></div>
          <span>High Risk / Old Sightings</span>
        </div>
        <div class="legend-item">
          <div class="legend-color" style="background: #d97706;"></div>
          <span>Medium Risk</span>
        </div>
        <div class="legend-item">
          <div class="legend-color" style="background: #FFD700;"></div>
          <span>Recent Sightings</span>
        </div>
        <div class="legend-item">
          <div class="legend-color" style="background: #10b981; border: 1px solid #ccc;"></div>
          <span>Non-invasive Species</span>
        </div>
      </div>
    </div>`;

  // Store sightings data globally for filtering
  window.clusterMapSightings = validSightings;

  // Initialize the map
  await initializeClusterMap(validSightings);
};

// === CLUSTER MAP FUNCTIONS ===

const initializeClusterMap = async (sightings) => {
  try {
    // Import the map loader instance
    const { mapProxy } = await import("../map.js");

    // Use the existing map instance
    const mapLoader = mapProxy;

    // Load Google Maps
    await mapLoader.loadGoogleMaps();

    // Calculate center point from sightings
    const lats = sightings.map((s) => s.location.coordinates[1]);
    const lngs = sightings.map((s) => s.location.coordinates[0]);
    const centerLat = lats.reduce((sum, lat) => sum + lat, 0) / lats.length;
    const centerLng = lngs.reduce((sum, lng) => sum + lng, 0) / lngs.length;

    // Initialize map
    const map = mapLoader.initMap(
      "cluster-map",
      { lat: centerLat, lng: centerLng },
      12
    );

    // Store map instance globally for filtering
    window.clusterMapInstance = mapLoader;

    // Add markers for all sightings
    addMarkersToMap(mapLoader, sightings);
  } catch (error) {
    console.error("Error initializing cluster map:", error);
    document.getElementById("cluster-map").innerHTML = `
      <div class="error-state">
        <p>Error loading map: ${error.message}</p>
        <p>Please check your Google Maps API configuration.</p>
      </div>`;
  }
};

const addMarkersToMap = (mapLoader, sightings) => {
  // Clear existing markers
  if (mapLoader.markers) {
    mapLoader.markers.forEach((marker) => {
      if (marker.setMap) marker.setMap(null);
    });
    mapLoader.markers = [];
    mapLoader.markerClusters.clear();
  }

  // Add markers for each sighting
  sightings.forEach((sighting) => {
    if (sighting.location?.coordinates) {
      const [lng, lat] = sighting.location.coordinates;

      mapLoader.addMarker({
        lat,
        lng,
        title: sighting.analysis?.predictedSpecies || "Unknown Species",
        data: { sighting },
      });
    }
  });
};

// Global functions for HTML onclick handlers
window.filterMapSightings = () => {
  const speciesFilter = document.getElementById("species-filter").value;
  const riskFilter = document.getElementById("risk-filter").value;

  if (!window.clusterMapSightings || !window.clusterMapInstance) return;

  let filteredSightings = window.clusterMapSightings;

  // Apply species filter
  if (speciesFilter !== "all") {
    filteredSightings = filteredSightings.filter(
      (s) => (s.analysis?.predictedSpecies || "Unknown") === speciesFilter
    );
  }

  // Apply risk filter
  if (riskFilter !== "all") {
    filteredSightings = filteredSightings.filter((s) => {
      const risk = s.analysis?.llm?.details?.risk_level || "Medium";
      return risk.toLowerCase().includes(riskFilter);
    });
  }

  // Update map markers
  addMarkersToMap(window.clusterMapInstance, filteredSightings);

  // Update stats
  updateMapStats(filteredSightings);
};

window.resetMapFilters = () => {
  document.getElementById("species-filter").value = "all";
  document.getElementById("risk-filter").value = "all";

  if (window.clusterMapSightings && window.clusterMapInstance) {
    addMarkersToMap(window.clusterMapInstance, window.clusterMapSightings);
    updateMapStats(window.clusterMapSightings);
  }
};

const updateMapStats = (sightings) => {
  const allSpecies = [
    ...new Set(sightings.map((s) => s.analysis?.predictedSpecies || "Unknown")),
  ];
  const invasiveCount = sightings.filter(
    (s) =>
      s.analysis?.predictedSpecies && s.analysis.predictedSpecies !== "Unknown"
  ).length;

  document.getElementById("visible-count").textContent = sightings.length;
  document.getElementById("species-count").textContent = allSpecies.length;
  document.getElementById("invasive-count").textContent = invasiveCount;
};

// === Timeline Event Generators ===
const renderPeriodSpecificData = (item, period) => {
  switch (period) {
    case "weekly":
      return `<div class="weekly-insights"><div class="insight-detail-text"><strong>Spread Rate:</strong> ${
        item.spreadRate
      }m</div><div class="insight-detail-text"><strong>Actions:</strong> ${item.actionItems
        .slice(0, 2)
        .join(", ")}</div></div>`;
    case "monthly":
      return `<div class="monthly-insights"><div class="insight-detail-text"><strong>Impact:</strong> ${item.environmentalSeverity}</div><div class="insight-detail-text"><strong>Effectiveness:</strong> ${item.controlEffectiveness}%</div><div class="insight-detail-text"><strong>Factor:</strong> ${item.seasonalFactors}</div></div>`;
    case "yearly":
      return `<div class="yearly-insights"><div class="insight-detail-text"><strong>Climate:</strong> ${item.climateCorrelation}</div><div class="insight-detail-text"><strong>Recovery:</strong> ${item.ecosystemRecovery}</div><div class="insight-detail-text"><strong>Prediction:</strong> ${item.predictiveInsights}</div></div>`;
    default:
      return "";
  }
};

export const generateInvasiveTimelineEvents = (sightings, period = "daily") => {
  const invasiveSightings = sightings.filter(
    (s) =>
      s.analysis?.predictedSpecies && s.analysis.predictedSpecies !== "Unknown"
  );
  const sortedSightings = invasiveSightings.sort(
    (a, b) => new Date(b.createdAt) - new Date(a.createdAt)
  );

  if (period === "daily") {
    return sortedSightings
      .slice(0, 12)
      .map((s) => {
        const riskLevel = s.analysis?.llm?.details?.risk_level || "Medium";
        const threatUrgency = analytics.getThreatUrgency(s);
        const riskColor = analytics.getThreatColor(riskLevel);
        return `
            <div class="timeline-item farmer-threat timeline-threat-item">
              <div class="timeline-marker" style="background: ${riskColor}; box-shadow-color: ${riskColor};"></div>
              <div class="timeline-content" style="background: linear-gradient(135deg, ${analytics.getThreatColorAlpha(
                riskLevel,
                0.1
              )} 0%, ${analytics.getThreatColorAlpha(
          riskLevel,
          0.02
        )} 100%); border-left-color: ${riskColor};">
                <div class="timeline-header"><div class="timeline-date">${utils.fmtDate(
                  s.createdAt
                )}</div><div class="threat-badge" style="background:${riskColor};">${riskLevel.toUpperCase()} RISK</div></div>
                <div class="timeline-title">${
                  s.analysis?.predictedSpecies || "Unknown"
                }</div>
                <div class="farmer-action">⚠ ${threatUrgency}</div>
                <div class="timeline-details">${utils.pct(
                  s.analysis?.confidence
                )} conf ${
          s.location?.coordinates
            ? ` • ${utils.fmtLatLng(s.location.coordinates)}`
            : ""
        }</div>
              </div>
            </div>`;
      })
      .join("");
  }

  let groupedData = [];
  if (period === "weekly") {
    const groups = {};
    sortedSightings.forEach((s) => {
      const d = new Date(s.createdAt);
      d.setDate(d.getDate() - d.getDay());
      const key = d.toISOString().split("T")[0];
      if (!groups[key]) groups[key] = [];
      groups[key].push(s);
    });
    groupedData = Object.entries(groups)
      .slice(0, 6)
      .map(([week, s_list]) => ({
        period: `Week of ${new Date(week).toLocaleDateString()}`,
        count: s_list.length,
        species: [...new Set(s_list.map((s) => s.analysis.predictedSpecies))],
        avgRisk: analytics.calculateAvgRisk(s_list),
        spreadRate: analytics.calculateWeeklySpread(s_list),
        actionItems: analytics.generateWeeklyActions(s_list),
      }));
  } else if (period === "monthly") {
    const groups = {};
    sortedSightings.forEach((s) => {
      const key = new Date(s.createdAt).toISOString().slice(0, 7);
      if (!groups[key]) groups[key] = [];
      groups[key].push(s);
    });
    groupedData = Object.entries(groups)
      .slice(0, 4)
      .map(([month, s_list]) => ({
        period: new Date(month + "-02").toLocaleDateString("en-US", {
          month: "long",
          year: "numeric",
        }),
        count: s_list.length,
        species: [...new Set(s_list.map((s) => s.analysis.predictedSpecies))],
        environmentalSeverity: analytics.calculateEnvironmentalSeverity(s_list),
        controlEffectiveness: analytics.calculateControlEffectiveness(),
        seasonalFactors: analytics.getSeasonalFactors(month),
      }));
  } else if (period === "yearly") {
    const groups = {};
    sortedSightings.forEach((s) => {
      const key = new Date(s.createdAt).getFullYear().toString();
      if (!groups[key]) groups[key] = [];
      groups[key].push(s);
    });
    groupedData = Object.entries(groups)
      .slice(0, 3)
      .map(([year, s_list]) => ({
        period: year,
        count: s_list.length,
        species: [...new Set(s_list.map((s) => s.analysis.predictedSpecies))],
        climateCorrelation: analytics.getClimateCorrelation(),
        predictiveInsights: analytics.generatePredictiveInsights(),
        ecosystemRecovery: analytics.calculateEcosystemRecovery(s_list),
      }));
  }

  return groupedData
    .map((item) => {
      const riskColor = item.avgRisk
        ? analytics.getThreatColor(item.avgRisk)
        : "#67d4a7";
      return `
        <div class="timeline-item farmer-strategic"><div class="timeline-marker" style="background:${riskColor};"></div><div class="timeline-content" style="border-left-color:${riskColor};">
          <div class="timeline-header"><div class="timeline-date">${
            item.period
          }</div><div class="timeline-count" style="background:${riskColor};">${
        item.count
      } Invasive${item.count !== 1 ? "s" : ""}</div></div>
          ${renderPeriodSpecificData(item, period)}
          <div class="timeline-species"><strong>Top Species:</strong> ${item.species
            .slice(0, 3)
            .join(", ")}${
        item.species.length > 3 ? ` +${item.species.length - 3} more` : ""
      }</div>
        </div></div>`;
    })
    .join("");
};

// === GLOBAL EVENT HANDLERS ===
export const toggleLLMDropdown = (sightingId) => {
  const dropdown = document.querySelector(
    `[data-sighting-id="${sightingId}"] .llm-dropdown`
  );
  const content = document.querySelector(
    `[data-sighting-id="${sightingId}"] .llm-dropdown-content`
  );
  const arrow = document.querySelector(
    `[data-sighting-id="${sightingId}"] .llm-dropdown-arrow`
  );
  if (dropdown && content) {
    const isHidden = content.classList.toggle("llm-dropdown-content-hidden");
    if (arrow) arrow.textContent = isHidden ? "▼" : "▲";
    dropdown.classList.toggle("open", !isHidden);
  }
};

export const showLLMSection = (sightingId, section) => {
  const container = document.querySelector(
    `[data-sighting-id="${sightingId}"]`
  );
  if (!container) return;
  container
    .querySelectorAll(".llm-section-btn")
    .forEach((btn) => btn.classList.remove("active"));
  container.querySelector(`[onclick*="'${section}'"]`).classList.add("active");
  const sighting = (window.sightingsData || []).find(
    (s) => s._id === sightingId
  );
  if (sighting) {
    const content = container.querySelector(`#llm-content-${sightingId}`);
    if (content)
      content.innerHTML = llmRenderer.formatLLMSection(
        sighting.analysis?.llm,
        section
      );
  }
};

// Removed duplicate functions - using the ones defined earlier in the file

export const updateTimelineView = (period, buttonElement) => {
  document
    .querySelectorAll(".chart-control-btn")
    .forEach((btn) => btn.classList.remove("active"));
  buttonElement.classList.add("active");
  const sightings = window.sightingsData || [];
  const timelineContainer = document.getElementById("timeline-events");
  if (timelineContainer) {
    timelineContainer.innerHTML = generateInvasiveTimelineEvents(
      sightings,
      period
    );
  }
  const timePatterns = analytics.analyzeTimePatterns(sightings);
  charts.generateTimelineChart(timePatterns, "timeline-chart", period);
};

export const handleAlertAction = (alertId) => {
  showNotificationToast("Alert acknowledged.", "info");
  const alertCard = document.querySelector(`[data-alert-id="${alertId}"]`);
  if (alertCard) alertCard.style.opacity = "0.6";
};

export const handleAlertDismiss = async (alertId) => {
  const alertCard = document.querySelector(`[data-alert-id="${alertId}"]`);
  const alertTitle =
    alertCard?.querySelector(".alert-title")?.textContent || "";

  if (alertTitle.includes("Weather")) {
    showNotificationToast("Weather alerts cannot be dismissed.", "warning");
    return;
  }

  // Determine alert type
  const alertType = alertTitle.includes("Weather") ? "weather" : "risk";

  // Dismiss alert via API
  const success = await notifications.dismissNotification(alertType, alertId);

  if (success) {
    if (alertCard) alertCard.remove();
    showNotificationToast("Alert dismissed.", "info");
  } else {
    showNotificationToast("Failed to dismiss alert.", "warning");
  }
};

export const triggerWeatherAlert = async () => {
  // Weather alerts are now generated server-side automatically
  await notifications.refreshAlerts();
  showNotificationToast("Weather alerts refreshed", "info");
};

// === REMOVAL FUNCTIONALITY ===

let currentRemovalModal = null;
export const closeRemovalModal = () => {
  if (currentRemovalModal) {
    currentRemovalModal.remove();
    currentRemovalModal = null;
  }
};

export const confirmRemoval = async (sightingId) => {
  try {
    const response = await fetch(`/api/sightings/${sightingId}/remove`, {
      method: "PATCH",
      credentials: "include",
    });
    if (!response.ok) throw new Error("Server responded with an error.");

    const card = document.querySelector(`[data-sighting-id="${sightingId}"]`);
    if (card) card.remove();

    showNotification("Plant removal recorded successfully!", "success");

    const scrollPosition = window.pageYOffset;
    await load();
    window.scrollTo(0, scrollPosition);
  } catch (error) {
    console.error("Error removing sighting:", error);
    showNotification(`Failed to record removal: ${error.message}`, "error");
  }
  closeRemovalModal();
};

export const removeSighting = (sightingId, speciesName) => {
  const modal = document.createElement("div");
  modal.className = "removal-modal";
  modal.innerHTML = `
    <div class="removal-modal-content">
      <h3>Confirm Removal</h3>
      <p>Are you sure you want to mark this <strong>${speciesName}</strong> sighting as removed?</p>
      <p><em>This action indicates the plant has been physically eradicated.</em></p>
      <div class="removal-modal-buttons">
        <button class="btn-confirm" onclick="confirmRemoval('${sightingId}')">Confirm Removal</button>
        <button class="btn-cancel" onclick="closeRemovalModal()">Cancel</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  currentRemovalModal = modal;
};

// === MAIN LOAD FUNCTION ===

export const loadSightings = async () => {
  return await load();
};

export const load = async () => {
  const container = document.getElementById("sightings-container");
  const empty = document.getElementById("sightings-empty");
  container.innerHTML = "";
  await notifications.initializeNotifications();

  try {
    // Force fresh data load to ensure companion detections appear
    const { data } = await SightingsAPI.list("", { cache: false });
    if (!data || data.length === 0) {
      empty.style.display = "block";
      return;
    }

    empty.style.display = "none";
    window.sightingsData = data; // Store for global access

    // Set up SSE listener for new sightings (companion detections)
    if (!window.sightingsSSESetup) {
      window.sightingsSSESetup = true;
      if (window.eventSource) {
        window.eventSource.addEventListener("new_sighting", (event) => {
          console.log(
            "New sighting detected via SSE, refreshing sightings list"
          );
          // Refresh sightings list to show new companion detection
          setTimeout(() => load(), 1000); // Small delay to ensure backend processing is complete
        });
      }
    }

    const isInvasive = (s) => {
      const species = s.analysis?.predictedSpecies || "Unknown";
      return species !== "Unknown" && !species.includes("Unknown");
    };

    const invasiveSightings = data
      .filter(isInvasive)
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    const nonInvasiveSightings = data
      .filter((s) => !isInvasive(s))
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

    const renderSightingCard = (sighting, isInv) => {
      const card = document.createElement("div");
      card.className = `sighting-card ${
        isInv ? "invasive-sighting" : "non-invasive-sighting"
      }`;
      card.setAttribute("data-sighting-id", sighting._id);
      const riskLevel = sighting.analysis?.llm?.details?.risk_level || "Medium";
      const riskColor = riskLevel.toLowerCase().includes("high")
        ? "#dc2626"
        : riskLevel.toLowerCase().includes("medium")
        ? "#d97706"
        : "#10b981";

      card.innerHTML = `
                <button class="remove-btn" onclick="removeSighting('${
                  sighting._id
                }', '${
        sighting.analysis?.predictedSpecies || "Unknown"
      }')">×</button>
                <div class="sighting-header">
                    ${
                      sighting.imageUrl || sighting.imagePath
                        ? `<img src="${
                            sighting.imageUrl || sighting.imagePath
                          }" alt="Sighting" class="sighting-thumbnail" onclick="showImageModalWrapper('${
                            sighting.imageUrl || sighting.imagePath
                          }')" />`
                        : ""
                    }
                    <div class="sighting-info">
                        <h3 class="sighting-species">${
                          isInv
                            ? `<span class="species-label invasive-label">[INVASIVE]</span>`
                            : `<span class="species-label non-invasive-label">[NON-INVASIVE]</span>`
                        } ${
        sighting.analysis?.predictedSpecies || "Unknown"
      }</h3>
                        <div class="sighting-meta"><span>${utils.fmtDate(
                          sighting.capturedAt || sighting.createdAt
                        )}</span>${
        sighting.location?.coordinates
          ? `<span>${utils.fmtLatLng(sighting.location.coordinates)}</span>`
          : ""
      }</div>
                        <div class="sighting-badges">
                            <span class="sighting-badge confidence">${utils.pct(
                              sighting.analysis?.confidence
                            )} conf</span>
                            <span class="sighting-badge source">${
                              sighting.fromVideo ? "Live Video" : "Upload"
                            }</span>
                            ${
                              isInv
                                ? `<span class="sighting-badge risk-level" style="background-color: ${riskColor}; color: white;">Risk: ${riskLevel}</span>`
                                : ""
                            }
                        </div>
                    </div>
                </div>
                ${llmRenderer.createLLMDropdown(sighting)}`;
      container.appendChild(card);
      // Wrapper function to keep showImageModal private to the module
      window.showImageModalWrapper = showImageModal;
    };

    if (invasiveSightings.length > 0) {
      container.innerHTML += `<div class="section-header"><h2 class="invasive-species-header">Invasive Species Detected (${invasiveSightings.length})</h2></div>`;
      invasiveSightings.forEach((s) => renderSightingCard(s, true));
    }

    if (nonInvasiveSightings.length > 0) {
      container.innerHTML += `<div class="section-header non-invasive-header"><h2 class="non-invasive-species-header">Non-Invasive Species (${nonInvasiveSightings.length})</h2></div>`;
      nonInvasiveSightings.forEach((s) => renderSightingCard(s, false));
    }
  } catch (e) {
    console.error("Failed to load sightings", e);
    empty.textContent = "Failed to load sightings.";
    empty.style.display = "block";
  }
};
