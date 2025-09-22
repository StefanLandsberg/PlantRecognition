// js/sightings/charts.js

export const generateSimpleBarChart = (data, container) => {
  container.innerHTML = '';
  const maxValue = Math.max(...data.map(d => d.value));
  if (maxValue === 0 || data.length === 0) {
    container.innerHTML = '<div class="no-data-display">No data available</div>';
    return;
  }

  data.forEach(item => {
    const barWrapper = document.createElement('div');
    barWrapper.className = 'chart-bar-wrapper';

    const bar = document.createElement('div');
    bar.className = 'chart-bar';
    bar.style.height = `${(item.value / maxValue) * 100}%`;
    bar.style.backgroundColor = item.color || '#67d4a7';

    const value = document.createElement('div');
    value.className = 'chart-bar-value';
    value.textContent = item.value;
    bar.appendChild(value);

    const label = document.createElement('div');
    label.className = 'chart-bar-label';
    label.textContent = item.label;

    barWrapper.appendChild(bar);
    barWrapper.appendChild(label);
    container.appendChild(barWrapper);
  });
};

export const generateRiskChart = (riskAnalysis, containerId) => {
  const container = document.getElementById(containerId);
  if (!container) return;
  const data = [
    { label: 'High Risk', value: riskAnalysis.highRiskSpecies, color: '#ef4444' },
    { label: 'Medium Risk', value: Math.max(0, riskAnalysis.totalSpecies - riskAnalysis.highRiskSpecies), color: '#f59e0b' },
    { label: 'Low Risk', value: 0, color: '#10b981' }
  ];
  generateSimpleBarChart(data, container);
};

export const generateRiskBreakdownChart = (sightings, containerId) => {
  const container = document.getElementById(containerId);
  if (!container) return;
  const riskCounts = { high: 0, medium: 0, low: 0, unknown: 0 };
  sightings.forEach(s => {
    const risk = s.analysis?.llm?.details?.risk_level?.toLowerCase() || 'unknown';
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
};

export const generateDensityDistributionChart = (sightings, containerId) => {
  const container = document.getElementById(containerId);
  if (!container) return;
  const speciesData = {};
  sightings.forEach(s => {
    const species = s.analysis?.predictedSpecies || 'Unknown';
    speciesData[species] = (speciesData[species] || 0) + 1;
  });
  const data = Object.entries(speciesData)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 8)
    .map(([species, count]) => ({
      label: species.substring(0, 12) + (species.length > 12 ? '...' : ''),
      value: count,
      color: '#67d4a7'
    }));
  generateSimpleBarChart(data, container);
};

export const generateCoverageAnalysisChart = (coverageStats, containerId) => {
  const container = document.getElementById(containerId);
  if (!container) return;
  const data = [
    { label: 'Total Area (km²)', value: Math.round(coverageStats.totalArea), color: '#3b82f6' },
    { label: 'Avg Distance (km)', value: Math.round(coverageStats.avgDistance), color: '#10b981' },
    { label: 'Coverage Score', value: Math.round(coverageStats.totalArea / Math.max(coverageStats.avgDistance, 1)), color: '#f59e0b' }
  ];
  generateSimpleBarChart(data, container);
};

export const generateDistanceDistributionChart = (sightings, containerId, calculateDistance) => {
  const container = document.getElementById(containerId);
  if (!container) return;
  const validSightings = sightings.filter(s => s.location?.coordinates);
  const distances = [];
  for (let i = 0; i < validSightings.length; i++) {
    for (let j = i + 1; j < validSightings.length && j < i + 10; j++) { // Limit comparisons
      distances.push(Math.round(calculateDistance(validSightings[i].location.coordinates, validSightings[j].location.coordinates)));
    }
  }
  const ranges = { '0-1km': 0, '1-5km': 0, '5-10km': 0, '10-20km': 0, '20km+': 0 };
  distances.forEach(dist => {
    if (dist <= 1) ranges['0-1km']++;
    else if (dist <= 5) ranges['1-5km']++;
    else if (dist <= 10) ranges['5-10km']++;
    else if (dist <= 20) ranges['10-20km']++;
    else ranges['20km+']++;
  });
  const data = Object.entries(ranges).map(([range, count]) => ({ label: range, value: count, color: '#67d4a7' }));
  generateSimpleBarChart(data, container);
};

export const generateClusterSizeChart = (locationClusters, containerId) => {
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
};

export const generateTimelineChart = (timePatterns, containerId, period = 'daily') => {
  const container = document.getElementById(containerId);
  if (!container) return;
  let data = [];
  switch (period) {
    case 'daily':
      const days = Object.keys(timePatterns.dayCounts).sort((a,b) => new Date(a) - new Date(b)).slice(-7);
      data = days.map(day => ({
        label: new Date(day).toLocaleDateString('en', { weekday: 'short' }),
        value: timePatterns.dayCounts[day] || 0,
      }));
      break;
    case 'weekly':
      const weeks = Object.keys(timePatterns.weekCounts || {}).sort().slice(-6);
      data = weeks.map(week => ({
        label: `W ${new Date(week).getMonth() + 1}/${new Date(week).getDate()}`,
        value: timePatterns.weekCounts[week] || 0,
      }));
      break;
    case 'monthly':
      const months = Object.keys(timePatterns.monthCounts || {}).sort().slice(-6);
      data = months.map(month => ({
        label: new Date(month + '-02').toLocaleDateString('en', { month: 'short' }),
        value: timePatterns.monthCounts[month] || 0,
      }));
      break;
    case 'yearly':
      const years = Object.keys(timePatterns.yearCounts || {}).sort().slice(-5);
      data = years.map(year => ({ label: year, value: timePatterns.yearCounts[year] || 0 }));
      break;
  }
  container.style.height = '200px';
  generateSimpleBarChart(data, container);
};

export const generateHourlyChart = (timePatterns, containerId) => {
  const container = document.getElementById(containerId);
  if (!container) return;
  const data = timePatterns.hourCounts.map((count, hour) => ({
    label: `${hour}h`,
    value: count,
    color: hour === timePatterns.peakHour ? '#ef4444' : '#67d4a7'
  })).filter((_, hour) => hour % 2 === 0);
  generateSimpleBarChart(data, container);
};

export const generateSeasonalChart = (seasonalData, containerId) => {
  const container = document.getElementById(containerId);
  if (!container) return;
  const data = Object.keys(seasonalData.seasonCounts).map(season => ({
    label: season,
    value: seasonalData.seasonCounts[season],
    color: season === seasonalData.dominantSeason ? '#ef4444' : '#67d4a7'
  }));
  generateSimpleBarChart(data, container);
};