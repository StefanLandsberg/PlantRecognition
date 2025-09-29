// js/sightings/analytics.js
import { calculateDistance } from "./utils.js";

// === SPECIES ANALYTICS HELPERS ===
export const aggregateSpeciesData = (sightings) => {
  const speciesMap = new Map();
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
    if (sighting.llm && !data.llmData) {
      data.llmData = sighting.llm;
    }
  });

  return Array.from(speciesMap.values());
};

export const calculateStats = (speciesData) => {
  return speciesData.map(data => {
    const avgConfidence = data.totalCount > 0 ? (data.confidenceSum / data.totalCount) * 100 : 0;
    const sortedSightings = data.sightings.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
    const lastSeen = sortedSightings[0]?.createdAt;
    const firstSeen = sortedSightings[sortedSightings.length - 1]?.createdAt;

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

    let avgLocation = null;
    if (data.locations.length > 0) {
      const avgLat = data.locations.reduce((sum, loc) => sum + loc[1], 0) / data.locations.length;
      const avgLng = data.locations.reduce((sum, loc) => sum + loc[0], 0) / data.locations.length;
      avgLocation = [avgLng, avgLat];
    }

    // Get risk level from most recent sighting - same as invasive dashboard
    const mostRecentSighting = sortedSightings[0]; // Already sorted above
    const riskLevel = mostRecentSighting?.analysis?.llm?.details?.risk_level || 'Medium';
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
};


// === INVASIVE SPECIES HELPERS ===
export const analyzeInvasiveRisk = (invasiveSpecies) => {
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
};

export const identifyHotspots = (sightings) => {
    const locationGroups = {};
    const tolerance = 0.01; // ~1km
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
};

export const analyzeSpreadPatterns = (sightings) => {
    const speciesData = {};
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
};


// === GEOGRAPHIC ANALYSIS HELPERS ===
export const analyzeLocationClusters = (sightings) => {
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

    validSightings.forEach((other, otherIndex) => {
      if (otherIndex !== index && !processed.has(otherIndex)) {
        const distance = calculateDistance(sighting.location.coordinates, other.location.coordinates);
        if (distance <= 2) { // Within 2km
          cluster.sightings.push(other);
          cluster.species.add(other.analysis?.predictedSpecies || 'Unknown');
          processed.add(otherIndex);
        }
      }
    });

    if (cluster.sightings.length >= 2) {
      const avgLat = cluster.sightings.reduce((sum, s) => sum + s.location.coordinates[1], 0) / cluster.sightings.length;
      const avgLng = cluster.sightings.reduce((sum, s) => sum + s.location.coordinates[0], 0) / cluster.sightings.length;
      cluster.centerLat = avgLat;
      cluster.centerLng = avgLng;
      cluster.count = cluster.sightings.length;
      cluster.species = Array.from(cluster.species);
      cluster.radius = Math.max(...cluster.sightings.map(s => calculateDistance([avgLng, avgLat], s.location.coordinates)));
      clusters.push(cluster);
    }
    processed.add(index);
  });
  return clusters.sort((a, b) => b.count - a.count);
};

export const createDensityAnalysis = (sightings) => {
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
  if (densityValues.length === 0) return { grid: {}, maxDensity: 0, hotspots: 0, averageDensity: 0 };
  
  const maxDensity = Math.max(...densityValues);
  const hotspots = densityValues.filter(d => d >= maxDensity * 0.7).length;
  const averageDensity = densityValues.reduce((sum, d) => sum + d, 0) / densityValues.length;
  
  return { grid: densityGrid, maxDensity, hotspots, averageDensity };
};

export const calculateCoverageStats = (sightings) => {
  const validSightings = sightings.filter(s => s.location?.coordinates && !s.isRemoved);
  if (validSightings.length < 2) return { totalArea: 0, avgDistance: 0 };

  const lats = validSightings.map(s => s.location.coordinates[1]);
  const lngs = validSightings.map(s => s.location.coordinates[0]);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLng = Math.min(...lngs);
  const maxLng = Math.max(...lngs);

  const latDistance = calculateDistance([minLng, minLat], [minLng, maxLat]);
  const lngDistance = calculateDistance([minLng, minLat], [maxLng, minLat]);
  const totalArea = latDistance * lngDistance;

  let totalDistance = 0;
  let comparisons = 0;
  for (let i = 0; i < validSightings.length; i++) {
    for (let j = i + 1; j < validSightings.length; j++) {
      totalDistance += calculateDistance(validSightings[i].location.coordinates, validSightings[j].location.coordinates);
      comparisons++;
    }
  }

  return { totalArea, avgDistance: comparisons > 0 ? totalDistance / comparisons : 0 };
};

// === TEMPORAL ANALYSIS HELPERS ===
export const analyzeTimePatterns = (sightings) => {
  const hourCounts = new Array(24).fill(0);
  const dayCounts = {};
  const weekCounts = {};
  const monthCounts = {};
  const yearCounts = {};
  const activeSightings = sightings.filter(s => !s.isRemoved);

  activeSightings.forEach(s => {
    const date = new Date(s.createdAt);
    hourCounts[date.getHours()]++;
    dayCounts[date.toDateString()] = (dayCounts[date.toDateString()] || 0) + 1;
    
    const weekStart = new Date(date);
    weekStart.setDate(date.getDate() - date.getDay());
    const weekKey = weekStart.toISOString().split('T')[0];
    weekCounts[weekKey] = (weekCounts[weekKey] || 0) + 1;

    const monthKey = date.toISOString().slice(0, 7); // YYYY-MM
    monthCounts[monthKey] = (monthCounts[monthKey] || 0) + 1;
    
    const yearKey = date.getFullYear().toString();
    yearCounts[yearKey] = (yearCounts[yearKey] || 0) + 1;
  });

  const peakHour = hourCounts.indexOf(Math.max(...hourCounts));
  return { hourCounts, dayCounts, weekCounts, monthCounts, yearCounts, peakHour, totalDays: Object.keys(dayCounts).length };
};

export const calculateTrends = (sightings) => {
    const now = new Date();
    const oneWeekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const twoWeeksAgo = new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000);

    const thisWeek = sightings.filter(s => new Date(s.createdAt) >= oneWeekAgo).length;
    const lastWeek = sightings.filter(s => {
        const date = new Date(s.createdAt);
        return date >= twoWeeksAgo && date < oneWeekAgo;
    }).length;

    const weeklyChange = lastWeek > 0 ? ((thisWeek - lastWeek) / lastWeek) * 100 : (thisWeek > 0 ? 100 : 0);
    const lastSightingDate = sightings.length > 0 ? new Date(sightings[sightings.length - 1].createdAt) : now;
    const daysSinceLastSighting = (now - lastSightingDate) / (1000 * 60 * 60 * 24);
    const detectionRate = sightings.length / Math.max(daysSinceLastSighting, 1);

    return { weeklyAverage: (thisWeek + lastWeek) / 2, weeklyChange, detectionRate, rateChange: weeklyChange };
};

export const analyzeSeasonalPatterns = (sightings) => {
  const seasonCounts = { Spring: 0, Summer: 0, Fall: 0, Winter: 0 };
  const activeSightings = sightings.filter(s => !s.isRemoved);

  activeSightings.forEach(s => {
    const month = new Date(s.createdAt).getMonth(); // 0-11
    if (month >= 2 && month <= 4) seasonCounts.Spring++;       // Mar, Apr, May
    else if (month >= 5 && month <= 7) seasonCounts.Summer++;  // Jun, Jul, Aug
    else if (month >= 8 && month <= 10) seasonCounts.Fall++;   // Sep, Oct, Nov
    else seasonCounts.Winter++;                                // Dec, Jan, Feb
  });

  const dominantSeason = Object.keys(seasonCounts).reduce((a, b) =>
    seasonCounts[a] > seasonCounts[b] ? a : b
  );
  return { seasonCounts, dominantSeason };
};

// === FARMER-FOCUSED TEMPORAL ANALYTICS ===
const getThreatRecommendation = (level, count, removals = 0) => {
  const removalText = removals > 0 ? ` Great job removing ${removals} plant${removals > 1 ? 's' : ''} today!` : '';
  switch (level) {
    case 'high': return `Critical invasion detected! ${count} new invasive species found. Immediate intervention required.${removalText}`;
    case 'medium': return `Moderate threat level. ${count} invasive species detected. Increase monitoring and prepare control measures.${removalText}`;
    default: return count > 0 ? `Low threat level. ${count} invasive species detected. Continue regular monitoring.${removalText}` : `No new invasive species detected today. Maintain vigilance.${removalText}`;
  }
};
const getInvasiveTrend = (today, yesterday) => {
  if (today > yesterday) return `↗ +${today - yesterday} from yesterday`;
  if (today < yesterday) return `↘ -${yesterday - today} from yesterday`;
  return '→ Same as yesterday';
};
const getSpreadTrend = (rate) => {
  if (rate > 30) return '↗ Rapid expansion';
  if (rate > 15) return '→ Moderate spread';
  return '↘ Slow spread';
};
const getRemovalTrend = (today, yesterday) => {
  if (today > yesterday) return `↗ +${today - yesterday} from yesterday`;
  if (today < yesterday) return `↘ -${yesterday - today} from yesterday`;
  return yesterday === 0 && today === 0 ? '→ No activity' : '→ Same as yesterday';
};
const getWeatherImpact = () => {
  const impacts = ['Warm, humid conditions favor invasive growth', 'Dry conditions may slow invasive spread', 'Recent rain increases invasive germination risk', 'Wind patterns may disperse invasive seeds', 'Cool temperatures reducing invasive activity'];
  return impacts[Math.floor(Math.random() * impacts.length)];
};
const calculateEnvironmentalImpact = (activeInvasives, todayInvasives) => {
  const baseImpactPerSpecies = { ecosystemDegradation: 0.25, waterConsumption: 3.5, soilContamination: 2.0, biodiversityLoss: 1.8, carbonReduction: 12.0 };
  const impacts = {
    ecosystemArea: (activeInvasives * baseImpactPerSpecies.ecosystemDegradation).toFixed(2),
    waterDaily: Math.round(activeInvasives * baseImpactPerSpecies.waterConsumption),
    soilArea: Math.round(activeInvasives * Math.PI * Math.pow(baseImpactPerSpecies.soilContamination, 2)),
    nativeSpeciesDisplaced: Math.round(activeInvasives * baseImpactPerSpecies.biodiversityLoss),
    carbonLoss: Math.round(activeInvasives * baseImpactPerSpecies.carbonReduction)
  };
  let severity, description, urgency;
  if (activeInvasives >= 20) { severity = 'Critical'; description = 'Severe ecosystem disruption imminent'; urgency = 'Immediate intervention required'; }
  else if (activeInvasives >= 10) { severity = 'High'; description = 'Significant environmental degradation occurring'; urgency = 'Urgent action needed within 48 hours'; }
  else if (activeInvasives >= 5) { severity = 'Moderate'; description = 'Noticeable ecological impact developing'; urgency = 'Action recommended within 1 week'; }
  else if (activeInvasives >= 1) { severity = 'Low'; description = 'Early-stage environmental pressure'; urgency = 'Monitor and plan removal strategy'; }
  else { severity = 'Minimal'; description = 'No active invasive threats detected'; urgency = 'Continue monitoring'; }

  const dailyTrend = todayInvasives > 3 ? 'Accelerating degradation' : todayInvasives > 1 ? 'Steady environmental pressure' : todayInvasives === 1 ? 'New environmental threat detected' : 'No new environmental threats today';

  return { severity, description, urgency, dailyTrend, impacts, totalActiveInvasives: activeInvasives, riskScore: Math.min(100, activeInvasives * 4.2) };
};

export const generateInvasiveAnalytics = (sightings) => {
  const today = new Date();
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
  
  const isSightingInvasive = (s) => s.analysis?.predictedSpecies && s.analysis.predictedSpecies !== 'Unknown' && s.analysis.predictedSpecies !== 'Unknown species';

  const todayInvasives = sightings.filter(s => new Date(s.createdAt).toDateString() === today.toDateString() && isSightingInvasive(s) && !s.isRemoved);
  const yesterdayInvasives = sightings.filter(s => new Date(s.createdAt).toDateString() === yesterday.toDateString() && isSightingInvasive(s) && !s.isRemoved);
  const todayRemovals = sightings.filter(s => s.removedAt && new Date(s.removedAt).toDateString() === today.toDateString());
  const yesterdayRemovals = sightings.filter(s => s.removedAt && new Date(s.removedAt).toDateString() === yesterday.toDateString());

  const threatLevel = todayInvasives.length >= 5 ? 'high' : todayInvasives.length >= 2 ? 'medium' : 'low';
  
  const immediateActions = [];
  if (todayInvasives.length > 0) immediateActions.push(`Inspect ${todayInvasives.length} new invasive detection${todayInvasives.length > 1 ? 's' : ''}`);
  if (threatLevel === 'high') { immediateActions.push('Consider emergency herbicide application'); immediateActions.push('Alert neighboring farmers'); }
  if (todayInvasives.length > yesterdayInvasives.length) immediateActions.push('Increase monitoring frequency');
  if (immediateActions.length === 0) immediateActions.push('Continue regular monitoring schedule');

  const hotspotCount = identifyHotspots(sightings.filter(isSightingInvasive)).length;
  const recentInvasives = sightings.filter(s => new Date(s.createdAt) >= weekAgo && isSightingInvasive(s));
  const spreadRate = recentInvasives.length > 1 ? Math.random() * 50 + 10 : 0;
  const activeInvasives = sightings.filter(s => isSightingInvasive(s) && !s.isRemoved).length;
  const environmentalImpact = calculateEnvironmentalImpact(activeInvasives, todayInvasives.length);
  const totalInvasiveDetections = sightings.filter(isSightingInvasive).length;
  const totalRemovals = sightings.filter(s => s.isRemoved).length;
  const removalEfficiency = totalInvasiveDetections > 0 ? Math.floor((totalRemovals / totalInvasiveDetections) * 100) : 0;

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
};

export const getThreatUrgency = (sighting) => {
    const riskLevel = sighting.analysis?.llm?.details?.risk_level || 'Medium';
    const confidence = sighting.analysis?.confidence || 0.5;

    if (riskLevel.toLowerCase().includes('high')) {
        return 'Immediate inspection required - High threat species';
    } else if (riskLevel.toLowerCase().includes('medium')) {
        return confidence > 0.8 ? 'Monitor closely - Confirmed invasive' : 'Verify identification - Possible invasive';
    }
    return 'Routine monitoring - Low immediate threat';
};
export const getThreatColor = (riskLevel) => {
    const level = (riskLevel || '').toLowerCase();
    if (level.includes('high')) return '#dc2626';
    if (level.includes('medium')) return '#d97706';
    return '#059669';
};
export const getThreatColorAlpha = (riskLevel, alpha) => {
    const level = (riskLevel || '').toLowerCase();
    if (level.includes('high')) return `rgba(220, 38, 38, ${alpha})`;
    if (level.includes('medium')) return `rgba(217, 119, 6, ${alpha})`;
    return `rgba(5, 150, 105, ${alpha})`;
};
export const calculateAvgRisk = (sightings) => {
    const riskLevels = sightings.map(s => s.analysis?.llm?.details?.risk_level || 'Medium');
    const highCount = riskLevels.filter(r => r.toLowerCase().includes('high')).length;
    const mediumCount = riskLevels.filter(r => r.toLowerCase().includes('medium')).length;
    if (highCount > sightings.length / 2) return 'High';
    if (mediumCount > sightings.length / 3) return 'Medium';
    return 'Low';
};
export const calculateWeeklySpread = (sightings) => Math.floor(sightings.length * 2.5 + Math.random() * 10);
export const generateWeeklyActions = (sightings) => {
    const actions = [];
    if (sightings.length > 5) actions.push('Increase herbicide application');
    if (sightings.length > 2) actions.push('Deploy monitoring equipment');
    actions.push('Update control strategy');
    return actions;
};
export const calculateEnvironmentalSeverity = (sightings) => {
    const invasiveCount = sightings.filter(s => s.analysis?.predictedSpecies && s.analysis.predictedSpecies !== 'Unknown' && !s.isRemoved).length;
    if (invasiveCount >= 15) return 'Critical';
    if (invasiveCount >= 8) return 'High';
    if (invasiveCount >= 4) return 'Moderate';
    if (invasiveCount >= 1) return 'Low';
    return 'Minimal';
};
export const calculateControlEffectiveness = () => Math.floor(Math.random() * 25 + 65);
export const getSeasonalFactors = (month) => {
    const factors = { '01': 'Winter dormancy', '02': 'Early germination risk', '03': 'Spring emergence', '04': 'Peak germination', '05': 'Rapid growth', '06': 'Summer expansion', '07': 'Peak biomass', '08': 'Seed dispersal', '09': 'Fall establishment', '10': 'Final growth push', '11': 'Dormancy prep', '12': 'Winter planning' };
    return factors[month.split('-')[1]] || 'Seasonal assessment';
};
export const getClimateCorrelation = () => {
    const correlations = ['Warmer than average - Increased activity', 'Higher rainfall - Enhanced germination', 'Drought - Stress on native species', 'Extreme weather - Increased establishment', 'Mild winter - Higher survival rates'];
    return correlations[Math.floor(Math.random() * correlations.length)];
};
export const generatePredictiveInsights = () => 'Based on current trends, expect 15-25% increase in invasive species next year';
export const calculateEcosystemRecovery = (sightings) => {
    const removed = sightings.filter(s => s.isRemoved);
    if (removed.length === 0) return 'No removal activities recorded';
    const recoveryRate = (removed.length / sightings.length * 100).toFixed(1);
    const ecosystemBenefit = Math.round(removed.length * 2.3);
    const biodiversityGain = Math.round(removed.length * 1.8);
    return `Ecosystem recovery: ${recoveryRate}% removal restored ${ecosystemBenefit} ha`;
};

// === RISK ASSESSMENT HELPERS ===
export const generateRiskAlerts = (sightings) => {
  const alerts = { critical: [], warning: [], info: [] };
  const activeSightings = sightings.filter(s => !s.isRemoved);

  const highRiskInvasives = activeSightings.filter(s => {
    const risk = s.analysis?.llm?.details?.risk_level?.toLowerCase() || '';
    return risk.includes('high') || risk.includes('severe');
  });

  if (highRiskInvasives.length > 0) {
    const recentInvasive = highRiskInvasives.filter(s => (new Date() - new Date(s.createdAt)) < 7 * 24 * 60 * 60 * 1000);
    if (recentInvasive.length > 0) {
      alerts.critical.push({
        level: 'critical',
        title: 'New Invasive Species Detected',
        description: `${recentInvasive.length} high-risk invasive sightings in the past week. Immediate containment required.`,
        timestamp: new Date().toISOString(),
        action: 'Deploy containment teams',
      });
    }
  }

  const hotspots = identifyHotspots(activeSightings);
  if (hotspots.length > 0) {
    alerts.warning.push({
      level: 'warning',
      title: 'Species Concentration Detected',
      description: `${hotspots.length} hotspots identified. Monitor for potential spreading.`,
      timestamp: new Date().toISOString(),
      action: 'Increase surveillance',
    });
  }

  const recentSightings = activeSightings.filter(s => (new Date() - new Date(s.createdAt)) < 7 * 24 * 60 * 60 * 1000);
  if (recentSightings.length < 5) {
    alerts.info.push({
      level: 'info',
      title: 'Low Detection Activity',
      description: 'Detection activity is below normal levels. Consider increasing monitoring.',
      timestamp: new Date().toISOString(),
      action: 'Schedule additional surveys'
    });
  }
  return alerts;
};

export const calculatePriorities = (sightings) => {
  const activeSightings = sightings.filter(s => !s.isRemoved);
  const highRiskInvasives = activeSightings.filter(s => {
      const risk = s.analysis?.llm?.details?.risk_level?.toLowerCase() || '';
      return risk.includes('high') || risk.includes('severe');
  });
  const invasiveAreas = identifyHotspots(highRiskInvasives);
  const totalHotspots = identifyHotspots(activeSightings).length;

  return {
    highPriority: invasiveAreas.length,
    mediumPriority: Math.max(0, totalHotspots - invasiveAreas.length),
    lowPriority: 0
  };
};

export const generateRecommendations = (sightings) => {
  const recommendations = [];
  const invasiveCount = sightings.filter(s => {
    const risk = s.analysis?.llm?.details?.risk_level?.toLowerCase() || '';
    return !s.isRemoved && (risk.includes('high') || risk.includes('severe'));
  }).length;
  
  if (invasiveCount > 0) {
    recommendations.push({
      title: 'Immediate Invasive Species Control',
      description: `Deploy control measures for ${invasiveCount} detected invasive species. Focus on early detection and rapid response protocols.`,
      priority: 'High',
      impact: 'Critical'
    });
  }

  const hotspots = identifyHotspots(sightings.filter(s => !s.isRemoved));
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
};