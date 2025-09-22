// js/sightings/utils.js

/**
 * Formats a date string into a localized string.
 * @param {string} s - The date string.
 * @returns {string} The formatted date string.
 */
export const fmtDate = (s) => {
  try {
    return new Date(s).toLocaleString();
  } catch {
    return s || "";
  }
};

/**
 * Formats coordinates into a "lat, lng" string.
 * @param {number[]} coords - An array containing [longitude, latitude].
 * @returns {string} The formatted coordinate string.
 */
export const fmtLatLng = (coords) => {
  if (!Array.isArray(coords) || coords.length < 2) return "";
  const [lng, lat] = coords;
  return `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
};

/**
 * Formats a number into a percentage string.
 * @param {number} n - The number (0 to 1).
 * @returns {string} The formatted percentage string.
 */
export const pct = (n) => {
  if (n == null || isNaN(n)) return "Unknown";
  return `${(Number(n) * 100).toFixed(1)}%`;
};

/**
 * Calculates the distance between two geographic coordinates using the Haversine formula.
 * @param {number[]} coord1 - The first coordinate [lng, lat].
 * @param {number[]} coord2 - The second coordinate [lng, lat].
 * @returns {number} The distance in kilometers.
 */
export const calculateDistance = (coord1, coord2) => {
  const R = 6371; // Earth's radius in km
  const dLat = ((coord2[1] - coord1[1]) * Math.PI) / 180;
  const dLon = ((coord2[0] - coord1[0]) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((coord1[1] * Math.PI) / 180) *
      Math.cos((coord2[1] * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
};

/**
 * Returns the CSS class for a risk level badge.
 * @param {string} riskLevel - The risk level string.
 * @returns {string} The corresponding CSS class.
 */
export const getRiskBadgeClass = (riskLevel) => {
  const risk = riskLevel.toLowerCase();
  if (risk.includes('high') || risk.includes('severe')) return 'risk-high';
  if (risk.includes('medium') || risk.includes('moderate')) return 'risk-medium';
  if (risk.includes('low') || risk.includes('minimal')) return 'risk-low';
  return 'risk-unknown';
};