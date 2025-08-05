// Map Tab Manager
import { api } from '../api.js';
import { appState } from '../state.js';
import { sanitizeHtml } from '../../utils/sanitize.js';
import { handleError } from '../utils.js';

class MapManager {
    constructor() {
        this.map = null;
        this.sightingsData = [];
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        try {
            console.log('Initializing Map Manager...');
            await this.loadSightingsData();
            this.initialized = true;
            console.log('Map Manager initialized successfully');
        } catch (error) {
            console.error('Error initializing Map Manager:', error);
            handleError(error, 'map_initialization');
        }
    }

    async loadSightingsData() {
        try {
            const response = await api.getSightings();
            if (response.success && response.data) {
                this.sightingsData = response.data;
                console.log(`Loaded ${this.sightingsData.length} sightings for map`);
            }
        } catch (error) {
            console.error('Error loading sightings data for map:', error);
        }
    }

    async refreshData() {
        await this.loadSightingsData();
        if (this.map) {
            this.updateMapPins();
        }
    }

    onTabActivated() {
        this.initializeMap();
    }

    initializeMap() {
        if (this.map) return;
        
        try {
            this.map = L.map('map').setView([-25.8408448, 28.2394624], 13);
            
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(this.map);
            
            this.updateMapPins();
            this.setupMapControls();
            
            setTimeout(() => {
                this.map.invalidateSize();
            }, 500);
            
        } catch (error) {
            console.error('Error initializing map:', error);
        }
    }

    updateMapPins() {
        if (!this.map) return;
        
        try {
            // Clear existing pins
            this.clearMapPins();
            
            // Add pins for all sightings
            this.sightingsData.forEach(sighting => {
                if (sighting.location && sighting.location.coordinates) {
                    const [lng, lat] = sighting.location.coordinates;
                    this.createDualCirclePin(sighting, lat, lng);
                }
            });
            
            // Update legend
            this.updateMapLegend();
        } catch (error) {
            console.error('Error updating map pins:', error);
        }
    }

    clearMapPins() {
        this.map.eachLayer((layer) => {
            if (layer instanceof L.CircleMarker || layer instanceof L.Circle || layer instanceof L.Marker) {
                this.map.removeLayer(layer);
            }
        });
    }

    createDualCirclePin(sighting, lat, lng) {
        const isInvasive = sighting.llmAnalysis?.invasive_status === true;
        const detectionDate = new Date(sighting.timestamp);
        const now = new Date();
        const ageInDays = (now - detectionDate) / (1000 * 60 * 60 * 24);
        
        // Determine pin class based on invasive status
        let pinClass = 'map-pin unknown';
        if (isInvasive) {
            pinClass = 'map-pin invasive';
        } else if (sighting.llmAnalysis?.invasive_status === false) {
            pinClass = 'map-pin native';
        }
        
        // Create custom HTML for the pin
        const pinHtml = `
            <div class="${pinClass}" style="width: 20px; height: 20px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 10px; font-weight: bold;">
                <i class="fas fa-leaf"></i>
            </div>
        `;
        
        // Create custom icon
        const customIcon = L.divIcon({
            html: pinHtml,
            className: 'custom-map-pin',
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });
        
        // Create marker with custom icon
        const marker = L.marker([lat, lng], { icon: customIcon }).addTo(this.map);
        
        // Bind popup to marker
        marker.bindPopup(this.createPinPopup(sighting));
    }

    updateMapLegend() {
        const legendContainer = document.getElementById('map-legend');
        if (!legendContainer) return;
        
        legendContainer.innerHTML = `
            <h4>Map Legend</h4>
            <div class="legend-section">
                <h5>Pin Colors (Invasive Status)</h5>
                <div class="legend-item">
                    <div class="legend-color map-pin invasive" style="width: 16px; height: 16px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 8px;">
                        <i class="fas fa-leaf"></i>
                    </div>
                    <span>Invasive Species</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color map-pin native" style="width: 16px; height: 16px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 8px;">
                        <i class="fas fa-leaf"></i>
                    </div>
                    <span>Native Species</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color map-pin unknown" style="width: 16px; height: 16px; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 8px;">
                        <i class="fas fa-leaf"></i>
                    </div>
                    <span>Unknown Status</span>
                </div>
            </div>
            <div class="legend-section">
                <h5>Pin Features</h5>
                <div class="legend-item">
                    <span><i class="fas fa-leaf" style="color: #28a745;"></i> Leaf Icon</span>
                </div>
                <div class="legend-item">
                    <span><i class="fas fa-pulse" style="color: #ff6b6b;"></i> Pulsing Animation</span>
                </div>
                <div class="legend-item">
                    <span><i class="fas fa-expand" style="color: #74c0fc;"></i> Hover to Zoom</span>
                </div>
            </div>
        `;
    }

    getPinColor(sighting) {
        const isInvasive = sighting.llmAnalysis?.invasive_status === true;
        const riskLevel = sighting.llmAnalysis?.risk_level;
        
        if (!isInvasive) {
            return sighting.llmAnalysis?.invasive_status === false ? 'green' : 'gray';
        }
        
        // For invasive species, calculate age and adjust color
        const detectionDate = new Date(sighting.timestamp);
        const now = new Date();
        const ageInDays = (now - detectionDate) / (1000 * 60 * 60 * 24);
        
        // Color progression: yellow -> orange -> red (older = redder)
        if (ageInDays <= 7) {
            return 'yellow'; // New detections (0-7 days)
        } else if (ageInDays <= 30) {
            return 'orange'; // Recent detections (8-30 days)
        } else {
            return 'red'; // Old detections (30+ days)
        }
    }

    getCircleColor(colorName) {
        const colors = {
            'red': '#dc3545',
            'orange': '#fd7e14',
            'yellow': '#ffc107',
            'green': '#28a745',
            'gray': '#6c757d'
        };
        return colors[colorName] || '#6c757d';
    }

    createPinPopup(sighting) {
        const status = sighting.llmAnalysis?.invasive_status === true ? 'Invasive' : 
                      sighting.llmAnalysis?.invasive_status === false ? 'Native' : 'Unknown';
        const riskLevel = sighting.llmAnalysis?.risk_level || 'Unknown';
        const nembaCategory = sighting.llmAnalysis?.advisory_content?.legal_status?.nemba_category || 'Unknown';
        const ageInDays = this.getDetectionAge(sighting.timestamp);
        
        return `
            <div class="pin-popup">
                <h3>${sanitizeHtml(sighting.species)}</h3>
                <p><strong>Status:</strong> ${sanitizeHtml(status)}</p>
                <p><strong>Risk Level:</strong> ${sanitizeHtml(riskLevel)}</p>
                <p><strong>NEMBA Category:</strong> ${sanitizeHtml(nembaCategory)}</p>
                <p><strong>Date:</strong> ${this.formatDate(sighting.timestamp)}</p>
                <p><strong>Age:</strong> ${ageInDays} days</p>
                <p><strong>Confidence:</strong> ${(sighting.confidence * 100).toFixed(1)}%</p>
                <p><strong>Location:</strong> ${this.formatLocation(sighting.latitude, sighting.longitude)}</p>
                ${sighting.imageUrl ? `<img src="${sighting.imageUrl}" alt="Detection" style="max-width: 200px; margin-top: 10px;">` : ''}
            </div>
        `;
    }

    getDetectionAge(timestamp) {
        const detectionDate = new Date(timestamp);
        const now = new Date();
        return Math.floor((now - detectionDate) / (1000 * 60 * 60 * 24));
    }

    setupMapControls() {
        // Center on me button
        const centerBtn = document.getElementById('center-map-btn');
        if (centerBtn) {
            centerBtn.addEventListener('click', () => {
                const location = appState.get('currentLocation');
                if (location && this.map) {
                    this.map.setView([location.lat, location.lng], 15);
                }
            });
        }
        
        // Clear pins button
        const clearBtn = document.getElementById('clear-pins-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearMapPins();
            });
        }
    }

    formatDate(dateString) {
        if (!dateString) return 'Unknown';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }

    formatLocation(lat, lng) {
        if (!lat || !lng) return 'Unknown';
        return `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
    }

    // Stats methods for admin manager
    getSightingsCount() {
        return this.sightingsData.length;
    }

    getPendingCount() {
        return this.sightingsData.filter(s => s.management?.status === 'pending').length;
    }

    getHighRiskCount() {
        return this.sightingsData.filter(s => 
            s.llmAnalysis?.risk_level === 'High' || s.llmAnalysis?.risk_level === 'Critical'
        ).length;
    }

    getRemovalSuccessCount() {
        return this.sightingsData.filter(s => s.management?.status === 'completed').length;
    }
}

// Create singleton instance
export const mapManager = new MapManager(); 