// Species Tab Manager
import { api } from '../api.js';
import { sanitizeHtml } from '../../utils/sanitize.js';
import { handleError } from '../utils.js';

class SpeciesManager {
    constructor() {
        this.speciesData = [];
        this.currentFilters = {};
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        try {
            console.log('Initializing Species Manager...');
            await this.loadSpeciesData();
            this.initialized = true;
            console.log('Species Manager initialized successfully');
        } catch (error) {
            console.error('Error initializing Species Manager:', error);
            handleError(error, 'species_initialization');
        }
    }

    async loadSpeciesData() {
        try {
            // Get sightings data to build species encyclopedia
            const response = await api.getSightings();
            if (response.success && response.data) {
                this.buildSpeciesFromSightings(response.data);
            }
        } catch (error) {
            console.error('Error loading species data:', error);
        }
    }

    buildSpeciesFromSightings(sightingsData) {
        try {
            // Group sightings by species to create species encyclopedia
            const speciesMap = new Map();
            
            sightingsData.forEach(sighting => {
                if (!speciesMap.has(sighting.species)) {
                    speciesMap.set(sighting.species, {
                        species: sighting.species,
                        scientificName: sighting.species,
                        commonName: sighting.llmAnalysis?.common_name || sighting.species,
                        family: sighting.llmAnalysis?.family || 'Unknown',
                        invasiveStatus: sighting.llmAnalysis?.invasive_status || false,
                        riskLevel: sighting.llmAnalysis?.risk_level || 'Unknown',
                        nembaCategory: sighting.llmAnalysis?.advisory_content?.legal_status?.nemba_category || 'Unknown',
                        description: sighting.llmAnalysis?.description || '',
                        treatment: sighting.llmAnalysis?.treatment || '',
                        origin: sighting.llmAnalysis?.origin || 'Unknown',
                        detectionCount: 0,
                        latestDetection: null,
                        images: []
                    });
                }
                
                const species = speciesMap.get(sighting.species);
                species.detectionCount++;
                
                if (!species.latestDetection || new Date(sighting.timestamp) > new Date(species.latestDetection)) {
                    species.latestDetection = sighting.timestamp;
                }
                
                if (sighting.imageUrl && !species.images.includes(sighting.imageUrl)) {
                    species.images.push(sighting.imageUrl);
                }
            });
            
            this.speciesData = Array.from(speciesMap.values());
            console.log(`Built ${this.speciesData.length} unique species from sightings`);
        } catch (error) {
            console.error('Error building species data:', error);
        }
    }

    async refreshData() {
        await this.loadSpeciesData();
    }

    onTabActivated() {
        this.updateSpeciesEncyclopedia();
    }

    updateSpeciesEncyclopedia() {
        const speciesGrid = document.getElementById('species-grid');
        if (!speciesGrid) return;
        
        const filteredSpecies = this.getFilteredSpecies();
        
        if (filteredSpecies.length === 0) {
            speciesGrid.innerHTML = `
                <div class="species-empty">
                    <i class="fas fa-search"></i>
                    <h3>No Species Found</h3>
                    <p>Try adjusting your search criteria</p>
                </div>
            `;
            return;
        }
        
        const speciesHTML = filteredSpecies.map(species => this.createSpeciesCard(species)).join('');
        speciesGrid.innerHTML = speciesHTML;
        
        // Add search functionality
        this.setupSpeciesSearch();
    }

    createSpeciesCard(species) {
        const statusClass = species.invasiveStatus ? 'invasive' : 'native';
        const statusText = species.invasiveStatus ? 'Invasive' : 'Native';
        const riskClass = species.riskLevel.toLowerCase();
        
        return `
            <div class="detection-item species-encyclopedia-card">
                <div class="detection-image">
                    ${species.images.length > 0 ? `
                        <img src="${species.images[0]}" alt="${species.species}" class="species-heading-image" 
                             onclick="this.classList.toggle('zoomed')">
                    ` : `
                        <div class="species-placeholder">
                            <i class="fas fa-seedling"></i>
                        </div>
                    `}
                </div>
                <div class="detection-content">
                    <div class="detection-header">
                        <h3 class="detection-species">${sanitizeHtml(species.species)}</h3>
                        <div class="detection-badges">
                            <span class="status-badge ${statusClass}">${statusText}</span>
                            <span class="risk-badge ${riskClass}">${species.riskLevel}</span>
                            <span class="nemba-badge">NEMBA ${species.nembaCategory}</span>
                        </div>
                    </div>
                    <div class="detection-details">
                        <div class="detail-row">
                            <span class="detail-label">Common Name:</span>
                            <span class="detail-value">${sanitizeHtml(species.commonName)}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Family:</span>
                            <span class="detail-value">${sanitizeHtml(species.family)}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Origin:</span>
                            <span class="detail-value">${sanitizeHtml(species.origin)}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Detections:</span>
                            <span class="detail-value">${species.detectionCount}</span>
                        </div>
                        <div class="detail-row">
                            <span class="detail-label">Latest Detection:</span>
                            <span class="detail-value">${this.formatDate(species.latestDetection)}</span>
                        </div>
                    </div>
                    <div class="detection-description">
                        <h4>Description:</h4>
                        <p>${sanitizeHtml(species.description.substring(0, 200))}...</p>
                    </div>
                    <div class="detection-treatment">
                        <h4>Treatment Recommendations:</h4>
                        <p>${sanitizeHtml(species.treatment || 'No treatment information available')}</p>
                    </div>
                </div>
            </div>
        `;
    }

    setupSpeciesSearch() {
        const searchInput = document.getElementById('species-search-input');
        if (!searchInput) return;
        
        searchInput.addEventListener('input', (e) => {
            const searchTerm = e.target.value.toLowerCase();
            this.currentFilters.speciesSearch = searchTerm;
            this.updateSpeciesEncyclopedia();
        });
    }

    getFilteredSpecies() {
        let filtered = this.speciesData;
        
        if (this.currentFilters.speciesSearch) {
            const searchTerm = this.currentFilters.speciesSearch.toLowerCase();
            filtered = filtered.filter(s => 
                s.species.toLowerCase().includes(searchTerm) ||
                s.commonName.toLowerCase().includes(searchTerm) ||
                s.family.toLowerCase().includes(searchTerm)
            );
        }
        
        return filtered;
    }

    formatDate(dateString) {
        if (!dateString) return 'Unknown';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }
}

// Create singleton instance
export const speciesManager = new SpeciesManager(); 