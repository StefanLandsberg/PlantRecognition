class MapLoaderProxy {
  constructor() {
    this.map = null;
    this.markers = [];
    this.isScriptLoaded = false;
    this.loadingPromise = null;
    this.markerClusters = new Map(); // Track clustered markers
    this.clusterTolerance = 0.0001; // ~10 meters tolerance for clustering
  }

  async loadGoogleMaps() {
    if (this.isScriptLoaded) {
      return this.loadingPromise;
    }

    if (this.loadingPromise) {
      return this.loadingPromise;
    }

    const key = window.APP_CONFIG?.GOOGLE_MAPS_API_KEY;
    if (!key) throw new Error('Missing Google Maps API Key');

    const script = document.createElement('script');
    script.id = 'maps-loader';
    script.src = `https://maps.googleapis.com/maps/api/js?key=${key}&loading=async&libraries=marker&callback=initMap`;

    this.loadingPromise = new Promise((resolve, reject) => {
      window.initMap = () => {
        this.isScriptLoaded = true;
        delete window.initMap;
        resolve();
      };
      script.onerror = reject;
      document.head.appendChild(script);
    });

    return this.loadingPromise;
  }

  initMap(elId = 'map', center = { lat: -25.8408, lng: 28.2395 }, zoom = 13) {
    if (!this.isScriptLoaded) {
      throw new Error('Google Maps API script is not loaded. Call loadGoogleMaps() first.');
    }
    this.map = new google.maps.Map(document.getElementById(elId), { center, zoom, mapId: 'DEMO_MAP_ID' });
    return this.map;
  }

  getAgeColor(createdAt) {
    if (!createdAt) return '#FFD700'; // Default yellow if no date

    const now = new Date();
    const created = new Date(createdAt);
    const daysDiff = (now - created) / (1000 * 60 * 60 * 24);

    if (daysDiff <= 0) return '#FFD700'; // Bright yellow for new
    if (daysDiff >= 30) return '#FF4444'; // Red for 30+ days

    // Linear interpolation from yellow to red over 30 days
    const ratio = daysDiff / 30;
    const r = Math.round(255);
    const g = Math.round(215 - (215 - 68) * ratio); // 215 to 68
    const b = Math.round(0 + 68 * (1 - ratio)); // 0 to 68, then back to 0

    return `rgb(${r}, ${g}, ${Math.max(0, b)})`;
  }

  getSightingColor(sighting) {
    const species = sighting.analysis?.predictedSpecies || 'Unknown';
    const isUnknown = species === 'Unknown' || species === 'Unknown species' || species.includes('Unknown');

    if (isUnknown) {
      return '#10b981'; // Green for unknown (non-invasive)
    } else {
      // Invasive species - use age-based coloring (yellow to red)
      return this.getAgeColor(sighting.createdAt);
    }
  }

  isInvasiveSpecies(sighting) {
    const species = sighting.analysis?.predictedSpecies || 'Unknown';
    return species !== 'Unknown' && species !== 'Unknown species' && !species.includes('Unknown');
  }

  getClusterKey(lat, lng) {
    // Round coordinates to tolerance level for clustering
    const roundedLat = Math.round(lat / this.clusterTolerance) * this.clusterTolerance;
    const roundedLng = Math.round(lng / this.clusterTolerance) * this.clusterTolerance;
    return `${roundedLat},${roundedLng}`;
  }

  addMarker({ lat, lng, title, data }) {
    if (!this.map) {
      throw new Error('Map is not initialized. Call initMap() first.');
    }

    if (title === "Your Location") {
      return this.createUserLocationMarker(lat, lng, title);
    }

    // Check if there are already markers at this location
    const clusterKey = this.getClusterKey(lat, lng);

    if (this.markerClusters.has(clusterKey)) {
      // Add to existing cluster
      const cluster = this.markerClusters.get(clusterKey);
      cluster.sightings.push(data.sighting);
      this.updateClusterMarker(cluster);
      return cluster.marker;
    } else {
      // Create new cluster
      const cluster = {
        lat,
        lng,
        sightings: [data.sighting],
        marker: null
      };
      this.markerClusters.set(clusterKey, cluster);
      cluster.marker = this.createClusterMarker(cluster);
      return cluster.marker;
    }
  }

  createUserLocationMarker(lat, lng, title) {
    const locationDiv = document.createElement('div');
    locationDiv.style.cssText = `
      width: 12px;
      height: 12px;
      background: #4285F4;
      border: 2px solid white;
      border-radius: 50%;
      box-shadow: 0 2px 6px rgba(0,0,0,0.3);
      cursor: pointer;
    `;

    const m = new google.maps.marker.AdvancedMarkerElement({
      position: { lat, lng },
      map: this.map,
      title,
      content: locationDiv
    });

    this.markers.push(m);
    return m;
  }

  createClusterMarker(cluster) {
    const markerContainer = document.createElement('div');
    markerContainer.className = 'marker-cluster';

    // Create the main marker element
    const sightingDiv = document.createElement('div');

    // Get color based on most recent sighting - prioritize invasive species
    const invasiveSightings = cluster.sightings.filter(s => this.isInvasiveSpecies(s));
    const mostRecentSighting = invasiveSightings.length > 0 ? invasiveSightings[0] : cluster.sightings[0];
    const pinColor = this.getSightingColor(mostRecentSighting);

    sightingDiv.style.cssText = `
      width: 10px;
      height: 10px;
      background: ${pinColor};
      border: 1px solid white;
      border-radius: 2px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
      cursor: pointer;
      position: relative;
    `;

    // Add count badge if multiple sightings
    if (cluster.sightings.length > 1) {
      const countBadge = document.createElement('div');
      countBadge.className = 'cluster-count-badge';
      countBadge.textContent = cluster.sightings.length;
      sightingDiv.appendChild(countBadge);
    }

    markerContainer.appendChild(sightingDiv);

    const m = new google.maps.marker.AdvancedMarkerElement({
      position: { lat: cluster.lat, lng: cluster.lng },
      map: this.map,
      title: `${cluster.sightings.length} sighting(s)`,
      content: markerContainer
    });

    // Add hover and click listeners
    this.addClusterListeners(m, cluster, markerContainer);

    this.markers.push(m);
    return m;
  }

  updateClusterMarker(cluster) {
    if (!cluster.marker) return;

    const markerElement = cluster.marker.content;
    const sightingDiv = markerElement.querySelector('div');

    // Update color to newest sighting
    const newestSighting = cluster.sightings.sort((a, b) =>
      new Date(b.createdAt) - new Date(a.createdAt)
    )[0];

    const ageColor = this.getAgeColor(newestSighting?.createdAt);
    sightingDiv.style.background = ageColor;

    // Update or add count badge
    let countBadge = sightingDiv.querySelector('.cluster-count-badge');
    if (cluster.sightings.length > 1) {
      if (!countBadge) {
        countBadge = document.createElement('div');
        countBadge.className = 'cluster-count-badge';
        sightingDiv.appendChild(countBadge);
      }
      countBadge.textContent = cluster.sightings.length;
    } else if (countBadge) {
      countBadge.remove();
    }

    cluster.marker.title = `${cluster.sightings.length} sighting(s)`;
  }

  addClusterListeners(marker, cluster, markerContainer) {
    let hoverTimeout;
    let selector;

    const showSelector = () => {
      if (cluster.sightings.length <= 1) {
        this.onMarkerClick(cluster.sightings[0]._id, cluster.sightings[0]);
        return;
      }

      // Remove existing selector
      this.hideClusterSelector();

      selector = this.createClusterSelector(cluster);
      markerContainer.appendChild(selector);

      requestAnimationFrame(() => {
        selector.classList.add('show');
      });
    };

    const hideSelector = () => {
      if (selector) {
        selector.classList.remove('show');
        setTimeout(() => {
          if (selector) {
            selector.remove();
            selector = null;
          }
        }, 200);
      }
    };

    // Click handler
    marker.addListener('click', showSelector);

    // Hover handlers
    markerContainer.addEventListener('mouseenter', () => {
      clearTimeout(hoverTimeout);
      if (cluster.sightings.length > 1) {
        hoverTimeout = setTimeout(showSelector, 500);
      }
    });

    markerContainer.addEventListener('mouseleave', () => {
      clearTimeout(hoverTimeout);
      hoverTimeout = setTimeout(hideSelector, 300);
    });

    if (selector) {
      selector.addEventListener('mouseenter', () => clearTimeout(hoverTimeout));
      selector.addEventListener('mouseleave', () => {
        hoverTimeout = setTimeout(hideSelector, 300);
      });
    }
  }

  createClusterSelector(cluster) {
    const selector = document.createElement('div');
    selector.className = 'cluster-selector';

    const header = document.createElement('div');
    header.className = 'cluster-header';
    header.textContent = `${cluster.sightings.length} Sightings at this Location`;

    selector.appendChild(header);

    // Sort sightings: invasive species first (by date), then non-invasive at bottom
    const invasiveSightings = cluster.sightings
      .filter(s => this.isInvasiveSpecies(s))
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

    const nonInvasiveSightings = cluster.sightings
      .filter(s => !this.isInvasiveSpecies(s))
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));

    const sortedSightings = [...invasiveSightings, ...nonInvasiveSightings];

    sortedSightings.forEach(sighting => {
      const item = document.createElement('div');
      item.className = 'cluster-item';
      const isInvasive = this.isInvasiveSpecies(sighting);

      const colorDiv = document.createElement('div');
      colorDiv.className = 'cluster-item-color';

      // Set color based on species type
      if (isInvasive) {
        colorDiv.style.background = this.getSightingColor(sighting); // Age-based coloring for invasive
      } else {
        colorDiv.style.background = 'transparent'; // No color for non-invasive
        colorDiv.style.border = '1px solid var(--muted)'; // Gray border
      }

      const infoDiv = document.createElement('div');
      infoDiv.className = 'cluster-item-info';

      const speciesDiv = document.createElement('div');
      speciesDiv.className = 'cluster-item-species';
      const speciesName = sighting.analysis?.predictedSpecies || 'Unknown Species';
      const statusLabel = isInvasive ? 'INVASIVE' : 'NON-INVASIVE';
      speciesDiv.innerHTML = `${speciesName} <span style="font-size: 0.6rem; color: ${isInvasive ? '#ef4444' : '#10b981'}; font-weight: 700;">[${statusLabel}]</span>`;

      const detailsDiv = document.createElement('div');
      detailsDiv.className = 'cluster-item-details';
      const confidence = (sighting.analysis?.confidence * 100)?.toFixed(1) || 'Unknown';
      const date = new Date(sighting.createdAt).toLocaleDateString();
      detailsDiv.textContent = `${confidence}% confidence • ${date}`;

      infoDiv.appendChild(speciesDiv);
      infoDiv.appendChild(detailsDiv);

      // Add remove button
      const removeBtn = document.createElement('button');
      removeBtn.className = 'cluster-remove-btn';
      removeBtn.innerHTML = '×';
      removeBtn.title = 'Mark as removed';
      removeBtn.style.cssText = `
        position: absolute;
        top: 5px;
        right: 5px;
        background: #dc2626;
        color: white;
        border: none;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        font-size: 12px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10;
      `;

      // Remove button click handler
      removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering the item click
        this.showRemovalConfirmation(sighting);
      });

      item.style.position = 'relative';
      item.appendChild(colorDiv);
      item.appendChild(infoDiv);
      item.appendChild(removeBtn);

      // Click handler for individual sighting
      item.addEventListener('click', () => {
        this.hideClusterSelector();
        this.onMarkerClick(sighting._id, sighting);
      });

      selector.appendChild(item);
    });

    // Prevent scroll events from bubbling to the map
    selector.addEventListener('wheel', (e) => {
      e.stopPropagation();
    });

    // Prevent touch scroll events from bubbling to the map
    selector.addEventListener('touchmove', (e) => {
      e.stopPropagation();
    });

    return selector;
  }

  hideClusterSelector() {
    const existingSelector = document.querySelector('.cluster-selector.show');
    if (existingSelector) {
      existingSelector.classList.remove('show');
      setTimeout(() => existingSelector.remove(), 200);
    }
  }

  formatLLMContent(llmData) {
    if (!llmData || !llmData.details) return '';

    // Handle different LLM data formats
    if (typeof llmData === 'string') {
      return `<div class="analysis-section"><p>${llmData}</p></div>`;
    }

    if (llmData.details && typeof llmData.details === 'object') {
      const analysisData = llmData.details;
      let formattedContent = '';

      // Species Information
      const speciesInfo = analysisData.advisory_content?.species_identification;
      if (speciesInfo?.scientific_name || analysisData.species || speciesInfo?.common_names || analysisData.common_name) {
        formattedContent += `
          <div class="analysis-section">
            <h4>Species Information</h4>
            <p><strong>Scientific Name:</strong> ${speciesInfo?.scientific_name || analysisData.species || 'Unknown'}</p>
            <p><strong>Common Names:</strong> ${speciesInfo?.common_names || analysisData.common_name || 'Unknown'}</p>
            <p><strong>Family:</strong> ${speciesInfo?.family || analysisData.family || 'Unknown'}</p>
          </div>
        `;
      }

      // Legal Status & Risk
      const legalInfo = analysisData.advisory_content?.legal_status;
      if (legalInfo?.nemba_category || analysisData.risk_level) {
        formattedContent += `
          <div class="analysis-section">
            <h4>Legal Status</h4>
            <p><strong>NEMBA Category:</strong> ${legalInfo?.nemba_category || 'Unknown'}</p>
            <p><strong>Legal Requirements:</strong> ${legalInfo?.legal_requirements || 'Unknown'}</p>
            <p><strong>Risk Level:</strong> ${analysisData.risk_level || 'Unknown'}</p>
            <p><strong>Invasive Status:</strong> ${analysisData.invasive_status ? 'Yes' : 'No'}</p>
          </div>
        `;
      }

      // Description
      const physicalDesc = analysisData.advisory_content?.physical_description;
      if (physicalDesc || analysisData.description) {
        formattedContent += `
          <div class="analysis-section">
            <h4>Physical Description</h4>
            <p>${physicalDesc || analysisData.description || 'No description available.'}</p>
            <p><strong>Origin:</strong> ${analysisData.origin || 'Unknown'}</p>
          </div>
        `;
      }

      // Distribution
      const distributionInfo = analysisData.advisory_content?.distribution || analysisData.where_found || analysisData.distribution;
      if (distributionInfo && distributionInfo !== 'Not found') {
        formattedContent += `
          <div class="analysis-section">
            <h4>Where Found</h4>
            <p>${distributionInfo}</p>
          </div>
        `;
      }

      // Control Methods
      const controlInfo = analysisData.advisory_content?.control_methods || analysisData.treatment || analysisData.control_methods;
      if (controlInfo && controlInfo !== 'Not found') {
        formattedContent += `
          <div class="analysis-section">
            <h4>Control Methods</h4>
            <p>${controlInfo}</p>
          </div>
        `;
      }

      // Action Required
      const actionInfo = analysisData.action_required || analysisData.advisory_content?.action_required;
      if (actionInfo && actionInfo !== 'Not found') {
        formattedContent += `
          <div class="analysis-section">
            <h4>Action Required</h4>
            <p>${actionInfo}</p>
          </div>
        `;
      }

      // Disclaimer
      if (analysisData.disclaimer) {
        formattedContent += `
          <div class="analysis-section disclaimer">
            <h4>Disclaimer</h4>
            <p><em>${analysisData.disclaimer}</em></p>
          </div>
        `;
      }

      return formattedContent || '<div class="analysis-section"><p>Analysis data available but format not recognized.</p></div>';
    }

    // Fallback for any other format
    return `<div class="analysis-section"><p>${JSON.stringify(llmData, null, 2)}</p></div>`;
  }

  onMarkerClick(sightingId, sightingData) {
    this.showSightingModal(sightingData);
  }

  showSightingModal(sighting) {
    // Remove existing modal if any
    const existingModal = document.getElementById('sighting-modal');
    if (existingModal) {
      existingModal.remove();
    }

    // Get LLM data from multiple sources
    let llmContent = '';
    let hasLLMData = false;

    // First try to get from the detection card
    const detectionCard = document.getElementById(`det-${sighting._id}`);
    if (detectionCard) {
      const llmDiv = detectionCard.querySelector('.llm-details');
      if (llmDiv) {
        llmContent = llmDiv.innerHTML;
        hasLLMData = true;
      }
    }

    // If no card data, try to get from sighting object directly
    if (!hasLLMData && sighting.analysis?.llm) {
      const llmData = sighting.analysis.llm;
      llmContent = this.formatLLMContent(llmData);
      hasLLMData = true;
    }

    // Check if analysis exists but LLM is missing
    if (!hasLLMData && sighting.analysis && !sighting.analysis?.llm) {
      llmContent = '<p style="color: var(--muted); font-style: italic;">AI analysis is being processed...</p>';
    }

    const modal = document.createElement('div');
    modal.id = 'sighting-modal';
    modal.className = 'sighting-modal';

    const createdDate = new Date(sighting.createdAt).toLocaleDateString();
    const confidence = (sighting.analysis?.confidence * 100)?.toFixed(1) || 'Unknown';

    modal.innerHTML = `
      <div class="modal-content">
        <div class="modal-header">
          <div class="modal-header-content">
            ${(sighting.imageUrl || sighting.imagePath) ? `<img src="${sighting.imageUrl || sighting.imagePath}" alt="Sighting" class="modal-header-image" />` : ''}
            <div class="modal-header-text">
              <h3>${sighting.analysis?.predictedSpecies || 'Unknown Species'}</h3>
              <p class="modal-subtitle">${confidence}% confidence • ${createdDate}</p>
            </div>
          </div>
          <button class="modal-close">&times;</button>
        </div>
        <div class="modal-body">
          <div class="modal-badges">
            ${sighting.location?.coordinates ? `<span class="modal-badge">GPS: ${sighting.location.coordinates[1].toFixed(4)}, ${sighting.location.coordinates[0].toFixed(4)}</span>` : ''}
          </div>

          ${hasLLMData ? `
            <div class="llm-analysis">
              <h4 style="color: var(--accent); margin-bottom: 0.5rem;">AI Analysis</h4>
              ${llmContent}
            </div>
          ` : `
            <div class="llm-analysis">
              ${llmContent || '<p style="color: var(--muted); font-style: italic;">AI analysis pending...</p>'}
            </div>
          `}
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    // Show modal
    requestAnimationFrame(() => {
      modal.classList.add('show');
    });

    // Close handlers
    const closeBtn = modal.querySelector('.modal-close');
    const closeModal = () => {
      modal.classList.remove('show');
      setTimeout(() => modal.remove(), 300);
    };

    closeBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });

    // ESC key to close
    const handleEsc = (e) => {
      if (e.key === 'Escape') {
        closeModal();
        document.removeEventListener('keydown', handleEsc);
      }
    };
    document.addEventListener('keydown', handleEsc);
  }

  fitToMarkers() {
    if (!this.map || this.markers.length === 0) {
      console.warn('Cannot fit to bounds: map not initialized or no markers present.');
      return;
    }
    const b = new google.maps.LatLngBounds();
    this.markers.forEach(m => b.extend(m.position));
    if (!b.isEmpty()) {
      this.map.fitBounds(b);
    }
  }

  showRemovalConfirmation(sighting) {
    const speciesName = sighting.analysis?.predictedSpecies || 'Unknown Species';

    // Create modal
    const modal = document.createElement('div');
    modal.className = 'map-removal-modal';
    modal.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    `;

    const modalContent = document.createElement('div');
    modalContent.style.cssText = `
      background: white;
      padding: 2rem;
      border-radius: 8px;
      max-width: 400px;
      width: 90%;
      text-align: center;
      color: black;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    `;

    modalContent.innerHTML = `
      <h3 style="margin-top: 0; color: #dc2626;">Confirm Removal</h3>
      <p style="color: black;">Are you sure you want to mark this <strong>${speciesName}</strong> sighting as removed?</p>
      <p style="font-size: 0.9rem; color: #666;"><em>This indicates the plant has been physically removed from the location.</em></p>
      <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 1.5rem;">
        <button id="confirm-removal" style="background: #dc2626; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 4px; cursor: pointer; font-weight: 600;">Mark as Removed</button>
        <button id="cancel-removal" style="background: #6b7280; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 4px; cursor: pointer;">Cancel</button>
      </div>
    `;

    modal.appendChild(modalContent);
    document.body.appendChild(modal);

    // Event handlers
    modalContent.querySelector('#confirm-removal').addEventListener('click', () => {
      this.performRemoval(sighting._id);
      modal.remove();
    });

    modalContent.querySelector('#cancel-removal').addEventListener('click', () => {
      modal.remove();
    });

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.remove();
      }
    });
  }

  async performRemoval(sightingId) {
    try {
      const response = await fetch(`/api/sightings/${sightingId}/remove`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          removedAt: new Date().toISOString(),
          removedBy: 'user'
        }),
        credentials: 'include' // Include cookies for authentication
      });

      // Handle both successful responses and redirects that indicate success
      if (response.ok || response.status === 200) {
        // Show success notification
        this.showNotification('Plant removal recorded successfully!', 'success');

        // Remove the sighting from the cluster display immediately
        this.removeSightingFromCluster(sightingId);

        // Hide cluster selector
        this.hideClusterSelector();

        // Update sightings page if it exists
        if (window.sightingsPageExists) {
          window.location.reload(); // Refresh sightings page to reflect removal
        }
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
        this.showNotification('Network error, but removal may have succeeded. Checking...', 'warning');
        // Wait a moment then just try to remove from cluster (optimistic update)
        setTimeout(() => {
          this.removeSightingFromCluster(sightingId);
          this.hideClusterSelector();
        }, 1000);
      } else {
        this.showNotification(`Failed to record removal: ${error.message}`, 'error');
      }
    }
  }

  removeSightingFromCluster(sightingId) {
    // Find and remove the sighting from all clusters
    for (const [key, cluster] of this.markerClusters) {
      const sightingIndex = cluster.sightings.findIndex(s => s._id === sightingId);
      if (sightingIndex !== -1) {
        // Remove the sighting from the cluster
        cluster.sightings.splice(sightingIndex, 1);

        // Also remove from the cluster selector if it's currently displayed
        const sightingElement = document.querySelector(`[data-sighting-id="${sightingId}"]`);
        if (sightingElement) {
          sightingElement.style.transition = 'all 0.3s ease';
          sightingElement.style.opacity = '0';
          sightingElement.style.transform = 'translateX(-100%)';
          setTimeout(() => {
            sightingElement.remove();

            // Update the cluster selector header with new count
            const selector = document.querySelector('.cluster-selector');
            if (selector) {
              const header = selector.querySelector('h3');
              if (header) {
                header.textContent = `${cluster.sightings.length} Sightings at this Location`;
              }
            }
          }, 300);
        }

        // If cluster is now empty, remove the marker
        if (cluster.sightings.length === 0) {
          if (cluster.marker) {
            cluster.marker.setMap(null);
          }
          this.markerClusters.delete(key);
        } else {
          // Update the cluster marker with new count
          this.updateClusterMarker(cluster);
        }
        break;
      }
    }
  }

  async refreshMapData() {
    try {
      // Clear existing clusters
      for (const [key, cluster] of this.markerClusters) {
        if (cluster.marker) {
          cluster.marker.setMap(null);
        }
      }
      this.markerClusters.clear();

      // Import SightingsAPI if not already available
      if (typeof SightingsAPI === 'undefined') {
        console.error('SightingsAPI not available for map refresh');
        return;
      }

      // Reload sightings data
      const box = "";
      const { data } = await SightingsAPI.list(box);

      // Re-add markers for non-removed sightings
      data.forEach((d) => {
        if (d.location?.coordinates && !d.isRemoved) {
          const [lng, lat] = d.location.coordinates;
          this.addMarker({
            lat,
            lng,
            title: d.analysis?.predictedSpecies || "Sighting",
            data: { sightingId: d._id, sighting: d }
          });
        }
      });

      this.fitToMarkers();
    } catch (error) {
      console.error('Error refreshing map data:', error);
    }
  }

  showNotification(message, type = 'info') {
    const notification = document.createElement('div');
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
      font-size: 0.9rem;
      max-width: 300px;
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
}

const mapProxy = new MapLoaderProxy();
export { mapProxy }; 