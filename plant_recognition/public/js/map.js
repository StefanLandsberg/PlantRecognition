class MapLoaderProxy {
  constructor() {
    this.map = null;
    this.markers = [];
    this.isScriptLoaded = false;
    this.loadingPromise = null;
    this.markerClusters = new Map(); // Track clustered markers
    this.clusterTolerance = 0.0001; // ~10 meters tolerance for clustering
  }

  // Helper function for safe image rendering
  getImageHTML(sighting, className = '', style = '') {
    const imageUrl = sighting.imageUrl || sighting.imagePath;

    // Debug logging and validation
    if (imageUrl && typeof imageUrl === 'string' && !imageUrl.startsWith('http') && !imageUrl.startsWith('/') && !imageUrl.startsWith('data:')) {
      console.warn('Invalid image URL detected:', {
        sightingId: sighting._id,
        imageUrl: imageUrl,
        species: sighting.analysis?.predictedSpecies
      });
      // Return placeholder for invalid URLs
      return '<div class="' + className + ' no-image-placeholder" style="' + style + '; border: 2px dashed var(--border); display: flex; align-items: center; justify-content: center; background: var(--panel);"></div>';
    }

    const isValidUrl = imageUrl && typeof imageUrl === 'string' &&
      (imageUrl.startsWith('http') || imageUrl.startsWith('/') || imageUrl.startsWith('data:'));

    if (isValidUrl) {
      return '<img src="' + imageUrl + '" alt="' + (sighting.analysis?.predictedSpecies || 'Plant sighting') + '" class="' + className + '" style="' + style + '" onerror="this.outerHTML=\'<div class=&quot;' + className + ' no-image-placeholder&quot; style=&quot;' + style + '; border: 2px dashed var(--border); display: flex; align-items: center; justify-content: center; background: var(--panel);&quot;></div>\'" />';
    } else {
      return '<div class="' + className + ' no-image-placeholder" style="' + style + '; border: 2px dashed var(--border); display: flex; align-items: center; justify-content: center; background: var(--panel);"></div>';
    }
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

      // Removal capability will be added inside the popup, not on the map pin

      return cluster.marker;
    }
  }

  createUserLocationMarker(lat, lng, title) {
    const locationDiv = document.createElement('div');
    locationDiv.style.cssText = `
      width: 12px;
      height: 12px;
      background: #4285F4;
      border: 2px solid var(--background);
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
      border: 1px solid var(--background);
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
    this.addNewClusterListeners(m, cluster, markerContainer);

    this.markers.push(m);
    return m;
  }

  updateClusterMarker(cluster) {
    if (!cluster.marker) return;

    const markerElement = cluster.marker.content;
    const sightingDiv = markerElement.querySelector('div');

    // Update color to newest sighting
    const newestSighting = cluster.sightings.sort((a, b) => {
      const dateA = a.createdAt ? new Date(a.createdAt) : new Date(0);
      const dateB = b.createdAt ? new Date(b.createdAt) : new Date(0);
      return dateB - dateA;
    })[0];

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
    let openedByClick = false;

    const showSelector = (clickTriggered = false) => {
      if (cluster.sightings.length <= 1) {
        this.onMarkerClick(cluster.sightings[0]._id, cluster.sightings[0]);
        return;
      }

      // Track if opened by click to prevent auto-hide
      openedByClick = clickTriggered;

      // Remove existing selector
      this.hideClusterPopup();

      selector = this.createClusterSelector(cluster);
      markerContainer.appendChild(selector);

      // Apply positioning immediately before showing
      try {
        const position = this.calculatePopupPosition(selector, markerContainer, 'bottom');
        selector.style.left = position.left;
        selector.style.top = position.top;
      } catch (e) {
        console.warn('Could not apply responsive positioning:', e);
        // Fallback to CSS positioning - let CSS handle it
      }

      // Add event listeners to the new selector
      selector.addEventListener('mouseenter', () => clearTimeout(hoverTimeout));
      selector.addEventListener('mouseleave', () => {
        if (!openedByClick) {
          hoverTimeout = setTimeout(hideSelector, 300);
        }
      });

      // Add close button event listener
      const closeBtn = selector.querySelector('.cluster-close');
      if (closeBtn) {
        closeBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          hideSelector();
        });
      }

      requestAnimationFrame(() => {
        selector.classList.add('show');
      });

      // Add click-outside-to-close for click-opened selectors
      if (clickTriggered) {
        const handleClickOutside = (e) => {
          if (!selector.contains(e.target) && !markerContainer.contains(e.target)) {
            document.removeEventListener('click', handleClickOutside);
            hideSelector();
          }
        };

        // Add listener after a brief delay to avoid immediate closing
        setTimeout(() => {
          document.addEventListener('click', handleClickOutside);
        }, 100);
      }
    };

    const hideSelector = () => {
      if (selector) {
        selector.classList.remove('show');
        setTimeout(() => {
          if (selector) {
            selector.remove();
            selector = null;
            openedByClick = false;
          }
        }, 200);
      }
    };

    // Click handler - prevent hover-based auto-hide
    marker.addListener('click', () => showSelector(true));

    // Hover handlers
    markerContainer.addEventListener('mouseenter', () => {
      clearTimeout(hoverTimeout);
      if (cluster.sightings.length > 1 && !selector && !openedByClick) {
        hoverTimeout = setTimeout(() => showSelector(false), 500);
      }
    });

    markerContainer.addEventListener('mouseleave', () => {
      clearTimeout(hoverTimeout);
      if (!openedByClick && selector) {
        hoverTimeout = setTimeout(hideSelector, 300);
      }
    });
  }

  // Calculate responsive positioning for popups within screen boundaries
  calculatePopupPosition(popupElement, markerElement, preferredPosition = 'bottom') {
    const mapContainer = document.getElementById('map');
    const mapRect = mapContainer.getBoundingClientRect();
    const markerRect = markerElement.getBoundingClientRect();

    // Get popup dimensions (temporarily show to measure)
    const wasHidden = popupElement.style.display === 'none';
    if (wasHidden) {
      popupElement.style.visibility = 'hidden';
      popupElement.style.display = 'block';
    }

    const popupRect = popupElement.getBoundingClientRect();

    if (wasHidden) {
      popupElement.style.display = 'none';
      popupElement.style.visibility = 'visible';
    }

    // Calculate 10% margin from screen boundaries
    const marginTop = window.innerHeight * 0.1;
    const marginBottom = window.innerHeight * 0.9;
    const marginLeft = window.innerWidth * 0.05;
    const marginRight = window.innerWidth * 0.95;

    // Calculate marker position relative to map
    const markerX = markerRect.left - mapRect.left;
    const markerY = markerRect.top - mapRect.top;

    let left = markerX;
    let top = markerY;

    // Determine best position based on available space
    if (preferredPosition === 'bottom') {
      top = markerY + 30; // Below marker

      // Check if popup would go below map boundary
      if (top + popupRect.height > mapRect.height - 20) {
        top = markerY - popupRect.height - 10; // Above marker instead
      }
    } else if (preferredPosition === 'top') {
      top = markerY - popupRect.height - 10; // Above marker

      // Check if popup would go above map boundary
      if (top < 20) {
        top = markerY + 30; // Below marker instead
      }
    }

    // Adjust horizontal position to stay within map bounds
    if (left + popupRect.width > mapRect.width - 20) {
      left = mapRect.width - popupRect.width - 20;
    }

    if (left < 20) {
      left = 20;
    }

    return { left: `${left}px`, top: `${top}px` };
  }

  // Position multiple popups in a responsive layout
  positionMultiplePopups(analysisPopup, timelinePopup, statsPopup, tidbitPopup) {
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const margin = {
      top: viewportHeight * 0.1,
      bottom: viewportHeight * 0.9,
      left: viewportWidth * 0.05,
      right: viewportWidth * 0.95
    };

    // Analysis popup - center
    analysisPopup.style.left = '50%';
    analysisPopup.style.top = '50%';
    analysisPopup.style.transform = 'translate(-50%, -50%)';

    // Timeline popup - left side, with boundary check
    const timelineLeft = Math.max(margin.left, 32); // 2rem = 32px
    timelinePopup.style.left = `${timelineLeft}px`;
    timelinePopup.style.top = '50%';
    timelinePopup.style.transform = 'translateY(-50%)';

    // Stats popup - right side, with boundary check
    const statsRect = statsPopup.getBoundingClientRect();
    const statsRight = Math.min(margin.right - statsRect.width, viewportWidth - statsRect.width - 32);
    statsPopup.style.left = `${statsRight}px`;
    statsPopup.style.top = '50%';
    statsPopup.style.transform = 'translateY(-50%)';

    // Tidbit popup - let CSS handle positioning

    // Adjust for mobile/small screens
    if (viewportWidth < 768) {
      // Stack vertically on mobile
      timelinePopup.style.position = 'relative';
      timelinePopup.style.left = 'auto';
      timelinePopup.style.top = 'auto';
      timelinePopup.style.transform = 'none';
      timelinePopup.style.marginBottom = '1rem';

      statsPopup.style.position = 'relative';
      statsPopup.style.left = 'auto';
      statsPopup.style.top = 'auto';
      statsPopup.style.transform = 'none';
      statsPopup.style.marginBottom = '1rem';

      tidbitPopup.style.position = 'relative';
      tidbitPopup.style.left = 'auto';
      tidbitPopup.style.top = 'auto';
      tidbitPopup.style.transform = 'none';
    }
  }

  createClusterSelector(cluster) {
    const selector = document.createElement('div');
    selector.className = 'cluster-popup';

    const header = document.createElement('div');
    header.className = 'cluster-header';
    header.innerHTML = `
      <span>${cluster.sightings.length} Sightings at this Location</span>
      <button class="cluster-close" title="Close">×</button>
    `;

    selector.appendChild(header);

    // Sort sightings: invasive species first (by date), then non-invasive at bottom
    const invasiveSightings = cluster.sightings
      .filter(s => this.isInvasiveSpecies(s))
      .sort((a, b) => {
        const dateA = a.createdAt ? new Date(a.createdAt) : new Date(0);
        const dateB = b.createdAt ? new Date(b.createdAt) : new Date(0);
        return dateB - dateA;
      });

    const nonInvasiveSightings = cluster.sightings
      .filter(s => !this.isInvasiveSpecies(s))
      .sort((a, b) => {
        const dateA = a.createdAt ? new Date(a.createdAt) : new Date(0);
        const dateB = b.createdAt ? new Date(b.createdAt) : new Date(0);
        return dateB - dateA;
      });

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
      speciesDiv.innerHTML = `${speciesName} <span class="species-status ${isInvasive ? 'invasive' : 'native'}">[${statusLabel}]</span>`;

      const detailsDiv = document.createElement('div');
      detailsDiv.className = 'cluster-item-details';
      const confidence = (sighting.analysis?.confidence * 100)?.toFixed(1) || 'Unknown';
      const date = sighting.createdAt ? new Date(sighting.createdAt).toLocaleDateString() : 'Unknown Date';
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
        color: var(--background);
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
        // Check if the same sighting is already displayed in main popup
        const currentPopup = document.querySelector('.analysis-popup');
        const isSameSighting = currentPopup && currentPopup.dataset.sightingId === sighting._id;

        this.hideClusterPopup();

        // Only show new popup if it's a different sighting
        if (!isSameSighting) {
          this.onMarkerClick(sighting._id, sighting);
        }
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
    const existingSelector = document.querySelector('.cluster-popup.show');
    if (existingSelector) {
      existingSelector.classList.remove('show');
      setTimeout(() => existingSelector.remove(), 200);
    }
  }

  formatLLMContent(llmData) {
    if (!llmData || !llmData.details) return '';

    // Handle different LLM data formats
    if (typeof llmData === 'string') {
      return `<div class="analysis-section"><p style="color: var(--text);">${llmData}</p></div>`;
    }

    if (llmData.details && typeof llmData.details === 'object') {
      const analysisData = llmData.details;
      let formattedContent = '';

      // Species Information
      const speciesInfo = analysisData.advisory_content?.species_identification;
      if (speciesInfo?.scientific_name || analysisData.species || speciesInfo?.common_names || analysisData.common_name) {
        formattedContent += `
          <div class="analysis-section">
            <h4 style="color: var(--accent);">Species Information</h4>
            <p style="color: var(--text);"><strong>Scientific Name:</strong> ${speciesInfo?.scientific_name || analysisData.species || 'Unknown'}</p>
            <p style="color: var(--text);"><strong>Common Names:</strong> ${speciesInfo?.common_names || analysisData.common_name || 'Unknown'}</p>
            <p style="color: var(--text);"><strong>Family:</strong> ${speciesInfo?.family || analysisData.family || 'Unknown'}</p>
          </div>
        `;
      }

      // Legal Status & Risk
      const legalInfo = analysisData.advisory_content?.legal_status;
      if (legalInfo?.nemba_category || analysisData.risk_level) {
        formattedContent += `
          <div class="analysis-section">
            <h4 style="color: var(--accent);">Legal Status</h4>
            <p style="color: var(--text);"><strong>NEMBA Category:</strong> ${legalInfo?.nemba_category || 'Unknown'}</p>
            <p style="color: var(--text);"><strong>Legal Requirements:</strong> ${legalInfo?.legal_requirements || 'Unknown'}</p>
            <p style="color: var(--text);"><strong>Risk Level:</strong> ${analysisData.risk_level || 'Unknown'}</p>
            <p style="color: var(--text);"><strong>Invasive Status:</strong> ${analysisData.invasive_status ? 'Yes' : 'No'}</p>
          </div>
        `;
      }

      // Description
      const physicalDesc = analysisData.advisory_content?.physical_description;
      if (physicalDesc || analysisData.description) {
        formattedContent += `
          <div class="analysis-section">
            <h4 style="color: var(--accent);">Physical Description</h4>
            <p style="color: var(--text);">${physicalDesc || analysisData.description || 'No description available.'}</p>
            <p style="color: var(--text);"><strong>Origin:</strong> ${analysisData.origin || 'Unknown'}</p>
          </div>
        `;
      }

      // Distribution
      const distributionInfo = analysisData.advisory_content?.distribution || analysisData.where_found || analysisData.distribution;
      if (distributionInfo && distributionInfo !== 'Not found') {
        formattedContent += `
          <div class="analysis-section">
            <h4 style="color: var(--accent);">Where Found</h4>
            <p style="color: var(--text);">${distributionInfo}</p>
          </div>
        `;
      }

      // Control Methods
      const controlInfo = analysisData.advisory_content?.control_methods || analysisData.treatment || analysisData.control_methods;
      if (controlInfo && controlInfo !== 'Not found') {
        formattedContent += `
          <div class="analysis-section">
            <h4 style="color: var(--accent);">Control Methods</h4>
            <p style="color: var(--text);">${controlInfo}</p>
          </div>
        `;
      }

      // Action Required
      const actionInfo = analysisData.action_required || analysisData.advisory_content?.action_required;
      if (actionInfo && actionInfo !== 'Not found') {
        formattedContent += `
          <div class="analysis-section">
            <h4 style="color: var(--accent);">Action Required</h4>
            <p style="color: var(--text);">${actionInfo}</p>
          </div>
        `;
      }

      // Disclaimer
      if (analysisData.disclaimer) {
        formattedContent += `
          <div class="analysis-section disclaimer">
            <h4 style="color: var(--accent);">Disclaimer</h4>
            <p style="color: var(--text);"><em>${analysisData.disclaimer}</em></p>
          </div>
        `;
      }

      return formattedContent || '<div class="analysis-section"><p style="color: var(--text);">Analysis data available but format not recognized.</p></div>';
    }

    // Fallback for any other format
    return `<div class="analysis-section"><p style="color: var(--text);">${JSON.stringify(llmData, null, 2)}</p></div>`;
  }

  onMarkerClick(sightingId, sightingData) {
    this.showSightingModal(sightingData);
  }

  // Add removal functionality for standalone pins
  addRemovalCapabilityToStandalone(marker, sighting) {
    const markerElement = marker.content;

    // Add removal button for standalone pins too
    const removeBtn = document.createElement('button');
    removeBtn.className = 'standalone-remove-btn';
    removeBtn.innerHTML = '×';
    removeBtn.title = 'Mark as removed';
    removeBtn.style.cssText = `
      position: absolute;
      top: -8px;
      right: -8px;
      background: #dc2626;
      color: var(--background);
      border: none;
      border-radius: 50%;
      width: 16px;
      height: 16px;
      font-size: 10px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10;
      box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    `;

    // Add hover effect
    removeBtn.addEventListener('mouseenter', () => {
      removeBtn.style.transform = 'scale(1.1)';
    });
    removeBtn.addEventListener('mouseleave', () => {
      removeBtn.style.transform = 'scale(1)';
    });

    // Remove button click handler
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.showRemovalConfirmation(sighting);
    });

    markerElement.style.position = 'relative';
    markerElement.appendChild(removeBtn);
  }

  getTimelineData(species, location, radiusKm = 1) {
    const timelineData = {
      detections: [],
      removals: []
    };

    // Get all clusters within radius
    for (const [key, cluster] of this.markerClusters) {
      const distance = this.calculateDistance(location.lat, location.lng, cluster.lat, cluster.lng);

      if (distance <= radiusKm) {
        cluster.sightings.forEach(sighting => {
          const sightingSpecies = sighting.analysis?.predictedSpecies || 'Unknown';

          // Match species (case insensitive)
          if (sightingSpecies.toLowerCase() === species.toLowerCase()) {
            // Handle invalid or missing createdAt dates
            let date;
            try {
              const createdDate = new Date(sighting.createdAt);
              if (isNaN(createdDate.getTime())) {
                // Invalid date, use current date as fallback
                date = new Date().toISOString().split('T')[0];
              } else {
                date = createdDate.toISOString().split('T')[0];
              }
            } catch (error) {
              // Any date parsing error, use current date
              date = new Date().toISOString().split('T')[0];
            }

            // Add to detections
            timelineData.detections.push({
              date: date,
              species: sightingSpecies,
              confidence: sighting.analysis?.confidence || 0,
              sightingId: sighting._id
            });

            // Add to removals if removed
            if (sighting.isRemoved && sighting.removedAt) {
              const removedDate = new Date(sighting.removedAt).toISOString().split('T')[0];
              timelineData.removals.push({
                date: removedDate,
                species: sightingSpecies,
                sightingId: sighting._id
              });
            }
          }
        });
      }
    }

    // Sort by date
    timelineData.detections.sort((a, b) => new Date(a.date) - new Date(b.date));
    timelineData.removals.sort((a, b) => new Date(a.date) - new Date(b.date));

    return timelineData;
  }

  calculateDistance(lat1, lng1, lat2, lng2) {
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLng = (lng2 - lng1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLng/2) * Math.sin(dLng/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  }

  createTimelineChart(containerId, timelineData, species) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Prepare data for chart
    const allDates = [...new Set([
      ...timelineData.detections.map(d => d.date),
      ...timelineData.removals.map(r => r.date)
    ])].sort();

    if (allDates.length === 0) {
      container.innerHTML = '<p style="text-align: center; color: var(--text); padding: 2rem;">No timeline data available</p>';
      return;
    }

    // Aggregate data by date
    const chartData = allDates.map(date => {
      const detectionsCount = timelineData.detections.filter(d => d.date === date).length;
      const removalsCount = timelineData.removals.filter(r => r.date === date).length;
      return { date, detections: detectionsCount, removals: removalsCount };
    });

    // Create simple line chart
    const maxValue = Math.max(...chartData.map(d => Math.max(d.detections, d.removals)));
    const chartHeight = 200;
    const chartWidth = container.offsetWidth - 60;

    let svg = `
      <svg width="${chartWidth + 60}" height="${chartHeight + 60}" style="overflow: visible;">
        <defs>
          <linearGradient id="detectionGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#ef4444;stop-opacity:0.3" />
            <stop offset="100%" style="stop-color:#ef4444;stop-opacity:0.1" />
          </linearGradient>
          <linearGradient id="removalGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" style="stop-color:#10b981;stop-opacity:0.3" />
            <stop offset="100%" style="stop-color:#10b981;stop-opacity:0.1" />
          </linearGradient>
        </defs>
    `;

    // Draw grid lines
    for (let i = 0; i <= 5; i++) {
      const y = chartHeight - (i * chartHeight / 5);
      const value = Math.round((maxValue * i) / 5);
      svg += `
        <line x1="40" y1="${y}" x2="${chartWidth + 40}" y2="${y}" stroke="#e5e7eb" stroke-width="1"/>
        <text x="35" y="${y + 4}" text-anchor="end" font-size="10" fill="var(--text)">${value}</text>
      `;
    }

    // Draw detection line
    let detectionPath = '';
    let removalPath = '';

    chartData.forEach((point, index) => {
      const x = 40 + (index * (chartWidth / (chartData.length - 1 || 1)));
      const detectionY = chartHeight - (point.detections / maxValue * chartHeight);
      const removalY = chartHeight - (point.removals / maxValue * chartHeight);

      if (index === 0) {
        detectionPath += `M ${x} ${detectionY}`;
        removalPath += `M ${x} ${removalY}`;
      } else {
        detectionPath += ` L ${x} ${detectionY}`;
        removalPath += ` L ${x} ${removalY}`;
      }

      // Add data points
      svg += `<circle cx="${x}" cy="${detectionY}" r="4" fill="#ef4444" stroke="var(--background)" stroke-width="2"/>`;
      svg += `<circle cx="${x}" cy="${removalY}" r="4" fill="#10b981" stroke="var(--background)" stroke-width="2"/>`;
    });

    svg += `
      <path d="${detectionPath}" stroke="#ef4444" stroke-width="2" fill="none"/>
      <path d="${removalPath}" stroke="#10b981" stroke-width="2" fill="none"/>
    `;

    // Add x-axis labels
    chartData.forEach((point, index) => {
      if (index % Math.ceil(chartData.length / 6) === 0) {
        const x = 40 + (index * (chartWidth / (chartData.length - 1 || 1)));
        const date = new Date(point.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        svg += `<text x="${x}" y="${chartHeight + 20}" text-anchor="middle" font-size="10" fill="var(--text)">${date}</text>`;
      }
    });

    svg += '</svg>';

    // Add legend
    const legend = `
      <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; font-size: 0.875rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
          <div style="width: 12px; height: 12px; background: #ef4444; border-radius: 50%;"></div>
          <span style="color: var(--text);">Detections</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
          <div style="width: 12px; height: 12px; background: #10b981; border-radius: 50%;"></div>
          <span style="color: var(--text);">Removals</span>
        </div>
      </div>
    `;

    container.innerHTML = `
      <div style="text-align: center;">
        <h4 style="margin: 0 0 1rem 0; color: var(--accent);">${species} Activity Timeline</h4>
        <p style="font-size: 0.875rem; color: var(--text); margin-bottom: 1rem;">1km radius from this location</p>
        ${svg}
        ${legend}
      </div>
    `;
  }

  async showSightingModal(sighting) {
    // Remove existing modals if any, but keep cluster selector
    const existingModals = document.querySelectorAll('.sighting-modal, .analysis-popup, .timeline-popup, .stats-popup, .multi-popup-backdrop');
    existingModals.forEach(modal => modal.remove());

    // Create multi-popup layout
    await this.createMultiPopupLayout(sighting);
  }

  async createMultiPopupLayout(sighting) {
    const species = sighting.analysis?.predictedSpecies || 'Unknown Species';
    const location = {
      lat: sighting.location?.coordinates[1] || 0,
      lng: sighting.location?.coordinates[0] || 0
    };

    // Create backdrop
    const backdrop = document.createElement('div');
    backdrop.className = 'multi-popup-backdrop';
    backdrop.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.8);
      backdrop-filter: blur(8px);
      z-index: 1000;
      display: flex;
      align-items: center;
      justify-content: center;
      opacity: 0;
      transition: opacity 0.3s ease;
    `;

    // 1. Main LLM Analysis (center)
    const analysisPopup = this.createAnalysisPopup(sighting);

    // 2. Timeline Chart (left side)
    const timelinePopup = this.createTimelinePopup(sighting, species, location);

    // 3. Location Stats (right side)
    const statsPopup = await this.createStatsPopup(sighting, species, location);

    // 4. Additional tidbit (top or bottom)
    const tidbitPopup = await this.createTidbitPopup(sighting, species, location);

    backdrop.appendChild(analysisPopup);
    backdrop.appendChild(timelinePopup);
    backdrop.appendChild(statsPopup);
    backdrop.appendChild(tidbitPopup);

    document.body.appendChild(backdrop);

    // Apply responsive positioning after DOM insertion
    this.positionMultiplePopups(analysisPopup, timelinePopup, statsPopup, tidbitPopup);

    // Create timeline chart after DOM insertion
    if (timelinePopup.dataset.species && timelinePopup.dataset.location) {
      const species = timelinePopup.dataset.species;
      const location = JSON.parse(timelinePopup.dataset.location);
      const timelineData = this.getTimelineData(species, location);
      this.createTimelineChart('timeline-chart-container', timelineData, species);
    }

    // Show with animation
    requestAnimationFrame(() => {
      backdrop.style.opacity = '1';
    });

    // Close handlers
    backdrop.addEventListener('click', (e) => {
      if (e.target === backdrop) this.closeMultiPopup();
    });

    document.addEventListener('keydown', this.handleEscClose = (e) => {
      if (e.key === 'Escape') this.closeMultiPopup();
    });
  }

  createAnalysisPopup(sighting) {
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
      llmContent = '<p style="color: var(--text); font-style: italic;">AI analysis is being processed...</p>';
    }

    if (!hasLLMData && !sighting.analysis) {
      llmContent = '<p style="color: var(--text); font-style: italic;">No analysis data available.</p>';
    }

    const createdDate = sighting.createdAt ? new Date(sighting.createdAt).toLocaleDateString() : 'Unknown Date';
    const confidence = (sighting.analysis?.confidence * 100)?.toFixed(1) || 'Unknown';

    const popup = document.createElement('div');
    popup.className = 'analysis-popup';
    popup.dataset.sightingId = sighting._id;
    popup.style.cssText = `
      position: absolute;
      background: var(--panel);
      backdrop-filter: blur(2px);
      border-radius: 12px;
      padding: 1.5rem;
      max-width: 500px;
      min-width: 300px;
      width: 90vw;
      max-height: 70vh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      transform: translateY(0px);
      opacity: 1;
      z-index: 1001;
    `;

    popup.innerHTML = `
      <div class="modal-header" style="display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #e5e7eb;">
${(sighting.imageUrl || sighting.imagePath) && typeof (sighting.imageUrl || sighting.imagePath) === 'string' && ((sighting.imageUrl || sighting.imagePath).startsWith('/') || (sighting.imageUrl || sighting.imagePath).startsWith('http')) ?
          `<img src="${sighting.imageUrl || sighting.imagePath}" alt="Plant" style="width: 80px; height: 80px; object-fit: cover; border-radius: 8px;" onerror="this.style.display='none'" />` :
          `<div style="width: 80px; height: 80px; border-radius: 8px; border: 2px dashed var(--border); display: flex; align-items: center; justify-content: center; background: var(--panel);"></div>`}
        <div style="flex: 1;">
          <h3 style="margin: 0; color: var(--text); font-size: 1.25rem;">${sighting.analysis?.predictedSpecies || 'Unknown Species'}</h3>
          <p style="margin: 0.25rem 0 0 0; color: var(--text); font-size: 0.875rem;">${confidence}% confidence • ${createdDate}</p>
          ${sighting.location?.coordinates ? `<p style="margin: 0.25rem 0 0 0; color: var(--text); font-size: 0.75rem;">GPS: ${sighting.location.coordinates[1].toFixed(4)}, ${sighting.location.coordinates[0].toFixed(4)}</p>` : ''}
        </div>
        <div style="display: flex; gap: 0.5rem;">
          <button class="removal-btn" style="background: #dc2626; color: var(--background); border: none; border-radius: 6px; padding: 0.5rem 0.75rem; font-size: 0.875rem; cursor: pointer; display: flex; align-items: center; gap: 0.25rem;" title="Mark as removed">
            <span style="font-size: 1rem;">×</span> Remove
          </button>
          <button class="modal-close" style="background: none; border: none; font-size: 1.5rem; cursor: pointer; color: var(--text); padding: 0.25rem;">&times;</button>
        </div>
      </div>
      <div class="modal-body" style="color: var(--text); flex: 1; overflow-y: auto; overflow-x: hidden; word-wrap: break-word;">
        ${hasLLMData ? `
          <div class="llm-analysis" style="overflow-wrap: break-word; word-break: break-word;">
            <h4 style="color: var(--accent); margin-bottom: 0.5rem;">AI Analysis</h4>
            <div style="color: var(--text); overflow-wrap: break-word; word-break: break-word;">${llmContent}</div>
          </div>
        ` : `
          <div class="llm-analysis" style="overflow-wrap: break-word; word-break: break-word;">
            <div style="color: var(--text); overflow-wrap: break-word; word-break: break-word;">${llmContent || '<p style="color: var(--text); font-style: italic;">AI analysis pending...</p>'}</div>
          </div>
        `}
      </div>
    `;

    // Add close handler
    const closeBtn = popup.querySelector('.modal-close');
    closeBtn.addEventListener('click', () => this.closeMultiPopup());

    // Add removal handler
    const removeBtn = popup.querySelector('.removal-btn');
    removeBtn.addEventListener('click', () => {
      this.closeMultiPopup();
      this.showRemovalConfirmation(sighting);
    });

    // Already visible - no animation needed

    return popup;
  }

  createTimelinePopup(sighting, species, location) {
    const popup = document.createElement('div');
    popup.className = 'timeline-popup';
    popup.style.cssText = `
      position: absolute;
      left: 2rem;
      top: 50%;
      transform: translateY(-50%);
      background: var(--panel);
      backdrop-filter: blur(2px);
      border-radius: 12px;
      padding: 1.5rem;
      width: 350px;
      max-height: 60vh;
      overflow-y: auto;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      opacity: 1;
      z-index: 1001;
    `;

    popup.innerHTML = `
      <div id="timeline-chart-container" style="min-height: 200px;"></div>
    `;

    // Store data for later chart creation (after DOM insertion)
    popup.dataset.species = species;
    popup.dataset.location = JSON.stringify(location);

    return popup;
  }

  async createStatsPopup(sighting, species, location) {
    const popup = document.createElement('div');
    popup.className = 'stats-popup';
    popup.style.cssText = `
      position: absolute;
      right: 2rem;
      top: 50%;
      transform: translateY(-50%);
      background: var(--panel);
      backdrop-filter: blur(2px);
      border-radius: 12px;
      padding: 1.5rem;
      width: 300px;
      max-height: 60vh;
      overflow-y: auto;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      opacity: 1;
      z-index: 1001;
    `;

    // Calculate local statistics
    const stats = await this.calculateLocalStats(species, location);

    popup.innerHTML = `
      <h4 style="margin: 0 0 1rem 0; color: var(--accent);">Local Statistics</h4>
      <div style="space-y: 1rem; color: var(--text);">
        <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: var(--border); border-radius: 8px; margin-bottom: 0.5rem;">
          <span style="font-weight: 600; color: var(--text);">Total Detections:</span>
          <span style="color: #ef4444; font-weight: 700;">${stats.totalDetections}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: var(--border); border-radius: 8px; margin-bottom: 0.5rem;">
          <span style="font-weight: 600; color: var(--text);">Successfully Removed:</span>
          <span style="color: #10b981; font-weight: 700;">${stats.totalRemovals}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: var(--border); border-radius: 8px; margin-bottom: 0.5rem;">
          <span style="font-weight: 600; color: var(--text);">Active Sightings:</span>
          <span style="color: #f59e0b; font-weight: 700;">${stats.activeSightings}</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: var(--border); border-radius: 8px; margin-bottom: 0.5rem;">
          <span style="font-weight: 600; color: var(--text);">Avg Confidence:</span>
          <span style="font-weight: 700; color: var(--text);">${stats.avgConfidence}%</span>
        </div>
        <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: var(--border); border-radius: 8px; margin-bottom: 1rem;">
          <span style="font-weight: 600; color: var(--text);">First Detected:</span>
          <span style="font-weight: 700; color: var(--text);">${stats.firstDetected}</span>
        </div>

        <h5 style="margin: 1rem 0 0.5rem 0; color: var(--accent);">Nearby Species</h5>
        <div style="max-height: 150px; overflow-y: auto;">
          ${stats.nearbySpecies.map(sp => `
            <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #e5e7eb; font-size: 0.875rem;">
              <span style="color: var(--text);">${sp.species}</span>
              <span style="color: var(--text);">${sp.count}</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;

    // Already visible - no animation needed

    return popup;
  }

  async createTidbitPopup(sighting, species, location) {
    const popup = document.createElement('div');
    popup.className = 'tidbit-popup';

    // Show loading state first
    popup.innerHTML = `
      <div class="tidbit-content">
        <h5 class="tidbit-title">Loading...</h5>
        <p class="tidbit-content-text">Analyzing species data...</p>
      </div>
    `;

    // Generate tidbit data asynchronously
    try {
      const tidbitData = await this.generateTidbit(sighting, species, location);

      // Update popup with actual tidbit data
      popup.innerHTML = `
        <div class="tidbit-content">
          <h5 class="tidbit-title">${tidbitData.title}</h5>
          <p class="tidbit-content-text">${tidbitData.content}</p>
        </div>
      `;
    } catch (error) {
      console.error('Error generating tidbit:', error);
      // Fallback content on error
      popup.innerHTML = `
        <div class="tidbit-content">
          <h5 class="tidbit-title">Plant Detection</h5>
          <p class="tidbit-content-text">${species} detected. Analysis available in detailed view.</p>
        </div>
      `;
    }

    return popup;
  }

  async calculateLocalStats(species, location) {
    let totalDetections = 0;
    let totalRemovals = 0;
    let activeSightings = 0;
    let confidences = [];
    let firstDate = new Date();
    const nearbySpeciesMap = new Map();

    try {
      // Import API here to avoid circular dependency
      const { SightingsAPI } = await import('./api.js');

      // Fetch ALL sightings including removed ones
      const { data: allSightings } = await SightingsAPI.list('', true); // includeRemoved = true

      // Filter sightings within 1km radius
      allSightings.forEach(sighting => {
        if (sighting.location?.coordinates) {
          const [lng, lat] = sighting.location.coordinates;
          const distance = this.calculateDistance(location.lat, location.lng, lat, lng);

          if (distance <= 1000) { // 1km radius in meters
            const sightingSpecies = sighting.analysis?.predictedSpecies || 'Unknown';

            // Count nearby species (only active ones for nearby species list)
            if (!sighting.isRemoved) {
              if (!nearbySpeciesMap.has(sightingSpecies)) {
                nearbySpeciesMap.set(sightingSpecies, 0);
              }
              nearbySpeciesMap.set(sightingSpecies, nearbySpeciesMap.get(sightingSpecies) + 1);
            }

            // Stats for this specific species
            if (sightingSpecies.toLowerCase() === species.toLowerCase()) {
              totalDetections++;

              if (sighting.isRemoved) {
                totalRemovals++;
              } else {
                activeSightings++;
              }

              if (sighting.analysis?.confidence) {
                confidences.push(sighting.analysis.confidence);
              }

              if (sighting.createdAt) {
                const sightingDate = new Date(sighting.createdAt);
                if (!isNaN(sightingDate.getTime()) && sightingDate < firstDate) {
                  firstDate = sightingDate;
                }
              }
            }
          }
        }
      });
    } catch (error) {
      console.warn('Failed to fetch sightings for stats:', error);
      // Fallback to existing cluster data if API fails
      for (const [key, cluster] of this.markerClusters) {
        const distance = this.calculateDistance(location.lat, location.lng, cluster.lat, cluster.lng);
        if (distance <= 1) {
          cluster.sightings.forEach(sighting => {
            const sightingSpecies = sighting.analysis?.predictedSpecies || 'Unknown';
            if (!nearbySpeciesMap.has(sightingSpecies)) {
              nearbySpeciesMap.set(sightingSpecies, 0);
            }
            nearbySpeciesMap.set(sightingSpecies, nearbySpeciesMap.get(sightingSpecies) + 1);
            if (sightingSpecies.toLowerCase() === species.toLowerCase()) {
              totalDetections++;
              if (sighting.isRemoved) {
                totalRemovals++;
              } else {
                activeSightings++;
              }
              if (sighting.analysis?.confidence) {
                confidences.push(sighting.analysis.confidence);
              }
            }
          });
        }
      }
    }

    // Convert nearby species to sorted array
    const nearbySpecies = Array.from(nearbySpeciesMap.entries())
      .map(([species, count]) => ({ species, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5); // Top 5

    // Calculate days since first detection
    const daysSinceFirst = totalDetections > 0 ? Math.floor((new Date() - firstDate) / (1000 * 60 * 60 * 24)) : null;

    return {
      totalDetections,
      totalRemovals,
      activeSightings,
      avgConfidence: confidences.length > 0 ? confidences.reduce((a, b) => a + b, 0) / confidences.length : 0,
      firstDetected: totalDetections > 0 ? firstDate.toLocaleDateString() : 'N/A',
      daysSinceFirst,
      nearbySpeciesCount: nearbySpeciesMap.size,
      nearbySpecies
    };
  }

  async generateTidbit(sighting, species, location) {
    const llmData = sighting.analysis?.llm?.details;
    const confidence = sighting.analysis?.confidence || 0;
    const speciesInfo = llmData?.advisory_content?.species_identification;
    const legalInfo = llmData?.advisory_content?.legal_status;
    const riskLevel = llmData?.risk_level;
    const invasiveStatus = llmData?.invasive_status;

    // Calculate local statistics for more accurate tidbits
    let localStats = null;
    try {
      localStats = await this.calculateLocalStats(species, location);
    } catch (error) {
      console.warn('Could not calculate local stats for tidbit:', error);
    }

    const tidbits = [
      // 1. Species Scientific Information
      {
        condition: () => speciesInfo?.scientific_name || speciesInfo?.family,
        title: 'Scientific Classification',
        content: () => {
          const parts = [];
          if (speciesInfo?.scientific_name) parts.push(`Scientific name: ${speciesInfo.scientific_name}`);
          if (speciesInfo?.family) parts.push(`Family: ${speciesInfo.family}`);
          if (speciesInfo?.common_names) parts.push(`Also known as: ${speciesInfo.common_names}`);
          return parts.join('. ') || 'Scientific classification data available in full analysis.';
        }
      },

      // 2. Legal Status & NEMBA Category
      {
        condition: () => legalInfo?.nemba_category || legalInfo?.legal_requirements,
        title: 'Legal Status',
        content: () => {
          const parts = [];
          if (legalInfo?.nemba_category) parts.push(`NEMBA Category: ${legalInfo.nemba_category}`);
          if (legalInfo?.legal_requirements) parts.push(`Legal requirement: ${legalInfo.legal_requirements}`);
          return parts.join('. ') || 'This species has specific legal requirements for management.';
        }
      },

      // 3. Risk Assessment
      {
        condition: () => riskLevel || invasiveStatus,
        title: 'Risk Assessment',
        content: () => {
          const parts = [];
          if (riskLevel) parts.push(`Risk level: ${riskLevel}`);
          if (invasiveStatus) parts.push('Confirmed invasive species');
          if (localStats && localStats.activeSightings > 1) {
            parts.push(`${localStats.activeSightings} active sightings in 1km radius`);
          }
          return parts.join('. ') || 'Risk assessment completed - see analysis for details.';
        }
      },

      // 4. Physical Description Insights
      {
        condition: () => llmData?.advisory_content?.physical_description || llmData?.description,
        title: 'Identification Features',
        content: () => {
          const desc = llmData?.advisory_content?.physical_description || llmData?.description;
          if (desc && desc.length > 100) {
            return desc.substring(0, 97) + '...';
          }
          return desc || 'Physical characteristics detailed in full analysis.';
        }
      },

      // 5. Origin and Distribution
      {
        condition: () => llmData?.origin || llmData?.advisory_content?.distribution,
        title: 'Origin & Distribution',
        content: () => {
          const parts = [];
          if (llmData?.origin) parts.push(`Native to: ${llmData.origin}`);
          const distribution = llmData?.advisory_content?.distribution || llmData?.where_found;
          if (distribution && distribution !== 'Not found') {
            const shortDist = distribution.length > 80 ? distribution.substring(0, 77) + '...' : distribution;
            parts.push(`Distribution: ${shortDist}`);
          }
          return parts.join('. ') || 'Origin and distribution information available.';
        }
      },

      // 6. Control Methods
      {
        condition: () => llmData?.advisory_content?.control_methods || llmData?.treatment,
        title: 'Control Methods',
        content: () => {
          const control = llmData?.advisory_content?.control_methods || llmData?.treatment;
          if (control && control !== 'Not found') {
            const shortControl = control.length > 120 ? control.substring(0, 117) + '...' : control;
            return shortControl;
          }
          return 'Control methods available in detailed analysis.';
        }
      },

      // 7. Detection Confidence & AI Analysis
      {
        condition: () => confidence > 0,
        title: 'AI Detection Analysis',
        content: () => {
          const confidencePercent = (confidence * 100).toFixed(1);
          let accuracy = 'lower accuracy';
          if (confidence > 0.9) accuracy = 'very high accuracy';
          else if (confidence > 0.8) accuracy = 'high accuracy';
          else if (confidence > 0.6) accuracy = 'moderate accuracy';

          const parts = [`AI confidence: ${confidencePercent}% (${accuracy})`];
          if (llmData) parts.push('Enhanced with expert knowledge analysis');
          return parts.join('. ');
        }
      },

      // 8. Local Area Impact
      {
        condition: () => localStats && (localStats.activeSightings > 0 || localStats.totalRemovals > 0),
        title: 'Local Area Impact',
        content: () => {
          if (!localStats) return 'Analyzing local area impact...';

          const parts = [];
          if (localStats.activeSightings > 1) {
            parts.push(`${localStats.activeSightings} active sightings within 1km`);
          }
          if (localStats.totalRemovals > 0) {
            parts.push(`${localStats.totalRemovals} previous removals in area`);
          }
          if (localStats.nearbySpeciesCount > 1) {
            parts.push(`${localStats.nearbySpeciesCount} different invasive species detected nearby`);
          }

          return parts.length > 0 ? parts.join('. ') : 'First detection in this immediate area.';
        }
      },

      // 9. Temporal Patterns & Monitoring
      {
        condition: () => localStats && localStats.daysSinceFirst !== null,
        title: 'Detection Patterns',
        content: () => {
          if (!localStats) return 'Analyzing detection patterns...';

          const parts = [];
          if (localStats.daysSinceFirst > 0) {
            parts.push(`First detected ${localStats.daysSinceFirst} days ago in this area`);
          }
          if (localStats.avgConfidence > 0 && localStats.avgConfidence <= 1) {
            parts.push(`Average detection confidence: ${(localStats.avgConfidence * 100).toFixed(1)}%`);
          }

          // Monitoring recommendation based on risk and local density
          const isHighRisk = riskLevel?.toLowerCase().includes('high') || riskLevel?.toLowerCase().includes('severe');
          const isDense = localStats.activeSightings > 3;

          if (isHighRisk || isDense) {
            parts.push('Weekly monitoring recommended');
          } else {
            parts.push('Bi-weekly monitoring recommended');
          }

          return parts.join('. ') || 'Monitoring schedule available in management recommendations.';
        }
      },

      // 10. Action Priority & Urgency
      {
        condition: () => invasiveStatus || riskLevel || localStats,
        title: 'Action Priority',
        content: () => {
          let priority = 'Medium';
          const reasons = [];

          // Calculate priority based on multiple factors
          if (riskLevel?.toLowerCase().includes('severe') || riskLevel?.toLowerCase().includes('high')) {
            priority = 'High';
            reasons.push(`${riskLevel} risk level`);
          }

          if (invasiveStatus) {
            if (priority !== 'High') priority = 'High';
            reasons.push('confirmed invasive status');
          }

          if (confidence > 0.9) {
            reasons.push('very high identification confidence');
          }

          if (localStats && localStats.activeSightings > 3) {
            if (priority === 'Medium') priority = 'High';
            reasons.push('multiple local sightings');
          }

          const reasonText = reasons.length > 0 ? ` due to ${reasons.join(', ')}` : '';
          return `${priority} priority for removal action${reasonText}.`;
        }
      }
    ];

    // Filter tidbits that meet their conditions and have valid content
    const validTidbits = tidbits.filter(tidbit => {
      try {
        return tidbit.condition() && tidbit.content();
      } catch (error) {
        console.warn('Error evaluating tidbit condition:', error);
        return false;
      }
    });

    // If no specific tidbits are available, return a fallback
    if (validTidbits.length === 0) {
      return {
        title: 'Plant Detection',
        content: `${species} detected with ${(confidence * 100).toFixed(1)}% confidence. Additional analysis may be available.`
      };
    }

    // Return a random valid tidbit
    const selectedTidbit = validTidbits[Math.floor(Math.random() * validTidbits.length)];
    return {
      title: selectedTidbit.title,
      content: selectedTidbit.content()
    };
  }

  closeMultiPopup() {
    const backdrop = document.querySelector('.multi-popup-backdrop');
    if (backdrop) {
      backdrop.style.opacity = '0';
      setTimeout(() => backdrop.remove(), 300);
    }

    if (this.handleEscClose) {
      document.removeEventListener('keydown', this.handleEscClose);
      this.handleEscClose = null;
    }
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
      background: var(--panel);
      backdrop-filter: blur(2px);
      padding: 2rem;
      border-radius: 8px;
      max-width: 400px;
      width: 90%;
      text-align: center;
      color: var(--text);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    `;

    modalContent.innerHTML = `
      <h3 style="margin-top: 0; color: #dc2626;">Confirm Removal</h3>
      <p style="color: var(--text);">Are you sure you want to mark this <strong>${speciesName}</strong> sighting as removed?</p>
      <p style="font-size: 0.9rem; color: var(--text);"><em>This indicates the plant has been physically removed from the location.</em></p>
      <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 1.5rem;">
        <button id="confirm-removal" style="background: #dc2626; color: var(--background); border: none; padding: 0.75rem 1.5rem; border-radius: 4px; cursor: pointer; font-weight: 600;">Mark as Removed</button>
        <button id="cancel-removal" style="background: #6b7280; color: var(--background); border: none; padding: 0.75rem 1.5rem; border-radius: 4px; cursor: pointer;">Cancel</button>
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
        this.hideClusterPopup();

        // Update sightings page and stats
        if (window.sightingsPageExists) {
          window.location.reload(); // Refresh sightings page to reflect removal
        } else {
          // If not on sightings page, trigger a data refresh for stats
          // This ensures removal stats are updated across all pages
          if (typeof window.refreshSightingsData === 'function') {
            window.refreshSightingsData();
          }
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
          this.hideClusterPopup();
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
            const selector = document.querySelector('.cluster-popup');
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
      color: var(--background);
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

  updateMarkerWithAnalysis(sightingId, llmData) {
    console.log('updateMarkerWithAnalysis called for:', sightingId, 'LLM data:', !!llmData);
    // Find the cluster that contains this sighting
    let found = false;
    for (const [key, cluster] of this.markerClusters) {
      const sightingIndex = cluster.sightings.findIndex(s => s._id === sightingId);
      if (sightingIndex !== -1) {
        console.log('Found sighting in cluster, updating with LLM data');
        // Update the sighting data with LLM analysis
        cluster.sightings[sightingIndex].analysis.llm = llmData;
        // No need to update the visual marker since it only shows basic info
        // The detailed analysis will be available when the marker is clicked
        found = true;
        break;
      }
    }
    if (!found) {
      console.warn('Could not find sighting in any cluster for LLM update:', sightingId);
      console.log('Available clusters:', Array.from(this.markerClusters.keys()));
    }
    return found;
  }

  // New simple cluster popup implementation
  showClusterPopup(cluster, markerContainer) {
    console.log('showClusterPopup called', cluster, markerContainer);

    try {
      // Remove any existing cluster popup
      this.hideClusterPopup();

      // Create popup container
      const popup = document.createElement('div');
      popup.className = 'cluster-popup';

      // Use shared date formatting function from sightings.js
      const formatDate = window.fmtDate || ((dateString) => {
        if (!dateString) return 'Unknown date';
        try {
          return new Date(dateString).toLocaleDateString();
        } catch (e) {
          return 'Invalid date';
        }
      });

      popup.innerHTML = `
        <div class="cluster-popup-header">
          <span>${cluster.sightings.length} Sightings Here</span>
          <button class="cluster-popup-close">&times;</button>
        </div>
        <div class="cluster-popup-content">
          ${cluster.sightings.map(sighting => `
            <div class="cluster-item" data-sighting-id="${sighting._id}">
${(sighting.imageUrl || sighting.imagePath) && typeof (sighting.imageUrl || sighting.imagePath) === 'string' && ((sighting.imageUrl || sighting.imagePath).startsWith('/') || (sighting.imageUrl || sighting.imagePath).startsWith('http')) ?
                `<img src="${sighting.imageUrl || sighting.imagePath}" alt="Plant" class="cluster-item-image" style="min-height: 60px;" onerror="this.style.display='none'" />` :
                `<div class="cluster-item-image no-image-placeholder" style="min-height: 60px;"></div>`}
              <div class="cluster-item-info">
                <div class="cluster-item-species">${sighting.analysis?.predictedSpecies || 'Unknown Species'}</div>
                <div class="cluster-item-date">${formatDate(sighting.createdAt)}</div>
                ${this.isInvasiveSpecies && this.isInvasiveSpecies(sighting) ? '<div class="cluster-item-invasive">INVASIVE</div>' : ''}
              </div>
            </div>
          `).join('')}
        </div>
      `;

      // Position popup near the marker
      if (markerContainer && markerContainer.getBoundingClientRect) {
        const markerRect = markerContainer.getBoundingClientRect();
        const mapElement = document.getElementById('map');
        if (mapElement) {
          const mapRect = mapElement.getBoundingClientRect();

          popup.style.position = 'absolute';
          popup.style.left = (markerRect.left - mapRect.left + 20) + 'px';
          popup.style.top = (markerRect.top - mapRect.top - 200) + 'px';
          popup.style.zIndex = '1000';

          // Add to map
          mapElement.appendChild(popup);
        } else {
          console.error('Map element not found');
          return;
        }
      } else {
        console.error('Invalid markerContainer', markerContainer);
        return;
      }

      // Add event listeners
      const closeBtn = popup.querySelector('.cluster-popup-close');
      if (closeBtn) {
        closeBtn.addEventListener('click', () => {
          this.hideClusterPopup();
        });
      }

      // Add click listeners to each item
      popup.querySelectorAll('.cluster-item').forEach(item => {
        item.addEventListener('click', () => {
          const sightingId = item.dataset.sightingId;
          const sighting = cluster.sightings.find(s => s._id === sightingId);
          if (sighting) {
            this.hideClusterPopup();
            this.onMarkerClick(sightingId, sighting);
          }
        });
      });

      // Close on outside click
      setTimeout(() => {
        const handleOutsideClick = (e) => {
          if (!popup.contains(e.target) && (!markerContainer || !markerContainer.contains(e.target))) {
            this.hideClusterPopup();
            document.removeEventListener('click', handleOutsideClick);
          }
        };
        document.addEventListener('click', handleOutsideClick);
      }, 100);

      console.log('Cluster popup created successfully');
    } catch (error) {
      console.error('Error in showClusterPopup:', error);
    }
  }

  hideClusterPopup() {
    const existingPopup = document.querySelector('.cluster-popup');
    if (existingPopup) {
      existingPopup.remove();
    }
  }

  // Update addClusterListeners to use new popup
  addNewClusterListeners(marker, cluster, markerContainer) {
    // Simple click-only popup for clusters
    marker.addListener('click', () => {
      if (cluster.sightings.length <= 1) {
        this.onMarkerClick(cluster.sightings[0]._id, cluster.sightings[0]);
        return;
      }
      this.showClusterPopup(cluster, markerContainer);
    });
  }
}

const mapProxy = new MapLoaderProxy();
export { mapProxy }; 