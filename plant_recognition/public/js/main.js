import { AuthAPI, AnalyzeAPI, SightingsAPI } from "./api.js";
import { startVideo, stopVideo } from "./video.js";
import { pickFile } from "./upload.js";
import { startSSE } from "./sse.js";
import { addDetectionCard, setLLMCompleted, showClassificationLoading, showLLMLoading } from "./ui.js";
// Import the singleton instance directly
import { mapProxy } from "./map.js";

// Session detection persistence
function saveSessionDetection(detection) {
  try {
    const sessionDetections = JSON.parse(sessionStorage.getItem('sessionDetections') || '[]');
    sessionDetections.push(detection);
    sessionStorage.setItem('sessionDetections', JSON.stringify(sessionDetections));
  } catch (error) {
    console.warn('Failed to save session detection:', error);
  }
}

function restoreSessionDetections() {
  try {
    const sessionDetections = JSON.parse(sessionStorage.getItem('sessionDetections') || '[]');

    // Filter out incomplete temp detections that are still "Analyzing..."
    const completedDetections = sessionDetections.filter(detection => {
      // Keep detections that have actual species data (not "Analyzing...")
      // and have real sighting IDs (not temp IDs starting with "temp-")
      return detection.predictedSpecies &&
             detection.predictedSpecies !== 'Analyzing...' &&
             detection.predictedSpecies !== 'Unknown' &&
             detection.sightingId &&
             !detection.sightingId.toString().startsWith('temp-');
    });

    completedDetections.forEach(detection => {
      // Check if a card with this ID already exists
      const existingCard = document.getElementById(`det-${detection.sightingId}`);
      if (!existingCard) {
        addDetectionCard(els.list, detection);
        if (detection.llmCompleted) {
          setTimeout(() => setLLMCompleted(detection.sightingId, detection.llm), 100);
        }
      }
    });

    // Update session storage to only keep completed detections
    sessionStorage.setItem('sessionDetections', JSON.stringify(completedDetections));

    console.log('Restored', completedDetections.length, 'completed session detections');
  } catch (error) {
    console.warn('Failed to restore session detections:', error);
  }
}

function updateSessionDetection(oldId, newDetection) {
  try {
    const sessionDetections = JSON.parse(sessionStorage.getItem('sessionDetections') || '[]');
    const index = sessionDetections.findIndex(d => d.sightingId === oldId);
    if (index !== -1) {
      sessionDetections[index] = newDetection;
      sessionStorage.setItem('sessionDetections', JSON.stringify(sessionDetections));
    }
  } catch (error) {
    console.warn('Failed to update session detection:', error);
  }
}

function clearSessionDetections() {
  try {
    sessionStorage.removeItem('sessionDetections');
  } catch (error) {
    console.warn('Failed to clear session detections:', error);
  }
}


const els = {
  btnLogout: document.getElementById("btn-logout"),
  btnVideo: document.getElementById("btn-video"),
  btnUpload: document.getElementById("btn-upload"),
  fileInput: document.getElementById("file-input"),
  videoPanel: document.getElementById("video-panel"),
  btnStop: document.getElementById("btn-stop"),
  list: document.getElementById("detections-list"),
};

// Debug: Log which elements were found
console.log('Elements found:', {
  btnLogout: !!els.btnLogout,
  btnVideo: !!els.btnVideo,
  btnUpload: !!els.btnUpload,
  fileInput: !!els.fileInput,
  videoPanel: !!els.videoPanel,
  btnStop: !!els.btnStop,
  list: !!els.list
});

let userLoc = null;
let videoAbortController = null; // Track active video analysis requests

async function geolocate() {
  return new Promise((res) => {
    if (!navigator.geolocation) {
      console.warn("Geolocation not supported");
      return res(null);
    }
    navigator.geolocation.getCurrentPosition(
      (p) => {
        console.log("Got user location:", p.coords.latitude, p.coords.longitude);
        res({ lat: p.coords.latitude, lng: p.coords.longitude });
      },
      (error) => {
        console.warn("Geolocation failed:", error.message);
        res(null);
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 300000 }
    );
  });
}

async function boot() {
  try {
    await AuthAPI.me().catch(() => {
      location.href = "/";
    });
    userLoc = (await geolocate()) || { lat: -25.8408, lng: 28.2395 };

    try {
      await mapProxy.loadGoogleMaps();
      const map = mapProxy.initMap("map", userLoc, 15);

      if (userLoc.lat !== -25.8408 || userLoc.lng !== 28.2395) {
        mapProxy.addMarker({
          lat: userLoc.lat,
          lng: userLoc.lng,
          title: "Your Location"
        });
      }
    } catch (mapError) {
      console.warn("Maps failed to load:", mapError);
      document.getElementById("map")?.remove();
    }

    try {
      const box = "";
      const { data } = await SightingsAPI.list(box);
      if (mapProxy.map) {
        data.forEach((d) => {
          if (d.location?.coordinates) {
            const [lng, lat] = d.location.coordinates;
            mapProxy.addMarker({
              lat,
              lng,
              title: d.analysis?.predictedSpecies || "Sighting",
              data: { sightingId: d._id, sighting: d }
            });
          }
        });
        mapProxy.fitToMarkers();
      }
    } catch {}

    startSSE((msg) => {
      if (msg.type === "analysis_done") {
        console.log('SSE analysis_done received for:', msg.sightingId, 'LLM:', !!msg.llm);
        setLLMCompleted(msg.sightingId, msg.llm);

        // Update session storage with LLM completion
        const sessionDetections = JSON.parse(sessionStorage.getItem('sessionDetections') || '[]');
        const detectionIndex = sessionDetections.findIndex(d => d.sightingId === msg.sightingId);
        if (detectionIndex !== -1) {
          sessionDetections[detectionIndex].llmCompleted = true;
          sessionDetections[detectionIndex].llm = msg.llm;
          sessionStorage.setItem('sessionDetections', JSON.stringify(sessionDetections));
        }

        // Update map marker with LLM analysis (with delay if needed)
        if (mapProxy.map) {
          // Try immediate update first
          const updateSuccess = mapProxy.updateMarkerWithAnalysis(msg.sightingId, msg.llm);

          // If not found, might be a race condition with marker creation
          if (!updateSuccess) {
            setTimeout(() => {
              console.log('Retrying map marker LLM update for:', msg.sightingId);
              mapProxy.updateMarkerWithAnalysis(msg.sightingId, msg.llm);
            }, 200);
          }
        }
      } else if (msg.type === "new_sighting") {
        // Check if detection card already exists (from upload/video handler)
        // Also check for temporary cards that might not have been updated yet
        const existingCard = document.getElementById(`det-${msg.sighting._id}`);
        const tempCards = document.querySelectorAll('[id^="det-temp-"]');
        console.log('SSE new_sighting event for:', msg.sighting._id, 'Existing card:', !!existingCard, 'Temp cards:', tempCards.length);

        // If we have temporary cards, wait a bit for them to be updated, then check again
        if (!existingCard && tempCards.length > 0) {
          console.log('Delaying SSE card creation due to temporary cards');
          setTimeout(() => {
            const existingCardDelayed = document.getElementById(`det-${msg.sighting._id}`);
            if (!existingCardDelayed) {
              console.log('Creating delayed detection card via SSE for:', msg.sighting._id);
              addDetectionCard(els.list, {
                sightingId: msg.sighting._id,
                predictedSpecies: msg.sighting.analysis?.predictedSpecies || "Unknown Species",
                confidence: msg.sighting.analysis?.confidence || 0,
                imageUrl: msg.sighting.imagePath || msg.sighting.imageUrl,
              });

              // Show LLM loading for the new detection only if it's a known species with confidence > 0
              if (msg.sighting.analysis?.predictedSpecies !== 'Unknown Species' &&
                  msg.sighting.analysis?.predictedSpecies !== 'Unknown species' &&
                  msg.sighting.analysis?.confidence > 0) {
                showLLMLoading(msg.sighting._id);
              }
            }
          }, 100); // 100ms delay
          return;
        }

        if (!existingCard) {
          // Only add detection card if it doesn't already exist
          console.log('Creating new detection card via SSE for:', msg.sighting._id);
          addDetectionCard(els.list, {
            sightingId: msg.sighting._id,
            predictedSpecies: msg.sighting.analysis?.predictedSpecies || "Unknown Species",
            confidence: msg.sighting.analysis?.confidence || 0,
            imageUrl: msg.sighting.imagePath || msg.sighting.imageUrl,
          });

          // Show LLM loading for all new detections
          showLLMLoading(msg.sighting._id);
        }

        // Add pin to map immediately
        if (mapProxy.map && msg.sighting.location?.coordinates) {
          const [lng, lat] = msg.sighting.location.coordinates;
          mapProxy.addMarker({
            lat,
            lng,
            title: msg.sighting.analysis?.predictedSpecies || "New Sighting",
            data: { sightingId: msg.sighting._id, sighting: msg.sighting }
          });
        }
      }
    });

    els.btnVideo.addEventListener("click", async () => {
      els.videoPanel.classList.remove("hidden");

      // Create new abort controller for this video session
      videoAbortController = new AbortController();

      await startVideo(async (blob) => {
        // Check if video session has been stopped
        if (videoAbortController?.signal.aborted) {
          console.log('Video session stopped, skipping frame analysis');
          return;
        }
        const tempSightingId = 'temp-' + Date.now();

        try {
          // 1. Immediate UI feedback with image preview
          const tempDetection = {
            sightingId: tempSightingId,
            predictedSpecies: 'Capturing...',
            confidence: 0,
            imageUrl: URL.createObjectURL(blob), // Show image immediately
          };
          addDetectionCard(els.list, tempDetection);
          saveSessionDetection(tempDetection);

          // 2. Update to "Classifying..." when API call starts
          setTimeout(() => {
            // Check if video session is still active
            if (videoAbortController?.signal.aborted) {
              return;
            }
            const card = document.getElementById(`det-${tempSightingId}`);
            if (card) {
              const speciesDiv = card.querySelector('.detection-species');
              if (speciesDiv) speciesDiv.innerHTML = 'Species: Classifying...';
              showClassificationLoading(tempSightingId);
            }
          }, 100);

          console.log('Analyzing video frame');

          // 3. Non-blocking API call with timeout
          // Use the video session abort controller
          const timeoutId = setTimeout(() => {
            if (videoAbortController && !videoAbortController.signal.aborted) {
              console.log('Video analysis timeout');
            }
          }, 10000); // 10s timeout

          const res = await AnalyzeAPI.analyze(blob, {
            lat: userLoc.lat,
            lng: userLoc.lng,
            fromVideo: true,
          }, { signal: videoAbortController.signal });

          clearTimeout(timeoutId);
          console.log('Video frame response:', res);

          // Check if video session is still active before processing response
          if (videoAbortController?.signal.aborted) {
            console.log('Video session stopped, ignoring analysis response');
            const tempCard = document.getElementById(`det-${tempSightingId}`);
            if (tempCard) {
              tempCard.remove();
            }
            return;
          }

          // Check if this is a duplicate detection
          if (res.duplicate) {
            // Check if there's already a duplicate card for this species
            const existingDuplicateCard = findExistingDuplicateCard(res.predictedSpecies);

            if (existingDuplicateCard) {
              // Remove the new detection card and move existing duplicate to top
              const newDetectionCard = document.getElementById(`det-${tempSightingId}`);
              if (newDetectionCard) {
                newDetectionCard.remove();
              }

              // Move existing duplicate card to top of the list
              const firstCard = els.list.firstChild;
              if (firstCard && firstCard !== existingDuplicateCard) {
                els.list.insertBefore(existingDuplicateCard, firstCard);
              }

              // Update the "days ago" text to reflect the most recent detection
              const confDiv = existingDuplicateCard.querySelector('.badges div:last-child');
              if (confDiv) {
                confDiv.textContent = `${res.originalDetection.daysAgo} days ago`;
              }

              return; // Exit early since we're reusing existing duplicate card
            }

            // Handle new duplicate detection (first duplicate for this species)
            const detectionCard = document.getElementById(`det-${tempSightingId}`);
            if (detectionCard) {
              detectionCard.id = `det-${res.sightingId}-duplicate`;

              // Update card to show "Previously Detected" status
              const speciesDiv = detectionCard.querySelector('.detection-species');
              const confDiv = detectionCard.querySelector('.badges div:last-child');
              const imgContainer = detectionCard.querySelector('img') ? detectionCard.querySelector('img').parentNode : detectionCard;

              if (speciesDiv) {
                speciesDiv.innerHTML = `Species: ${res.predictedSpecies} <span class="previous-detection-label">[PREVIOUSLY DETECTED]</span>`;
                // Link to original detection
                speciesDiv.setAttribute('onclick', `viewOriginalDetection('${res.originalDetection.id}')`);
                speciesDiv.style.cursor = 'pointer';
              }
              if (confDiv) {
                confDiv.textContent = `${res.originalDetection.daysAgo} days ago`;
                confDiv.style.color = '#f59e0b';
              }

              // Add current image if available
              if (res.imageUrl && !detectionCard.querySelector('img')) {
                const img = document.createElement('img');
                img.src = res.imageUrl;
                img.alt = 'duplicate detection';
                img.style.filter = 'grayscale(0.3) opacity(0.8)'; // Slightly faded to indicate duplicate
                imgContainer.insertBefore(img, detectionCard.querySelector('.classification-loading'));
              }

              // Hide classification loading and show duplicate status
              const loadingDiv = detectionCard.querySelector('.classification-loading');
              if (loadingDiv) {
                loadingDiv.style.display = 'none';
              }

              // Add "View Original" button
              const duplicateInfo = document.createElement('div');
              duplicateInfo.className = 'duplicate-info';
              duplicateInfo.style.cssText = `
                padding: 0.5rem;
                background: rgba(245, 158, 11, 0.1);
                border-radius: 6px;
                margin-top: 0.5rem;
                font-size: 0.75rem;
                color: #f59e0b;
                text-align: center;
              `;
              duplicateInfo.innerHTML = `
                <div>Original: ${(res.originalDetection.confidence * 100).toFixed(1)}% confidence</div>
                <button onclick="viewOriginalDetection('${res.originalDetection.id}')" class="warning-button">
                  View Original
                </button>
              `;
              detectionCard.appendChild(duplicateInfo);
            }
          } else {
            // Handle normal (new) detection with optimized updates
            updateDetectionCardFromResponse(tempSightingId, res);

            // Update session storage with real detection data
            updateSessionDetection(tempSightingId, {
              sightingId: res.sightingId,
              predictedSpecies: res.predictedSpecies,
              confidence: res.confidence,
              imageUrl: res.imageUrl,
            });
          }

          // Only add map marker for new detections (not duplicates)
          if (mapProxy.map && !res.duplicate) {
            mapProxy.addMarker({
              lat: userLoc.lat,
              lng: userLoc.lng,
              title: res.predictedSpecies,
              data: {
                sightingId: res.sightingId,
                sighting: {
                  _id: res.sightingId,
                  analysis: {
                    predictedSpecies: res.predictedSpecies,
                    confidence: res.confidence
                  },
                  location: {
                    coordinates: [userLoc.lng, userLoc.lat]
                  },
                  imagePath: res.imageUrl
                }
              }
            });
          }
        } catch (error) {
          // Don't show error for intentionally aborted video analysis
          if (error.name === 'AbortError' && videoAbortController?.signal.aborted) {
            console.log('Video analysis aborted by user');
            // Remove the temp detection card since analysis was cancelled
            const tempCard = document.getElementById(`det-${tempSightingId}`);
            if (tempCard) {
              tempCard.remove();
            }
          } else {
            handleAnalysisError(tempSightingId, error);
          }
        }
      });
    });

    els.btnStop.addEventListener("click", () => {
      console.log('Stopping video and aborting all analysis requests');

      // Abort all ongoing video analysis requests
      if (videoAbortController) {
        videoAbortController.abort();
        videoAbortController = null;
      }

      // Stop video capture
      stopVideo();
      els.videoPanel.classList.add("hidden");

      console.log('Video stopped and all requests aborted');
    });

    els.btnUpload.addEventListener("click", () => {
      pickFile("file-input", async (file) => {
        const tempSightingId = 'temp-' + Date.now();

        try {
          // 1. Immediate UI feedback with file preview
          const tempDetection = {
            sightingId: tempSightingId,
            predictedSpecies: 'Processing...',
            confidence: 0,
            imageUrl: URL.createObjectURL(file), // Show uploaded image immediately
          };
          addDetectionCard(els.list, tempDetection);
          saveSessionDetection(tempDetection);

          // 2. Update to "Classifying..." when API call starts
          setTimeout(() => {
            const card = document.getElementById(`det-${tempSightingId}`);
            if (card) {
              const speciesDiv = card.querySelector('.detection-species');
              if (speciesDiv) speciesDiv.innerHTML = 'Species: Classifying...';
              showClassificationLoading(tempSightingId);
            }
          }, 100);

          console.log('Uploading file:', file.name);

          // 3. Non-blocking API call with timeout
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 15000); // 15s timeout for uploads

          const res = await AnalyzeAPI.analyze(file, {
            lat: userLoc.lat,
            lng: userLoc.lng,
            fromVideo: false,
          }, { signal: controller.signal });

          clearTimeout(timeoutId);
          console.log('Upload response:', res);

          // Check if this is a duplicate detection (same logic as video)
          if (res.duplicate) {
            // Check if there's already a duplicate card for this species
            const existingDuplicateCard = findExistingDuplicateCard(res.predictedSpecies);

            if (existingDuplicateCard) {
              // Remove the new detection card and move existing duplicate to top
              const newDetectionCard = document.getElementById(`det-${tempSightingId}`);
              if (newDetectionCard) {
                newDetectionCard.remove();
              }

              // Move existing duplicate card to top of the list
              const firstCard = els.list.firstChild;
              if (firstCard && firstCard !== existingDuplicateCard) {
                els.list.insertBefore(existingDuplicateCard, firstCard);
              }

              // Update the "days ago" text to reflect the most recent detection
              const confDiv = existingDuplicateCard.querySelector('.badges div:last-child');
              if (confDiv) {
                confDiv.textContent = `${res.originalDetection.daysAgo} days ago`;
              }

              return; // Exit early since we're reusing existing duplicate card
            }

            // Handle new duplicate detection (first duplicate for this species)
            const detectionCard = document.getElementById(`det-${tempSightingId}`);
            if (detectionCard) {
              detectionCard.id = `det-${res.sightingId}-duplicate`;

              // Update card to show "Previously Detected" status
              const speciesDiv = detectionCard.querySelector('.detection-species');
              const confDiv = detectionCard.querySelector('.badges div:last-child');
              const imgContainer = detectionCard.querySelector('img') ? detectionCard.querySelector('img').parentNode : detectionCard;

              if (speciesDiv) {
                speciesDiv.innerHTML = `Species: ${res.predictedSpecies} <span class="previous-detection-label">[PREVIOUSLY DETECTED]</span>`;
                speciesDiv.setAttribute('onclick', `viewOriginalDetection('${res.originalDetection.id}')`);
                speciesDiv.style.cursor = 'pointer';
              }
              if (confDiv) {
                confDiv.textContent = `${res.originalDetection.daysAgo} days ago`;
                confDiv.style.color = '#f59e0b';
              }

              // Add current image if available (with duplicate styling)
              if (res.imageUrl && !detectionCard.querySelector('img')) {
                const img = document.createElement('img');
                img.src = res.imageUrl;
                img.alt = 'duplicate detection';
                img.style.filter = 'grayscale(0.3) opacity(0.8)'; // Slightly faded to indicate duplicate
                imgContainer.insertBefore(img, detectionCard.querySelector('.classification-loading'));
              }

              // Hide classification loading and show duplicate status
              const loadingDiv = detectionCard.querySelector('.classification-loading');
              if (loadingDiv) {
                loadingDiv.style.display = 'none';
              }

              // Add "View Original" button
              const duplicateInfo = document.createElement('div');
              duplicateInfo.className = 'duplicate-info';
              duplicateInfo.style.cssText = `
                padding: 0.5rem;
                background: rgba(245, 158, 11, 0.1);
                border-radius: 6px;
                margin-top: 0.5rem;
                font-size: 0.75rem;
                color: #f59e0b;
                text-align: center;
              `;
              duplicateInfo.innerHTML = `
                <div>Original: ${(res.originalDetection.confidence * 100).toFixed(1)}% confidence</div>
                <button onclick="viewOriginalDetection('${res.originalDetection.id}')" class="warning-button">
                  View Original
                </button>
              `;
              detectionCard.appendChild(duplicateInfo);
            }
          } else {
            // Handle normal (new) detection
            console.log('Upload response received for:', tempSightingId, '->', res.sightingId);
            updateDetectionCardFromResponse(tempSightingId, res);

            // Update session storage with real detection data
            updateSessionDetection(tempSightingId, {
              sightingId: res.sightingId,
              predictedSpecies: res.predictedSpecies,
              confidence: res.confidence,
              imageUrl: res.imageUrl,
            });
          }

          // Only add map marker for new detections (not duplicates)
          if (mapProxy.map && !res.duplicate) {
            mapProxy.addMarker({
              lat: userLoc.lat,
              lng: userLoc.lng,
              title: res.predictedSpecies,
              data: {
                sightingId: res.sightingId,
                sighting: {
                  _id: res.sightingId,
                  analysis: {
                    predictedSpecies: res.predictedSpecies,
                    confidence: res.confidence
                  },
                  location: {
                    coordinates: [userLoc.lng, userLoc.lat]
                  },
                  imagePath: res.imageUrl
                }
              }
            });
          }
        } catch (error) {
          handleAnalysisError(tempSightingId, error);
        }
      });
    });

    console.log('Reached event listener setup section');
    // Mobile QR Code button functionality moved to nav.js

  } catch (e) {
    console.error("Boot error", e);
    location.href = "/";
  }
}

// showQRModal function moved to nav.js


// Optimized helper functions for better classification flow performance
function updateDetectionCardFromResponse(tempId, res) {
  const detectionCard = document.getElementById(`det-${tempId}`);
  if (!detectionCard) return;

  // Batch DOM updates for performance
  requestAnimationFrame(() => {
    detectionCard.id = `det-${res.sightingId}`;

    const speciesDiv = detectionCard.querySelector('.detection-species');
    const confDiv = detectionCard.querySelector('.badges div:last-child');
    const imgContainer = detectionCard.querySelector('img') ? detectionCard.querySelector('img').parentNode : detectionCard;

    if (speciesDiv) {
      speciesDiv.innerHTML = `Species: ${res.predictedSpecies}`;
      speciesDiv.setAttribute('onclick', `toggleMainDetectionLLM('${res.sightingId}')`);
    }
    if (confDiv) confDiv.textContent = `Conf: ${(res.confidence*100).toFixed(1)}%`;

    // Add image if available
    if (res.imageUrl && !detectionCard.querySelector('img')) {
      const img = document.createElement('img');
      img.src = res.imageUrl;
      img.alt = 'detection';
      imgContainer.insertBefore(img, detectionCard.querySelector('.classification-loading'));
    }

    // Show LLM loading immediately after classification
    showLLMLoading(res.sightingId);
  });
}

function handleAnalysisError(tempId, error) {
  console.error('Classification failed:', error);

  const detectionCard = document.getElementById(`det-${tempId}`);
  if (detectionCard) {
    requestAnimationFrame(() => {
      const speciesDiv = detectionCard.querySelector('.detection-species');
      const loadingDiv = detectionCard.querySelector('.classification-loading');

      if (speciesDiv) {
        if (error.name === 'AbortError') {
          speciesDiv.innerHTML = 'Species: Request Timeout';
        } else {
          speciesDiv.innerHTML = 'Species: Analysis Failed';
        }
      }
      if (loadingDiv) loadingDiv.style.display = 'none';
    });
  }
}

// Helper function to find existing duplicate card for a species
function findExistingDuplicateCard(species) {
  const allCards = document.querySelectorAll('[id^="det-"][id$="-duplicate"]');
  for (const card of allCards) {
    const speciesDiv = card.querySelector('.detection-species');
    if (speciesDiv && speciesDiv.textContent.includes(species)) {
      return card;
    }
  }
  return null;
}

// Global function to view original detection (for duplicate detection buttons)
window.viewOriginalDetection = function(sightingId) {
  // Navigate to sightings page with the specific sighting highlighted
  window.location.href = `/sightings#${sightingId}`;
};

boot().then(() => {
  // Restore any session detections after the page loads
  restoreSessionDetections();
});
