import { AuthAPI, AnalyzeAPI, SightingsAPI, VideoSessionAPI } from "./api.js";
import { startVideo, stopVideo } from "./video.js";
import { pickFile } from "./upload.js";
import { storageAwareUpload, pickFileWithStorage } from "./storageAwareUpload.js";
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

function upsertSessionDetection(detection) {
  try {
    const sessionDetections = JSON.parse(sessionStorage.getItem('sessionDetections') || '[]');
    const index = sessionDetections.findIndex((d) => d.sightingId === detection.sightingId);
    if (index === -1) {
      sessionDetections.push(detection);
    } else {
      sessionDetections[index] = detection;
    }
    sessionStorage.setItem('sessionDetections', JSON.stringify(sessionDetections));
  } catch (error) {
    console.warn('Failed to persist detection:', error);
  }
}

function setSessionDetectionCache(detections) {
  try {
    sessionStorage.setItem('sessionDetections', JSON.stringify(detections));
  } catch (error) {
    console.warn('Failed to update detection cache:', error);
  }
}

function restoreSessionDetections() {
  try {
    const container = document.getElementById('detections-list');
    if (!container) return;
    container.innerHTML = '';

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
      addDetectionCard(container, detection);
      // Always add LLM dropdown for all restored detections
      setTimeout(() => {
        if (detection.llm) {
          // If we have LLM data, show it (regardless of llmCompleted flag)
          setLLMCompleted(detection.sightingId, detection.llm);
        } else {
          // If no LLM data, still add the dropdown but show it's pending
          setLLMCompleted(detection.sightingId, null);
        }
      }, 100);
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
    } else {
      sessionDetections.push(newDetection);
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

function removeSessionDetection(sightingId) {
  if (!sightingId) return;
  try {
    const sessionDetections = JSON.parse(sessionStorage.getItem('sessionDetections') || '[]');
    const filtered = sessionDetections.filter(d => d.sightingId !== sightingId);
    sessionStorage.setItem('sessionDetections', JSON.stringify(filtered));
  } catch (error) {
    console.warn('Failed to remove session detection:', error);
  }
}

function removeDetectionCard(cardId) {
  if (!cardId) return;
  const domId = cardId.startsWith('det-') ? cardId : `det-${cardId}`;
  const card = document.getElementById(domId);
  if (card) {
    card.remove();
  }
}

function highlightExistingDetection(sightingId) {
  if (!sightingId) return false;
  const card = document.getElementById(`det-${sightingId}`);
  if (!card) return false;

  if (els.list && card !== els.list.firstChild) {
    els.list.insertBefore(card, els.list.firstChild);
  }

  card.classList.add('duplicate-highlight');
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  setTimeout(() => card.classList.remove('duplicate-highlight'), 2000);
  return true;
}

function handleDuplicateDetection(tempSightingId, res) {
  removeDetectionCard(tempSightingId);
  removeSessionDetection(tempSightingId);

  const foundExisting = highlightExistingDetection(res.originalDetection?.id);
  if (mapProxy && typeof mapProxy.showNotification === 'function') {
    const days = res.originalDetection?.daysAgo;
    const relative = typeof days === 'number'
      ? `${days} day${days === 1 ? '' : 's'} ago`
      : 'recently';
    const message = foundExisting
      ? `${res.predictedSpecies} already logged nearby. Showing the earlier sighting.`
      : `${res.predictedSpecies} was already logged ${relative} within the duplicate window.`;
    mapProxy.showNotification(message, 'warning');
  }

  return foundExisting;
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
let currentVideoSession = null; // Track current video session
let pendingVideoDetections = 0; // Frames currently being analyzed
let waitingToEndVideoSession = false;

function logVideoSessionWaitStatus() {
  if (waitingToEndVideoSession && pendingVideoDetections > 0) {
    console.log(`Waiting for ${pendingVideoDetections} pending frame(s) before ending video session`);
  }
}

async function finalizeVideoSessionIfNeeded() {
  if (!waitingToEndVideoSession) return;
  if (pendingVideoDetections > 0) {
    logVideoSessionWaitStatus();
    return;
  }

  if (!currentVideoSession) {
    waitingToEndVideoSession = false;
    return;
  }

  try {
    await VideoSessionAPI.end(currentVideoSession);
    console.log('Ended video session:', currentVideoSession);
  } catch (error) {
    console.error('Failed to end video session:', error);
    return;
  }

  currentVideoSession = null;
  waitingToEndVideoSession = false;
}

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
              const newDetection = {
                sightingId: msg.sighting._id,
                predictedSpecies: msg.sighting.analysis?.predictedSpecies || "Unknown Species",
                confidence: msg.sighting.analysis?.confidence || 0,
                imageUrl: msg.sighting.imagePath || msg.sighting.imageUrl,
                llmCompleted: false,
                llm: null
              };
              
              addDetectionCard(els.list, newDetection);

              // Save to session storage for persistence
              saveSessionDetection(newDetection);

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
          const newDetection = {
            sightingId: msg.sighting._id,
            predictedSpecies: msg.sighting.analysis?.predictedSpecies || "Unknown Species",
            confidence: msg.sighting.analysis?.confidence || 0,
            imageUrl: msg.sighting.imagePath || msg.sighting.imageUrl,
            llmCompleted: false,
            llm: null
          };
          
          addDetectionCard(els.list, newDetection);

          // Save to session storage for persistence
          saveSessionDetection(newDetection);

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
      } else if (msg.type === "video_session_updated") {
        console.log('Video session updated:', msg.sessionId, 'sighting:', msg.sightingId, 'status:', msg.newStatus);
        
        // If we're currently viewing this video session, refresh the timeline
        if (window.currentVideoSessionId === msg.sessionId) {
          console.log('Refreshing current video session timeline');
          // Trigger a refresh of the video review if it's open
          if (window.refreshVideoReview) {
            window.refreshVideoReview();
          }
        }
      }
    });

    els.btnVideo.addEventListener("click", async () => {
      els.videoPanel.classList.remove("hidden");

      waitingToEndVideoSession = false;
      pendingVideoDetections = 0;

      // Create new abort controller for this video session
      videoAbortController = new AbortController();

      // Start video session recording
      try {
        const sessionResponse = await VideoSessionAPI.start('live_video', userLoc.lat, userLoc.lng);
        currentVideoSession = sessionResponse.sessionId;
        console.log('Started video session:', currentVideoSession);
      } catch (error) {
        console.error('Failed to start video session:', error);
      }

      await startVideo(async (blob, timestamp) => {
        // Check if video session has been stopped
        if (videoAbortController?.signal.aborted || els.videoPanel.classList.contains("hidden")) {
          console.log('Video session stopped, skipping frame analysis');
          return;
        }
        if (!currentVideoSession) {
          console.warn('No active video session ID, skipping frame analysis');
          return;
        }
        const frameSessionId = currentVideoSession;
        pendingVideoDetections += 1;
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

          // Handle storage-aware frame processing
          let frameResult = null;
          try {
            frameResult = await storageAwareUpload(blob, {
              originalName: 'video-frame.jpg',
              uploadType: 'video_frame',
              sessionId: frameSessionId,
              timestamp: timestamp
            });
          } catch (error) {
            console.warn('Storage-aware upload failed, using server upload:', error);
          }

          const formData = new FormData();
          formData.append('image', blob, 'video-frame.jpg');
          formData.append('lat', userLoc.lat);
          formData.append('lng', userLoc.lng);
          formData.append('fromVideo', 'true');
          formData.append('videoSessionId', frameSessionId || '');
          formData.append('videoTimestamp', typeof timestamp === 'number' ? Math.round(timestamp) : '');
          
          // Add storage information if using local storage
          if (frameResult && frameResult.storageType === 'local') {
            formData.append('localFileId', frameResult.fileId);
            formData.append('storageType', 'local');
          }

          const res = await AnalyzeAPI.analyzeOnce(formData, { signal: videoAbortController.signal });

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
            handleDuplicateDetection(tempSightingId, res);
            return;
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

            // Add detection to video session
            if (frameSessionId) {
              try {
                console.log(`Adding detection to video session: ${frameSessionId}, sightingId: ${res.sightingId}, timestamp: ${timestamp}`);
                const result = await VideoSessionAPI.addDetection(frameSessionId, timestamp, res.imageUrl, res.sightingId);
                console.log(`✓ Detection added to video session:`, result);
              } catch (error) {
                console.error('✗ Failed to add detection to video session:', error);
              }
            } else {
              console.log('No captured video session ID - detection not added to session');
            }
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
          if (error.name === 'AbortError') {
            console.log('Video analysis aborted by user');
            // Remove the temp detection card since analysis was cancelled
            const tempCard = document.getElementById(`det-${tempSightingId}`);
            if (tempCard) {
              tempCard.remove();
            }
            return; // Don't log this as an error
          } else {
            handleAnalysisError(tempSightingId, error);
          }
        } finally {
          pendingVideoDetections = Math.max(0, pendingVideoDetections - 1);
          finalizeVideoSessionIfNeeded();
        }
      });
    });

    els.btnStop.addEventListener("click", async () => {
      console.log('Stopping video capture - allowing all pending analysis to complete');

      // Stop video capture and get recorded video
      const videoBlob = await stopVideo();
      els.videoPanel.classList.add("hidden");

      // Upload the recorded video if available
      if (videoBlob && currentVideoSession) {
        console.log('Uploading recorded video for session:', currentVideoSession);
        try {
          const { uploadVideoFile } = await import('./video.js');
          const videoUrl = await uploadVideoFile(videoBlob, currentVideoSession);
          if (videoUrl) {
            console.log('Video uploaded successfully:', videoUrl);
          } else {
            console.warn('Video upload failed');
          }
        } catch (error) {
          console.error('Error uploading video:', error);
        }
      }

      waitingToEndVideoSession = true;
      await finalizeVideoSessionIfNeeded();

      // Reset the abort controller to prevent new analysis requests
      // but don't abort existing ones - let them complete naturally
      if (videoAbortController) {
        videoAbortController = null;
      }

      console.log('Video stopped - all pending analysis will complete naturally');
    });

    els.btnUpload.addEventListener("click", () => {
      pickFileWithStorage("file-input", async (fileOrResult) => {
        // Handle both regular files and storage-aware results
        const file = fileOrResult.file || fileOrResult;
        const isLocalStorage = fileOrResult.storageType === 'local';
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

          const formData = new FormData();
          formData.append('image', file);
          formData.append('lat', userLoc.lat);
          formData.append('lng', userLoc.lng);
          formData.append('fromVideo', 'false');
          formData.append('videoSessionId', '');
          formData.append('videoTimestamp', '');
          
          // Add storage information if using local storage
          if (isLocalStorage && fileOrResult.fileId) {
            formData.append('localFileId', fileOrResult.fileId);
            formData.append('storageType', 'local');
          }

          const res = await AnalyzeAPI.analyzeOnce(formData, { signal: controller.signal });

          clearTimeout(timeoutId);
          console.log('Upload response:', res);

          // Check if this is a duplicate detection (same logic as video)
          if (res.duplicate) {
            handleDuplicateDetection(tempSightingId, res);
            return;
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
    if (error.name === 'AbortError') {
      // For abort errors, just remove the card instead of showing timeout
      detectionCard.remove();
      return;
    }
    
    requestAnimationFrame(() => {
      const speciesDiv = detectionCard.querySelector('.detection-species');
      const loadingDiv = detectionCard.querySelector('.classification-loading');

      if (speciesDiv) {
        speciesDiv.innerHTML = 'Species: Analysis Failed';
      }
      if (loadingDiv) loadingDiv.style.display = 'none';
    });
  }
}

boot().then(() => {
  // Restore any session detections after the page loads
  restoreSessionDetections();
});
