import { AuthAPI, AnalyzeAPI, SightingsAPI } from "./api.js";
import { startVideo, stopVideo } from "./video.js";
import { pickFile } from "./upload.js";
import { startSSE } from "./sse.js";
import { addDetectionCard, setLLMCompleted, showClassificationLoading, showLLMLoading } from "./ui.js";
// Import the singleton instance directly
import { mapProxy } from "./map.js";


const els = {
  btnLogout: document.getElementById("btn-logout"),
  btnVideo: document.getElementById("btn-video"),
  btnUpload: document.getElementById("btn-upload"),
  fileInput: document.getElementById("file-input"),
  videoPanel: document.getElementById("video-panel"),
  btnStop: document.getElementById("btn-stop"),
  list: document.getElementById("detections-list"),
};

let userLoc = null;

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
        setLLMCompleted(msg.sightingId, msg.llm);
      } else if (msg.type === "new_sighting") {
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
      await startVideo(async (blob) => {
        const tempSightingId = 'temp-' + Date.now();

        try {
          // Show classification loading first
          addDetectionCard(els.list, {
            sightingId: tempSightingId,
            predictedSpecies: 'Analyzing...',
            confidence: 0,
            imageUrl: null,
          });
          showClassificationLoading(tempSightingId);

          console.log('Analyzing video frame');
          const res = await AnalyzeAPI.analyze(blob, {
            lat: userLoc.lat,
            lng: userLoc.lng,
            fromVideo: true,
          });
          console.log('Video frame response:', res);

          // Update the card with classification results and show LLM loading
          const detectionCard = document.getElementById(`det-${tempSightingId}`);
          if (detectionCard) {
            detectionCard.id = `det-${res.sightingId}`;
            const speciesDiv = detectionCard.querySelector('.detection-species');
            const confDiv = detectionCard.querySelector('.badges div:last-child');
            const imgContainer = detectionCard.querySelector('img') ? detectionCard.querySelector('img').parentNode : detectionCard;

            if (speciesDiv) {
              speciesDiv.innerHTML = `Species: ${res.predictedSpecies}`;
              // Update the onclick to use the correct sighting ID
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

            // Show LLM loading
            showLLMLoading(res.sightingId);
          }

          if (mapProxy.map) {
            mapProxy.addMarker({
              lat: userLoc.lat,
              lng: userLoc.lng,
              title: res.predictedSpecies,
            });
          }
        } catch (error) {
          console.error('Video frame analysis failed:', error);
          // Update with error state
          const detectionCard = document.getElementById(`det-${tempSightingId}`);
          if (detectionCard) {
            const speciesDiv = detectionCard.querySelector('.detection-species');
            if (speciesDiv) speciesDiv.innerHTML = 'Species: Analysis Failed';
            detectionCard.querySelector('.classification-loading').style.display = 'none';
          }
        }
      });
    });

    els.btnStop.addEventListener("click", () => {
      stopVideo();
      els.videoPanel.classList.add("hidden");
    });

    els.btnUpload.addEventListener("click", () => {
      pickFile("file-input", async (file) => {
        const tempSightingId = 'temp-' + Date.now();

        try {
          // Show classification loading first
          addDetectionCard(els.list, {
            sightingId: tempSightingId,
            predictedSpecies: 'Analyzing...',
            confidence: 0,
            imageUrl: null,
          });
          showClassificationLoading(tempSightingId);

          console.log('Uploading file:', file.name);
          const res = await AnalyzeAPI.analyze(file, {
            lat: userLoc.lat,
            lng: userLoc.lng,
            fromVideo: false,
          });
          console.log('Upload response:', res);

          // Update the card with classification results and show LLM loading
          const detectionCard = document.getElementById(`det-${tempSightingId}`);
          if (detectionCard) {
            detectionCard.id = `det-${res.sightingId}`;
            const speciesDiv = detectionCard.querySelector('.detection-species');
            const confDiv = detectionCard.querySelector('.badges div:last-child');
            const imgContainer = detectionCard.querySelector('img') ? detectionCard.querySelector('img').parentNode : detectionCard;

            if (speciesDiv) {
              speciesDiv.innerHTML = `Species: ${res.predictedSpecies}`;
              // Update the onclick to use the correct sighting ID
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

            // Show LLM loading
            showLLMLoading(res.sightingId);
          }

          if (mapProxy.map) {
            mapProxy.addMarker({
              lat: userLoc.lat,
              lng: userLoc.lng,
              title: res.predictedSpecies,
            });
          }
        } catch (error) {
          console.error('Upload failed:', error);
          // Update with error state
          const detectionCard = document.getElementById(`det-${tempSightingId}`);
          if (detectionCard) {
            const speciesDiv = detectionCard.querySelector('.detection-species');
            if (speciesDiv) speciesDiv.innerHTML = 'Species: Upload Failed';
            detectionCard.querySelector('.classification-loading').style.display = 'none';
          }
        }
      });
    });
  } catch (e) {
    console.error("Boot error", e);
    location.href = "/";
  }
}

boot();
