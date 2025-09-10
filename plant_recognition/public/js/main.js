import { AuthAPI, AnalyzeAPI, SightingsAPI } from "./api.js";
import { loadGoogleMaps, initMap, addMarker } from "./map.js";
import { startVideo, stopVideo } from "./video.js";
import { pickFile } from "./upload.js";
import { startSSE } from "./sse.js";
import { addDetectionCard, setLLMCompleted } from "./ui.js";

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
    if (!navigator.geolocation) return res(null);
    navigator.geolocation.getCurrentPosition(
      (p) => res({ lat: p.coords.latitude, lng: p.coords.longitude }),
      () => res(null),
      { enableHighAccuracy: true, timeout: 7000 }
    );
  });
}

async function boot() {
  try {
    await AuthAPI.me().catch(() => {
      location.href = "/";
    });
    userLoc = (await geolocate()) || { lat: -25.8408, lng: 28.2395 };
    await loadGoogleMaps();
    const map = initMap("map", userLoc, 13);

    try {
      const box = "";
      const { data } = await SightingsAPI.list(box);
      data.forEach((d) => {
        const [lng, lat] = d.location?.coordinates || [
          userLoc.lng,
          userLoc.lat,
        ];
        addMarker({
          lat,
          lng,
          title: d.analysis?.predictedSpecies || "Sighting",
        });
      });
    } catch {}

    startSSE((msg) => {
      if (msg.type === "analysis_done") {
        setLLMCompleted(msg.sightingId, msg.llm);
      }
    });

    els.btnVideo.addEventListener("click", async () => {
      els.videoPanel.classList.remove("hidden");
      await startVideo(async (blob) => {
        try {
          console.log('Analyzing video frame');
          const res = await AnalyzeAPI.analyze(blob, {
            lat: userLoc.lat,
            lng: userLoc.lng,
            fromVideo: true,
          });
          console.log('Video frame response:', res);
          addDetectionCard(els.list, {
            ...res,
            predictedSpecies: res.predictedSpecies,
            confidence: res.confidence,
            imageUrl: res.imageUrl,
          });
          addMarker({
            lat: userLoc.lat,
            lng: userLoc.lng,
            title: res.predictedSpecies,
          });
        } catch (error) {
          console.error('Video frame analysis failed:', error);
          // Add error card
          addDetectionCard(els.list, {
            sightingId: 'error-' + Date.now(),
            predictedSpecies: 'Analysis Failed',
            confidence: 0,
            imageUrl: null,
          });
        }
      });
    });

    els.btnStop.addEventListener("click", () => {
      stopVideo();
      els.videoPanel.classList.add("hidden");
    });

    els.btnUpload.addEventListener("click", () => {
      pickFile("file-input", async (file) => {
        try {
          console.log('Uploading file:', file.name);
          const res = await AnalyzeAPI.analyze(file, {
            lat: userLoc.lat,
            lng: userLoc.lng,
            fromVideo: false,
          });
          console.log('Upload response:', res);
          addDetectionCard(els.list, {
            ...res,
            predictedSpecies: res.predictedSpecies,
            confidence: res.confidence,
            imageUrl: res.imageUrl,
          });
          addMarker({
            lat: userLoc.lat,
            lng: userLoc.lng,
            title: res.predictedSpecies,
          });
        } catch (error) {
          console.error('Upload failed:', error);
          // Add error card to show user what happened
          addDetectionCard(els.list, {
            sightingId: 'error-' + Date.now(),
            predictedSpecies: 'Upload Failed',
            confidence: 0,
            imageUrl: null,
          });
        }
      });
    });
  } catch (e) {
    console.error("Boot error", e);
    location.href = "/";
  }
}

boot();
