import { AuthAPI, AnalyzeAPI, SightingsAPI } from './api.js';
import { loadGoogleMaps, initMap, addMarker } from './map.js';
import { startVideo, stopVideo } from './video.js';
import { pickFile } from './upload.js';
import { startSSE } from './sse.js';
import { addDetectionCard, setLLMCompleted } from './ui.js';
 
const els = {
  btnLogout: document.getElementById('btn-logout'),
  btnVideo: document.getElementById('btn-video'),
  btnUpload: document.getElementById('btn-upload'),
  fileInput: document.getElementById('file-input'),
  videoPanel: document.getElementById('video-panel'),
  btnStop: document.getElementById('btn-stop'),
  list: document.getElementById('detections-list')
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
    // Require auth
    await AuthAPI.me().catch(() => { location.href = '/'; });
    userLoc = await geolocate() || { lat: -25.8408, lng: 28.2395 };
    await loadGoogleMaps();
    const map = initMap('map', userLoc, 13);
 
    // load existing markers
    try {
      const box = ''; // you can compute bbox from viewport if desired
      const { data } = await SightingsAPI.list(box);
      data.forEach(d => {
        const [lng, lat] = d.location?.coordinates || [userLoc.lng, userLoc.lat];
        addMarker({ lat, lng, title: d.analysis?.predictedSpecies || 'Sighting' });
      });
    } catch {}
 
    // SSE for LLM completion
    startSSE((msg) => {
      if (msg.type === 'analysis_done') {
        setLLMCompleted(msg.sightingId, msg.llm);
      }
    });
 
    // actions
    els.btnLogout.addEventListener('click', async () => {
      await AuthAPI.logout();
      location.href = '/';
    });
 
    els.btnVideo.addEventListener('click', async () => {
      els.videoPanel.classList.remove('hidden');
      await startVideo(async (blob) => {
        const res = await AnalyzeAPI.analyze(blob, { lat: userLoc.lat, lng: userLoc.lng, fromVideo: true });
        addDetectionCard(els.list, { ...res, predictedSpecies: res.predictedSpecies, confidence: res.confidence, imageUrl: res.imageUrl });
        addMarker({ lat: userLoc.lat, lng: userLoc.lng, title: res.predictedSpecies });
      });
    });
 
    els.btnStop.addEventListener('click', () => {
      stopVideo();
      els.videoPanel.classList.add('hidden');
    });
 
    els.btnUpload.addEventListener('click', () => {
      pickFile('file-input', async (file) => {
        const res = await AnalyzeAPI.analyze(file, { lat: userLoc.lat, lng: userLoc.lng, fromVideo: false });
        addDetectionCard(els.list, { ...res, predictedSpecies: res.predictedSpecies, confidence: res.confidence, imageUrl: res.imageUrl });
        addMarker({ lat: userLoc.lat, lng: userLoc.lng, title: res.predictedSpecies });
      });
    });

        document.getElementById('menu-btn').addEventListener('click', () => {
      const menu = document.getElementById('menu-dropdown');
      menu.classList.toggle('hidden');
    });

    document.addEventListener('click', (e) => {
      const menu = document.getElementById('menu-dropdown');
      const btn = document.getElementById('menu-btn');
      if (!btn.contains(e.target) && !menu.contains(e.target)) {
        menu.classList.add('hidden');
      }
    });
 
  } catch (e) {
    console.error('Boot error', e);
    location.href = '/';
  }
}
 
boot();
 