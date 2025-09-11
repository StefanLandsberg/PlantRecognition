export class MapLoaderProxy {
  constructor() {
    this.map = null;
    this.markers = [];
    this.isScriptLoaded = false;
    this.loadingPromise = null;
  }

  async loadGoogleMaps() {
    if (this.isScriptLoaded) {
      return this.loadingPromise; // Return the existing promise
    }

    if (this.loadingPromise) {
      return this.loadingPromise; // Still loading, return the same promise
    }

    const key = window.APP_CONFIG?.GOOGLE_MAPS_API_KEY;
    if (!key) throw new Error('Missing Google Maps API Key');

    const script = document.createElement('script');
    script.id = 'maps-loader';
    script.src = `https://maps.googleapis.com/maps/api/js?key=${key}`;
    document.head.appendChild(script);

    this.loadingPromise = new Promise((resolve, reject) => {
      script.onload = () => {
        this.isScriptLoaded = true;
        resolve();
      };
      script.onerror = reject;
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

  addMarker({ lat, lng, title }) {
    if (!this.map) {
      throw new Error('Map is not initialized. Call initMap() first.');
    }
    const m = new google.maps.Marker({ position: { lat, lng }, map: this.map, title });
    this.markers.push(m);
    return m;
  }

  fitToMarkers() {
    if (!this.map || this.markers.length === 0) {
      console.warn('Cannot fit to bounds: map not initialized or no markers present.');
      return;
    }
    const b = new google.maps.LatLngBounds();
    this.markers.forEach(m => b.extend(m.getPosition()));
    if (!b.isEmpty()) {
      this.map.fitBounds(b);
    }
  }
}