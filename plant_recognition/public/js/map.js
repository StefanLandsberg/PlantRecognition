let map, markers = [];

export async function loadGoogleMaps() {
  const key = window.APP_CONFIG?.GOOGLE_MAPS_API_KEY;
  if (!key) throw new Error('Missing Google Maps API Key');
  const script = document.getElementById('maps-loader');
  script.src = `https://maps.googleapis.com/maps/api/js?key=${key}`;
  await new Promise((res, rej) => {
    script.onload = res; script.onerror = rej;
  });
}

export function initMap(elId='map', center={lat:-25.8408, lng:28.2395}, zoom=13) {
  map = new google.maps.Map(document.getElementById(elId), { center, zoom, mapId: 'DEMO_MAP_ID' });
  return map;
}

export function addMarker({ lat, lng, title }) {
  const m = new google.maps.Marker({ position: { lat, lng }, map, title });
  markers.push(m);
  return m;
}

export function fitToMarkers() {
  const b = new google.maps.LatLngBounds();
  markers.forEach(m => b.extend(m.getPosition()));
  if (!b.isEmpty()) map.fitBounds(b);
}
