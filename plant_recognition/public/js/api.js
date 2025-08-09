const base = '';
const common = { credentials: 'include', headers: { 'Content-Type': 'application/json' } };

export async function postJSON(url, body) {
  const res = await fetch(url, { ...common, method: 'POST', body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
export async function getJSON(url) {
  const res = await fetch(url, { ...common, method: 'GET' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
export async function postForm(url, formData) {
  const res = await fetch(url, { method: 'POST', body: formData, credentials: 'include' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Auth
export const AuthAPI = {
  register: (email, password) => postJSON('/api/auth/register', { email, password }),
  login: (email, password) => postJSON('/api/auth/login', { email, password }),
  logout: () => postJSON('/api/auth/logout', {}),
  me: () => getJSON('/api/auth/me')
};

// Sightings
export const SightingsAPI = {
  list: (bbox) => getJSON('/api/sightings' + (bbox ? `?bbox=${bbox}` : ''))
};

// Analyze
export const AnalyzeAPI = {
  analyze: (blob, { lat, lng, fromVideo }) => {
    const fd = new FormData();
    fd.append('image', blob, 'frame.jpg');
    if (lat != null) fd.append('lat', String(lat));
    if (lng != null) fd.append('lng', String(lng));
    if (fromVideo != null) fd.append('fromVideo', String(fromVideo));
    return postForm('/api/analyze', fd);
  }
};
