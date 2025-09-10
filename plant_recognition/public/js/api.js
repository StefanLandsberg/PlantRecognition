const base = '';
const common = { credentials: 'include', headers: { 'Content-Type': 'application/json' } };

export async function postJSON(url, body) {
  console.log('postJSON called:', url, body);
  try {
    const res = await fetch(url, { ...common, method: 'POST', body: JSON.stringify(body) });
    console.log('fetch response:', res.status, res.statusText);
    if (!res.ok) {
      const errorText = await res.text();
      console.log('API error response:', errorText);
      throw new Error(errorText);
    }
    const result = await res.json();
    console.log('API success response:', result);
    return result;
  } catch (error) {
    console.error('postJSON error:', error);
    throw error;
  }
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
  register: (username, email, password) => postJSON('/api/auth/register', { username, email, password }),
  login: (username, password) => postJSON('/api/auth/login', { username, password }),
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
