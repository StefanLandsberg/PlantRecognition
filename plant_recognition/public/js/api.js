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
export async function postForm(url, formData, options = {}) {
  const fetchOptions = {
    method: 'POST',
    body: formData,
    credentials: 'include',
    ...options // Allow passing AbortController signal
  };

  const res = await fetch(url, fetchOptions);
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
  list: (bbox, includeRemoved = false) => {
    let url = '/api/sightings';
    const params = [];
    if (bbox) params.push(`bbox=${bbox}`);
    if (includeRemoved) params.push('includeRemoved=true');
    if (params.length > 0) url += '?' + params.join('&');
    return getJSON(url);
  }
};

// Analyze
export const AnalyzeAPI = {
  analyze: (blob, { lat, lng, fromVideo }, options = {}) => {
    const fd = new FormData();
    fd.append('image', blob, 'frame.jpg');
    if (lat != null) fd.append('lat', String(lat));
    if (lng != null) fd.append('lng', String(lng));
    if (fromVideo != null) fd.append('fromVideo', String(fromVideo));
    return postForm('/api/analyze', fd, options);
  }
};

// Storage
export const StorageAPI = {
  getStatus: () => getJSON('/api/storage/status'),
  cleanup: (days = 30) => postJSON('/api/storage/cleanup', { days })
};

// Account Storage Preferences
export const AccountAPI = {
  updateStoragePreference: (storagePreference) => postJSON('/api/account/storage-preference', { storagePreference })
};
