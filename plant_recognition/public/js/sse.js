export function startSSE(onMsg) {
  const ev = new EventSource('/api/events', { withCredentials: true });
  ev.onmessage = (e) => {
    try { onMsg(JSON.parse(e.data)); } catch {}
  };
  ev.onerror = () => { /* network hiccups auto-reconnect */ };
  return ev;
}
