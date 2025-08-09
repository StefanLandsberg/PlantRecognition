const clientsByUser = new Map(); // userId -> Set(res)

export function sseHandler(req, res) {
  const userId = req.auth.userId;
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  let set = clientsByUser.get(userId);
  if (!set) { set = new Set(); clientsByUser.set(userId, set); }
  set.add(res);

  req.on('close', () => {
    set.delete(res);
    if (set.size === 0) clientsByUser.delete(userId);
  });
}

export function publish(userId, payload) {
  const set = clientsByUser.get(userId?.toString());
  if (!set) return;
  const data = `data: ${JSON.stringify(payload)}\n\n`;
  for (const res of set) res.write(data);
}
