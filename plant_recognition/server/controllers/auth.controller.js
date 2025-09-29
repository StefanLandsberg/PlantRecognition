import * as Auth from '../services/auth.service.js';

const cookieOpts = {
  httpOnly: true,
  sameSite: 'lax',
  secure: false // set true if behind HTTPS
};

export async function register(req, res, next) {
  try {
    const { username, email, password } = req.body;
    if (!email || !password || !username) return res.status(400).json({ error: 'Email and password required' });
    const user = await Auth.register(username, email, password);
    res.json({ success: true, user: { id: user._id, email: user.email } });
  } catch (e) { next(e); }
}

export async function login(req, res, next) {
  try {
    const { username, password } = req.body;
    const { user, token } = await Auth.login(username, password);
    res.cookie('token', token, cookieOpts);
    res.json({ success: true, user: { id: user._id, username: user.username } });
  } catch (e) { next(e); }
}

export async function me(req, res) {
  res.json({ user: { id: req.auth.userId } });
}

export async function logout(req, res) {
  res.clearCookie('token');
  res.json({ success: true });
}

export async function cleanup(req, res, next) {
  try {
    const User = (await import('../models/User.js')).default;
    await User.deleteMany({});
    res.json({ success: true, message: 'All users deleted' });
  } catch (e) { next(e); }
}
