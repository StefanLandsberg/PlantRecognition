import * as Auth from '../services/auth.service.js';

const cookieOpts = {
  httpOnly: true,
  sameSite: 'lax',
  secure: false // set true if behind HTTPS
};

export async function register(req, res, next) {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: 'Email and password required' });
    const user = await Auth.register(email, password);
    res.json({ success: true, user: { id: user._id, email: user.email } });
  } catch (e) { next(e); }
}

export async function login(req, res, next) {
  try {
    const { email, password } = req.body;
    const { user, token } = await Auth.login(email, password);
    res.cookie('token', token, cookieOpts);
    res.json({ success: true, user: { id: user._id, email: user.email } });
  } catch (e) { next(e); }
}

export async function me(req, res) {
  res.json({ user: { id: req.auth.userId } });
}

export async function logout(req, res) {
  res.clearCookie('token');
  res.json({ success: true });
}
