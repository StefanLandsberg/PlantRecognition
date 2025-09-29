import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import User from '../models/User.js';
import { CONFIG } from '../utils/config.js';

export async function register(username, email, password) {
  let existing = await User.findOne({ email });
  if (existing) throw new Error('Email already in use');
  existing = await User.findOne({ username });
  if (existing) throw new Error('Email already in use');
  const hash = await bcrypt.hash(password, 10);
  const user = await User.create({ username, email, passwordHash: hash });
  return user;
}

export async function login(username, password) {
  if (!username || !password) {
    throw new Error('Username and password are required');
  }
  
  const user = await User.findOne({ username });
  if (!user) throw new Error('Invalid credentials. No User found');
  
  if (!user.passwordHash) {
    throw new Error('User account is corrupted - no password hash found');
  }
  
  const ok = await bcrypt.compare(password, user.passwordHash);
  if (!ok) throw new Error('Invalid credentials. Wrong password');
  const token = jwt.sign(
    { sub: user._id.toString(), role: user.role },
    CONFIG.JWT_SECRET,
    { expiresIn: CONFIG.JWT_EXPIRES }
  );
  return { user, token };
}
