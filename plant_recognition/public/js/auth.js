import { AuthAPI } from './api.js';

const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');

loginForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(loginForm);
  const email = fd.get('email'); const password = fd.get('password');
  const err = document.getElementById('login-error');
  try {
    await AuthAPI.login(email, password);
    location.href = '/app';
  } catch (e) { err.textContent = 'Login failed'; }
});

registerForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(registerForm);
  const email = fd.get('email'); const password = fd.get('password');
  const err = document.getElementById('register-error');
  try {
    await AuthAPI.register(email, password);
    err.textContent = 'Registered. You can login now.';
  } catch (e) { err.textContent = 'Registration failed'; }
});
