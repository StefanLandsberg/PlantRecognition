import { AuthAPI } from './api.js';

const loginForm = document.getElementById('login-form');
const registerForm = document.getElementById('register-form');
// links to toggle between logIn and register
const showRegisterLink = document.getElementById('show-register-link');
const showLoginLink = document.getElementById('show-login-link');

// Event listeners to toggle form visibility
showRegisterLink?.addEventListener('click', (e) => {
    e.preventDefault();
    loginForm?.classList.add('hidden');
    registerForm?.classList.remove('hidden');
});

showLoginLink?.addEventListener('click', (e) => {
    e.preventDefault();
    loginForm?.classList.remove('hidden');
    registerForm?.classList.add('hidden');
});

loginForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(loginForm);
  const username = formData.get('username'); 
  const password = formData.get('password');
  const err = document.getElementById('login-error');

  try {
    await AuthAPI.login(username, password);
    location.href = '/app';
  } catch (e) { err.textContent = 'Login failed. ' + e; }
});

registerForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(registerForm);
  const username = formData.get('username'); 
  const email = formData.get('email');
  const password = formData.get('password');
  const confirmPassword = formData.get('confirm-password')
  const err = document.getElementById('register-error');

  try{
    if (password === confirmPassword){
        try {
        await AuthAPI.register(username, email, password);
        err.textContent = 'Registered. You can login now.';
      } catch (e) { err.textContent = 'Registration failed. ' + e; }
    }
    else err.textContent = 'Passwords did not match';
  }catch (e) { err.textContent = 'Passwords did not match'; }
});
