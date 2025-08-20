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

//code for password recommendations
// Get a reference to the input field and all requirement elements
const passwordInput = document.getElementById('pwd');
const lengthRequirement = document.getElementById('length');
const uppercaseRequirement = document.getElementById('uppercase');
const numberRequirement = document.getElementById('number');
const specialRequirement = document.getElementById('special');

// Add an event listener for the 'keyup' event
passwordInput.addEventListener('keyup', () => {
    const password = passwordInput.value;

    // 1. Check for password length
    if (password.length >= 8) {
        lengthRequirement.classList.remove('invalid');
        lengthRequirement.classList.add('valid');
    } else {
        lengthRequirement.classList.remove('valid');
        lengthRequirement.classList.add('invalid');
    }

    // 2. Check for at least one uppercase letter (A-Z)
    if (/[A-Z]/.test(password)) {
        uppercaseRequirement.classList.remove('invalid');
        uppercaseRequirement.classList.add('valid');
    } else {
        uppercaseRequirement.classList.remove('valid');
        uppercaseRequirement.classList.add('invalid');
    }

    // 3. Check for at least one number (0-9)
    if (/\d/.test(password)) {
        numberRequirement.classList.remove('invalid');
        numberRequirement.classList.add('valid');
    } else {
        numberRequirement.classList.remove('valid');
        numberRequirement.classList.add('invalid');
    }

    // 4. Check for at least one special character
    // This regular expression matches common special characters
    if (/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>/?]/.test(password)) {
        specialRequirement.classList.remove('invalid');
        specialRequirement.classList.add('valid');
    } else {
        specialRequirement.classList.remove('valid');
        specialRequirement.classList.add('invalid');
    }
});

