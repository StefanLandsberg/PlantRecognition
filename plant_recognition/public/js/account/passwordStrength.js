// /js/account/passwordStrength.js

/**
 * Initializes the real-time password strength checker UI.
 */
export const initializePasswordStrengthChecker = () => {
  const passwordInput = document.getElementById('pwd-new');
  if (!passwordInput) return; // Exit if the element doesn't exist

  const requirements = {
    length: document.getElementById('length'),
    uppercase: document.getElementById('uppercase'),
    number: document.getElementById('number'),
    special: document.getElementById('special'),
  };

  const validate = (el, isValid) => {
    if (!el) return;
    el.classList.toggle('valid', isValid);
    el.classList.toggle('invalid', !isValid);
  };

  passwordInput.addEventListener('keyup', () => {
    const password = passwordInput.value;
    validate(requirements.length, password.length >= 8);
    validate(requirements.uppercase, /[A-Z]/.test(password));
    validate(requirements.number, /\d/.test(password));
    validate(requirements.special, /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>/?]/.test(password));
  });
};