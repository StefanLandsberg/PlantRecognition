// /js/account/utils.js

/**
 * Validates if a string is a correctly formatted email address.
 * @param {string} v The email string to validate.
 * @returns {boolean} True if the email is valid, false otherwise.
 */
export const isValidEmail = (v) => {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(v);
};