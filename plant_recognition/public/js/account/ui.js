// /js/account/ui.js

/**
 * Toggles the loading state of a button element.
 * @param {HTMLButtonElement} btn The button element.
 * @param {boolean} on True to set loading state, false to remove it.
 */
export const setLoading = (btn, on) => {
  if (!btn) return;
  btn.disabled = !!on;
  btn.classList.toggle("is-loading", !!on);
  if (on) {
    btn.dataset._label = btn.textContent;
    btn.textContent = "Please wait…";
  } else if (btn.dataset._label) {
    btn.textContent = btn.dataset._label;
  }
};

/**
 * Displays an error message in a specified element.
 * @param {string} id The ID of the element to display the error in.
 * @param {string} msg The error message to show.
 */
export const showErr = (id, msg) => {
  const el = document.getElementById(id);
  if (el) el.textContent = msg || "";
};