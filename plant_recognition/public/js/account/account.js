// /js/account/account.js

import { initializeAccountPage } from "./handlers.js";
import { initializePasswordStrengthChecker } from "./passwordStrength.js";

/**
 * Bootstraps the entire account page functionality.
 */
const boot = () => {
  initializeAccountPage();

    initializePasswordStrengthChecker();
};

// Run the bootstrap function once the DOM is fully loaded.
document.addEventListener("DOMContentLoaded", boot);