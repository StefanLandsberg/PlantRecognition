// Application Entry Point
import { MobileCompanionApp } from './app.js';

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.companionApp = new MobileCompanionApp();
});