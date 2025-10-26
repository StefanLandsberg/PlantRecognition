// js/sightings/sightings.js

import {
  load,
  toggleLLMDropdown,
  showLLMSection,
  toggleDropdown,
  switchTab,
  updateTimelineView,
  handleAlertAction,
  handleAlertDismiss,
  removeSighting,
  confirmRemoval,
  closeRemovalModal,
  triggerWeatherAlert,
} from "./ui.js";

import { loadVideoSessions } from "./videoReview.js";

// Attach functions to the global window object for inline HTML event handlers
window.toggleLLMDropdown = toggleLLMDropdown;
window.showLLMSection = showLLMSection;
window.toggleDropdown = toggleDropdown;
window.switchTab = switchTab;
window.updateTimelineView = updateTimelineView;
window.handleAlertAction = handleAlertAction;
window.handleAlertDismiss = handleAlertDismiss;
window.removeSighting = removeSighting;
window.confirmRemoval = confirmRemoval;
window.closeRemovalModal = closeRemovalModal;
window.loadVideoSessions = loadVideoSessions;

// Attach global test functions
window.triggerWeatherAlert = triggerWeatherAlert;

// Initial load of the application
load();