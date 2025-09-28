import { AuthAPI, AccountAPI } from './api.js';

document.addEventListener('DOMContentLoaded', () => {
    // 1. Get all the necessary elements from the page
    const themeToggle = document.getElementById('theme-toggle');
    const colorBlindnessSelect = document.getElementById('color-blindness');
    const languageSelect = document.getElementById('language-select');
    const storagePreferenceSelect = document.getElementById('storage-preference');
    const storageWarning = document.getElementById('storage-warning');
    const applyBtn = document.getElementById('apply-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    const body = document.body;

    let currentUser = null;
    let originalStoragePreference = null;

    // --- Helper function to apply styles instantly ---
    function applySettings() {
        // Apply Dark/Light mode
        if (themeToggle.checked) {
            body.classList.remove('light-mode');
            body.classList.add('dark-mode');
        } else {
            body.classList.remove('dark-mode');
            body.classList.add('light-mode');
        }

        // Apply Color Blindness filter
        // First, remove any existing filter classes
        body.classList.remove('protanopia', 'deuteranopia', 'tritanopia');
        
        // Then, add the selected one (if it's not 'none')
        const selectedFilter = colorBlindnessSelect.value;
        if (selectedFilter !== 'none') {
            body.classList.add(selectedFilter);
        }
    }

    // 2. Load and display the current saved settings when the page opens
    async function loadCurrentSettings() {
        // Check if dark mode is saved in localStorage, or detect from body class
        let isDarkModeSaved;

        if (localStorage.getItem('darkMode') !== null) {
            // If setting exists in localStorage, use it
            isDarkModeSaved = localStorage.getItem('darkMode') === 'true';
        } else {
            // If no setting exists, detect current theme from body classes
            // Default to dark mode (true) since that's the application default
            isDarkModeSaved = !body.classList.contains('light-mode');
        }

        const colorBlindModeSaved = localStorage.getItem('colorBlindMode') || 'none';
        const languageSaved = localStorage.getItem('language') || 'en';

        // Load user data for storage preference
        try {
            currentUser = await AuthAPI.me();
            originalStoragePreference = currentUser.storagePreference || 'server';
            storagePreferenceSelect.value = originalStoragePreference;
        } catch (error) {
            console.error('Failed to load user data:', error);
            storagePreferenceSelect.value = 'server'; // Default
        }

        // Update the controls on the page to reflect the saved settings
        themeToggle.checked = isDarkModeSaved;
        colorBlindnessSelect.value = colorBlindModeSaved;
        languageSelect.value = languageSaved;

        // Apply the current settings to the settings page itself for consistency
        applySettings();
    }

    // Handle storage preference change to show warning
    function handleStoragePreferenceChange() {
        const newPreference = storagePreferenceSelect.value;

        if (originalStoragePreference === 'server' && newPreference === 'local') {
            storageWarning.classList.remove('hidden');
        } else {
            storageWarning.classList.add('hidden');
        }
    }
    
    // --- Event Listeners ---

    // 3. Handle the "Apply" button click
    applyBtn.addEventListener('click', async (event) => {
        event.preventDefault(); // Prevent the link from navigating immediately

        // Save other settings (like dark mode) to localStorage as before
        localStorage.setItem('darkMode', themeToggle.checked);
        localStorage.setItem('colorBlindMode', colorBlindnessSelect.value);

        // Get the newly selected language from the dropdown
        const selectedLanguage = languageSelect.value;
        localStorage.setItem('language', selectedLanguage);

        // Handle storage preference change
        const newStoragePreference = storagePreferenceSelect.value;
        if (newStoragePreference !== originalStoragePreference) {
            try {
                // Show confirmation if switching from server to local
                if (originalStoragePreference === 'server' && newStoragePreference === 'local') {
                    const confirmed = await showModal.confirm(
                        'Switch to Local Storage',
                        'This will delete all your server-stored images permanently. Your images will only be stored on this device going forward. Are you sure?',
                        {
                            danger: true,
                            confirmText: 'Switch to Local',
                            cancelText: 'Keep Server Storage'
                        }
                    );

                    if (!confirmed) {
                        storagePreferenceSelect.value = originalStoragePreference;
                        return;
                    }
                }

                // Update storage preference
                const result = await AccountAPI.updateStoragePreference(newStoragePreference);

                if (result.cleanupResult) {
                    await showModal.alert(
                        'Storage Updated',
                        `Storage preference updated to ${newStoragePreference}. ${result.cleanupResult.message}`
                    );
                } else {
                    await showModal.success(
                        'Storage Updated',
                        `Storage preference updated to ${newStoragePreference} storage.`
                    );
                }

                originalStoragePreference = newStoragePreference;
                storageWarning.classList.add('hidden');
            } catch (error) {
                console.error('Failed to update storage preference:', error);
                await showModal.error(
                    'Update Failed',
                    'Failed to update storage preference. Please try again.'
                );
                storagePreferenceSelect.value = originalStoragePreference;
                return;
            }
        }

        // Redirect with the new language parameter.
        // The server will see '?lang=...' and set the cookie automatically (from Step 1).
        window.location.href = `/settings?lang=${selectedLanguage}`;
    });

    // 4. Handle the "Cancel" button click
    cancelBtn.addEventListener('click', () => {
        // Go back to the previous page without saving anything
        window.history.back();
    });

    // Add storage preference change listener
    storagePreferenceSelect.addEventListener('change', handleStoragePreferenceChange);

    // Add immediate preview for settings changes
    themeToggle.addEventListener('change', applySettings);
    colorBlindnessSelect.addEventListener('change', applySettings);


    // --- Initial Execution ---
    // Load the current settings as soon as the page is ready
    loadCurrentSettings();
});