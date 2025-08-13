document.addEventListener('DOMContentLoaded', () => {
    // 1. Get all the necessary elements from the page
    const themeToggle = document.getElementById('theme-toggle');
    const colorBlindnessSelect = document.getElementById('color-blindness');
    const applyBtn = document.getElementById('apply-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    const body = document.body;

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
    function loadCurrentSettings() {
        // Get saved settings, providing defaults if none exist
        const isDarkModeSaved = localStorage.getItem('darkMode') === 'true';
        const colorBlindModeSaved = localStorage.getItem('colorBlindMode') || 'none';

        // Update the controls on the page to reflect the saved settings
        themeToggle.checked = isDarkModeSaved;
        colorBlindnessSelect.value = colorBlindModeSaved;

        // Apply the current settings to the settings page itself for consistency
        applySettings();
    }
    
    // --- Event Listeners ---

    // 3. Handle the "Apply" button click
    applyBtn.addEventListener('click', () => {
        // Save the current values from the controls to localStorage
        localStorage.setItem('darkMode', themeToggle.checked);
        localStorage.setItem('colorBlindMode', colorBlindnessSelect.value);

        // Apply the settings to the current page for immediate feedback
        applySettings();

        // Let the user know the settings were saved
        alert('Settings applied successfully!');
    });

    // 4. Handle the "Cancel" button click
    cancelBtn.addEventListener('click', () => {
        // Go back to the previous page without saving anything
        window.history.back();
    });
    
    // --- Initial Execution ---
    // Load the current settings as soon as the page is ready
    loadCurrentSettings();
});