// UI Management and Screen Navigation
export class UIManager {
    constructor() {
        this.currentScreen = 'connection-screen';
        this.permissionCallbacks = new Map();
    }

    showScreen(screenId) {
        console.log('Switching to screen:', screenId);

        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });

        // Show target screen
        const targetScreen = document.getElementById(screenId);
        if (targetScreen) {
            targetScreen.classList.add('active');
            this.currentScreen = screenId;
            console.log('Screen switched successfully to:', screenId);
        } else {
            console.error('Screen not found:', screenId);
        }
    }

    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('connection-status');
        if (statusDiv) {
            statusDiv.textContent = message;
            statusDiv.className = `status-message ${type}`;
        }
    }

    showPermissionStatus(message, type = 'info') {
        const connectionStatus = document.getElementById('connection-status');
        if (connectionStatus) {
            connectionStatus.textContent = message;
            connectionStatus.className = `status-message ${type}`;
        }
    }

    showQRStatus(message, type = 'info') {
        const qrStatus = document.getElementById('qr-status-message');
        if (qrStatus) {
            qrStatus.textContent = message;
            qrStatus.className = `status-message ${type}`;
        }
    }

    clearPermissionStatus() {
        const connectionStatus = document.getElementById('connection-status');
        if (connectionStatus) {
            connectionStatus.textContent = '';
            connectionStatus.className = 'status-message';
        }
    }

    updateCameraStatus(status) {
        const statusElement = document.getElementById('classification-status');
        if (statusElement) {
            const statusText = statusElement.querySelector('.status-text');
            if (statusText) {
                statusText.textContent = status;
            }
        }
    }

    showError(title, message) {
        document.getElementById('error-title').textContent = title;
        document.getElementById('error-message').textContent = message;
        this.showScreen('error-screen');
    }

    showClassificationPopup(species, confidence) {
        // Create small notification
        let notification = document.getElementById('classification-notification');
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'classification-notification';
            notification.className = 'classification-notification';
            document.body.appendChild(notification);
        }

        notification.innerHTML = `
            <div class="notification-content">
                <div class="species-name">${species}</div>
                <div class="confidence-text">${(confidence * 100).toFixed(0)}%</div>
            </div>
        `;

        // Show notification with slide-in animation
        notification.classList.add('show');

        // Auto-hide after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    }

    // Settings UI Management
    loadCurrentSettings(resolution, storage) {
        // Load current resolution setting
        const resolutionSelect = document.getElementById('resolution-select');
        if (resolutionSelect) {
            const currentRes = `${resolution.width}x${resolution.height}`;
            resolutionSelect.value = currentRes;
        }

        // Load current storage setting
        const storageSelect = document.getElementById('storage-select');
        if (storageSelect) {
            storageSelect.value = storage;
        }
    }

    getSettingsFromUI() {
        const settings = {};

        // Get resolution setting
        const resolutionSelect = document.getElementById('resolution-select');
        if (resolutionSelect) {
            const [width, height] = resolutionSelect.value.split('x').map(Number);
            settings.resolution = { width, height };
        }

        // Get storage setting
        const storageSelect = document.getElementById('storage-select');
        if (storageSelect) {
            settings.storage = storageSelect.value;
        }

        return settings;
    }

    clearConnectionForm() {
        const codeInput = document.getElementById('companion-code');
        const statusDiv = document.getElementById('connection-status');

        if (codeInput) codeInput.value = '';
        if (statusDiv) statusDiv.textContent = '';
    }

    formatCodeInput(input) {
        // Format input as 6 digits
        input.value = input.value.replace(/\D/g, '').slice(0, 6);
    }

    setConnectionButtonState(connecting) {
        const connectBtn = document.getElementById('connect-btn');
        if (connectBtn) {
            connectBtn.disabled = connecting;
            connectBtn.textContent = connecting ? 'Connecting...' : 'Connect';
        }
    }
}