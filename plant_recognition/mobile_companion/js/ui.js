// UI Management and Screen Navigation
export class UIManager {
    constructor() {
        this.currentScreen = 'connection-screen';
        this.permissionCallbacks = new Map();
        this.notificationTimeout = null;
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

    setConnectedUser(username) {
        const connectionLabel = document.getElementById('connection-user');
        if (!connectionLabel) return;

        if (!username || username === 'Disconnected') {
            connectionLabel.textContent = 'Disconnected';
        } else {
            connectionLabel.textContent = `Connected as ${username}`;
        }
    }

    showClassificationPopup(result = {}) {
        const species = result.predictedSpecies || 'Unknown Species';
        const confidence =
            typeof result.confidence === 'number'
                ? `${(result.confidence * 100).toFixed(1)}%`
                : 'Confidence unavailable';
        const duplicate = !!result.duplicate;
        const originalInfo = result.originalDetection;
        const duplicateMeta = duplicate
            ? originalInfo?.daysAgo != null
                ? `Already logged ${originalInfo.daysAgo} day${originalInfo.daysAgo === 1 ? '' : 's'} ago`
                : 'Already logged nearby'
            : 'New sighting saved to your dashboard';

        let notification = document.getElementById('classification-notification');
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'classification-notification';
            notification.className = 'classification-notification';
            document.body.appendChild(notification);
        }

        notification.innerHTML = `
            <div class="notification-content ${duplicate ? 'duplicate' : ''}">
                <div class="notification-header">
                    <div class="species-name">${species}</div>
                    <div class="confidence-text">${confidence}</div>
                </div>
                <div class="notification-meta">
                    <span class="status-pill ${duplicate ? 'warning' : 'success'}">
                        ${duplicate ? 'Previously Detected' : 'New Detection'}
                    </span>
                    <span>${duplicateMeta}</span>
                </div>
            </div>
        `;

        // Show notification with slide-in animation
        notification.classList.remove('show');
        // Force reflow for restart animation
        void notification.offsetWidth;
        notification.classList.add('show');
        notification.onclick = () => {
            notification.classList.remove('show');
        };

        if (this.notificationTimeout) {
            clearTimeout(this.notificationTimeout);
        }

        this.notificationTimeout = setTimeout(() => {
            notification.classList.remove('show');
        }, 4000);
    }

    // Settings UI Management
    loadCurrentSettings(resolution, storage) {
        // Load current resolution setting
        const resolutionSelect = document.getElementById('resolution-select');
        if (resolutionSelect) {
            const currentRes = `${resolution.width}x${resolution.height}`;
            resolutionSelect.value = currentRes;
        }

        // Display current storage setting (read-only)
        const storageDisplay = document.getElementById('storage-display');
        if (storageDisplay) {
            const storageText = storage === 'local' 
                ? 'Local Storage (Unlimited, device only)' 
                : 'Server Storage (2GB total, 90 days, backed up)';
            storageDisplay.textContent = storageText;
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

        // Storage setting is no longer user-configurable in companion app
        // It's inherited from the main app via QR code and WebSocket connection

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
