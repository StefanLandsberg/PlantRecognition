// Main Application Controller
import { CompanionConfig } from './config.js';
import { CameraManager } from './camera.js';
import { ConnectionManager } from './connection.js';
import { StreamingManager } from './streaming.js';
import { UIManager } from './ui.js';

export class MobileCompanionApp {
    constructor() {
        // Initialize modules
        this.config = new CompanionConfig();
        this.ui = new UIManager();
        this.camera = new CameraManager(this.config);
        this.connection = new ConnectionManager();
        this.streaming = new StreamingManager(this.camera, this.connection, this.config);

        // Application state
        this.resolution = this.config.loadResolutionSetting();
        this.storagePreference = this.config.loadStorageSetting();

        this.initializeApp();
    }

    initializeApp() {
        this.setupMessageHandlers();
        this.bindEvents();

        // Check if accessed via QR code (companion code in URL)
        const urlParams = new URLSearchParams(window.location.search);
        const qrCompanionCode = urlParams.get('code');
        const qrStoragePreference = urlParams.get('storage');

        console.log('Mobile companion initializing...');
        console.log('URL parameters:', window.location.search);
        console.log('QR companion code:', qrCompanionCode);
        console.log('QR storage preference:', qrStoragePreference);

        // If storage preference is provided via QR code, use it and save it
        if (qrStoragePreference && (qrStoragePreference === 'local' || qrStoragePreference === 'server')) {
            this.storagePreference = qrStoragePreference;
            this.config.saveStorageSetting(this.storagePreference);
            console.log('Using storage preference from QR code:', this.storagePreference);
        }

        if (qrCompanionCode && qrCompanionCode.length === 6) {
            console.log('QR code detected, auto-connecting...');
            this.handleQRConnection(qrCompanionCode);
        } else {
            console.log('No QR code detected, showing manual connection screen');
            this.ui.showScreen('connection-screen');
            this.ui.showPermissionStatus('Tap "Request Permissions" button when ready');
        }
    }

    setupMessageHandlers() {
        this.connection.registerMessageHandler('classification_result', (data) => {
            const result = data?.result || {};
            this.ui.showClassificationPopup(result);
            this.ui.updateCameraStatus(result.duplicate ? 'Previously detected nearby' : 'Classification complete');
        });

        this.connection.registerMessageHandler('analysis_complete', (data) => {
            // For mobile, we don't need to update analysis status since we use popup
        });

        this.connection.registerMessageHandler('error', (data) => {
            this.ui.showStatus(data.message, 'error');
        });

        this.connection.registerMessageHandler('connection_lost', (message) => {
            this.ui.showError('Connection Lost', message);
            this.ui.setConnectedUser('Disconnected');
        });

        this.connection.registerMessageHandler('session_replaced', (message) => {
            this.ui.showError('Session Replaced', message || 'Another companion session has been started. This session will be disconnected.');
            this.ui.setConnectedUser('Disconnected');
            // Optionally redirect to connection screen after a delay
            setTimeout(() => {
                this.ui.showScreen('connection');
            }, 3000);
        });

        this.connection.registerMessageHandler('reconnected', (message) => {
            this.ui.showStatus(message || 'Reconnected successfully', 'success');
            console.log('Companion reconnected successfully');
        });

        this.connection.registerMessageHandler('connection_confirmed', (data) => {
            if (data.user?.username) {
                this.ui.setConnectedUser(data.user.username);
            }
            if (data.storagePreference && data.storagePreference !== this.storagePreference) {
                this.storagePreference = data.storagePreference;
                this.config.saveStorageSetting(this.storagePreference);
                console.log('Storage preference updated from main app:', this.storagePreference);
                
                // Update the storage display if settings screen is visible
                const storageDisplay = document.getElementById('storage-display');
                if (storageDisplay) {
                    const storageText = this.storagePreference === 'local' 
                        ? 'Local Storage (Unlimited, device only)' 
                        : 'Server Storage (2GB total, 90 days, backed up)';
                    storageDisplay.textContent = storageText;
                }
            }
        });
    }

    bindEvents() {
        // Permission events
        document.getElementById('request-permissions-btn').addEventListener('click', () => this.requestPermissions());

        // Connection events
        document.getElementById('connect-btn').addEventListener('click', () => this.handleConnect());
        document.getElementById('companion-code').addEventListener('input', (e) => {
            this.ui.formatCodeInput(e.target);
        });

        // Camera events
        document.getElementById('capture-btn').addEventListener('click', () => this.captureImage());
        document.getElementById('gallery-btn').addEventListener('click', () => this.openGallery());
        document.getElementById('file-input').addEventListener('change', (e) => this.handleFileSelect(e));
        document.getElementById('switch-camera-btn').addEventListener('click', () => this.switchCamera());
        document.getElementById('stream-toggle-btn').addEventListener('click', () => this.toggleLiveStream());

        // Settings events
        const settingsBtn = document.getElementById('settings-btn');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => this.showSettings());
        }
        const saveSettingsBtn = document.getElementById('save-settings-btn');
        if (saveSettingsBtn) {
            saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        }
        const cancelSettingsBtn = document.getElementById('cancel-settings-btn');
        if (cancelSettingsBtn) {
            cancelSettingsBtn.addEventListener('click', () => this.cancelSettings());
        }

        // Navigation events
        document.getElementById('disconnect-btn').addEventListener('click', () => {
            this.disconnect().catch((error) => console.warn('Disconnect warning:', error));
        });
        document.getElementById('close-results').addEventListener('click', () => this.hideResults());
        document.getElementById('capture-another').addEventListener('click', () => this.hideResults());
        document.getElementById('retry-btn').addEventListener('click', () => this.ui.showScreen('connection-screen'));

        // Enter key on companion code input
        document.getElementById('companion-code').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleConnect();
            }
        });
    }

    async handleQRConnection(qrCompanionCode) {
        this.ui.showScreen('qr-connection-screen');
        this.ui.showQRStatus('Connected via QR code. Requesting permissions...');

        try {
            await this.requestPermissions();
            setTimeout(() => {
                this.connectToMainAppDirect(qrCompanionCode);
            }, 1000);
        } catch (error) {
            this.ui.showQRStatus('Permission request failed', 'error');
        }
    }

    async requestPermissions() {
        this.ui.showPermissionStatus('Requesting permissions... Tap Allow if prompted');
        this.ui.showQRStatus('Requesting permissions... Tap Allow if prompted');

        // Request camera permission first
        this.ui.showPermissionStatus('Testing camera access...');
        this.ui.showQRStatus('Testing camera access...');

        const cameraGranted = await this.camera.requestPermissions();

        if (cameraGranted) {
            this.ui.showPermissionStatus('Camera access granted', 'success');
            this.ui.showQRStatus('Camera access granted', 'success');
        } else {
            this.ui.showPermissionStatus('Camera denied - can still upload photos manually', 'warning');
            this.ui.showQRStatus('Camera denied - can still upload photos manually', 'warning');
        }

        // Wait before requesting location
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Request location permission
        this.ui.showPermissionStatus('Testing location access...');
        this.ui.showQRStatus('Testing location access...');

        let locationGranted = false;
        if (navigator.geolocation) {
            try {
                await new Promise((resolve, reject) => {
                    navigator.geolocation.getCurrentPosition(
                        (position) => {
                            console.log('Location permission granted');
                            this.ui.showPermissionStatus('Location access granted', 'success');
                            this.ui.showQRStatus('Location access granted', 'success');
                            locationGranted = true;
                            if (this.connection) {
                                this.connection.lastKnownLocation = {
                                    lat: position.coords.latitude,
                                    lng: position.coords.longitude
                                };
                            }
                            resolve(position);
                        },
                        (error) => {
                            console.log('Location permission denied:', error);
                            this.ui.showPermissionStatus('Location denied - GPS coordinates unavailable', 'warning');
                            this.ui.showQRStatus('Location denied - GPS coordinates unavailable', 'warning');
                            reject(error);
                        },
                        { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 }
                    );
                });
            } catch (error) {
                // Location permission failed, but continue anyway
            }
        } else {
            this.ui.showPermissionStatus('Location not supported on this device', 'warning');
            this.ui.showQRStatus('Location not supported on this device', 'warning');
        }

        // Show final status
        setTimeout(() => {
            if (cameraGranted && locationGranted) {
                this.ui.showPermissionStatus('All permissions granted! Ready to connect.', 'success');
                this.ui.showQRStatus('All permissions granted! Ready to connect.', 'success');
            } else if (cameraGranted || locationGranted) {
                this.ui.showPermissionStatus('Partial permissions granted. Ready to connect.', 'info');
                this.ui.showQRStatus('Partial permissions granted. Ready to connect.', 'info');
            } else {
                this.ui.showPermissionStatus('App will work without permissions. File upload available.', 'info');
                this.ui.showQRStatus('App will work without permissions. File upload available.', 'info');
            }
        }, 1500);

        // Clear permission status after showing ready message
        setTimeout(() => {
            this.ui.clearPermissionStatus();
        }, 5000);
    }

    async handleConnect() {
        const codeInput = document.getElementById('companion-code');
        const code = codeInput.value.trim();

        if (code.length !== 6) {
            this.ui.showStatus('Please enter a 6-digit code', 'error');
            return;
        }

        this.ui.setConnectionButtonState(true);
        this.ui.showStatus('Connecting to main app...', 'info');

        try {
            await this.connection.connectToMainApp(code);
            this.ui.showScreen('camera-screen');
            this.ui.showStatus('Connected successfully!', 'success');
            await this.initializeCamera();
        } catch (error) {
            this.ui.showStatus(error.message || 'Connection failed', 'error');
        } finally {
            this.ui.setConnectionButtonState(false);
        }
    }

    async connectToMainAppDirect(code) {
        this.ui.showStatus('Connecting to main app...', 'info');
        this.ui.showQRStatus('Connecting to main app...', 'info');

        try {
            await this.connection.connectToMainApp(code);
            this.ui.showScreen('camera-screen');
            this.ui.showStatus('Connected successfully!', 'success');
            this.ui.showQRStatus('Connected successfully!', 'success');
            await this.initializeCamera();
        } catch (error) {
            this.ui.showStatus(error.message || 'Auto-connection failed', 'error');
            this.ui.showQRStatus(error.message || 'Auto-connection failed', 'error');

            // If auto-connect fails, fall back to manual connection screen
            setTimeout(() => {
                this.ui.showScreen('connection-screen');
                document.getElementById('companion-code').value = code; // Pre-fill the code
                this.ui.showStatus('Auto-connect failed. Please try manually.', 'warning');
            }, 2000);
        }
    }

    async initializeCamera() {
        try {
            await this.camera.startCamera(this.resolution);
            this.ui.updateCameraStatus('Ready to capture');
        } catch (error) {
            console.error('Camera initialization failed:', error);
            this.ui.updateCameraStatus('Camera not available');
        }
    }

    async switchCamera() {
        await this.camera.switchCamera(this.resolution);
    }

    async captureImage() {
        try {
            const blob = await this.camera.captureFrame(this.config.defaults.captureQuality);
            if (blob) {
                this.ui.updateCameraStatus('Processing image...');
                await this.connection.sendImageToMainApp(blob, this.storagePreference);
            }
        } catch (error) {
            this.ui.showStatus('Failed to capture image', 'error');
        }
    }

    async toggleLiveStream() {
        try {
            this.streaming.setStoragePreference(this.storagePreference);
            await this.streaming.toggleLiveStream();
        } catch (error) {
            this.ui.showStatus(error.message || 'Live stream error', 'error');
        }
    }

    openGallery() {
        document.getElementById('file-input').click();
    }

    async handleFileSelect(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            try {
                this.ui.updateCameraStatus('Processing image...');
                await this.connection.sendImageToMainApp(file, this.storagePreference);
            } catch (error) {
                this.ui.showStatus('Failed to upload image', 'error');
            }
        }
    }

    // Settings management
    showSettings() {
        this.ui.showScreen('settings-screen');
        this.ui.loadCurrentSettings(this.resolution, this.storagePreference);
    }

    async saveSettings() {
        const settings = this.ui.getSettingsFromUI();

        if (settings.resolution) {
            this.resolution = settings.resolution;
            this.config.saveResolutionSetting(this.resolution);
        }

        // Storage preference is no longer user-configurable in companion app
        // It's automatically inherited from the main app

        this.ui.updateCameraStatus('Settings saved successfully');
        this.ui.showScreen('camera-screen');

        // Restart camera with new resolution if currently active
        if (this.camera.currentStream) {
            await this.camera.startCamera(this.resolution);
        }
    }

    cancelSettings() {
        this.ui.showScreen('camera-screen');
    }

    hideResults() {
        document.getElementById('results-display').classList.add('hidden');
        this.ui.updateCameraStatus('Ready to capture');
    }

    async disconnect() {
        // Stop live streaming if active
        await this.streaming.cleanup();

        // Disconnect from main app
        this.connection.disconnect();

        // Stop camera
        this.camera.stopCamera();

        // Reset UI
        this.ui.showScreen('connection-screen');
        this.ui.clearConnectionForm();
        this.ui.setConnectedUser('Disconnected');
    }
}

// Handle page visibility changes for battery optimization
document.addEventListener('visibilitychange', () => {
    if (window.companionApp && window.companionApp.camera) {
        if (document.hidden) {
            window.companionApp.camera.pauseCamera();
        } else {
            window.companionApp.camera.resumeCamera();
        }
    }
});
