// Mobile Companion App JavaScript
class MobileCompanionApp {
    constructor() {
        this.companionCode = null;
        this.ws = null;
        this.currentStream = null;
        this.currentCamera = 'environment'; // Start with back camera

        this.initializeApp();
    }

    initializeApp() {
        this.bindEvents();
        this.showScreen('connection-screen');
        // Don't auto-request permissions, let user trigger them
        this.showPermissionStatus('Tap "Request Permissions" button when ready');
    }

    async requestPermissions() {
        // Show permission status to user
        this.showPermissionStatus('Requesting permissions... Tap Allow if prompted');

        // Request camera permission first
        this.showPermissionStatus('Testing camera access...');
        let cameraGranted = false;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });
            stream.getTracks().forEach(track => track.stop()); // Stop immediately, just testing permission
            this.showPermissionStatus('✓ Camera access granted', 'success');
            cameraGranted = true;
        } catch (error) {
            console.log('Camera permission denied:', error);
            this.showPermissionStatus('⚠ Camera denied - can still upload photos manually', 'warning');
        }

        // Wait before requesting location
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Request location permission
        this.showPermissionStatus('Testing location access...');
        let locationGranted = false;
        if (navigator.geolocation) {
            try {
                await new Promise((resolve, reject) => {
                    navigator.geolocation.getCurrentPosition(
                        (position) => {
                            console.log('Location permission granted');
                            this.showPermissionStatus('✓ Location access granted', 'success');
                            locationGranted = true;
                            resolve(position);
                        },
                        (error) => {
                            console.log('Location permission denied:', error);
                            this.showPermissionStatus('⚠ Location denied - GPS coordinates unavailable', 'warning');
                            reject(error);
                        },
                        { enableHighAccuracy: true, timeout: 8000, maximumAge: 60000 }
                    );
                });
            } catch (error) {
                // Location permission failed, but continue anyway
            }
        } else {
            this.showPermissionStatus('⚠ Location not supported on this device', 'warning');
        }

        // Show final status
        setTimeout(() => {
            if (cameraGranted && locationGranted) {
                this.showPermissionStatus('🎉 All permissions granted! Ready to connect.', 'success');
            } else if (cameraGranted || locationGranted) {
                this.showPermissionStatus('✓ Partial permissions granted. Ready to connect.', 'info');
            } else {
                this.showPermissionStatus('App will work without permissions. File upload available.', 'info');
            }
        }, 1500);

        // Clear permission status after showing ready message
        setTimeout(() => {
            this.clearPermissionStatus();
        }, 5000);
    }

    showPermissionStatus(message, type = 'info') {
        const connectionStatus = document.getElementById('connection-status');
        if (connectionStatus) {
            connectionStatus.textContent = message;
            connectionStatus.className = `status-message ${type}`;
        }
    }

    clearPermissionStatus() {
        const connectionStatus = document.getElementById('connection-status');
        if (connectionStatus) {
            connectionStatus.textContent = '';
            connectionStatus.className = 'status-message';
        }
    }

    bindEvents() {
        // Permission events
        document.getElementById('request-permissions-btn').addEventListener('click', () => this.requestPermissions());

        // Connection events
        document.getElementById('connect-btn').addEventListener('click', () => this.handleConnect());
        document.getElementById('companion-code').addEventListener('input', (e) => {
            // Format input as 6 digits
            e.target.value = e.target.value.replace(/\D/g, '').slice(0, 6);
        });

        // Camera events
        document.getElementById('capture-btn').addEventListener('click', () => this.captureImage());
        document.getElementById('gallery-btn').addEventListener('click', () => this.openGallery());
        document.getElementById('file-input').addEventListener('change', (e) => this.handleFileSelect(e));
        document.getElementById('switch-camera-btn').addEventListener('click', () => this.switchCamera());

        // Navigation events
        document.getElementById('disconnect-btn').addEventListener('click', () => this.disconnect());
        document.getElementById('close-results').addEventListener('click', () => this.hideResults());
        document.getElementById('capture-another').addEventListener('click', () => this.hideResults());
        document.getElementById('retry-btn').addEventListener('click', () => this.showScreen('connection-screen'));

        // Enter key on companion code input
        document.getElementById('companion-code').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleConnect();
            }
        });
    }

    showScreen(screenId) {
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });

        // Show target screen
        document.getElementById(screenId).classList.add('active');

        // Initialize camera if showing camera screen
        if (screenId === 'camera-screen') {
            this.initializeCamera();
        }
    }

    async handleConnect() {
        const codeInput = document.getElementById('companion-code');
        const connectBtn = document.getElementById('connect-btn');
        const statusDiv = document.getElementById('connection-status');

        const code = codeInput.value.trim();

        if (code.length !== 6) {
            this.showStatus('Please enter a 6-digit code', 'error');
            return;
        }

        connectBtn.disabled = true;
        connectBtn.textContent = 'Connecting...';
        this.showStatus('Connecting to main app...', 'info');

        try {
            await this.connectToMainApp(code);
            this.companionCode = code;
            this.showScreen('camera-screen');
            this.showStatus('Connected successfully!', 'success');
        } catch (error) {
            this.showStatus(error.message || 'Connection failed', 'error');
        } finally {
            connectBtn.disabled = false;
            connectBtn.textContent = 'Connect';
        }
    }

    async connectToMainApp(code) {
        return new Promise((resolve, reject) => {
            // Create WebSocket connection to main app
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

            // Use the current hostname for WebSocket connection
            const host = window.location.hostname;
            const port = window.location.port || '3000';

            // For development, connect to main app server
            const wsUrl = `${protocol}//${host}:${port}/mobile-companion`;

            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                // Send companion code for verification
                this.ws.send(JSON.stringify({
                    type: 'connect',
                    companionCode: code
                }));
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);

                if (data.type === 'connection_confirmed') {
                    resolve();
                } else if (data.type === 'connection_failed') {
                    reject(new Error(data.message || 'Invalid companion code'));
                }
            };

            this.ws.onerror = () => {
                reject(new Error('Unable to connect to main app'));
            };

            this.ws.onclose = () => {
                if (this.companionCode) {
                    this.showError('Connection Lost', 'Lost connection to main app');
                }
            };

            // Timeout after 10 seconds
            setTimeout(() => {
                if (this.ws.readyState === WebSocket.CONNECTING) {
                    this.ws.close();
                    reject(new Error('Connection timeout'));
                }
            }, 10000);
        });
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'classification_result':
                this.displayClassificationResult(data.result);
                break;
            case 'analysis_complete':
                this.updateAnalysisStatus(data.analysis);
                break;
            case 'error':
                this.showStatus(data.message, 'error');
                break;
        }
    }

    async initializeCamera() {
        try {
            await this.startCamera();
            this.updateCameraStatus('Ready to capture');
        } catch (error) {
            console.error('Camera initialization failed:', error);
            this.updateCameraStatus('Camera not available');
        }
    }

    async startCamera() {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
        }

        const video = document.getElementById('camera-video');

        try {
            // First try with specific facing mode
            let constraints = {
                video: {
                    facingMode: this.currentCamera,
                    width: { ideal: 1280, max: 1920 },
                    height: { ideal: 720, max: 1080 }
                }
            };

            this.currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = this.currentStream;

            // Wait for video to be ready
            return new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play();
                    resolve();
                };
            });

        } catch (error) {
            console.log('Camera error with facing mode, trying fallback:', error);

            try {
                // Fallback: try without facing mode
                const fallbackConstraints = {
                    video: {
                        width: { ideal: 1280, max: 1920 },
                        height: { ideal: 720, max: 1080 }
                    }
                };

                this.currentStream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
                video.srcObject = this.currentStream;

                return new Promise((resolve) => {
                    video.onloadedmetadata = () => {
                        video.play();
                        resolve();
                    };
                });

            } catch (fallbackError) {
                console.log('Fallback camera error, trying basic constraints:', fallbackError);

                try {
                    // Final fallback: most basic constraints
                    this.currentStream = await navigator.mediaDevices.getUserMedia({ video: true });
                    video.srcObject = this.currentStream;

                    return new Promise((resolve) => {
                        video.onloadedmetadata = () => {
                            video.play();
                            resolve();
                        };
                    });

                } catch (basicError) {
                    console.error('All camera attempts failed:', basicError);
                    this.updateCameraStatus('Camera access denied or not available');
                    throw basicError;
                }
            }
        }
    }

    async switchCamera() {
        this.currentCamera = this.currentCamera === 'environment' ? 'user' : 'environment';
        await this.startCamera();
    }

    captureImage() {
        const video = document.getElementById('camera-video');
        const canvas = document.getElementById('camera-canvas');
        const ctx = canvas.getContext('2d');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        ctx.drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            this.sendImageToMainApp(blob);
        }, 'image/jpeg', 0.9);

        this.updateCameraStatus('Processing image...');
    }

    openGallery() {
        document.getElementById('file-input').click();
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('image/')) {
            this.sendImageToMainApp(file);
            this.updateCameraStatus('Processing image...');
        }
    }

    sendImageToMainApp(imageBlob) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            this.showStatus('Not connected to main app', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = () => {
            const base64Data = reader.result.split(',')[1];

            // Try to get current location
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        // Send with GPS coordinates
                        this.ws.send(JSON.stringify({
                            type: 'image_capture',
                            image: base64Data,
                            lat: position.coords.latitude,
                            lng: position.coords.longitude,
                            timestamp: Date.now()
                        }));
                    },
                    (error) => {
                        // Send without GPS coordinates if location fails
                        this.ws.send(JSON.stringify({
                            type: 'image_capture',
                            image: base64Data,
                            lat: 0,
                            lng: 0,
                            timestamp: Date.now()
                        }));
                    },
                    { timeout: 5000, enableHighAccuracy: true }
                );
            } else {
                // Send without GPS if not supported
                this.ws.send(JSON.stringify({
                    type: 'image_capture',
                    image: base64Data,
                    lat: 0,
                    lng: 0,
                    timestamp: Date.now()
                }));
            }
        };
        reader.readAsDataURL(imageBlob);
    }

    displayClassificationResult(result) {
        // Show a small popup with just the essential info
        this.showClassificationPopup(result.predictedSpecies || 'Unknown Species', result.confidence || 0);
        this.updateCameraStatus('Classification complete');
    }

    showClassificationPopup(species, confidence) {
        // Create or update popup
        let popup = document.getElementById('classification-popup');
        if (!popup) {
            popup = document.createElement('div');
            popup.id = 'classification-popup';
            popup.className = 'classification-popup';
            document.body.appendChild(popup);
        }

        popup.innerHTML = `
            <div class="popup-content">
                <div class="popup-header">
                    <h3>Classification Result</h3>
                    <button class="popup-close" onclick="this.parentElement.parentElement.parentElement.style.display='none'">×</button>
                </div>
                <div class="popup-body">
                    <div class="species-result">${species}</div>
                    <div class="confidence-result">${(confidence * 100).toFixed(1)}% confidence</div>
                </div>
            </div>
        `;

        popup.style.display = 'flex';

        // Auto-hide after 4 seconds
        setTimeout(() => {
            if (popup) {
                popup.style.display = 'none';
            }
        }, 4000);
    }

    updateAnalysisStatus(analysis) {
        // For mobile, we don't need to update analysis status since we use popup
    }

    showResults() {
        document.getElementById('results-display').classList.remove('hidden');
    }

    hideResults() {
        document.getElementById('results-display').classList.add('hidden');
        this.updateCameraStatus('Ready to capture');
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

    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('connection-status');
        statusDiv.textContent = message;
        statusDiv.className = `status-message ${type}`;
    }

    showError(title, message) {
        document.getElementById('error-title').textContent = title;
        document.getElementById('error-message').textContent = message;
        this.showScreen('error-screen');
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }

        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
            this.currentStream = null;
        }

        this.companionCode = null;
        this.showScreen('connection-screen');

        // Clear form
        document.getElementById('companion-code').value = '';
        document.getElementById('connection-status').textContent = '';
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.companionApp = new MobileCompanionApp();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden && window.companionApp && window.companionApp.currentStream) {
        // Pause camera when app is hidden to save battery
        window.companionApp.currentStream.getTracks().forEach(track => {
            track.enabled = false;
        });
    } else if (!document.hidden && window.companionApp && window.companionApp.currentStream) {
        // Resume camera when app is visible
        window.companionApp.currentStream.getTracks().forEach(track => {
            track.enabled = true;
        });
    }
});