// Live Video Streaming Management
export class StreamingManager {
    constructor(cameraManager, connectionManager, config) {
        this.cameraManager = cameraManager;
        this.connectionManager = connectionManager;
        this.config = config;
        this.isStreaming = false;
        this.streamInterval = null;
        this.storagePreference = 'server';
    }

    setStoragePreference(preference) {
        this.storagePreference = preference;
    }

    toggleLiveStream() {
        if (this.isStreaming) {
            this.stopLiveStream();
        } else {
            this.startLiveStream();
        }
    }

    startLiveStream() {
        if (!this.connectionManager.isConnected) {
            throw new Error('Not connected to main app');
        }

        if (!this.cameraManager.isInitialized || !this.cameraManager.currentStream) {
            throw new Error('Camera not ready for streaming');
        }

        this.isStreaming = true;
        this.updateStreamButton(true);
        this.updateStatus('Live streaming to main app...');
        console.log('Live streaming started');

        // Stream frames every 3 seconds (like the main app)
        this.streamInterval = setInterval(() => {
            if (this.isStreaming && this.connectionManager.isConnected) {
                this.captureAndStreamFrame();
            } else {
                this.stopLiveStream();
            }
        }, this.config.defaults.streamInterval);
    }

    stopLiveStream() {
        this.isStreaming = false;

        if (this.streamInterval) {
            clearInterval(this.streamInterval);
            this.streamInterval = null;
        }

        this.updateStreamButton(false);
        this.updateStatus('Live streaming stopped');
    }

    async captureAndStreamFrame() {
        try {
            if (!this.cameraManager.isInitialized) {
                return;
            }

            const blob = await this.cameraManager.captureFrame(this.config.defaults.streamQuality);
            if (blob && this.isStreaming) {
                await this.connectionManager.sendImageToMainApp(blob, this.storagePreference, true);
            }
        } catch (error) {
            // Continue streaming despite individual frame failures
        }
    }

    updateStreamButton(streaming) {
        const streamBtn = document.getElementById('stream-btn-text');
        const streamBtnContainer = document.getElementById('stream-toggle-btn');

        if (streamBtn) {
            streamBtn.textContent = streaming ? 'Stop Stream' : 'Start Live Stream';
        }

        if (streamBtnContainer) {
            if (streaming) {
                streamBtnContainer.classList.add('streaming');
            } else {
                streamBtnContainer.classList.remove('streaming');
            }
        }
    }

    updateStatus(message) {
        const statusElement = document.getElementById('classification-status');
        if (statusElement) {
            const statusText = statusElement.querySelector('.status-text');
            if (statusText) {
                statusText.textContent = message;
            }
        }
    }

    cleanup() {
        this.stopLiveStream();
    }
}