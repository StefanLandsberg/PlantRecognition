// Live Video Streaming Management
export class StreamingManager {
    constructor(cameraManager, connectionManager, config) {
        this.cameraManager = cameraManager;
        this.connection = connectionManager;
        this.config = config;
        this.isStreaming = false;
        this.streamInterval = null;
        this.storagePreference = 'server';
        this.videoStoragePreference = 'server';
        this.activeSessionId = null;
        this.sessionStartTime = null;
        this.sessionLocation = null;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.recorderMimeType = null;
        this.recorderSupported = typeof window !== 'undefined' && 'MediaRecorder' in window;
        this.frameInFlight = false;
    }

    setStoragePreference(preference) {
        this.storagePreference = preference;
    }

    async toggleLiveStream() {
        if (this.isStreaming) {
            await this.stopLiveStream();
        } else {
            await this.startLiveStream();
        }
    }

    async startLiveStream() {
        if (this.isStreaming) return;
        if (!this.connection.isConnected) throw new Error('Not connected to main app');
        if (!this.cameraManager.isInitialized || !this.cameraManager.currentStream) {
            throw new Error('Camera not ready for streaming');
        }

        this.updateStreamButton(true, true);
        this.updateStatus('Starting live stream...');

        try {
        this.sessionLocation = await this.connection.getPosition(this.config.defaults.locationTimeout || 2000, true);
        if (!this.sessionLocation && this.connection.lastKnownLocation) {
            this.sessionLocation = this.connection.lastKnownLocation;
        }

        const enforcedStorage = this.videoStoragePreference;
        const sessionResponse = await this.connection.startVideoSession({
            lat: this.sessionLocation?.lat,
            lng: this.sessionLocation?.lng,
            storagePreference: enforcedStorage,
            sessionType: 'live_video'
        });

            this.activeSessionId = sessionResponse.sessionId;
            this.sessionStartTime = Date.now();

            await this.startRecorder();

            this.isStreaming = true;
            this.updateStreamButton(true);
            this.updateStatus('Live streaming to main app...');

            await this.captureAndStreamFrame();
            this.streamInterval = setInterval(() => {
                this.captureAndStreamFrame();
            }, this.config.defaults.streamInterval);
        } catch (error) {
            this.updateStreamButton(false);
            this.updateStatus(error.message || 'Unable to start live stream');
            throw error;
        }
    }

    async stopLiveStream() {
        if (!this.isStreaming && !this.activeSessionId) {
            this.updateStreamButton(false);
            return;
        }

        this.isStreaming = false;
        if (this.streamInterval) {
            clearInterval(this.streamInterval);
            this.streamInterval = null;
        }

        this.updateStreamButton(false, true);
        this.updateStatus('Stopping live stream...');

        let pendingError = null;
        try {
            await this.stopRecorderAndUpload();
        } catch (error) {
            pendingError = error;
            console.warn('Video upload failed:', error);
        }

        if (this.activeSessionId) {
            const duration = Math.max(1, Math.floor((Date.now() - (this.sessionStartTime || Date.now())) / 1000));
            try {
                await this.connection.stopVideoSession(this.activeSessionId, duration);
            } catch (error) {
                pendingError = pendingError || error;
                console.warn('Video session stop failed:', error);
            }
        }

        if (pendingError) {
            this.updateStatus(pendingError.message || 'Live stream stopped with warnings');
        } else {
            this.updateStatus('Live streaming stopped');
        }

        this.activeSessionId = null;
        this.sessionStartTime = null;
        this.sessionLocation = null;
        this.updateStreamButton(false);

        if (pendingError) {
            throw pendingError;
        }
    }

    async captureAndStreamFrame() {
        if (!this.isStreaming || !this.activeSessionId || this.frameInFlight) {
            return;
        }

        this.frameInFlight = true;
        try {
            const blob = await this.cameraManager.captureFrame(this.config.defaults.streamQuality);
            if (!blob) {
                return;
            }

            const elapsedSeconds = this.sessionStartTime
                ? Math.max(0, Math.round((Date.now() - this.sessionStartTime) / 1000))
                : 0;

            const payloadLocation = this.sessionLocation || this.connection.lastKnownLocation || null;

            await this.connection.sendImageToMainApp(blob, this.storagePreference, {
                isLiveStream: true,
                videoSessionId: this.activeSessionId,
                videoTimestamp: elapsedSeconds,
                location: payloadLocation
            });
        } catch (error) {
            console.warn('Streaming frame failed:', error);
            this.updateStatus('Frame send failed, retrying...');
        } finally {
            this.frameInFlight = false;
        }
    }

    async startRecorder() {
        if (!this.recorderSupported || !this.cameraManager.currentStream) {
            this.recorderMimeType = null;
            this.recordedChunks = [];
            return;
        }

        const mimeType = this.getSupportedMimeType();
        if (!mimeType) {
            this.recorderSupported = false;
            this.updateStatus('Live stream running (recording unavailable on this device)');
            return;
        }

        this.recordedChunks = [];
        this.recorderMimeType = mimeType;
        this.mediaRecorder = new MediaRecorder(this.cameraManager.currentStream, { mimeType });
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                this.recordedChunks.push(event.data);
            }
        };
        this.mediaRecorder.start(5000);
    }

    getSupportedMimeType() {
        if (!this.recorderSupported) return null;
        const candidates = [
            'video/webm;codecs=vp9',
            'video/webm;codecs=vp8',
            'video/webm',
            'video/mp4;codecs=h264',
            'video/mp4'
        ];
        return candidates.find((type) => {
            try {
                return window.MediaRecorder.isTypeSupported(type);
            } catch {
                return false;
            }
        }) || null;
    }

    async stopRecorderAndUpload() {
        if (!this.mediaRecorder) {
            return;
        }

        const chunks = await new Promise((resolve) => {
            this.mediaRecorder.onstop = () => resolve([...this.recordedChunks]);
            this.mediaRecorder.stop();
        });

        this.mediaRecorder = null;
        this.recordedChunks = [];

        if (!chunks.length || !this.activeSessionId) {
            return;
        }

        const recordedBlob = new Blob(chunks, { type: this.recorderMimeType || 'video/webm' });
                await this.connection.uploadVideoRecording(this.activeSessionId, recordedBlob, this.videoStoragePreference);
    }

    updateStreamButton(streaming, pending = false) {
        const streamBtn = document.getElementById('stream-btn-text');
        const streamBtnContainer = document.getElementById('stream-toggle-btn');

        if (streamBtn) {
            streamBtn.textContent = streaming ? 'Stop Stream' : 'Start Live Stream';
        }

        if (streamBtnContainer) {
            streamBtnContainer.disabled = pending;
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

    async cleanup() {
        try {
            await this.stopLiveStream();
        } catch (error) {
            console.warn('Live stream cleanup warning:', error);
        }
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.frameInFlight = false;
    }
}
