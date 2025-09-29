// Camera Management Module
export class CameraManager {
    constructor(config) {
        this.config = config;
        this.currentStream = null;
        this.currentCamera = 'environment';
        this.isInitialized = false;
    }

    async requestPermissions() {
        let cameraGranted = false;
        try {
            // Use most basic constraints for mobile compatibility
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach(track => track.stop());
            cameraGranted = true;
        } catch (error) {
            // Permission denied
        }
        return cameraGranted;
    }

    async startCamera(resolution) {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
        }

        const video = document.getElementById('camera-video');

        try {
            // Use basic constraints that work on all mobile devices
            this.currentStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: this.currentCamera
                }
            });
            video.srcObject = this.currentStream;

            // Wait for video to be ready
            return new Promise((resolve) => {
                video.onloadedmetadata = () => {
                    video.play();
                    this.isInitialized = true;
                    resolve();
                };
            });

        } catch (error) {
            // Fallback to most basic constraints
            try {
                this.currentStream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = this.currentStream;

                return new Promise((resolve) => {
                    video.onloadedmetadata = () => {
                        video.play();
                        this.isInitialized = true;
                        resolve();
                    };
                });
            } catch (fallbackError) {
                throw fallbackError;
            }
        }
    }

    async switchCamera(resolution) {
        this.currentCamera = this.currentCamera === 'environment' ? 'user' : 'environment';
        await this.startCamera(resolution);
    }

    captureFrame(quality = 0.9) {
        const video = document.getElementById('camera-video');
        const canvas = document.getElementById('camera-canvas');
        const ctx = canvas.getContext('2d');

        if (!video || !video.videoWidth || !video.videoHeight) {
            return null;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        return new Promise((resolve) => {
            canvas.toBlob((blob) => {
                resolve(blob);
            }, 'image/jpeg', quality);
        });
    }

    stopCamera() {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => track.stop());
            this.currentStream = null;
        }
        this.isInitialized = false;
    }

    pauseCamera() {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => {
                track.enabled = false;
            });
        }
    }

    resumeCamera() {
        if (this.currentStream) {
            this.currentStream.getTracks().forEach(track => {
                track.enabled = true;
            });
        }
    }
}