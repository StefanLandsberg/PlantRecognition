// Configuration and Settings Management
export class CompanionConfig {
    constructor() {
        this.defaults = {
            resolution: '1280x720',
            storage: 'server',
            streamInterval: 3000,
            streamQuality: 0.8,
            captureQuality: 0.9,
            locationTimeout: 2000
        };
    }

    // Resolution and storage settings management
    loadResolutionSetting() {
        const saved = localStorage.getItem('mobileCompanionResolution');
        return saved ? saved : this.defaults.resolution;
    }

    saveResolutionSetting(resolution) {
        localStorage.setItem('mobileCompanionResolution', resolution);
    }

    loadStorageSetting() {
        return localStorage.getItem('mobileCompanionStorage') || this.defaults.storage;
    }

    saveStorageSetting(storage) {
        localStorage.setItem('mobileCompanionStorage', storage);
    }

    getResolutionConstraints(resolution) {
        // Handle both string format ("1280x720") and object format ({width: 1280, height: 720})
        let width, height;

        if (typeof resolution === 'string') {
            const [w, h] = resolution.split('x').map(Number);
            width = w;
            height = h;
        } else if (resolution && resolution.width && resolution.height) {
            width = resolution.width;
            height = resolution.height;
        } else {
            // Default fallback
            width = 1280;
            height = 720;
        }

        // Ensure minimum reasonable resolution
        const minWidth = 640;
        const minHeight = 360;

        return {
            width: {
                ideal: Math.max(width, minWidth),
                min: minWidth,
                max: 1920
            },
            height: {
                ideal: Math.max(height, minHeight),
                min: minHeight,
                max: 1080
            }
        };
    }
}