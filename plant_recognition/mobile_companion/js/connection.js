// WebSocket Connection Management
export class ConnectionManager {
    constructor() {
        this.ws = null;
        this.companionCode = null;
        this.isConnected = false;
        this.messageHandlers = new Map();
    }

    registerMessageHandler(type, handler) {
        this.messageHandlers.set(type, handler);
    }

    async connectToMainApp(code) {
        return new Promise((resolve, reject) => {
            // Create WebSocket connection to main app
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.hostname;
            const port = window.location.port || '3000';
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
                    this.isConnected = true;
                    this.companionCode = code;
                    resolve();
                } else if (data.type === 'connection_failed') {
                    reject(new Error(data.message || 'Invalid companion code'));
                }
            };

            this.ws.onerror = () => {
                reject(new Error('Unable to connect to main app'));
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                if (this.companionCode) {
                    // Connection was established but lost
                    const errorHandler = this.messageHandlers.get('connection_lost');
                    if (errorHandler) {
                        errorHandler('Lost connection to main app');
                    }
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
        const handler = this.messageHandlers.get(data.type);
        if (handler) {
            handler(data);
        }
    }

    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
            return true;
        }
        return false;
    }

    sendImageToMainApp(imageBlob, storagePreference, isLiveStream = false) {
        if (!this.isConnected) {
            return Promise.reject(new Error('Not connected to main app'));
        }

        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const base64Data = reader.result.split(',')[1];
                const messageType = 'image_capture'; // Use same message type for both photos and live stream

                // Try to get current location
                if (navigator.geolocation) {
                    const timeout = isLiveStream ? 2000 : 5000;
                    const highAccuracy = !isLiveStream;

                    navigator.geolocation.getCurrentPosition(
                        (position) => {
                            const success = this.sendMessage({
                                type: messageType,
                                image: base64Data,
                                lat: position.coords.latitude,
                                lng: position.coords.longitude,
                                timestamp: Date.now(),
                                storagePreference: storagePreference,
                                isLiveStream: isLiveStream
                            });
                            if (success) resolve(); else reject(new Error('Send failed'));
                        },
                        (error) => {
                            // Send without GPS coordinates if location fails
                            const success = this.sendMessage({
                                type: messageType,
                                image: base64Data,
                                lat: 0,
                                lng: 0,
                                timestamp: Date.now(),
                                storagePreference: storagePreference,
                                isLiveStream: isLiveStream
                            });
                            if (success) resolve(); else reject(new Error('Send failed'));
                        },
                        { timeout: timeout, enableHighAccuracy: highAccuracy }
                    );
                } else {
                    // Send without GPS if not supported
                    const success = this.sendMessage({
                        type: messageType,
                        image: base64Data,
                        lat: 0,
                        lng: 0,
                        timestamp: Date.now(),
                        storagePreference: storagePreference,
                        isLiveStream: isLiveStream
                    });
                    if (success) resolve(); else reject(new Error('Send failed'));
                }
            };
            reader.readAsDataURL(imageBlob);
        });
    }

    disconnect() {
        this.isConnected = false;
        this.companionCode = null;

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}