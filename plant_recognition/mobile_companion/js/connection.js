// WebSocket Connection Management
export class ConnectionManager {
    constructor() {
        this.ws = null;
        this.companionCode = null;
        this.isConnected = false;
        this.messageHandlers = new Map();
        this.pendingRequests = new Map();
        this.requestCounter = 0;
        this.lastKnownLocation = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.heartbeatInterval = null;
        this.connectionTimeout = null;
        this.isReconnecting = false;
    }

    registerMessageHandler(type, handler) {
        this.messageHandlers.set(type, handler);
    }

    async connectToMainApp(code) {
        this.companionCode = code;
        this.reconnectAttempts = 0;
        return this.establishConnection();
    }

    async establishConnection() {
        return new Promise((resolve, reject) => {
            if (this.connectionTimeout) {
                clearTimeout(this.connectionTimeout);
            }

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.hostname;
            const port = window.location.port || '3000';
            const wsUrl = `${protocol}//${host}:${port}/mobile-companion`;

            console.log(`Attempting connection to ${wsUrl} (attempt ${this.reconnectAttempts + 1})`);

            // Clean up existing connection
            if (this.ws) {
                this.ws.onopen = null;
                this.ws.onmessage = null;
                this.ws.onerror = null;
                this.ws.onclose = null;
                if (this.ws.readyState === WebSocket.OPEN) {
                    this.ws.close();
                }
            }

            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket opened, sending connect message');
                this.ws.send(JSON.stringify({
                    type: 'connect',
                    companionCode: this.companionCode
                }));
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);

                    if (data.type === 'connection_confirmed') {
                        console.log('Connection confirmed');
                        this.isConnected = true;
                        this.reconnectAttempts = 0;
                        this.startHeartbeat();
                        resolve();
                    } else if (data.type === 'connection_failed') {
                        console.error('Connection failed:', data.message);
                        reject(new Error(data.message || 'Invalid companion code'));
                    } else if (data.type === 'session_replaced') {
                        console.log('Session replaced by another companion');
                        const sessionReplacedHandler = this.messageHandlers.get('session_replaced');
                        if (sessionReplacedHandler) {
                            sessionReplacedHandler(data.message);
                        }
                        this.disconnect();
                    } else if (data.type === 'pong') {
                        // Heartbeat response - connection is alive
                        console.log('Heartbeat pong received');
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                if (!this.isConnected && !this.isReconnecting) {
                    reject(new Error('Unable to connect to main app'));
                }
            };

            this.ws.onclose = (event) => {
                console.log('WebSocket closed:', event.code, event.reason);
                this.isConnected = false;
                this.stopHeartbeat();
                this.rejectAllPendingRequests('Connection closed');
                
                if (this.companionCode && !this.isReconnecting) {
                    this.attemptReconnection();
                }
            };

            // Connection timeout
            this.connectionTimeout = setTimeout(() => {
                if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
                    console.log('Connection timeout');
                    this.ws.close();
                    reject(new Error('Connection timeout'));
                }
            }, 10000);
        });
    }

    startHeartbeat() {
        this.stopHeartbeat();
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // Send ping every 30 seconds
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    async attemptReconnection() {
        if (this.isReconnecting || this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnection attempts reached or already reconnecting');
            const errorHandler = this.messageHandlers.get('connection_lost');
            if (errorHandler) {
                errorHandler('Lost connection to main app - max retries exceeded');
            }
            return;
        }

        this.isReconnecting = true;
        this.reconnectAttempts++;
        
        console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
        
        const errorHandler = this.messageHandlers.get('connection_lost');
        if (errorHandler) {
            errorHandler(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        }

        await new Promise(resolve => setTimeout(resolve, this.reconnectDelay * this.reconnectAttempts));

        try {
            await this.establishConnection();
            console.log('Reconnection successful');
            const reconnectedHandler = this.messageHandlers.get('reconnected');
            if (reconnectedHandler) {
                reconnectedHandler('Reconnected successfully');
            }
        } catch (error) {
            console.error('Reconnection failed:', error);
            // Will try again due to onclose handler
        } finally {
            this.isReconnecting = false;
        }
    }

    disconnect() {
        console.log('Disconnecting companion');
        this.isConnected = false;
        this.companionCode = null;
        this.reconnectAttempts = 0;
        this.isReconnecting = false;
        
        this.stopHeartbeat();
        
        if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
        }

        if (this.ws) {
            this.ws.onopen = null;
            this.ws.onmessage = null;
            this.ws.onerror = null;
            this.ws.onclose = null;
            
            if (this.ws.readyState === WebSocket.OPEN) {
                this.ws.close();
            }
            this.ws = null;
        }

        this.rejectAllPendingRequests('Disconnected');
    }

    handleWebSocketMessage(data) {
        if (data.requestId && this.pendingRequests.has(data.requestId)) {
            const pending = this.pendingRequests.get(data.requestId);
            clearTimeout(pending.timeout);
            this.pendingRequests.delete(data.requestId);

            if (data.success === false) {
                pending.reject(new Error(data.message || 'Request failed'));
            } else {
                pending.resolve(data);
            }
            return;
        }

        const handler = this.messageHandlers.get(data.type);
        if (handler) {
            handler(data);
        }
    }

    sendMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN && this.isConnected) {
            try {
                this.ws.send(JSON.stringify(message));
                return true;
            } catch (error) {
                console.error('Error sending message:', error);
                return false;
            }
        }
        console.warn('Cannot send message - not connected');
        return false;
    }

    sendRPCMessage(type, payload = {}, timeoutMs = 15000) {
        if (!this.isConnected || !this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return Promise.reject(new Error('Not connected to main app'));
        }

        const requestId = `${type}-${Date.now()}-${++this.requestCounter}`;

        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                this.pendingRequests.delete(requestId);
                reject(new Error('Request timed out'));
            }, timeoutMs);

            this.pendingRequests.set(requestId, { resolve, reject, timeout });

            const success = this.sendMessage({ type, requestId, ...payload });
            if (!success) {
                clearTimeout(timeout);
                this.pendingRequests.delete(requestId);
                reject(new Error('Connection unavailable'));
            }
        });
    }

    async sendImageToMainApp(imageBlob, storagePreference, options = {}) {
        if (!this.isConnected) {
            throw new Error('Not connected to main app');
        }

        const base64Data = await this.blobToBase64(imageBlob);
        const payload = {
            type: 'image_capture',
            image: base64Data,
            lat: 0,
            lng: 0,
            timestamp: Date.now(),
            storagePreference,
            isLiveStream: !!options.isLiveStream
        };

        if (options.videoSessionId) {
            payload.videoSessionId = options.videoSessionId;
        }
        if (Number.isFinite(options.videoTimestamp)) {
            payload.videoTimestamp = options.videoTimestamp;
        }
        if (options.isLiveStream) {
            payload.fromVideo = 'true';
        }
        if (options.localFileId) {
            payload.localFileId = options.localFileId;
        }

        const timeout = options.isLiveStream ? 2000 : 5000;
        let coords = options.location;
        if (!coords) {
            coords = await this.getPosition(timeout, !options.isLiveStream);
        }
        if (!coords && this.lastKnownLocation) {
            coords = this.lastKnownLocation;
        }

        if (coords) {
            payload.lat = coords.lat;
            payload.lng = coords.lng;
            this.lastKnownLocation = coords;
        }

        const sent = this.sendMessage(payload);
        if (!sent) {
            throw new Error('Send failed');
        }
    }

    async startVideoSession({ lat, lng, storagePreference, sessionType = 'live_video' }) {
        return this.sendRPCMessage('video_session_start', {
            lat,
            lng,
            storagePreference,
            sessionType
        });
    }

    async stopVideoSession(sessionId, duration) {
        return this.sendRPCMessage('video_session_stop', {
            sessionId,
            duration
        });
    }

    async uploadVideoRecording(sessionId, blob, storagePreference) {
        if (!blob) return;
        const base64Video = await this.blobToBase64(blob);
        return this.sendRPCMessage('video_upload', {
            sessionId,
            video: base64Video,
            mimeType: blob.type || 'video/webm',
            storagePreference
        }, 30000);
    }

    async blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(',')[1]);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    async getPosition(timeout = 5000, enableHighAccuracy = true) {
        if (!navigator.geolocation) {
            return null;
        }

        return new Promise((resolve) => {
            navigator.geolocation.getCurrentPosition(
                (position) => resolve({
                    lat: position.coords.latitude,
                    lng: position.coords.longitude
                }),
                () => resolve(null),
                { timeout, enableHighAccuracy }
            );
        });
    }

    rejectAllPendingRequests(reason) {
        for (const [id, pending] of this.pendingRequests.entries()) {
            clearTimeout(pending.timeout);
            pending.reject(new Error(reason));
            this.pendingRequests.delete(id);
        }
    }

    disconnect() {
        this.isConnected = false;
        this.companionCode = null;
        this.rejectAllPendingRequests('Disconnected');

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}
