/**
 * Centralized Error Handling Module
 * Provides consistent error handling, logging, and user feedback across the application
 */

class ErrorHandler {
    constructor() {
        this.errorCount = 0;
        this.maxErrors = 10;
        this.errorLog = [];
        this.isInitialized = false;
    }

    /**
     * Initialize error handling system
     */
    initialize() {
        if (this.isInitialized) return;

        // Global error handlers
        window.addEventListener('error', (event) => {
            this.handleError(event.error || new Error(event.message), {
                type: 'global',
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno
            });
        });

        window.addEventListener('unhandledrejection', (event) => {
            this.handleError(event.reason, {
                type: 'unhandledrejection',
                promise: event.promise
            });
        });

        // Network error handling
        window.addEventListener('online', () => {
            this.showNotification('Connection restored', 'success');
        });

        window.addEventListener('offline', () => {
            this.handleError(new Error('Network connection lost'), {
                type: 'network',
                severity: 'warning'
            });
        });

        this.isInitialized = true;
        console.log('Error handling system initialized');
    }

    /**
     * Handle errors with consistent logging and user feedback
     */
    handleError(error, context = {}) {
        const errorInfo = {
            message: error.message || 'Unknown error',
            stack: error.stack,
            timestamp: new Date().toISOString(),
            context: context,
            userAgent: navigator.userAgent,
            url: window.location.href
        };

        // Log error
        this.logError(errorInfo);

        // Increment error count
        this.errorCount++;

        // Determine severity
        const severity = context.severity || this.determineSeverity(error, context);

        // Show user feedback based on severity
        this.showUserFeedback(error, severity, context);

        // Handle critical errors
        if (severity === 'critical') {
            this.handleCriticalError(error, context);
        }

        // Prevent error spam
        if (this.errorCount > this.maxErrors) {
            console.warn('Too many errors, suppressing further error messages');
            return;
        }

        return errorInfo;
    }

    /**
     * Determine error severity based on error type and context
     */
    determineSeverity(error, context) {
        // Network errors are usually warnings
        if (context.type === 'network') return 'warning';
        
        // Authentication errors are important
        if (error.message.includes('auth') || error.message.includes('login')) return 'important';
        
        // File upload errors are important
        if (error.message.includes('upload') || error.message.includes('file')) return 'important';
        
        // Database errors are critical
        if (error.message.includes('database') || error.message.includes('db')) return 'critical';
        
        // LLM errors are important
        if (error.message.includes('llm') || error.message.includes('analysis')) return 'important';
        
        // Default to error
        return 'error';
    }

    /**
     * Show appropriate user feedback based on error severity
     */
    showUserFeedback(error, severity, context) {
        const message = this.getUserFriendlyMessage(error, context);
        
        switch (severity) {
            case 'critical':
                this.showNotification(message, 'error', 10000);
                break;
            case 'important':
                this.showNotification(message, 'warning', 5000);
                break;
            case 'warning':
                this.showNotification(message, 'info', 3000);
                break;
            default:
                // Don't show user feedback for minor errors
                break;
        }
    }

    /**
     * Get user-friendly error message
     */
    getUserFriendlyMessage(error, context) {
        const message = error.message.toLowerCase();
        
        if (message.includes('network') || message.includes('fetch')) {
            return 'Network connection issue. Please check your internet connection.';
        }
        
        if (message.includes('auth') || message.includes('login')) {
            return 'Authentication error. Please log in again.';
        }
        
        if (message.includes('upload') || message.includes('file')) {
            return 'File upload failed. Please try again with a different file.';
        }
        
        if (message.includes('database') || message.includes('db')) {
            return 'Database connection issue. Please try again later.';
        }
        
        if (message.includes('llm') || message.includes('analysis')) {
            return 'Analysis service temporarily unavailable. Please try again.';
        }
        
        return 'An unexpected error occurred. Please try again.';
    }

    /**
     * Handle critical errors that require special attention
     */
    handleCriticalError(error, context) {
        // Log critical error to server if possible
        this.logToServer(error, context);
        
        // Show critical error modal
        this.showCriticalErrorModal(error);
    }

    /**
     * Show notification to user
     */
    showNotification(message, type = 'info', duration = 3000) {
        try {
            // Validate parameters
            if (!message || typeof message !== 'string') {
                console.warn('Invalid notification message:', message);
                return;
            }
            
            if (!['info', 'success', 'warning', 'error'].includes(type)) {
                console.warn('Invalid notification type:', type);
                type = 'info';
            }
            
            if (typeof duration !== 'number' || duration < 0) {
                console.warn('Invalid notification duration:', duration);
                duration = 3000;
            }
            
            // Sanitize message to prevent XSS
            const sanitizedMessage = this.sanitizeMessage(message);
            
            // Create notification element
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.innerHTML = `
                <div class="notification-content">
                    <i class="fas fa-${this.getNotificationIcon(type)}"></i>
                    <span>${sanitizedMessage}</span>
                </div>
                <button class="notification-close" onclick="this.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            `;

            // Add to page
            const container = document.querySelector('.notifications-container') || document.body;
            container.appendChild(notification);

            // Auto-remove after duration
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, duration);
        } catch (error) {
            console.error('Error showing notification:', error);
        }
    }

    /**
     * Get appropriate icon for notification type
     */
    getNotificationIcon(type) {
        const validTypes = ['success', 'error', 'warning', 'info'];
        if (!validTypes.includes(type)) {
            console.warn('Invalid notification type for icon:', type);
            type = 'info';
        }
        
        switch (type) {
            case 'success': return 'check-circle';
            case 'error': return 'exclamation-triangle';
            case 'warning': return 'exclamation-circle';
            case 'info': return 'info-circle';
            default: return 'info-circle';
        }
    }

    /**
     * Show critical error modal
     */
    showCriticalErrorModal(error) {
        try {
            // Sanitize error message to prevent XSS
            const sanitizedMessage = this.sanitizeMessage(error.message || 'Unknown error');
            
            const modal = document.createElement('div');
            modal.className = 'modal-overlay critical-error-modal';
            modal.innerHTML = `
                <div class="modal">
                    <div class="modal-header">
                        <h3><i class="fas fa-exclamation-triangle"></i> Critical Error</h3>
                    </div>
                    <div class="modal-body">
                        <p>A critical error has occurred. Please refresh the page or contact support if the problem persists.</p>
                        <div class="error-details">
                            <strong>Error:</strong> ${sanitizedMessage}
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button onclick="window.location.reload()" class="btn btn-primary">Refresh Page</button>
                        <button onclick="this.closest('.modal-overlay').remove()" class="btn btn-secondary">Close</button>
                    </div>
                </div>
            `;

            document.body.appendChild(modal);
        } catch (modalError) {
            console.error('Error showing critical error modal:', modalError);
            // Fallback: just show a simple alert
            alert('A critical error has occurred. Please refresh the page.');
        }
    }

    /**
     * Log error to console and internal log
     */
    logError(errorInfo) {
        // Console logging
        console.error('Application Error:', errorInfo);
        
        // Internal log
        this.errorLog.push(errorInfo);
        
        // Keep log size manageable
        if (this.errorLog.length > 100) {
            this.errorLog = this.errorLog.slice(-50);
        }
    }

    /**
     * Log error to server (if available)
     */
    async logToServer(error, context) {
        try {
            const response = await fetch('/api/errors', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    error: error.message,
                    stack: error.stack,
                    context: context,
                    timestamp: new Date().toISOString(),
                    userAgent: navigator.userAgent,
                    url: window.location.href
                })
            });
            
            if (!response.ok) {
                console.warn('Failed to log error to server');
            }
        } catch (logError) {
            console.warn('Could not log error to server:', logError);
        }
    }

    /**
     * Get error statistics
     */
    getErrorStats() {
        return {
            totalErrors: this.errorCount,
            recentErrors: this.errorLog.length,
            isHealthy: this.errorCount < this.maxErrors
        };
    }

    /**
     * Clear error log
     */
    clearErrorLog() {
        this.errorLog = [];
        this.errorCount = 0;
    }

    /**
     * Sanitize message to prevent XSS attacks
     */
    sanitizeMessage(message) {
        try {
            if (!message || typeof message !== 'string') {
                return '';
            }
            
            // Create a temporary div element to escape HTML
            const div = document.createElement('div');
            div.textContent = message;
            return div.innerHTML;
        } catch (error) {
            console.warn('Error sanitizing message:', error);
            return '';
        }
    }

    /**
     * Create a wrapped function with error handling
     */
    wrapFunction(fn, context = {}) {
        return async (...args) => {
            try {
                return await fn(...args);
            } catch (error) {
                this.handleError(error, {
                    ...context,
                    function: fn.name || 'anonymous',
                    args: args.length
                });
                throw error; // Re-throw to allow caller to handle
            }
        };
    }
}

// Create singleton instance
const errorHandler = new ErrorHandler();

// Export for use in other modules
export default errorHandler; 