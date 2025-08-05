// HTML Sanitization Utility
// Simple XSS protection for innerHTML assignments

/**
 * Sanitize HTML string to prevent XSS attacks
 * @param {string} str - The string to sanitize
 * @returns {string} - Sanitized HTML string
 */
export function sanitizeHtml(str) {
    try {
        if (!str || typeof str !== 'string') {
            return '';
        }
        
        // Create a temporary div element
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    } catch (error) {
        console.warn('Error sanitizing HTML:', error);
        return '';
    }
}

/**
 * Sanitize and truncate text for safe display
 * @param {string} text - The text to sanitize and truncate
 * @param {number} maxLength - Maximum length before truncation
 * @returns {string} - Sanitized and truncated text
 */
export function sanitizeAndTruncate(text, maxLength = 100) {
    if (!text) return '';
    
    const sanitized = sanitizeHtml(text);
    if (sanitized.length <= maxLength) {
        return sanitized;
    }
    
    return sanitized.substring(0, maxLength) + '...';
}

/**
 * Create safe HTML element
 * @param {string} tag - HTML tag name
 * @param {string} className - CSS class name
 * @param {string} innerHTML - Inner HTML content (will be sanitized)
 * @returns {HTMLElement} - Safe HTML element
 */
export function createSafeElement(tag, className, innerHTML) {
    const element = document.createElement(tag);
    if (className) {
        element.className = className;
    }
    if (innerHTML) {
        element.innerHTML = sanitizeHtml(innerHTML);
    }
    return element;
}

/**
 * Safely set innerHTML of an element
 * @param {HTMLElement} element - The element to update
 * @param {string} html - The HTML content to set (will be sanitized)
 */
export function setSafeInnerHTML(element, html) {
    try {
        if (element && html && typeof html === 'string') {
            element.innerHTML = sanitizeHtml(html);
        }
    } catch (error) {
        console.warn('Error setting safe innerHTML:', error);
    }
}

/**
 * Safely append HTML to an element
 * @param {HTMLElement} element - The element to append to
 * @param {string} html - The HTML content to append (will be sanitized)
 */
export function appendSafeHTML(element, html) {
    if (element && html) {
        const sanitized = sanitizeHtml(html);
        element.insertAdjacentHTML('beforeend', sanitized);
    }
} 