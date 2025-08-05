// Image Storage Module
// Handles local storage of images and videos in the database folder

class ImageStorage {
    constructor() {
        this.baseUrl = '/api/storage';
    }

    // Convert base64 data URL to Blob
    base64ToBlob(base64Data) {
        const parts = base64Data.split(',');
        const mimeType = parts[0].match(/:(.*?);/)[1];
        const byteCharacters = atob(parts[1]);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }

    // Save image locally with database ID
    async saveImageLocally(imageBlob, databaseId, type = 'detection') {
        try {
            const formData = new FormData();
            formData.append('image', imageBlob, `${databaseId}_${type}.jpg`);
            formData.append('databaseId', databaseId);
            formData.append('type', type);

            const response = await fetch(`${this.baseUrl}/save-image`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Failed to save image: ${response.statusText}`);
            }

            const result = await response.json();
            return result.localPath;
        } catch (error) {
            console.error('Error saving image locally:', error);
            throw error;
        }
    }

    // Save video locally
    async saveVideoLocally(videoBlob, databaseId, type = 'streaming') {
        try {
            const formData = new FormData();
            formData.append('video', videoBlob, `${databaseId}_${type}.mp4`);
            formData.append('databaseId', databaseId);
            formData.append('type', type);

            const response = await fetch(`${this.baseUrl}/save-video`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Failed to save video: ${response.statusText}`);
            }

            const result = await response.json();
            return result.localPath;
        } catch (error) {
            console.error('Error saving video locally:', error);
            throw error;
        }
    }

    // Get local image URL
    getLocalImageUrl(databaseId, type = 'detection') {
        return `${this.baseUrl}/images/${databaseId}_${type}.jpg`;
    }

    // Get local video URL
    getLocalVideoUrl(databaseId, type = 'streaming') {
        return `${this.baseUrl}/videos/${databaseId}_${type}.mp4`;
    }

    // Check if file exists locally
    async fileExists(databaseId, type = 'detection', fileType = 'image') {
        try {
            const url = fileType === 'image' 
                ? this.getLocalImageUrl(databaseId, type)
                : this.getLocalVideoUrl(databaseId, type);
            
            const response = await fetch(url, { method: 'HEAD' });
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    // Delete local file
    async deleteFile(databaseId, type = 'detection', fileType = 'image') {
        try {
            const response = await fetch(`${this.baseUrl}/delete`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    databaseId,
                    type,
                    fileType
                })
            });

            return response.ok;
        } catch (error) {
            console.error('Error deleting file:', error);
            return false;
        }
    }
}

// Export the ImageStorage class
export const imageStorage = new ImageStorage(); 