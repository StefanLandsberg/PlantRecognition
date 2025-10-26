// Storage-aware upload service that respects user preferences
import { localStorage } from './localStorage.js';
import { AuthAPI } from './api.js';

class StorageAwareUploadService {
  constructor() {
    this.userStoragePreference = null;
    this.initialized = false;
  }

  async init() {
    if (this.initialized) return;

    try {
      const user = await AuthAPI.me();
      this.userStoragePreference = user.storagePreference || 'server';
      this.initialized = true;
    } catch (error) {
      console.warn('Failed to get user storage preference, defaulting to server:', error);
      this.userStoragePreference = 'server';
      this.initialized = true;
    }
  }

  async getStoragePreference() {
    await this.init();
    return this.userStoragePreference;
  }

  async handleFileUpload(file, metadata = {}) {
    await this.init();

    if (this.userStoragePreference === 'local') {
      // Store locally using IndexedDB
      const fileId = await localStorage.saveFile(file, {
        ...metadata,
        uploadedAt: new Date().toISOString(),
        storageType: 'local'
      });

      // Return local storage reference
      return {
        success: true,
        fileId: fileId,
        storageType: 'local',
        imageUrl: null, // No server URL for local storage
        localFileId: fileId
      };
    } else {
      // Upload to server (existing behavior)
      return null; // Let the server handle it normally
    }
  }

  async getFileUrl(fileId, storageType = null) {
    if (storageType === 'local' || (!storageType && this.userStoragePreference === 'local')) {
      return await localStorage.createBlobUrl(fileId);
    }
    
    // For server storage, the URL is already provided by the server
    return fileId; // Assume fileId is actually the server URL
  }

  async deleteFile(fileId, storageType = null) {
    if (storageType === 'local' || (!storageType && this.userStoragePreference === 'local')) {
      await localStorage.deleteFile(fileId);
    }
    // For server storage, deletion is handled by the server
  }

  async getStorageStats() {
    await this.init();

    if (this.userStoragePreference === 'local') {
      return await localStorage.getStorageStats();
    } else {
      // For server storage, we'd need to call the server API
      // This would be implemented based on existing storage API
      return {
        totalSize: 0,
        totalFiles: 0,
        byType: {},
        totalSizeFormatted: '0 Bytes',
        note: 'Server storage stats not implemented'
      };
    }
  }

  async selectLocalStorageDirectory() {
    return await localStorage.selectStorageDirectory();
  }

  getLocalStorageDirectoryPreference() {
    return localStorage.getStorageDirectoryPreference();
  }

  async clearLocalStorage() {
    await localStorage.clearAllData();
  }
}

// Enhanced file picker that respects storage preferences
export async function pickFileWithStorage(inputId, callback) {
  const storageService = new StorageAwareUploadService();
  const input = document.getElementById(inputId);
  
  input.onchange = async () => {
    const file = input.files?.[0];
    if (!file) return;

    try {
      const storagePreference = await storageService.getStoragePreference();
      
      if (storagePreference === 'local') {
        // Handle local storage
        const result = await storageService.handleFileUpload(file, {
          originalName: file.name,
          uploadType: 'manual'
        });
        
        if (result) {
          // Create a blob URL for immediate display
          const blobUrl = await storageService.getFileUrl(result.fileId, 'local');
          
          callback({
            ...result,
            imageUrl: blobUrl,
            file: file // Keep original file for processing
          });
        }
      } else {
        // Let server handle it (existing behavior)
        callback(file);
      }
    } catch (error) {
      console.error('Storage-aware upload failed:', error);
      // Fallback to regular file handling
      callback(file);
    }
    
    input.value = '';
  };
  
  input.click();
}

// Storage-aware upload function for programmatic uploads
export async function storageAwareUpload(file, metadata = {}) {
  const storageService = new StorageAwareUploadService();
  return await storageService.handleFileUpload(file, metadata);
}

// Export the service class for direct use
export { StorageAwareUploadService };