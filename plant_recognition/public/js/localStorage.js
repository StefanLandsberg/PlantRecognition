// Client-side local storage service using IndexedDB
class LocalStorageService {
  constructor() {
    this.dbName = 'PlantRecognitionStorage';
    this.version = 1;
    this.db = null;
  }

  async init() {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        // Create object stores for different file types
        if (!db.objectStoreNames.contains('images')) {
          const imageStore = db.createObjectStore('images', { keyPath: 'id' });
          imageStore.createIndex('timestamp', 'timestamp', { unique: false });
          imageStore.createIndex('sightingId', 'sightingId', { unique: false });
        }

        if (!db.objectStoreNames.contains('videos')) {
          const videoStore = db.createObjectStore('videos', { keyPath: 'id' });
          videoStore.createIndex('timestamp', 'timestamp', { unique: false });
          videoStore.createIndex('sessionId', 'sessionId', { unique: false });
        }

        if (!db.objectStoreNames.contains('frames')) {
          const frameStore = db.createObjectStore('frames', { keyPath: 'id' });
          frameStore.createIndex('timestamp', 'timestamp', { unique: false });
          frameStore.createIndex('sessionId', 'sessionId', { unique: false });
        }
      };
    });
  }

  async saveFile(file, metadata = {}) {
    await this.init();

    const fileData = {
      id: metadata.id || `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      file: file,
      filename: file.name || metadata.filename || 'unknown',
      type: file.type || metadata.type || 'application/octet-stream',
      size: file.size || metadata.size || 0,
      timestamp: Date.now(),
      ...metadata
    };

    const storeName = this.getStoreNameForType(file.type);
    const transaction = this.db.transaction([storeName], 'readwrite');
    const store = transaction.objectStore(storeName);

    return new Promise((resolve, reject) => {
      const request = store.add(fileData);
      request.onsuccess = () => resolve(fileData.id);
      request.onerror = () => reject(request.error);
    });
  }

  async getFile(id, storeName = null) {
    await this.init();

    if (!storeName) {
      // Try all stores if not specified
      const stores = ['images', 'videos', 'frames'];
      for (const store of stores) {
        try {
          const file = await this.getFileFromStore(id, store);
          if (file) return file;
        } catch (e) {
          // Continue to next store
        }
      }
      return null;
    }

    return this.getFileFromStore(id, storeName);
  }

  async getFileFromStore(id, storeName) {
    const transaction = this.db.transaction([storeName], 'readonly');
    const store = transaction.objectStore(storeName);

    return new Promise((resolve, reject) => {
      const request = store.get(id);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async deleteFile(id, storeName = null) {
    await this.init();

    if (!storeName) {
      // Try all stores if not specified
      const stores = ['images', 'videos', 'frames'];
      for (const store of stores) {
        try {
          await this.deleteFileFromStore(id, store);
        } catch (e) {
          // Continue to next store
        }
      }
      return;
    }

    return this.deleteFileFromStore(id, storeName);
  }

  async deleteFileFromStore(id, storeName) {
    const transaction = this.db.transaction([storeName], 'readwrite');
    const store = transaction.objectStore(storeName);

    return new Promise((resolve, reject) => {
      const request = store.delete(id);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async listFiles(storeName, limit = 50) {
    await this.init();

    const transaction = this.db.transaction([storeName], 'readonly');
    const store = transaction.objectStore(storeName);
    const index = store.index('timestamp');

    return new Promise((resolve, reject) => {
      const request = index.openCursor(null, 'prev'); // Most recent first
      const results = [];
      let count = 0;

      request.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor && count < limit) {
          results.push(cursor.value);
          count++;
          cursor.continue();
        } else {
          resolve(results);
        }
      };

      request.onerror = () => reject(request.error);
    });
  }

  async getStorageStats() {
    await this.init();

    const stores = ['images', 'videos', 'frames'];
    const stats = { totalSize: 0, totalFiles: 0, byType: {} };

    for (const storeName of stores) {
      const files = await this.listFiles(storeName, 1000); // Get all files
      let storeSize = 0;

      files.forEach(file => {
        storeSize += file.size || 0;
      });

      stats.byType[storeName] = {
        count: files.length,
        size: storeSize,
        sizeFormatted: this.formatBytes(storeSize)
      };

      stats.totalFiles += files.length;
      stats.totalSize += storeSize;
    }

    stats.totalSizeFormatted = this.formatBytes(stats.totalSize);
    return stats;
  }

  async clearAllData() {
    await this.init();

    const stores = ['images', 'videos', 'frames'];
    for (const storeName of stores) {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      await new Promise((resolve, reject) => {
        const request = store.clear();
        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    }
  }

  getStoreNameForType(mimeType) {
    if (mimeType.startsWith('image/')) return 'images';
    if (mimeType.startsWith('video/')) return 'videos';
    return 'frames'; // Default for unknown types
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // Create a blob URL for a stored file
  async createBlobUrl(id, storeName = null) {
    const fileData = await this.getFile(id, storeName);
    if (!fileData) return null;

    const blob = new Blob([fileData.file], { type: fileData.type });
    return URL.createObjectURL(blob);
  }

  // Get user's preferred local storage directory (if supported)
  async selectStorageDirectory() {
    if ('showDirectoryPicker' in window) {
      try {
        const dirHandle = await window.showDirectoryPicker();
        localStorage.setItem('localStorageDirectory', dirHandle.name);
        return dirHandle;
      } catch (error) {
        console.warn('Directory selection cancelled or failed:', error);
        return null;
      }
    } else {
      console.warn('Directory picker not supported in this browser');
      return null;
    }
  }

  // Get the current storage directory preference
  getStorageDirectoryPreference() {
    return localStorage.getItem('localStorageDirectory') || 'Browser Storage';
  }
}

// Export singleton instance
export const localStorage = new LocalStorageService();