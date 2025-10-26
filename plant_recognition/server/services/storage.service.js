import fs from 'fs';
import path from 'path';
import { CONFIG } from '../utils/config.js';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const UPLOADS_DIR = path.resolve(PROJECT_ROOT, 'uploads');

// Storage limits and thresholds
const STORAGE_CONFIG = {
  MAX_STORAGE_MB: 2000,       // 2GB total storage limit
  WARNING_THRESHOLD: 0.85,    // Warning at 85% (1.7GB)
  CRITICAL_THRESHOLD: 0.95,   // Critical at 95% (1.9GB)
  MAX_FILES: 5000,           // Maximum number of files
  FILE_WARNING_THRESHOLD: 0.9 // Warning at 90% of max files
};

// Get directory size in bytes
function getDirectorySize(dirPath) {
  let totalSize = 0;
  let fileCount = 0;

  if (!fs.existsSync(dirPath)) {
    return { size: 0, count: 0 };
  }

  const files = fs.readdirSync(dirPath);

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    try {
      const stats = fs.statSync(filePath);

      if (stats.isFile()) {
        totalSize += stats.size;
        fileCount++;
      }
    } catch (error) {
      // Skip files that can't be accessed
      console.warn(`Could not access file ${filePath}:`, error.message);
    }
  }

  return { size: totalSize, count: fileCount };
}

// Format bytes to human readable
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Get storage status
export function getStorageStatus() {
  const { size, count } = getDirectorySize(UPLOADS_DIR);
  const maxBytes = STORAGE_CONFIG.MAX_STORAGE_MB * 1024 * 1024;

  const usagePercent = (size / maxBytes) * 100;
  const fileUsagePercent = (count / STORAGE_CONFIG.MAX_FILES) * 100;

  let status = 'ok';
  let warnings = [];

  // Check storage usage
  if (usagePercent >= STORAGE_CONFIG.CRITICAL_THRESHOLD * 100) {
    status = 'critical';
    warnings.push(`Storage critically low: ${formatBytes(size)} of ${formatBytes(maxBytes)} used (${usagePercent.toFixed(1)}%)`);
  } else if (usagePercent >= STORAGE_CONFIG.WARNING_THRESHOLD * 100) {
    status = 'warning';
    warnings.push(`Storage warning: ${formatBytes(size)} of ${formatBytes(maxBytes)} used (${usagePercent.toFixed(1)}%)`);
  }

  // Check file count
  if (fileUsagePercent >= STORAGE_CONFIG.FILE_WARNING_THRESHOLD * 100) {
    warnings.push(`High file count: ${count} of ${STORAGE_CONFIG.MAX_FILES} files (${fileUsagePercent.toFixed(1)}%)`);
    if (status === 'ok') status = 'warning';
  }

  return {
    status, // 'ok', 'warning', 'critical'
    warnings,
    usage: {
      bytes: size,
      formatted: formatBytes(size),
      percent: usagePercent,
      maxBytes,
      maxFormatted: formatBytes(maxBytes)
    },
    files: {
      count,
      percent: fileUsagePercent,
      max: STORAGE_CONFIG.MAX_FILES
    },
    limits: STORAGE_CONFIG,
    cleanupInfo: {
      message: usagePercent > 85 ?
        'Images older than 90 days are automatically removed to free space.' :
        'Images are kept for 90 days before automatic cleanup.',
      daysKept: usagePercent > 95 ? 60 : usagePercent > 85 ? 75 : 90
    }
  };
}

// Clean old files when approaching limits
export function cleanOldFiles(keepRecentDays = 30) {
  const cutoffTime = Date.now() - (keepRecentDays * 24 * 60 * 60 * 1000);
  let removedCount = 0;
  let freedBytes = 0;

  if (!fs.existsSync(UPLOADS_DIR)) {
    return { removedCount: 0, freedBytes: 0 };
  }

  const files = fs.readdirSync(UPLOADS_DIR);

  for (const file of files) {
    const filePath = path.join(UPLOADS_DIR, file);
    try {
      const stats = fs.statSync(filePath);

      if (stats.isFile() && stats.mtime.getTime() < cutoffTime) {
        const fileSize = stats.size;
        fs.unlinkSync(filePath);
        removedCount++;
        freedBytes += fileSize;
        console.log(`Cleaned old file: ${file} (${formatBytes(fileSize)})`);
      }
    } catch (error) {
      console.error(`Failed to process file ${file}:`, error);
    }
  }

  return {
    removedCount,
    freedBytes,
    freedFormatted: formatBytes(freedBytes)
  };
}

// Check if upload is allowed
export function canUpload(fileSize) {
  const status = getStorageStatus();
  const newTotalBytes = status.usage.bytes + fileSize;
  const newTotalPercent = (newTotalBytes / status.usage.maxBytes) * 100;

  // Block if would exceed critical threshold
  if (newTotalPercent > STORAGE_CONFIG.CRITICAL_THRESHOLD * 100) {
    return {
      allowed: false,
      reason: 'Storage limit would be exceeded',
      suggestion: 'Please wait for old files to be cleaned up automatically, or try a smaller file.'
    };
  }

  // Check file count
  if (status.files.count >= STORAGE_CONFIG.MAX_FILES) {
    return {
      allowed: false,
      reason: 'Maximum file count reached',
      suggestion: 'Old files are being cleaned up automatically.'
    };
  }

  return { allowed: true };
}

// Auto-cleanup when storage gets full
export function autoCleanup() {
  const status = getStorageStatus();

  if (status.status === 'critical') {
    console.log('Storage critical - running emergency cleanup...');
    const result = cleanOldFiles(60); // Keep only 2 months
    console.log(`Emergency cleanup: removed ${result.removedCount} files, freed ${result.freedFormatted}`);
    return result;
  } else if (status.status === 'warning') {
    console.log('Storage warning - running cleanup...');
    const result = cleanOldFiles(75); // Keep only 2.5 months
    console.log(`Cleanup: removed ${result.removedCount} files, freed ${result.freedFormatted}`);
    return result;
  }

  return { removedCount: 0, freedBytes: 0 };
}

// Clean up all server files for a specific user
export async function cleanupUserServerImages(userId) {
  if (!userId) return { removedCount: 0, freedBytes: 0 };

  let removedCount = 0;
  let freedBytes = 0;

  try {
    // Import here to avoid circular dependency
    const Sighting = (await import('../models/Sighting.js')).default;
    const VideoSession = (await import('../models/VideoSession.js')).default;

    // Clean up sighting images
    const userSightings = await Sighting.find({
      owner: userId,
      imagePath: { $exists: true, $ne: null }
    });

    for (const sighting of userSightings) {
      if (sighting.imagePath && sighting.imagePath.startsWith('/uploads/')) {
        const filename = path.basename(sighting.imagePath);
        const filepath = path.join(UPLOADS_DIR, filename);

        try {
          if (fs.existsSync(filepath)) {
            const stats = fs.statSync(filepath);
            const fileSize = stats.size;
            fs.unlinkSync(filepath);
            removedCount++;
            freedBytes += fileSize;
            console.log(`Cleaned user ${userId} server image: ${filename} (${formatBytes(fileSize)})`);
          }

          // Remove imagePath from sighting since it's no longer valid
          await Sighting.updateOne(
            { _id: sighting._id },
            { $set: { imagePath: null } }
          );
        } catch (error) {
          console.error(`Failed to remove user image ${filepath}:`, error);
        }
      }
    }

    // Clean up video session files
    const userVideoSessions = await VideoSession.find({
      userId: userId,
      $or: [
        { videoPath: { $exists: true, $ne: null } },
        { thumbnailPath: { $exists: true, $ne: null } }
      ]
    });

    for (const session of userVideoSessions) {
      // Clean up video file
      if (session.videoPath && session.videoPath.startsWith('/uploads/')) {
        const filename = path.basename(session.videoPath);
        const filepath = path.join(UPLOADS_DIR, filename);

        try {
          if (fs.existsSync(filepath)) {
            const stats = fs.statSync(filepath);
            const fileSize = stats.size;
            fs.unlinkSync(filepath);
            removedCount++;
            freedBytes += fileSize;
            console.log(`Cleaned user ${userId} server video: ${filename} (${formatBytes(fileSize)})`);
          }
        } catch (error) {
          console.error(`Failed to remove user video ${filepath}:`, error);
        }
      }

      // Clean up thumbnail file
      if (session.thumbnailPath && session.thumbnailPath.startsWith('/uploads/')) {
        const filename = path.basename(session.thumbnailPath);
        const filepath = path.join(UPLOADS_DIR, filename);

        try {
          if (fs.existsSync(filepath)) {
            const stats = fs.statSync(filepath);
            const fileSize = stats.size;
            fs.unlinkSync(filepath);
            removedCount++;
            freedBytes += fileSize;
            console.log(`Cleaned user ${userId} server thumbnail: ${filename} (${formatBytes(fileSize)})`);
          }
        } catch (error) {
          console.error(`Failed to remove user thumbnail ${filepath}:`, error);
        }
      }

      // Remove file paths from session since they're no longer valid
      await VideoSession.updateOne(
        { _id: session._id },
        { $set: { videoPath: null, thumbnailPath: null } }
      );
    }

    return {
      removedCount,
      freedBytes,
      freedFormatted: formatBytes(freedBytes)
    };
  } catch (error) {
    console.error('Error cleaning up user server files:', error);
    return { removedCount: 0, freedBytes: 0 };
  }
}

export async function saveFile(file, userId = null, storagePreference = 'server') {
  // If user prefers local storage, don't save to server
  if (storagePreference === 'local') {
    // Delete the uploaded file since we're not storing it
    try {
      fs.unlinkSync(file.path);
    } catch (e) {
      console.warn('Could not delete local-preference upload:', e);
    }
    // Return null to indicate local storage should be used
    return null;
  }

  // Check storage before saving (server storage only)
  const fileSize = fs.statSync(file.path).size;
  const uploadCheck = canUpload(fileSize);

  if (!uploadCheck.allowed) {
    // Delete the uploaded file since we can't keep it
    try {
      fs.unlinkSync(file.path);
    } catch (e) {
      console.warn('Could not delete rejected upload:', e);
    }
    throw new Error(`Upload denied: ${uploadCheck.reason}. ${uploadCheck.suggestion}`);
  }

  // Auto-cleanup if needed
  autoCleanup();

  // Multer already saved to disk; return the URL path
  const rel = '/uploads/' + path.basename(file.path);
  return rel;
}
