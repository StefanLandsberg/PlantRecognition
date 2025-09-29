// File upload constants
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB (matches server limit)
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];

// Format file size for display
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Validate file before upload
function validateFile(file) {
  const errors = [];

  // Check file size
  if (file.size > MAX_FILE_SIZE) {
    errors.push(`File size (${formatFileSize(file.size)}) exceeds the ${formatFileSize(MAX_FILE_SIZE)} limit.`);
  }

  // Check file type
  if (!ALLOWED_TYPES.includes(file.type)) {
    errors.push(`File type "${file.type}" is not supported. Please use: JPG, PNG, WebP, or GIF.`);
  }

  // Check if file is corrupted (0 bytes)
  if (file.size === 0) {
    errors.push('File appears to be empty or corrupted.');
  }

  return errors;
}

export function pickFile(inputId, cb) {
  const input = document.getElementById(inputId);
  input.onchange = async () => {
    const f = input.files?.[0];
    if (!f) return;

    // Validate file
    const errors = validateFile(f);
    if (errors.length > 0) {
      // Show error modal with file validation issues
      if (window.showModal) {
        await window.showModal.error('Upload Error', errors.join('<br>'));
      } else {
        alert(errors.join('\n'));
      }
      input.value = '';
      return;
    }

    // Show success info and proceed
    if (window.showModal && f.size > 5 * 1024 * 1024) { // Show info for files > 5MB
      const proceed = await window.showModal.confirm(
        'Large File Upload',
        `You're uploading a ${formatFileSize(f.size)} file. This may take a moment to process.`,
        {
          confirmText: 'Continue',
          cancelText: 'Cancel'
        }
      );

      if (!proceed) {
        input.value = '';
        return;
      }
    }

    cb(f);
    input.value = '';
  };
  input.click();
}

// Enhanced file picker with drag and drop support
export function setupDropZone(elementId, onFileDrop) {
  const element = document.getElementById(elementId);
  if (!element) return;

  // Add visual feedback for drag and drop
  element.addEventListener('dragover', (e) => {
    e.preventDefault();
    element.classList.add('drag-over');
  });

  element.addEventListener('dragleave', (e) => {
    e.preventDefault();
    if (!element.contains(e.relatedTarget)) {
      element.classList.remove('drag-over');
    }
  });

  element.addEventListener('drop', async (e) => {
    e.preventDefault();
    element.classList.remove('drag-over');

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    const file = files[0]; // Only handle first file
    const errors = validateFile(file);

    if (errors.length > 0) {
      if (window.showModal) {
        await window.showModal.error('Upload Error', errors.join('<br>'));
      } else {
        alert(errors.join('\n'));
      }
      return;
    }

    onFileDrop(file);
  });
}

// Add upload progress tracking
export function uploadWithProgress(file, url, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('image', file);

    // Track upload progress
    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const percentComplete = (e.loaded / e.total) * 100;
        onProgress(percentComplete);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const response = JSON.parse(xhr.responseText);
          resolve(response);
        } catch (e) {
          reject(new Error('Invalid response format'));
        }
      } else {
        reject(new Error(`Upload failed: ${xhr.statusText}`));
      }
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Upload failed: Network error'));
    });

    xhr.open('POST', url);
    xhr.setRequestHeader('credentials', 'include');
    xhr.send(formData);
  });
}
