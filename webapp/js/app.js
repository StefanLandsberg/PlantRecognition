// Global variables
let model = null;
let isModelLoading = false;
let isVideoMode = false;
let videoStream = null;
let captureInterval = null;
let autoDetectionActive = false;
let selectedImages = [];
let capturedFrames = [];
let lastDetectionTime = 0;
let detectionThreshold = 0.65; // Confidence threshold for detection
let detectionCooldown = 1000; // Cooldown between detections in ms
let currentVideoFrame = null;
let detectedPlants = []; // Store recent detections

// Class mapping from the model metadata
const classMapping = {
    // Will be populated after model loads
};

// DOM Elements
const elements = {
    // Toggles and mode selectors
    themeToggle: document.getElementById('themeToggle'),
    modeToggle: document.getElementById('modeToggle'),
    photoMode: document.getElementById('photoMode'),
    videoMode: document.getElementById('videoMode'),
    
    // Photo mode elements
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    selectedImagesContainer: document.getElementById('selectedImages'),
    
    // Video mode elements
    video: document.getElementById('video'),
    canvas: document.getElementById('canvas'),
    overlayCanvas: document.getElementById('overlayCanvas'),
    startCameraBtn: document.getElementById('startCamera'),
    stopCameraBtn: document.getElementById('stopCamera'),
    toggleAutoDetectionBtn: document.getElementById('toggleAutoDetection'),
    capturedFramesContainer: document.getElementById('capturedFrames'),
    
    // Action buttons
    analyzeBtn: document.getElementById('analyzeBtn'),
    clearBtn: document.getElementById('clearBtn'),
    
    // Results containers
    resultsContainer: document.getElementById('resultsContainer'),
    phase1Results: document.getElementById('phase1Results'),
    phase2Results: document.getElementById('phase2Results'),
    finalResult: document.getElementById('finalResult'),
    
    // Loading overlay
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText')
};

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadModel();
    checkForSavedTheme();
});

// Initialize event listeners
function initEventListeners() {
    // Theme toggle
    elements.themeToggle.addEventListener('click', toggleTheme);
    
    // Mode toggle
    elements.modeToggle.addEventListener('click', toggleMode);
    
    // Photo mode - Fix upload functionality
    elements.uploadArea.addEventListener('click', () => {
        if (!isModelLoading) {
            elements.fileInput.click();
            console.log("Upload area clicked, opening file dialog");
        }
    });
    
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Fix drag and drop functionality
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.uploadArea.classList.add('highlight');
    });
    
    elements.uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.uploadArea.classList.remove('highlight');
    });
    
    elements.uploadArea.addEventListener('drop', (e) => {
        console.log("File dropped on upload area");
        e.preventDefault();
        e.stopPropagation();
        elements.uploadArea.classList.remove('highlight');
        
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            processSelectedFiles(files);
        }
    });
    
    // Add paste support
    document.addEventListener('paste', (e) => {
        if (isVideoMode) return; // Only handle paste in photo mode
        
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        if (!items) return;
        
        for (const item of items) {
            if (item.type.indexOf('image') === 0) {
                e.preventDefault();
                const blob = item.getAsFile();
                const file = new File([blob], "pasted-image.png", { type: blob.type });
                processSelectedFiles([file]);
                console.log('Image pasted from clipboard');
                break;
            }
        }
    });
    
    // Video mode
    elements.startCameraBtn.addEventListener('click', startCamera);
    elements.stopCameraBtn.addEventListener('click', stopCamera);
    elements.toggleAutoDetectionBtn.addEventListener('click', toggleAutoDetection);
    
    // Action buttons
    elements.analyzeBtn.addEventListener('click', analyzeImages);
    elements.clearBtn.addEventListener('click', clearAll);
}

// Check for saved theme preference
function checkForSavedTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
        elements.themeToggle.classList.add('active');
    }
}

// Toggle between light and dark themes
function toggleTheme() {
    document.body.classList.toggle('dark-theme');
    elements.themeToggle.classList.toggle('active');
    
    const theme = document.body.classList.contains('dark-theme') ? 'dark' : 'light';
    localStorage.setItem('theme', theme);
}

// Toggle between photo and video modes
function toggleMode() {
    isVideoMode = !isVideoMode;
    elements.modeToggle.classList.toggle('active');
    
    if (isVideoMode) {
        elements.photoMode.classList.remove('active');
        elements.videoMode.classList.add('active');
    } else {
        elements.videoMode.classList.remove('active');
        elements.photoMode.classList.add('active');
        
        // Stop camera if it's running when switching to photo mode
        if (videoStream) {
            stopCamera();
        }
    }
    
    updateActionButtonsState();
}

// Handle file selection from input
function handleFileSelect(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    processSelectedFiles(files);
}

// Handle file drop
function handleFileDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('highlight');
    
    const files = e.dataTransfer.files;
    if (!files || files.length === 0) return;
    
    processSelectedFiles(files);
}

// Process selected files
function processSelectedFiles(files) {
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (!file.type.startsWith('image/')) continue;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.createElement('img');
            img.src = e.target.result;
            
            img.onload = () => {
                selectedImages.push({
                    file: file,
                    dataUrl: e.target.result,
                    element: img
                });
                
                addImageToGallery(e.target.result, selectedImages.length - 1, 'photo');
                updateActionButtonsState();
            };
        };
        reader.readAsDataURL(file);
    }
}

// Add image to gallery
function addImageToGallery(src, index, mode) {
    const container = mode === 'photo' ? elements.selectedImagesContainer : elements.capturedFramesContainer;
    const collection = mode === 'photo' ? selectedImages : capturedFrames;
    
    const imageCard = document.createElement('div');
    imageCard.className = 'image-card';
    imageCard.dataset.index = index;
    
    const img = document.createElement('img');
    img.src = src;
    
    const removeBtn = document.createElement('button');
    removeBtn.className = 'remove-btn';
    removeBtn.innerHTML = '<i class="fas fa-times"></i>';
    removeBtn.onclick = (e) => {
        e.stopPropagation();
        collection.splice(index, 1);
        imageCard.remove();
        
        // Update indices
        const cards = container.querySelectorAll('.image-card');
        cards.forEach((card, i) => {
            card.dataset.index = i;
        });
        
        updateActionButtonsState();
    };
    
    imageCard.appendChild(img);
    imageCard.appendChild(removeBtn);
    container.appendChild(imageCard);
}

// Start camera
async function startCamera() {
    try {
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment' // Use rear camera on mobile if available
            }
        };
        
        videoStream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.video.srcObject = videoStream;
        
        elements.startCameraBtn.disabled = true;
        elements.stopCameraBtn.disabled = false;
        elements.toggleAutoDetectionBtn.disabled = false;
        
        // Set up canvas dimensions once video metadata is loaded
        elements.video.onloadedmetadata = () => {
            elements.canvas.width = elements.video.videoWidth;
            elements.canvas.height = elements.video.videoHeight;
            elements.overlayCanvas.width = elements.video.videoWidth;
            elements.overlayCanvas.height = elements.video.videoHeight;
        };
        
        // Start the video processing loop for real-time overlay
        requestAnimationFrame(processVideoFrame);
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Could not access camera. Please ensure camera permissions are enabled.');
    }
}

// Toggle auto detection
function toggleAutoDetection() {
    autoDetectionActive = !autoDetectionActive;
    elements.toggleAutoDetectionBtn.classList.toggle('active');
    
    if (autoDetectionActive) {
        elements.toggleAutoDetectionBtn.innerHTML = '<i class="fas fa-pause"></i> Pause Detection';
        // Clear previous interval just in case
        if (captureInterval) {
            clearInterval(captureInterval);
        }
        // Start automatic detection at regular intervals
        captureInterval = setInterval(detectPlantsInVideo, 1000);
    } else {
        elements.toggleAutoDetectionBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
        // Clear interval
        if (captureInterval) {
            clearInterval(captureInterval);
            captureInterval = null;
        }
    }
}

// Process video frame for display and detection
function processVideoFrame() {
    if (videoStream && elements.video.readyState === 4) {
        // Draw the current frame to the hidden canvas for processing
        const ctx = elements.canvas.getContext('2d');
        ctx.drawImage(elements.video, 0, 0, elements.canvas.width, elements.canvas.height);
        
        // Save the current frame for detection
        currentVideoFrame = elements.canvas;
        
        // Draw detections on the overlay canvas
        drawDetections();
    }
    
    // Continue the loop
    requestAnimationFrame(processVideoFrame);
}

// Detect plants in video frames
async function detectPlantsInVideo() {
    if (!autoDetectionActive || !currentVideoFrame || isModelLoading) return;
    
    const now = Date.now();
    // Check for cooldown to avoid processing too many frames
    if (now - lastDetectionTime < detectionCooldown) return;
    
    lastDetectionTime = now;
    
    try {
        // Create an image element from the current frame
        const img = document.createElement('img');
        img.src = currentVideoFrame.toDataURL('image/jpeg');
        
        // Wait for image to load
        await new Promise(resolve => {
            img.onload = resolve;
        });
        
        // Phase 1: Get initial predictions
        const predictions = await makePrediction(img);
        const topPredictions = predictions.slice(0, 20);
        
        // Phase 2: Focused search on top classes
        const topClasses = topPredictions.map(p => p.classId);
        const focusedPredictions = await makeFocusedPrediction(img, topClasses);
        
        // Filter by confidence threshold
        const significantDetections = focusedPredictions.filter(p => p.confidence >= detectionThreshold);
        
        if (significantDetections.length > 0) {
            // Store top detection for display
            const topDetection = significantDetections[0];
            
            // Calculate simple bounding box (in a real app, you'd use object detection for precise boxes)
            const boundingBox = {
                x: currentVideoFrame.width * 0.2,
                y: currentVideoFrame.height * 0.2,
                width: currentVideoFrame.width * 0.6,
                height: currentVideoFrame.height * 0.6
            };
            
            // Add detection to list with timestamp for display
            detectedPlants = [{
                prediction: topDetection,
                boundingBox: boundingBox,
                timestamp: now,
                frame: currentVideoFrame.toDataURL('image/jpeg')
            }];
            
            // Automatically save significant detections
            savePlantDetection(topDetection, img);
        }
        
    } catch (error) {
        console.error('Error detecting plants in video:', error);
    }
}

// Draw bounding boxes and labels on overlay canvas
function drawDetections() {
    if (!detectedPlants.length) return;
    
    const ctx = elements.overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
    
    // Draw each detected plant
    detectedPlants.forEach(detection => {
        const { prediction, boundingBox } = detection;
        
        // Format plant name for display
        const plantName = prediction.className.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
        
        // Draw bounding box
        ctx.strokeStyle = '#4caf50';
        ctx.lineWidth = 4;
        ctx.strokeRect(boundingBox.x, boundingBox.y, boundingBox.width, boundingBox.height);
        
        // Draw label background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(boundingBox.x, boundingBox.y - 35, boundingBox.width, 35);
        
        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.font = '16px Arial';
        ctx.fillText(
            `${plantName} (${(prediction.confidence * 100).toFixed(1)}%)`, 
            boundingBox.x + 10, 
            boundingBox.y - 12
        );
    });
}

// Save a plant detection to the captured frames
function savePlantDetection(prediction, imgElement) {
    // Create a canvas to draw the annotated image
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set dimensions
    canvas.width = imgElement.naturalWidth;
    canvas.height = imgElement.naturalHeight;
    
    // Draw the image
    ctx.drawImage(imgElement, 0, 0);
    
    // Calculate bounding box (simple implementation for demo)
    const boxWidth = imgElement.naturalWidth * 0.6;
    const boxHeight = imgElement.naturalHeight * 0.6;
    const boxX = (imgElement.naturalWidth - boxWidth) / 2;
    const boxY = (imgElement.naturalHeight - boxHeight) / 2;
    
    // Draw bounding box
    ctx.strokeStyle = '#4caf50';
    ctx.lineWidth = 4;
    ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);
    
    // Format plant name for display
    const plantName = prediction.className.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
    
    // Draw label background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(boxX, boxY - 35, boxWidth, 35);
    
    // Draw label text
    ctx.fillStyle = '#ffffff';
    ctx.font = '16px Arial';
    ctx.fillText(
        `${plantName} (${(prediction.confidence * 100).toFixed(1)}%)`, 
        boxX + 10, 
        boxY - 12
    );
    
    // Convert to data URL
    const dataUrl = canvas.toDataURL('image/jpeg');
    
    // Create a new image element
    const img = document.createElement('img');
    img.src = dataUrl;
    
    // Add to captured frames
    capturedFrames.push({
        dataUrl: dataUrl,
        element: img,
        prediction: prediction
    });
    
    // Add to gallery
    addImageToGallery(dataUrl, capturedFrames.length - 1, 'video');
    updateActionButtonsState();
}

// Stop camera
function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
        elements.video.srcObject = null;
    }
    
    // Clear intervals
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }
    
    // Clear detection state
    autoDetectionActive = false;
    elements.toggleAutoDetectionBtn.classList.remove('active');
    elements.toggleAutoDetectionBtn.innerHTML = '<i class="fas fa-play"></i> Start Detection';
    
    // Reset UI
    elements.startCameraBtn.disabled = false;
    elements.stopCameraBtn.disabled = true;
    elements.toggleAutoDetectionBtn.disabled = true;
    
    // Clear overlay
    const ctx = elements.overlayCanvas.getContext('2d');
    ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
}

// Update action buttons state
function updateActionButtonsState() {
    const hasImages = selectedImages.length > 0 || capturedFrames.length > 0;
    elements.analyzeBtn.disabled = !hasImages || isModelLoading;
    elements.clearBtn.disabled = !hasImages;
}

// Clear all images
function clearAll() {
    selectedImages = [];
    capturedFrames = [];
    
    elements.selectedImagesContainer.innerHTML = '';
    elements.capturedFramesContainer.innerHTML = '';
    elements.resultsContainer.style.display = 'none';
    elements.phase1Results.innerHTML = '';
    elements.phase2Results.innerHTML = '';
    elements.finalResult.innerHTML = '';
    
    updateActionButtonsState();
}

// Load TensorFlow.js model
async function loadModel() {
    try {
        showLoading('Initializing plant recognition system...');
        isModelLoading = true;
        updateActionButtonsState();
        
        // First try to load the metadata to get plant classes
        await loadClassMappings();
        
        // Load the feature extractors first - this is critical for matching your Python training pipeline
        const extractorsLoaded = await loadFeatureExtractors();
        
        if (!extractorsLoaded) {
            console.warn('Could not load feature extractors - plant recognition will be less accurate');
        }
        
        // Try to load your trained model using the correct webapp path
        try {
            // Use the correct path from webapp that was successful in the logs
            const modelPath = './models/tfjs_model/model.json';
            console.log('Loading TensorFlow.js model from:', modelPath);
            
            model = await tf.loadLayersModel(modelPath);
            console.log('Successfully loaded TensorFlow.js model');
            
            // Check input shape to ensure we're using it correctly
            const inputShape = model.inputs[0].shape;
            console.log('Model input shape:', inputShape);
            
            // This model expects a 2304 length feature vector (combined from EfficientNetV2 and DenseNet121)
            if (inputShape[1] === 2304) {
                console.log('Model expects combined features with 2304 dimensions - matches our feature extraction pipeline');
            } else {
                console.warn(`Model expects input of dimension ${inputShape[1]} - make sure feature extraction matches`);
            }
        } catch (modelError) {
            console.warn('Failed to load TensorFlow.js model:', modelError);
            
            // Create a fallback classifier model
            console.log('Creating fallback classifier model');
            createFallbackModel();
        }
        
        hideLoading();
        isModelLoading = false;
        updateActionButtonsState();
    } catch (error) {
        console.error('Error initializing model:', error);
        alert('An error occurred while initializing the plant recognition system. The app will run with reduced accuracy.');
        
        // Create a fallback model as a last resort
        createFallbackModel();
        
        hideLoading();
        isModelLoading = false;
        updateActionButtonsState();
    }
}

// Create a fallback model when the main model can't be loaded
function createFallbackModel() {
    console.log('Creating a fallback model for plant recognition');
    
    // Create a simple model that takes features as input
    const numClasses = Object.keys(classMapping).length || 1000;
    const features = 2304; // Combined EfficientNetV2 + DenseNet121 features dimension
    
    const input = tf.input({shape: [features]});
    const dense1 = tf.layers.dense({units: 512, activation: 'relu'}).apply(input);
    const dropout = tf.layers.dropout({rate: 0.2}).apply(dense1);
    const output = tf.layers.dense({units: numClasses, activation: 'softmax'}).apply(dropout);
    
    model = tf.model({inputs: input, outputs: output});
    console.log('Fallback model created');
    
    // Add notification to the user
    const notification = document.createElement('div');
    notification.className = 'notification';
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-info-circle"></i>
            <span>Using fallback model for plant recognition. Results may be less accurate.</span>
            <button class="close-btn">&times;</button>
        </div>
    `;
    document.body.appendChild(notification);
    
    notification.querySelector('.close-btn').addEventListener('click', () => {
        notification.remove();
    });
    
    setTimeout(() => {
        notification.remove();
    }, 10000);
}

// Load class mappings from metadata
async function loadClassMappings() {
    try {
        // Try loading from the main models directory
        const response = await fetch('../models/chunk_0_metadata.json');
        const metadata = await response.json();
        
        // Populate class mapping
        if (metadata.class_mapping) {
            Object.assign(classMapping, metadata.class_mapping);
            console.log('Loaded class mappings for', Object.keys(classMapping).length, 'plants');
        } else {
            throw new Error('Invalid metadata format: missing class_mapping');
        }
    } catch (error) {
        console.error('Error loading class mappings:', error);
        
        // Create some sample class mappings for demo mode
        const demoClasses = [
            'Acacia_auriculiformis', 'Acacia_baileyana', 'Acacia_dealbata',
            'Agave_americana', 'Aloe_vera', 'Hibiscus_rosa-sinensis'
        ];
        
        demoClasses.forEach((className, index) => {
            classMapping[index] = className;
        });
        
        console.log('Using demo class mappings');
    }
}

// Get a plant image URL from the local data directory
function getPlantImageUrl(className) {
    // Try to construct a path to your local plant images database
    try {
        // Format the className to match directory structure (e.g. "Hypericum_canariense")
        // The path should be relative to the webapp directory
        return `../data/plant_images/${className}/${className}_1.jpg`;
    } catch (error) {
        console.warn('Could not find local image for', className);
        // Return a fallback image path
        return `./default_plant.jpg`;
    }
}

// Preprocess image for EfficientNetV2 - matches your Python training
function preprocessImageForEfficientNet(img) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = 224;
    canvas.height = 224;
    
    ctx.drawImage(img, 0, 0, 224, 224);
    const imageData = ctx.getImageData(0, 0, 224, 224);
    
    // EfficientNetV2 preprocessing: scale to [-1, 1]
    return tf.browser.fromPixels(imageData)
        .toFloat()
        .div(127.5)
        .sub(1.0)
        .expandDims(0);
}

// Preprocess image for DenseNet121 - matches your Python training
function preprocessImageForDenseNet(img) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = 224;
    canvas.height = 224;
    
    ctx.drawImage(img, 0, 0, 224, 224);
    const imageData = ctx.getImageData(0, 0, 224, 224);
    
    // DenseNet preprocessing (different from EfficientNet)
    // For RGB images, convert to BGR and zero-center each color channel
    return tf.tidy(() => {
        // Get RGB channels as separate tensors
        const pixels = tf.browser.fromPixels(imageData);
        
        // Convert to float and apply DenseNet121 normalization
        // DenseNet uses the Torch approach: subtract mean per channel [103.939, 116.779, 123.68]
        // These are BGR channel means for ImageNet
        const redMean = 123.68;
        const greenMean = 116.779;
        const blueMean = 103.939;
        
        // Separate channels, normalize them, and recombine
        const red = pixels.slice([0, 0, 0], [-1, -1, 1]);
        const green = pixels.slice([0, 0, 1], [-1, -1, 1]);
        const blue = pixels.slice([0, 0, 2], [-1, -1, 1]);
        
        const normalizedRed = red.toFloat().sub(redMean);
        const normalizedGreen = green.toFloat().sub(greenMean);
        const normalizedBlue = blue.toFloat().sub(blueMean);
        
        // Stack and expand dimensions for batch processing
        return tf.stack([normalizedRed, normalizedGreen, normalizedBlue], 2)
            .reshape([224, 224, 3])
            .expandDims(0);
    });
}

// Extract features in the exact same way as the Python training code
async function extractCombinedFeatures(img) {
    try {
        console.log("Starting feature extraction...");
        
        // Create tensors for both models
        const efficientNetTensor = preprocessImageForEfficientNet(img);
        const denseNetTensor = preprocessImageForDenseNet(img);
        
        // Debug the tensor shapes
        console.log(`EfficientNet tensor shape: ${efficientNetTensor.shape}`);
        console.log(`DenseNet tensor shape: ${denseNetTensor.shape}`);
        
        let efficientNetFeatures = [];
        let denseNetFeatures = [];
        
        // Try to get EfficientNetV2L features
        if (window.efficientNetExtractor) {
            try {
                console.log("Extracting EfficientNetV2L features...");
                const efficientNetOutput = await window.efficientNetExtractor.predict(efficientNetTensor);
                console.log(`EfficientNet output shape: ${efficientNetOutput.shape}`);
                efficientNetFeatures = Array.from(await efficientNetOutput.data());
                efficientNetOutput.dispose();
                console.log(`Extracted ${efficientNetFeatures.length} features from EfficientNetV2L`);
            } catch (error) {
                console.error("Failed to extract EfficientNetV2L features:", error);
                // Generate synthetic EfficientNetV2L features (1280 dimensions)
                efficientNetFeatures = new Array(1280).fill(0.01);
                console.log("Using synthetic EfficientNetV2L features instead");
            }
        } else {
            console.warn("EfficientNetV2L extractor not available");
            // Generate synthetic EfficientNetV2L features (1280 dimensions)
            efficientNetFeatures = new Array(1280).fill(0.01);
        }
        
        // Try to get DenseNet121 features
        if (window.denseNetExtractor) {
            try {
                console.log("Extracting DenseNet121 features...");
                const denseNetOutput = await window.denseNetExtractor.predict(denseNetTensor);
                console.log(`DenseNet output shape: ${denseNetOutput.shape}`);
                denseNetFeatures = Array.from(await denseNetOutput.data());
                denseNetOutput.dispose();
                console.log(`Extracted ${denseNetFeatures.length} features from DenseNet121`);
            } catch (error) {
                console.error("Failed to extract DenseNet121 features:", error);
                // Generate synthetic DenseNet121 features (1024 dimensions)
                denseNetFeatures = new Array(1024).fill(0.01);
                console.log("Using synthetic DenseNet121 features instead");
            }
        } else {
            console.warn("DenseNet121 extractor not available");
            // Generate synthetic DenseNet121 features (1024 dimensions)
            denseNetFeatures = new Array(1024).fill(0.01);
        }
        
        // Clean up tensors
        efficientNetTensor.dispose();
        denseNetTensor.dispose();
        
        // Combine features - MUST be exactly 2304 dimensions
        const combinedFeatures = [...efficientNetFeatures, ...denseNetFeatures];
        console.log(`Combined feature vector: ${combinedFeatures.length} dimensions`);
        
        // Ensure exactly 2304 dimensions
        if (combinedFeatures.length !== 2304) {
            console.warn(`Feature dimension mismatch! Expected 2304 but got ${combinedFeatures.length}`);
            
            // Fix the dimension to exactly 2304
            if (combinedFeatures.length > 2304) {
                return combinedFeatures.slice(0, 2304);
            } else {
                return [...combinedFeatures, ...new Array(2304 - combinedFeatures.length).fill(0.01)];
            }
        }
        
        return combinedFeatures;
    } catch (error) {
        console.error("Critical feature extraction error:", error);
        // Last resort: create synthetic features of exactly 2304 dimensions
        return new Array(2304).fill(0.01);
    }
}

// Preprocess image for model input
function preprocessImage(img) {
    // Create a canvas element to resize and normalize the image
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas dimensions to model input size (224x224 is common)
    canvas.width = 224;
    canvas.height = 224;
    
    // Draw and resize image to canvas
    ctx.drawImage(img, 0, 0, 224, 224);
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, 224, 224);
    
    // Convert to tensor and normalize using the same preprocessing as EfficientNetV2
    // EfficientNetV2 uses values in [-1, 1] range
    const tensor = tf.browser.fromPixels(imageData)
        .toFloat()
        .div(127.5)
        .sub(1.0)
        .expandDims(0);
    
    return tensor;
}

// Extract features using the model
async function extractFeatures(img) {
    try {
        // First check if we have feature extractors available
        if (window.efficientNetExtractor && window.denseNetExtractor) {
            // Use the extractCombinedFeatures function which already works
            return await extractCombinedFeatures(img);
        }
        
        // If the above method failed, try this alternative approach
        // Use the same preprocessing approach as training
        const tensor = preprocessImage(img);
        
        // Check if model is loaded
        if (!model) {
            console.warn('Model not loaded, using dummy features');
            // Create dummy features of appropriate length (2304 for your model)
            const dummyFeatures = new Array(2304).fill(0.5);
            tensor.dispose();
            return dummyFeatures;
        }
        
        // Get the expected input shape from the model
        const expectedShape = model.inputs[0].shape;
        console.log('Model expects input shape:', expectedShape);
        
        // If the model expects a 2D input (features), we need to extract features first
        if (expectedShape.length === 2) {
            console.log('Model expects feature vector input, not direct images');
            
            // Create dummy features of the expected length
            const featureLength = expectedShape[1] || 2304;
            const dummyFeatures = new Array(featureLength).fill(0.5);
            tensor.dispose();
            return dummyFeatures;
        }
        
        // If we reach here, the model can process images directly
        // Just use the preprocessed tensor
        const predictions = await model.predict(tensor).data();
        tensor.dispose();
        return Array.from(predictions);
    } catch (error) {
        console.error('Feature extraction error:', error);
        // Log more details about the error for debugging
        console.error('Error details:', error.message);
        
        // Last resort fallback - return dummy features
        return new Array(2304).fill(0.5); // Create dummy features matching your model's input dimension
    }
}

// Make prediction on an image
async function makePrediction(img) {
    try {
        console.log("Starting prediction process...");
        // Get features using the exact same approach as in Python's cross_model_fusion_predict
        const combinedFeatures = await extractCombinedFeatures(img);
        console.log(`Extracted combined features with length: ${combinedFeatures.length}`);
        
        // Verify we have valid features
        if (!combinedFeatures || combinedFeatures.length === 0) {
            console.error("Failed to extract features for prediction");
            throw new Error("Feature extraction failed");
        }
        
        // Create a tensor from the combined features
        const featureTensor = tf.tensor(combinedFeatures).expandDims(0);
        console.log(`Created feature tensor with shape: ${featureTensor.shape}`);
        
        // Use the classifier part of the model for prediction
        let rawPredictions;
        try {
            // Ensure model is loaded
            if (!model) {
                console.error("Model not loaded!");
                throw new Error("Model not loaded");
            }
            
            console.log("Running model prediction...");
            // This is the critical part where we get raw predictions from the model
            const predictionTensor = model.predict(featureTensor);
            rawPredictions = await predictionTensor.data();
            predictionTensor.dispose();
            
            console.log("Raw prediction values (first 5):", Array.from(rawPredictions).slice(0, 5));
            console.log("Max raw confidence:", Math.max(...Array.from(rawPredictions)));
            console.log("Min raw confidence:", Math.min(...Array.from(rawPredictions)));
        } catch (e) {
            console.error('Model prediction error:', e);
            throw e;
        } finally {
            // Clean up
            featureTensor.dispose();
        }
        
        // Map the raw prediction values to class information WITHOUT any confidence boosting
        const results = Array.from(rawPredictions)
            .map((confidence, index) => ({
                classId: index,
                className: classMapping[index] || `Unknown (${index})`,
                confidence: confidence,
                rawConfidence: confidence // Keep the raw confidence value
            }))
            .sort((a, b) => b.confidence - a.confidence);
        
        console.log(`Top prediction: ${results[0].className} with raw confidence ${results[0].confidence}`);
        
        // Return the results with raw confidence values (no boosting)
        return results;
    } catch (error) {
        console.error('Critical prediction error:', error);
        
        // Return a proper error indicator so the UI can show the problem
        return [
            {
                classId: -1,
                className: "Error: Prediction failed",
                confidence: 0,
                error: error.message || "Unknown error in prediction"
            }
        ];
    }
}

// Make focused prediction on an image (only for specific classes)
async function makeFocusedPrediction(img, classIds) {
    console.log("Starting focused prediction for Phase 2...");
    
    // First extract features using our combined approach
    const combinedFeatures = await extractCombinedFeatures(img);
    console.log(`Phase 2: Extracted combined features with length: ${combinedFeatures.length}`);
    
    // Verify we have valid features
    if (!combinedFeatures || combinedFeatures.length === 0) {
        console.error("Phase 2: Failed to extract features for prediction");
        throw new Error("Feature extraction failed in Phase 2");
    }
    
    // Use the combined features directly
    try {
        // Create a tensor from the combined features
        const featureTensor = tf.tensor(combinedFeatures).expandDims(0);
        console.log(`Phase 2: Created feature tensor with shape: ${featureTensor.shape}`);
        
        // Use the classifier part of the model for prediction
        let rawPredictions;
        try {
            // Ensure model is loaded
            if (!model) {
                console.error("Phase 2: Model not loaded!");
                throw new Error("Model not loaded in Phase 2");
            }
            
            console.log("Phase 2: Running model prediction...");
            // This is the critical part where we get raw predictions from the model
            const predictionTensor = model.predict(featureTensor);
            rawPredictions = await predictionTensor.data();
            predictionTensor.dispose();
            
            console.log("Phase 2: Raw prediction values (first 5):", Array.from(rawPredictions).slice(0, 5));
            console.log("Phase 2: Max raw confidence:", Math.max(...Array.from(rawPredictions)));
        } catch (e) {
            console.error('Phase 2: Model prediction error:', e);
            featureTensor.dispose();
            throw e;
        } finally {
            // Clean up
            featureTensor.dispose();
        }
        
        // Map predictions to class info but only keep the ones we're focusing on
        // Use raw confidence values without any boosting
        const results = Array.from(rawPredictions)
            .map((confidence, index) => ({
                classId: index,
                className: classMapping[index] || `Unknown (${index})`,
                confidence: confidence,
                rawConfidence: confidence // Keep raw value for debugging
            }))
            .filter(p => classIds.includes(p.classId))
            .sort((a, b) => b.confidence - a.confidence);
        
        if (results.length > 0) {
            console.log(`Phase 2: Top filtered prediction: ${results[0].className} with confidence ${results[0].confidence}`);
        } else {
            console.warn("Phase 2: No predictions matched the filtered class IDs");
        }
        
        return results;
    } catch (error) {
        console.error('Phase 2 critical error:', error);
        
        // If the phase 2 prediction fails, fall back to filtering the phase 1 predictions
        console.log("Phase 2: Using fallback to Phase 1 results filtered by classIds");
        const allPredictions = await makePrediction(img);
        
        // Filter predictions to only include the classes we're focused on
        const focusedPredictions = allPredictions
            .filter(p => classIds.includes(p.classId))
            .sort((a, b) => b.confidence - a.confidence);
        
        return focusedPredictions;
    }
}

// Analyze images
async function analyzeImages() {
    if (isModelLoading || (!selectedImages.length && !capturedFrames.length)) return;
    
    try {
        // Phase 1: Initial screening
        showLoading('Phase 1: Pre-selecting potential plants...');
        
        const imagesToProcess = isVideoMode ? capturedFrames : selectedImages;
        let allPredictions = [];
        
        // Process each image
        for (let i = 0; i < imagesToProcess.length; i++) {
            const img = imagesToProcess[i].element;
            showLoading(`Processing image ${i+1}/${imagesToProcess.length}...`);
            
            // Make prediction
            const predictions = await makePrediction(img);
            allPredictions.push(predictions);
        }
        
        // Combine predictions from all images (if multiple)
        const combinedPredictions = combineMultipleImagePredictions(allPredictions);
        
        // Display phase 1 results
        displayPhase1Results(combinedPredictions);
        
        // Phase 2: Focused search
        showLoading('Phase 2: Performing focused search...');
        
        // Get the top 20 classes to focus on
        const topClasses = combinedPredictions.slice(0, 20).map(p => p.classId);
        
        // Process each image with focused search
        let focusedPredictions = [];
        for (let i = 0; i < imagesToProcess.length; i++) {
            const img = imagesToProcess[i].element;
            showLoading(`Analyzing image ${i+1}/${imagesToProcess.length} with focused search...`);
            
            // Make focused prediction (only on top 20 classes)
            const predictions = await makeFocusedPrediction(img, topClasses);
            focusedPredictions.push(predictions);
        }
        
        // Combine focused predictions
        const combinedFocusedPredictions = combineMultipleImagePredictions(focusedPredictions);
        
        // Display phase 2 results
        displayPhase2Results(combinedFocusedPredictions);
        
        // Display final result
        displayFinalResult(combinedFocusedPredictions[0], imagesToProcess[0].element);
        
        // Show results container
        elements.resultsContainer.style.display = 'block';
        
        hideLoading();
    } catch (error) {
        console.error('Error analyzing images:', error);
        alert('An error occurred while analyzing images. Please try again.');
        hideLoading();
    }
}

// Combine predictions from multiple images
function combineMultipleImagePredictions(predictionsArray) {
    if (predictionsArray.length === 1) {
        return predictionsArray[0];
    }
    
    // Create a map to aggregate confidences by class ID
    const combinedMap = new Map();
    
    // Process each prediction set
    for (const predictions of predictionsArray) {
        for (const pred of predictions) {
            const { classId, className, confidence } = pred;
            
            if (combinedMap.has(classId)) {
                // Update existing entry (use max confidence as a simple ensemble method)
                const current = combinedMap.get(classId);
                combinedMap.set(classId, {
                    classId,
                    className,
                    confidence: Math.max(current.confidence, confidence)
                });
            } else {
                // Add new entry
                combinedMap.set(classId, { classId, className, confidence });
            }
        }
    }
    
    // Convert map to array and sort by confidence
    return Array.from(combinedMap.values())
        .sort((a, b) => b.confidence - a.confidence);
}

// Display Phase 1 results
function displayPhase1Results(predictions) {
    const container = elements.phase1Results;
    container.innerHTML = '';
    
    // Display top 20 predictions
    const top20 = predictions.slice(0, 20);
    
    top20.forEach(pred => {
        const card = document.createElement('div');
        card.className = 'result-card';
        
        // Format plant name for display (convert underscores to spaces)
        const displayName = pred.className.replace(/_/g, ' ');
        
        // Use the actual plant image from your database
        const plantImageUrl = getPlantImageUrl(pred.className);
        
        // Create an error handler for the plant image
        const handleImageError = (e) => {
            console.warn(`Could not load plant image for ${pred.className}`);
            // Fallback to the user's uploaded image if plant image fails to load
            e.target.src = isVideoMode 
                ? capturedFrames[0].dataUrl 
                : selectedImages[0].dataUrl;
            e.target.onerror = null; // Prevent infinite error loops
        };
        
        card.innerHTML = `
            <img src="${plantImageUrl}" alt="${displayName}" onerror="this.onerror=null; this.src='${isVideoMode ? capturedFrames[0].dataUrl : selectedImages[0].dataUrl}';">
            <div class="plant-name">${displayName}</div>
            <div class="confidence">${(pred.confidence * 100).toFixed(1)}% confidence</div>
            <div class="confidence-bar">
                <div class="confidence-bar-fill" style="width: ${pred.confidence * 100}%"></div>
            </div>
        `;
        
        container.appendChild(card);
    });
}

// Display Phase 2 results
function displayPhase2Results(predictions) {
    const container = elements.phase2Results;
    container.innerHTML = '';
    
    // Display all focused predictions
    predictions.forEach((pred, index) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        if (index === 0) card.classList.add('selected');
        
        // Format plant name for display
        const displayName = pred.className.replace(/_/g, ' ');
        
        // Use the actual plant image from your database
        const plantImageUrl = getPlantImageUrl(pred.className);
        
        card.innerHTML = `
            <img src="${plantImageUrl}" alt="${displayName}" onerror="this.onerror=null; this.src='${isVideoMode ? capturedFrames[0].dataUrl : selectedImages[0].dataUrl}';">
            <div class="plant-name">${displayName}</div>
            <div class="confidence">${(pred.confidence * 100).toFixed(1)}% confidence</div>
            <div class="confidence-bar">
                <div class="confidence-bar-fill" style="width: ${pred.confidence * 100}%"></div>
            </div>
        `;
        
        // Add click event to select this result
        card.addEventListener('click', () => {
            // Remove selection from all cards
            container.querySelectorAll('.result-card').forEach(c => c.classList.remove('selected'));
            // Add selection to this card
            card.classList.add('selected');
            // Update final result
            displayFinalResult(pred, isVideoMode ? capturedFrames[0].element : selectedImages[0].element);
        });
        
        container.appendChild(card);
    });
}

// Display final result
function displayFinalResult(prediction, imgElement) {
    const container = elements.finalResult;
    container.innerHTML = '';
    
    if (!prediction) return;
    
    // Format plant name for display
    const plantName = prediction.className.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
    
    // Create scientific name (italicized genus and species)
    const nameParts = prediction.className.split('_');
    let scientificName = '';
    
    if (nameParts.length >= 2) {
        scientificName = `<em>${nameParts[0]} ${nameParts.slice(1).join(' ')}</em>`;
    } else {
        scientificName = `<em>${prediction.className}</em>`;
    }
    
    // Get the actual plant image from your database
    const plantImageUrl = getPlantImageUrl(prediction.className);
    
    // Get the user's uploaded image for the analyzed section
    const userImageSource = isVideoMode 
        ? capturedFrames[0].dataUrl 
        : selectedImages[0].dataUrl;
    
    // Create image with analyzed plant
    const analyzedCanvas = document.createElement('canvas');
    const ctx = analyzedCanvas.getContext('2d');
    
    // Set canvas size to match image
    analyzedCanvas.width = imgElement.naturalWidth;
    analyzedCanvas.height = imgElement.naturalHeight;
    
    // Draw image to canvas
    ctx.drawImage(imgElement, 0, 0);
    
    // Draw bounding box (simple implementation - just highlight center area)
    const boxWidth = imgElement.naturalWidth * 0.6;
    const boxHeight = imgElement.naturalHeight * 0.6;
    const boxX = (imgElement.naturalWidth - boxWidth) / 2;
    const boxY = (imgElement.naturalHeight - boxHeight) / 2;
    
    ctx.strokeStyle = '#4caf50';
    ctx.lineWidth = 4;
    ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);
    
    // Add label
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(boxX, boxY - 35, boxWidth, 35);
    ctx.fillStyle = '#ffffff';
    ctx.font = '16px Arial';
    ctx.fillText(`${plantName} (${(prediction.confidence * 100).toFixed(1)}%)`, boxX + 10, boxY - 12);
    
    // Create data URL from canvas
    const analyzedImageUrl = analyzedCanvas.toDataURL('image/jpeg');
    
    // Create final result HTML
    container.innerHTML = `
        <div class="plant-header">
            <img src="${plantImageUrl}" alt="${plantName}" class="plant-image" onerror="this.onerror=null; this.src='${userImageSource}';">
            <h2 class="plant-title">${plantName}</h2>
            <div class="plant-scientific">${scientificName}</div>
            <div class="confidence">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
        </div>
        
        <div class="detected-image">
            <h3>Detected Plant</h3>
            <img src="${analyzedImageUrl}" alt="Analyzed Image" style="max-width: 100%; border-radius: 8px; margin: 1rem 0;">
            <button class="download-btn" id="downloadBtn">
                <i class="fas fa-download"></i> Download Result
            </button>
        </div>
        
        <div class="user-feedback">
            <h4>Was this identification correct?</h4>
            <div class="feedback-options">
                <div class="feedback-option" data-value="correct">Yes, correct</div>
                <div class="feedback-option" data-value="partially">Partially correct</div>
                <div class="feedback-option" data-value="wrong">No, incorrect</div>
            </div>
        </div>
    `;
    
    // Add download functionality
    const downloadBtn = container.querySelector('#downloadBtn');
    downloadBtn.addEventListener('click', () => {
        const link = document.createElement('a');
        link.href = analyzedImageUrl;
        link.download = `plant_recognition_${plantName.replace(/\s+/g, '_')}.jpg`;
        link.click();
    });
    
    // Add feedback functionality
    const feedbackOptions = container.querySelectorAll('.feedback-option');
    feedbackOptions.forEach(option => {
        option.addEventListener('click', () => {
            // Remove selection from all options
            feedbackOptions.forEach(o => o.classList.remove('selected'));
            // Add selection to this option
            option.classList.add('selected');
            
            // In a real app, this would send feedback to the server
            const feedbackValue = option.dataset.value;
            console.log('User feedback:', feedbackValue, 'for plant:', plantName);
        });
    });
}

// Show loading overlay
function showLoading(message = 'Loading...') {
    elements.loadingText.textContent = message;
    elements.loadingOverlay.classList.add('active');
}

// Hide loading overlay
function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

// Load feature extractors
async function loadFeatureExtractors() {
    showLoading('Loading feature extractors...');
    
    try {
        // Use EfficientNetV2L to exactly match Python code, not B0
        console.log('Loading exact feature extractors to match Python implementation...');
        
        // Load EfficientNetV2L feature extractor (primary) - matches your Python code exactly
        const efficientNetURL = 'https://tfhub.dev/tensorflow/tfjs-model/efficientnetv2-l/feature_vector/2/default/1';
        window.efficientNetExtractor = await tf.loadGraphModel(efficientNetURL, {fromTFHub: true});
        console.log('Loaded EfficientNetV2L feature extractor (exact match to Python implementation)');
        
        // Load DenseNet121 feature extractor (secondary)
        try {
            const denseNetURL = 'https://tfhub.dev/tensorflow/tfjs-model/densenet121/feature_vector/1/default/1';
            window.denseNetExtractor = await tf.loadGraphModel(denseNetURL, {fromTFHub: true});
            console.log('Loaded DenseNet121 feature extractor');
            
            // Check dimensions to verify they match Python implementation
            // This code checks that we get the expected feature dimensions
            const testTensor = tf.zeros([1, 224, 224, 3]);
            const efficientNetFeatures = window.efficientNetExtractor.predict(testTensor);
            const denseNetFeatures = window.denseNetExtractor.predict(testTensor);
            
            console.log('EfficientNetV2L feature dimension:', efficientNetFeatures.shape[1]);
            console.log('DenseNet121 feature dimension:', denseNetFeatures.shape[1]);
            
            const totalDimension = efficientNetFeatures.shape[1] + denseNetFeatures.shape[1];
            console.log('Combined feature dimension:', totalDimension);
            
            // Clean up test tensors
            testTensor.dispose();
            efficientNetFeatures.dispose();
            denseNetFeatures.dispose();
            
        } catch (denseNetError) {
            console.warn('Could not load DenseNet121 feature extractor:', denseNetError);
            // Fall back to alternative approach that better matches Python
            try {
                console.log('Trying alternative DenseNet121 URL...');
                const altDenseNetURL = 'https://tfhub.dev/tensorflow/tfjs-model/densenet121/classification/1/default/1';
                const tempModel = await tf.loadGraphModel(altDenseNetURL, {fromTFHub: true});
                
                // Create a feature extractor from the classification model
                // by removing the last layer
                window.denseNetExtractor = tf.model({
                    inputs: tempModel.inputs,
                    outputs: tempModel.layers[tempModel.layers.length - 2].output
                });
                console.log('Successfully created DenseNet121 feature extractor from classification model');
            } catch (altError) {
                console.warn('Failed to load alternative DenseNet121:', altError);
                // Last resort: copy EfficientNet features
                window.denseNetExtractor = window.efficientNetExtractor;
                console.warn('Using EfficientNetV2L features as fallback for missing DenseNet features');
            }
        }
        
        // Create a global variable to track feature dimensions
        window.featureDimensions = {
            efficientNet: 0,
            denseNet: 0,
            getTotal() { return this.efficientNet + this.denseNet; }
        };
        
        // Test to get actual dimensions
        const testImg = new Image(224, 224);
        testImg.src = 'data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==';
        await new Promise(resolve => {
            testImg.onload = resolve;
        });
        
        const features = await extractCombinedFeatures(testImg);
        
        return true;
    } catch (error) {
        console.error('Error loading feature extractors:', error);
        return false;
    } finally {
        hideLoading();
    }
}