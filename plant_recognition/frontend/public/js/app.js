// Import modules
import { CONFIG } from './modules/config.js';
import { appState } from './modules/state.js';
import { api } from './modules/api.js';
import { authManager } from './modules/auth.js';
import { uiManager } from './modules/ui.js';
import { adminManager } from './modules/admin.js';
import { imageStorage } from './modules/imageStorage.js';
import { sanitizeHtml, sanitizeAndTruncate } from './utils/sanitize.js';
import errorHandler from './modules/errorHandler.js';

// Main application class - replaces global functions
class PlantRecognitionApp {
    constructor() {
        this.initialized = false;
        this.eventListenersSetup = false;
        this.map = null;
        this.stream = null;
        this.captureInterval = null;
        this.weatherInterval = null;
    }

    async initialize() {
        if (this.initialized) return;
        
        console.log('Initializing Plant Recognition Application...');
        
        try {
            // Initialize error handling first
            errorHandler.initialize();
            console.log('Error handler initialized');
            
            // Initialize managers
            await authManager.initialize();
            console.log('Auth manager initialized');
            uiManager.initialize();
            console.log('UI manager initialized');
            await adminManager.initialize();
            console.log('Admin manager initialized');
            
            // Wait for DOM to be completely ready
            if (document.readyState !== 'complete') {
                console.log('Waiting for DOM to be complete...');
                await new Promise(resolve => {
                    window.addEventListener('load', resolve, { once: true });
                });
            }
            
            // Additional wait to ensure all elements are rendered
            await new Promise(resolve => setTimeout(resolve, 200));
            console.log('DOM ready, setting up event listeners...');
            
            // Set up event listeners only once
            this.setupEventListeners();
            
            // Initialize UI
            authManager.updateAuthUI();
            console.log('Auth UI updated');
            
            // Initialize map if needed
            if (appState.get('currentTab') === 'map') {
                this.initializeMap();
            }
            
            // Start weather checking
            this.startWeatherChecking();
            
            this.initialized = true;
            console.log('Application initialization completed successfully');
        } catch (error) {
            console.error('Application initialization failed:', error);
            errorHandler.handleError(error, {
                context: 'application_initialization',
                severity: 'critical'
            });
        }
    }

    setupEventListeners() {
        // Guard against multiple calls
        if (this.eventListenersSetup) return;
        this.eventListenersSetup = true;
        
        try {
            console.log('Setting up event listeners...');
            console.log('Document ready state:', document.readyState);
            
            // File upload events
            const fileInput = document.getElementById('file-input');
            const uploadForm = document.getElementById('upload-form');
            const modeToggleBtn = document.getElementById('mode-toggle-btn');
            const startCameraBtn = document.getElementById('start-camera-btn');
            const stopCameraBtn = document.getElementById('stop-camera-btn');
            
            // Also check for the mode-icon specifically
            const modeIcon = document.getElementById('mode-icon');
            
            console.log('Found elements:', {
                fileInput: !!fileInput,
                uploadForm: !!uploadForm,
                modeToggleBtn: !!modeToggleBtn,
                modeIcon: !!modeIcon,
                startCameraBtn: !!startCameraBtn,
                stopCameraBtn: !!stopCameraBtn
            });
            
            // Debug: Check if the elements exist in the DOM
            if (!modeToggleBtn) {
                console.error('mode-toggle-btn not found in DOM');
                console.log('All buttons in document:', document.querySelectorAll('button').length);
                console.log('All elements with id containing "mode":', document.querySelectorAll('[id*="mode"]'));
            }
            
            if (!modeIcon) {
                console.error('mode-icon not found in DOM');
                console.log('All i elements in document:', document.querySelectorAll('i').length);
                console.log('All elements with id containing "icon":', document.querySelectorAll('[id*="icon"]'));
                
                // Debug: Check the actual button content
                if (modeToggleBtn) {
                    console.log('mode-toggle-btn innerHTML:', modeToggleBtn.innerHTML);
                    console.log('mode-toggle-btn children:', modeToggleBtn.children.length);
                    console.log('mode-toggle-btn first child:', modeToggleBtn.firstElementChild);
                    console.log('mode-toggle-btn first child id:', modeToggleBtn.firstElementChild?.id);
                }
                
                // Try alternative selectors
                const iconByClass = document.querySelector('.mode-toggle-btn i');
                console.log('Icon found by class selector:', !!iconByClass);
                if (iconByClass) {
                    console.log('Icon by class id:', iconByClass.id);
                }
            }
            
            if (fileInput) {
                fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
            }
            if (uploadForm) {
                uploadForm.addEventListener('submit', (e) => this.handleUploadSubmit(e));
            }
            if (modeToggleBtn) {
                modeToggleBtn.addEventListener('click', () => this.toggleMode());
            }
            if (startCameraBtn) {
                startCameraBtn.addEventListener('click', () => this.startCameraHandler());
            }
            if (stopCameraBtn) {
                stopCameraBtn.addEventListener('click', () => this.stopStreaming());
            }
            
            // Upload functionality
            const browseLink = document.getElementById('browse-link');
            const dropZone = document.getElementById('drop-zone');
            
            if (browseLink) {
                browseLink.addEventListener('click', (e) => {
                    e.preventDefault();
                    fileInput.click();
                });
            }
            
            if (dropZone) {
                dropZone.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    dropZone.classList.add('drag-over');
                });
                
                dropZone.addEventListener('dragleave', (e) => {
                    e.preventDefault();
                    dropZone.classList.remove('drag-over');
                });
                
                dropZone.addEventListener('drop', (e) => {
                    e.preventDefault();
                    dropZone.classList.remove('drag-over');
                    
                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        fileInput.files = files;
                        this.handleFileSelect({ target: { files: files } });
                    }
                });
            }
            
            // Authentication event listeners
            const loginBtn = document.getElementById('login-btn');
            const signupBtn = document.getElementById('signup-btn');
            const logoutBtn = document.getElementById('logout-btn');
            const loginForm = document.getElementById('login-form');
            const signupForm = document.getElementById('signup-form');
            const loginClose = document.getElementById('login-close');
            const signupClose = document.getElementById('signup-close');
            
            if (loginBtn) {
                loginBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.showLoginModal();
                });
            }
            
            if (signupBtn) {
                signupBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.showSignupModal();
                });
            }
            
            if (logoutBtn) {
                logoutBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.handleLogout();
                });
            }
            
            if (loginForm) {
                loginForm.addEventListener('submit', (e) => this.handleLogin(e));
            }
            
            if (signupForm) {
                signupForm.addEventListener('submit', (e) => this.handleSignup(e));
            }
            
            if (loginClose) {
                loginClose.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.closeLoginModal();
                });
            }
            
            if (signupClose) {
                signupClose.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.closeSignupModal();
                });
            }
            
            // Close modals when clicking outside
            const loginModal = document.getElementById('login-modal');
            const signupModal = document.getElementById('signup-modal');
            
            if (loginModal) {
                loginModal.addEventListener('click', (e) => {
                    if (e.target === loginModal) {
                        this.closeLoginModal();
                    }
                });
            }
            
            if (signupModal) {
                signupModal.addEventListener('click', (e) => {
                    if (e.target === signupModal) {
                        this.closeSignupModal();
                    }
                });
            }
            
            // Admin toggle event listener
            const adminToggle = document.getElementById('mode-toggle');
            console.log('Admin toggle element found:', !!adminToggle);
            if (adminToggle) {
                adminToggle.addEventListener('change', (e) => this.handleAdminToggle(e));
                console.log('Admin toggle event listener added');
            } else {
                console.error('Admin toggle element not found');
            }
            
            // LLM control buttons
            const clearHistoryBtn = document.getElementById('clear-history-btn');
            const exportAnalysisBtn = document.getElementById('export-analysis-btn');
            const testLlmBtn = document.getElementById('test-llm-btn');
            const clearProcessedBtn = document.getElementById('clear-processed-btn');
            
            if (clearHistoryBtn) {
                clearHistoryBtn.addEventListener('click', () => {
                    appState.clearDetectionHistory();
                    this.updateDetectionHistoryDisplay();
                });
            }
            
            if (exportAnalysisBtn) {
                exportAnalysisBtn.addEventListener('click', () => {
                    this.exportAnalysis();
                });
            }
            
            if (testLlmBtn) {
                testLlmBtn.addEventListener('click', () => {
                    this.testLLM();
                });
            }
            
            if (clearProcessedBtn) {
                clearProcessedBtn.addEventListener('click', () => {
                    this.clearProcessedDetections();
                });
            }
            
            // Admin tab switching
            const adminTabs = document.querySelectorAll('.admin-tab');
            adminTabs.forEach(tab => {
                tab.addEventListener('click', (e) => {
                    e.preventDefault();
                    const tabName = tab.getAttribute('data-tab');
                    this.switchAdminTab(tabName);
                });
            });
            
            console.log('Event listeners set up successfully');
            
        } catch (error) {
            errorHandler.handleError(error, {
                context: 'setupEventListeners',
                severity: 'error'
            });
        }
    }

    toggleMode() {
        const videoContent = document.getElementById('video-content');
        const uploadContent = document.getElementById('upload-content');
        const modeToggleBtn = document.getElementById('mode-toggle-btn');
        
        // Try to find the icon element by ID first, then by class selector as fallback
        let modeIcon = document.getElementById('mode-icon');
        if (!modeIcon && modeToggleBtn) {
            modeIcon = modeToggleBtn.querySelector('i');
        }
        
        // Check if all required elements exist
        if (!videoContent || !uploadContent || !modeIcon || !modeToggleBtn) {
            console.error('Required elements not found for mode toggle:', {
                videoContent: !!videoContent,
                uploadContent: !!uploadContent,
                modeIcon: !!modeIcon,
                modeToggleBtn: !!modeToggleBtn
            });
            
            // Try multiple times with increasing delays
            this.retryFindElements(0);
            return;
        }
        
        this.performToggle(videoContent, uploadContent, modeIcon, modeToggleBtn);
    }
    
    retryFindElements(attempt) {
        const maxAttempts = 5;
        const delays = [100, 200, 300, 500, 1000]; // Increasing delays
        
        if (attempt >= maxAttempts) {
            console.error('Failed to find elements after', maxAttempts, 'attempts');
            return;
        }
        
        setTimeout(() => {
            const videoContent = document.getElementById('video-content');
            const uploadContent = document.getElementById('upload-content');
            const modeToggleBtn = document.getElementById('mode-toggle-btn');
            
            // Try to find the icon element by ID first, then by class selector as fallback
            let modeIcon = document.getElementById('mode-icon');
            if (!modeIcon && modeToggleBtn) {
                modeIcon = modeToggleBtn.querySelector('i');
            }
            
            console.log(`Retry attempt ${attempt + 1}:`, {
                videoContent: !!videoContent,
                uploadContent: !!uploadContent,
                modeIcon: !!modeIcon,
                modeToggleBtn: !!modeToggleBtn
            });
            
            if (videoContent && uploadContent && modeIcon && modeToggleBtn) {
                console.log('Elements found on retry, proceeding with toggle');
                this.performToggle(videoContent, uploadContent, modeIcon, modeToggleBtn);
            } else {
                this.retryFindElements(attempt + 1);
            }
        }, delays[attempt]);
    }
    
    performToggle(videoContent, uploadContent, modeIcon, modeToggleBtn) {
        // Check if we're currently in video mode by looking at the active class
        const isVideoMode = videoContent.classList.contains('active');
        console.log('Toggle mode - isVideoMode:', isVideoMode);
        
        if (isVideoMode) {
            // Switch to upload mode
            videoContent.style.display = 'none';
            videoContent.classList.remove('active');
            uploadContent.style.display = 'block';
            uploadContent.classList.add('active');
            // Show camera icon when in upload mode (to switch back to streaming)
            modeIcon.className = 'fas fa-video';
            modeToggleBtn.innerHTML = '<i class="fas fa-video"></i>';
            console.log('Switched to upload mode, showing camera icon');
        } else {
            // Switch to video mode
            videoContent.style.display = 'block';
            videoContent.classList.add('active');
            uploadContent.style.display = 'none';
            uploadContent.classList.remove('active');
            // Show upload icon when in streaming mode (to switch to upload)
            modeIcon.className = 'fas fa-upload';
            modeToggleBtn.innerHTML = '<i class="fas fa-upload"></i>';
            console.log('Switched to video mode, showing upload icon');
        }
    }

    async handleFileSelect(e) {
        const file = e.target.files[0];
        const error = this.validateFile(file);
        
        if (error) {
            this.showError(error);
            return;
        }
        
        this.updateUploadArea(true);
        
        // Show preview of selected image
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.displayUploadedImage(e.target.result);
            };
            reader.readAsDataURL(file);
        }
    }

    validateFile(file) {
        if (!file) return 'No file selected';
        if (file.size > CONFIG.MAX_FILE_SIZE) return 'File too large (max 10MB)';
        if (!CONFIG.ALLOWED_FILE_TYPES || !Array.isArray(CONFIG.ALLOWED_FILE_TYPES)) {
            return 'Invalid configuration: allowed file types not defined';
        }
        if (!CONFIG.ALLOWED_FILE_TYPES.includes(file.type)) return 'Invalid file type';
        return null;
    }

    updateUploadArea(hasFile) {
        const dropZone = document.getElementById('drop-zone');
        const submitBtn = document.getElementById('submit-btn');
        
        if (hasFile) {
            dropZone.classList.add('has-file');
            submitBtn.disabled = false;
        } else {
            dropZone.classList.remove('has-file');
            submitBtn.disabled = true;
        }
    }

    async handleUploadSubmit(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('file-input');
        const submitBtn = document.getElementById('submit-btn');
        const statusElement = document.querySelector('.ai-analysis h2') || document.querySelector('.ai-analysis h3');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('Please select a file');
            return;
        }
        
        try {
            // Update button state for classification
            submitBtn.disabled = true;
            submitBtn.textContent = 'Analysing image...';
            
            // Update status
            if (statusElement) {
                statusElement.textContent = 'Processing with classification model...';
            }
            
            const formData = new FormData();
            formData.append('image', file);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            
            // Display the uploaded image
            this.displayUploadedImage(result.imageUrl || result.frameImage);
            
            // Update button state for LLM processing
            submitBtn.textContent = 'Processing with LLM...';
            
            // Update status
            if (statusElement) {
                statusElement.textContent = 'Processing with LLM...';
            }
            
            await this.addDetectionToHistory(result);
            
        } catch (error) {
            this.showError('Upload failed: ' + error.message);
        } finally {
            // Reset button and status
            submitBtn.disabled = false;
            submitBtn.textContent = 'Analyze Image';
            
            if (statusElement) {
                statusElement.textContent = 'Ready for analysis';
            }
        }
    }

    async addDetectionToHistory(result) {
        try {
            // Save image locally if we have a database ID
            let localImageUrl = result.imageUrl;
            let localFrameImage = result.frameImage;
            
            // Create the display detection object
            const detection = {
                id: Date.now().toString(),
                timestamp: new Date().toISOString(),
                species: result.predicted_species,
                confidence: result.confidence,
                imageUrl: localImageUrl,
                frameImage: localFrameImage,
                isVideo: result.isVideo || false,
                videoFile: result.videoFile,
                status: 'pending',
                llmAnalysis: null
            };
            
            // Add to display history
            appState.addDetection(detection);
            this.updateDetectionHistoryDisplay();
            
            // Save to database first if logged in
            if (appState.isAuthenticated()) {
                try {
                    // Create temp detection with all required database fields
                    const tempDetection = this.createTempDetection(
                        result.predicted_species,
                        result.confidence,
                        localImageUrl,
                        new Date().toISOString()
                    );
                    
                    console.log('Created temp detection for database:', tempDetection);
                    
                    // Save temp detection to database
                    const saveResult = await this.saveDetectionToDatabase(tempDetection);
                    
                    if (saveResult && saveResult.success && saveResult.databaseId) {
                        detection.databaseId = saveResult.databaseId;
                        console.log('Successfully saved detection to database with ID:', saveResult.databaseId);
                        
                        // Now save images locally with the database ID
                        try {
                            if (result.imageUrl && result.imageUrl.startsWith('data:')) {
                                // Convert base64 to blob and save
                                const imageBlob = imageStorage.base64ToBlob(result.imageUrl);
                                const localPath = await imageStorage.saveImageLocally(imageBlob, saveResult.databaseId, 'detection');
                                detection.imageUrl = localPath;
                                localImageUrl = localPath;
                            }
                            
                            if (result.frameImage && result.frameImage.startsWith('data:')) {
                                // Convert base64 to blob and save
                                const frameBlob = imageStorage.base64ToBlob(result.frameImage);
                                const localFramePath = await imageStorage.saveImageLocally(frameBlob, saveResult.databaseId, 'frame');
                                detection.frameImage = localFramePath;
                                localFrameImage = localFramePath;
                            }
                            
                            // Update the detection in the database with local paths
                            await this.updateDetectionInDatabase(detection);
                            
                        } catch (imageError) {
                            console.error('Error saving images locally:', imageError);
                            // Continue without local storage if it fails
                        }
                    } else {
                        console.error('Failed to save detection to database:', saveResult);
                    }
                } catch (error) {
                    console.error('Error saving detection to database:', error);
                    errorHandler.handleError(error, {
                        context: 'addDetectionToHistory_database_save',
                        severity: 'warning'
                    });
                }
            }
            
            // Always trigger LLM analysis for all detections
            try {
                await this.triggerLLMAnalysis(detection);
            } catch (error) {
                console.error('Error triggering LLM analysis:', error);
                // Don't fail the detection if LLM analysis fails
            }
            
            // Notify admin managers of new detection for real-time updates
            this.notifyAdminManagers(detection);
            
        } catch (error) {
            errorHandler.handleError(error, {
                context: 'addDetectionToHistory',
                severity: 'error'
            });
        }
    }

    // Create a temporary detection object with all required database fields
    createTempDetection(species, confidence, imageUrl, timestamp) {
        const currentLocation = appState.get('currentLocation') || { lat: 0, lng: 0 };
        
        return {
            species: species,
            confidence: confidence,
            imageUrl: imageUrl,
            latitude: currentLocation.lat,
            longitude: currentLocation.lng,
            location: {
                type: 'Point',
                coordinates: [currentLocation.lng, currentLocation.lat]
            },
            timestamp: timestamp,
            description: '',
            detection: {
                method: 'image_upload',
                confidence: confidence,
                processingTime: 0
            },
            management: {
                status: 'pending',
                priority: 'medium'
            },
            llmAnalysis: null,
            createdBy: appState.get('currentUser')?.id || null
        };
    }

    async saveDetectionToDatabase(tempDetection) {
        console.log('Saving temp detection to database:', tempDetection);
        
        if (!appState.isAuthenticated()) {
            console.log('User not authenticated, skipping database save');
            return { success: false, message: 'User not authenticated' };
        }

        try {
            console.log('Sending temp detection data to database:', tempDetection);
            const response = await api.createSighting(tempDetection);
            console.log('Database save response:', response);
            
            if (response && response.success && response.data && response.data._id) {
                console.log('Successfully saved detection with ID:', response.data._id);
                return { ...response, databaseId: response.data._id };
            }
            return response;
        } catch (error) {
            console.error('Error saving detection to database:', error);
            return { success: false, error: error.message };
        }
    }

    async triggerLLMAnalysis(detection) {
        try {
            console.log(`Triggering LLM analysis for species: ${detection.species}, confidence: ${detection.confidence}`);
            
            // Update status to show LLM processing
            const statusElement = document.querySelector('.ai-analysis h2') || document.querySelector('.ai-analysis h3');
            if (statusElement) {
                statusElement.textContent = 'Processing with LLM...';
            }
            
            const response = await api.getLLMAnalysis(detection.species, detection.confidence);
            
            console.log('LLM analysis response:', response);
            
            if (response && response.success && response.analysis) {
                detection.llmAnalysis = response.analysis;
                detection.status = 'completed';
                
                // Update in database if authenticated
                if (detection.databaseId && appState.isAuthenticated()) {
                    await this.updateDetectionInDatabase(detection);
                }
                
                // Update display
                this.updateDetectionHistoryDisplay();
                console.log('LLM analysis completed and displayed');
            } else {
                console.log('LLM analysis failed or returned no analysis data:', response);
                detection.status = 'llm_failed';
                this.updateDetectionHistoryDisplay();
            }
            
            // Reset status to ready
            if (statusElement) {
                statusElement.textContent = 'Ready for analysis';
            }
            
        } catch (error) {
            console.error('Error triggering LLM analysis:', error);
            detection.status = 'llm_failed';
            this.updateDetectionHistoryDisplay();
            
            // Reset status to ready even on error
            const statusElement = document.querySelector('.ai-analysis h2') || document.querySelector('.ai-analysis h3');
            if (statusElement) {
                statusElement.textContent = 'Ready for analysis';
            }
        }
    }

    async updateDetectionInDatabase(detection) {
        if (!detection.databaseId || !appState.isAuthenticated()) {
            console.log('Cannot update detection in database:', {
                hasDatabaseId: !!detection.databaseId,
                isAuthenticated: appState.isAuthenticated()
            });
            return;
        }
        
        try {
            console.log('Updating detection in database:', detection.databaseId);
            const updates = {
                llmAnalysis: detection.llmAnalysis,
                'management.status': detection.status === 'completed' ? 'completed' : 'pending'
            };
            
            console.log('Updates to apply:', updates);
            const response = await api.updateSighting(detection.databaseId, updates);
            console.log('Database update response:', response);
        } catch (error) {
            console.error('Error updating detection in database:', error);
        }
    }

    notifyAdminManagers(detection) {
        try {
            // Import admin managers dynamically to avoid circular dependencies
            import('./modules/admin/map.js').then(({ mapManager }) => {
                if (mapManager && mapManager.refreshData) {
                    mapManager.refreshData();
                }
            }).catch(err => console.log('Map manager not available:', err));

            import('./modules/admin/analytics.js').then(({ analyticsManager }) => {
                if (analyticsManager && analyticsManager.refreshData) {
                    analyticsManager.refreshData();
                }
            }).catch(err => console.log('Analytics manager not available:', err));

            import('./modules/admin/species.js').then(({ speciesManager }) => {
                if (speciesManager && speciesManager.refreshData) {
                    speciesManager.refreshData();
                }
            }).catch(err => console.log('Species manager not available:', err));

            import('./modules/admin/notifications.js').then(({ notificationManager }) => {
                if (notificationManager && notificationManager.refreshData) {
                    notificationManager.refreshData();
                }
            }).catch(err => console.log('Notification manager not available:', err));

            import('./modules/admin/video.js').then(({ videoManager }) => {
                if (videoManager && videoManager.refreshData) {
                    videoManager.refreshData();
                }
            }).catch(err => console.log('Video manager not available:', err));

            console.log('Notified all admin managers of new detection');
        } catch (error) {
            console.error('Error notifying admin managers:', error);
        }
    }

    updateDetectionHistoryDisplay() {
        try {
            const history = appState.get('detectionHistoryData') || [];
            const detectionHistory = document.getElementById('detection-history');
            
            if (!detectionHistory) {
                errorHandler.handleError(new Error('Detection history element not found'), {
                    context: 'updateDetectionHistoryDisplay',
                    severity: 'warning'
                });
                return;
            }
            
            if (history.length === 0) {
                detectionHistory.innerHTML = sanitizeHtml('<p>No detections yet. Upload an image or start streaming to begin.</p>');
                return;
            }
            
            const historyHTML = history.map(detection => {
                if (!detection) return '';
                
                let statusClass = 'pending';
                let statusText = 'Pending Analysis';
                
                if (detection.status === 'completed') {
                    statusClass = 'completed';
                    statusText = 'Analysis Complete';
                } else if (detection.status === 'llm_failed') {
                    statusClass = 'failed';
                    statusText = 'LLM Analysis Failed';
                }
                
                const imageUrl = detection.frameImage || detection.imageUrl || 'placeholder.jpg';
                
                let llmContent = '';
                if (detection.llmAnalysis && typeof detection.llmAnalysis === 'object') {
                    const analysis = detection.llmAnalysis;
                    
                    // Get NEMBA category from the correct path in the response
                    const nembaCategory = analysis.advisory_content?.legal_status?.nemba_category || 'Unknown';
                    
                    // Get action required (management priority)
                    const actionRequired = analysis.action_required || 'Unknown';
                    
                    // Format invasive status
                    const invasiveStatus = analysis.invasive_status === true ? 'Invasive' : 
                                         analysis.invasive_status === false ? 'Non-Invasive' : 'Unknown';
                    
                                         llmContent = `
                         <div class="llm-analysis">
                             <h4>AI Analysis</h4>
                             <p><strong>Common Name:</strong> ${sanitizeHtml(analysis.common_name || 'Not found')}</p>
                             <p><strong>Family:</strong> ${sanitizeHtml(analysis.family || 'Not found')}</p>
                             <p><strong>Origin:</strong> ${sanitizeHtml(analysis.origin || 'Not found')}</p>
                             <p><strong>Description:</strong> ${sanitizeHtml(analysis.description || 'Not found')}</p>
                             <p><strong>Where Found:</strong> ${sanitizeHtml(analysis.where_found || 'Not found')}</p>
                             <p><strong>Treatment:</strong> ${sanitizeHtml(analysis.treatment || 'Not found')}</p>
                             <p><strong>Identification:</strong> ${sanitizeHtml(analysis.identification || 'Not found')}</p>
                             <p><strong>Poisonous:</strong> ${sanitizeHtml(analysis.poisonous || 'Not found')}</p>
                             <p><strong>Action Required:</strong> ${sanitizeHtml(analysis.action_required || 'Not found')}</p>
                         </div>
                     `;
                }
                
                // Determine status badge text and class
                let statusBadgeText = 'PENDING';
                let statusBadgeClass = 'status-pending';
                
                if (detection.status === 'completed') {
                    statusBadgeText = 'INVASIVE';
                    statusBadgeClass = 'status-invasive';
                } else if (detection.status === 'llm_failed') {
                    statusBadgeText = 'FAILED';
                    statusBadgeClass = 'status-failed';
                }
                
                                 return `
                     <div class="detection-item ${statusClass}">
                         <div class="detection-header">
                             <div class="detection-title">
                                 <h3>${sanitizeHtml(detection.species || 'Unknown Species')}</h3>
                             </div>
                             <div class="detection-badges">
                                 <div class="confidence-badge">${((detection.confidence || 0) * 100).toFixed(1)}%</div>
                                 <div class="timestamp">${this.formatTime(detection.timestamp)}</div>
                                 <div class="status-badge ${statusBadgeClass}">${statusBadgeText}</div>
                             </div>
                         </div>
                         <div class="detection-content">
                             <div class="detection-image-container">
                                 <img src="${imageUrl}" alt="Detection" onerror="this.src='placeholder.jpg'">
                                 <div class="captured-frame-label">Captured Frame</div>
                             </div>
                             <div class="detection-description">
                                 ${llmContent}
                             </div>
                         </div>
                     </div>
                 `;
            }).join('');
            
            // Fix: Don't sanitize the HTML twice - the individual elements are already sanitized
            detectionHistory.innerHTML = historyHTML;
        } catch (error) {
            errorHandler.handleError(error, {
                context: 'updateDetectionHistoryDisplay',
                severity: 'error'
            });
        }
    }

    displayUploadedImage(imageUrl) {
        try {
            const uploadArea = document.querySelector('.upload-area') || document.querySelector('.file-upload-area');
            if (!uploadArea || !imageUrl) return;
            
            // Create or update image display
            let imageDisplay = uploadArea.querySelector('.uploaded-image-display');
            if (!imageDisplay) {
                imageDisplay = document.createElement('div');
                imageDisplay.className = 'uploaded-image-display';
                imageDisplay.style.cssText = `
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.8);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 10;
                    border-radius: 8px;
                `;
                uploadArea.style.position = 'relative';
                uploadArea.appendChild(imageDisplay);
            }
            
            // Create image element
            const img = document.createElement('img');
            img.src = imageUrl;
            img.style.cssText = `
                max-width: 90%;
                max-height: 90%;
                object-fit: contain;
                border-radius: 4px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            `;
            img.onerror = () => {
                imageDisplay.innerHTML = '<p style="color: white;">Image preview not available</p>';
            };
            
            imageDisplay.innerHTML = '';
            imageDisplay.appendChild(img);
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                if (imageDisplay && imageDisplay.parentNode) {
                    imageDisplay.remove();
                }
            }, 5000);
            
        } catch (error) {
            console.error('Error displaying uploaded image:', error);
        }
    }

    formatTime(date) {
        if (!date) return 'Unknown';
        const d = new Date(date);
        return d.toLocaleTimeString('en-US', { 
            hour12: false, 
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit' 
        });
    }

    formatDate(date) {
        if (!date) return 'Unknown';
        const d = new Date(date);
        return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
    }

    switchAdminTab(tabName) {
        // Use the admin manager to handle tab switching
        adminManager.switchTab(tabName);
        appState.set('currentTab', tabName);
    }

    async startCameraHandler() {
        try {
            const startCameraBtn = document.getElementById('start-camera-btn');
            const videoStream = document.getElementById('video-stream');
            const stopCameraBtn = document.getElementById('stop-camera-btn');
            const cameraOverlay = document.querySelector('.camera-overlay');
            const statusIndicator = document.getElementById('status-indicator');
            
            startCameraBtn.disabled = true;
            startCameraBtn.textContent = 'Starting...';
            
            if (statusIndicator) {
                statusIndicator.innerHTML = '<span class="status-text">Starting camera...</span>';
            }
            
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 1280 }, 
                    height: { ideal: 720 } 
                } 
            });
            
            videoStream.srcObject = this.stream;
            await videoStream.play();
            
            // Hide the camera placeholder text
            if (cameraOverlay) {
                cameraOverlay.style.display = 'none';
            }
            
            // Update button states
            startCameraBtn.style.display = 'none';
            stopCameraBtn.style.display = 'block';
            stopCameraBtn.disabled = false;
            
            // Update status
            if (statusIndicator) {
                statusIndicator.innerHTML = '<span class="status-text">Camera active - Analyzing...</span>';
            }
            
            this.startAutomaticCapture();
            
        } catch (error) {
            this.showError('Failed to start camera: ' + error.message);
            const startCameraBtn = document.getElementById('start-camera-btn');
            startCameraBtn.disabled = false;
            startCameraBtn.textContent = 'Start Camera';
            
            const statusIndicator = document.getElementById('status-indicator');
            if (statusIndicator) {
                statusIndicator.innerHTML = '<span class="status-text">Camera failed to start</span>';
            }
        }
    }

    startAutomaticCapture() {
        this.captureInterval = setInterval(() => {
            this.captureAndAnalyzeFrame();
        }, 5000); // Capture every 5 seconds
    }

    stopStreaming() {
        if (this.captureInterval) {
            clearInterval(this.captureInterval);
            this.captureInterval = null;
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        const videoStream = document.getElementById('video-stream');
        const startCameraBtn = document.getElementById('start-camera-btn');
        const stopCameraBtn = document.getElementById('stop-camera-btn');
        const cameraOverlay = document.querySelector('.camera-overlay');
        const statusIndicator = document.getElementById('status-indicator');
        
        if (videoStream) {
            videoStream.srcObject = null;
        }
        
        // Show camera placeholder text again
        if (cameraOverlay) {
            cameraOverlay.style.display = 'block';
        }
        
        // Update button states
        startCameraBtn.style.display = 'block';
        startCameraBtn.disabled = false;
        startCameraBtn.textContent = 'Start Camera';
        stopCameraBtn.style.display = 'none';
        stopCameraBtn.disabled = true;
        
        // Update status
        if (statusIndicator) {
            statusIndicator.innerHTML = '<span class="status-text">Ready to start</span>';
        }
    }

    async captureAndAnalyzeFrame() {
        if (!this.stream) return;
        
        try {
            const canvas = document.getElementById('canvas-capture');
            const video = document.getElementById('video-stream');
            
            if (!canvas || !video) {
                throw new Error('Canvas or video element not found');
            }
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);
            
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // TODO: Replace with actual ML analysis
            // For now, create a placeholder result for streaming
            const result = {
                predicted_species: 'Streaming Detection (Analysis Pending)',
                confidence: 0.85,
                imageUrl: imageData,
                frameImage: imageData,
                isVideo: true
            };
            
            await this.addDetectionToHistory(result);
            
        } catch (error) {
            errorHandler.handleError(error, {
                context: 'captureAndAnalyzeFrame',
                severity: 'error'
            });
        }
    }

    initializeMap() {
        if (this.map) return; // Already initialized
        
        try {
            this.map = L.map('map').setView([-25.8408448, 28.2394624], 13);
            
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(this.map);
            
            // Invalidate size after a delay to ensure proper rendering
            setTimeout(() => {
                this.map.invalidateSize();
            }, 500);
            
        } catch (error) {
            console.error('Error initializing map:', error);
        }
    }



    startWeatherChecking() {
        // Get current location and start weather monitoring
        this.getCurrentLocation().then(() => {
            this.updateWeatherNotification();
            this.weatherInterval = setInterval(() => {
                this.updateWeatherNotification();
            }, CONFIG.WEATHER_CHECK_INTERVAL);
        });
    }

    async getCurrentLocation() {
        try {
            const position = await new Promise((resolve, reject) => {
                navigator.geolocation.getCurrentPosition(resolve, reject, {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 60000
                });
            });
            
            const location = {
                lat: position.coords.latitude,
                lng: position.coords.longitude
            };
            
            appState.set('currentLocation', location);
            // Location obtained successfully
            
        } catch (error) {
            // Failed to get location
        }
    }

    async updateWeatherNotification() {
        const location = appState.get('currentLocation');
        if (!location) return;
        
        try {
            const response = await fetch(
                `https://api.open-meteo.com/v1/forecast?latitude=${location.lat}&longitude=${location.lng}&current=temperature_2m,weather_code&timezone=auto`
            );
            
            if (!response.ok) {
                throw new Error(`Weather API error: ${response.status}`);
            }
            
            const data = await response.json();
            const weather = data.current;
            
            const description = this.getWeatherDescription(weather.weather_code);
            const temperature = weather.temperature_2m;
            
            appState.set('weatherData', { description, temperature });
            
            // Update weather notification in UI
            const weatherNotification = document.querySelector('.weather-notification');
            if (weatherNotification) {
                const weatherHTML = `
                    <div class="weather-card">
                        <h3>Current Weather</h3>
                        <p><strong>Temperature:</strong> ${temperature}°C</p>
                        <p><strong>Conditions:</strong> ${description}</p>
                        <p><strong>Location:</strong> ${location.lat.toFixed(4)}, ${location.lng.toFixed(4)}</p>
                    </div>
                `;
                weatherNotification.innerHTML = sanitizeHtml(weatherHTML);
            }
            
        } catch (error) {
            console.error('Error updating weather notification:', error);
        }
    }

    getWeatherDescription(code) {
        const weatherCodes = {
            0: 'Clear sky',
            1: 'Mainly clear',
            2: 'Partly cloudy',
            3: 'Overcast',
            45: 'Foggy',
            48: 'Depositing rime fog',
            51: 'Light drizzle',
            53: 'Moderate drizzle',
            55: 'Dense drizzle',
            61: 'Slight rain',
            63: 'Moderate rain',
            65: 'Heavy rain',
            71: 'Slight snow',
            73: 'Moderate snow',
            75: 'Heavy snow',
            95: 'Thunderstorm'
        };
        
        return weatherCodes[code] || 'Unknown';
    }

    showError(message) {
        console.error(message);
        // Could implement a proper error notification system
    }

    // Authentication methods
    showLoginModal() {
        const loginModal = document.getElementById('login-modal');
        if (loginModal) {
            loginModal.style.display = 'flex';
            loginModal.classList.add('active');
        }
    }

    closeLoginModal() {
        const loginModal = document.getElementById('login-modal');
        if (loginModal) {
            loginModal.classList.remove('active');
            loginModal.style.display = 'none';
            const form = document.getElementById('login-form');
            if (form) form.reset();
            const error = document.getElementById('login-error');
            if (error) error.textContent = '';
        }
    }

    showSignupModal() {
        const signupModal = document.getElementById('signup-modal');
        if (signupModal) {
            signupModal.style.display = 'flex';
            signupModal.classList.add('active');
        }
    }

    closeSignupModal() {
        const signupModal = document.getElementById('signup-modal');
        if (signupModal) {
            signupModal.classList.remove('active');
            signupModal.style.display = 'none';
            const form = document.getElementById('signup-form');
            if (form) form.reset();
            const error = document.getElementById('signup-error');
            if (error) error.textContent = '';
        }
    }

    async handleLogin(e) {
        e.preventDefault();
        
        const username = document.getElementById('login-username').value;
        const password = document.getElementById('login-password').value;
        const errorDiv = document.getElementById('login-error');
        
        if (!username || !password) {
            if (errorDiv) errorDiv.textContent = 'Username and password are required';
            return;
        }
        
        try {
            const response = await authManager.login(username, password);
            
            if (response.success) {
                this.closeLoginModal();
                authManager.updateAuthUI();
            } else {
                if (errorDiv) errorDiv.textContent = response.message || 'Login failed';
            }
        } catch (error) {
            if (errorDiv) errorDiv.textContent = 'Login error occurred';
            console.error('Login error:', error);
        }
    }

    async handleSignup(e) {
        e.preventDefault();
        
        const username = document.getElementById('signup-username').value;
        const email = document.getElementById('signup-email').value;
        const password = document.getElementById('signup-password').value;
        const confirmPassword = document.getElementById('signup-confirm-password').value;
        const errorDiv = document.getElementById('signup-error');
        
        if (!username || !email || !password || !confirmPassword) {
            if (errorDiv) errorDiv.textContent = 'All fields are required';
            return;
        }
        
        if (password !== confirmPassword) {
            if (errorDiv) errorDiv.textContent = 'Passwords do not match';
            return;
        }
        
        try {
            const response = await authManager.register(username, email, password, confirmPassword);
            
            if (response.success) {
                this.closeSignupModal();
                authManager.updateAuthUI();
            } else {
                if (errorDiv) errorDiv.textContent = response.message || 'Registration failed';
            }
        } catch (error) {
            if (errorDiv) errorDiv.textContent = 'Registration error occurred';
            console.error('Registration error:', error);
        }
    }

    async handleLogout() {
        try {
            await authManager.logout();
            authManager.updateAuthUI();
        } catch (error) {
            console.error('Logout error:', error);
        }
    }

    handleAdminToggle(e) {
        console.log('Admin toggle triggered:', e.target.checked);
        const isAdminMode = e.target.checked;
        
        if (isAdminMode && !authManager.isAuthenticated()) {
            console.log('User not authenticated, showing login modal');
            // If trying to access admin without being logged in, show login modal
            e.target.checked = false; // Revert the toggle
            this.showLoginModal();
            return;
        }
        
        // Switch between classify and admin sections
        const classifySection = document.getElementById('classify-section');
        const adminSection = document.getElementById('admin-section');
        
        console.log('Found sections:', {
            classifySection: !!classifySection,
            adminSection: !!adminSection
        });
        
        if (classifySection && adminSection) {
            if (isAdminMode) {
                console.log('Switching to admin mode');
                classifySection.style.display = 'none';
                adminSection.style.display = 'block';
            } else {
                console.log('Switching to classify mode');
                classifySection.style.display = 'block';
                adminSection.style.display = 'none';
            }
        } else {
            console.error('Required sections not found');
        }
    }

    cleanup() {
        if (this.captureInterval) {
            clearInterval(this.captureInterval);
        }
        
        if (this.weatherInterval) {
            clearInterval(this.weatherInterval);
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        
        if (this.map) {
            this.map.remove();
        }
    }
}

// Create and initialize the application
const app = new PlantRecognitionApp();

// Initialize app when DOM is ready - only once
document.addEventListener('DOMContentLoaded', () => {
    app.initialize().catch(error => {
        console.error('Failed to initialize application:', error);
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    app.cleanup();
});

// Application initialized successfully
// Note: Removed global window pollution for security 