const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
require('dotenv').config({ path: path.join(__dirname, '..', 'env.example') });

const app = express();
// Configuration
const PORT = process.env.PORT || 3000;
const DB_API_BASE = process.env.DB_API_BASE || 'http://localhost:3001';
const CONFIDENCE_THRESHOLD = process.env.CONFIDENCE_THRESHOLD || 0.8;



// LLM Service Management - Clean Implementation
let llmProcess = null;
let llmReady = false;
let llmInitializing = false;
let llmProcessId = null;
let pendingRequests = new Map();
let processedRequests = new Map();
let resolveFunctions = new Map();

// LLM Service Health Check
function isLLMProcessHealthy() {
    return llmProcess && !llmProcess.killed && llmProcess.pid && llmReady;
}

// LLM Service Lifecycle Management
function initializeLLMService() {
    if (llmInitializing || llmProcess) {
        console.log('LLM service already initializing or running, skipping...');
        return;
    }

    try {
        console.log('Initializing LLM service...');
        llmInitializing = true;
        
        const llmPath = path.join(__dirname, '..', 'backend', 'llm_integration.py');
        
        // Spawn LLM process
        llmProcess = spawn('python', [llmPath, 'server']);
        llmProcessId = llmProcess.pid;
        
        console.log(`LLM process started with PID: ${llmProcessId}`);
        
        let output = '';
        let errorOutput = '';

        // Handle stdout (LLM responses)
        llmProcess.stdout.on('data', (data) => {
            const dataStr = data.toString();
            console.log('LLM stdout:', dataStr);
            output += dataStr;
            
            // Check for complete JSON responses
            const lines = output.split('\n');
            output = lines.pop() || ''; // Keep incomplete line
            
            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const response = JSON.parse(line.trim());
                        console.log('Parsed LLM response:', response);
                        handleLLMResponse(response);
                    } catch (e) {
                        console.log('Non-JSON LLM output:', line.trim());
                    }
                }
            }
        });

        // Handle stderr (LLM status messages)
        llmProcess.stderr.on('data', (data) => {
            const dataStr = data.toString();
            console.log('LLM stderr:', dataStr);
            errorOutput += dataStr;
        });
        
        // Handle process events
        llmProcess.on('close', (code) => {
            console.log(`LLM process closed with code: ${code}`);
            llmProcess = null;
            llmProcessId = null;
            llmReady = false;
            llmInitializing = false;
            
            // Clear all pending requests
            pendingRequests.clear();
            resolveFunctions.clear();
        });

        llmProcess.on('error', (error) => {
            console.error('LLM process error:', error);
            llmProcess = null;
            llmProcessId = null;
            llmReady = false;
            llmInitializing = false;
            
            // Clear all pending requests
            pendingRequests.clear();
            resolveFunctions.clear();
        });

        // Wait for initialization
        setTimeout(() => {
            llmReady = true;
            llmInitializing = false;
            console.log('LLM service ready - waiting for input');
        }, process.env.LLM_INIT_TIMEOUT || 3000);

    } catch (error) {
        console.error('Error initializing LLM service:', error);
        llmReady = false;
        llmProcess = null;
        llmProcessId = null;
        llmInitializing = false;
    }
}

// LLM Response Handler
async function handleLLMResponse(response) {
    if (response.success && response.analysis) {
        console.log('LLM analysis completed:', response.analysis.species);
        
        const confidence = response.analysis.confidence_score || response.analysis.confidence;
        const key = `${response.analysis.species}_${confidence}`;
        const resolveFn = resolveFunctions.get(key);
        
        if (resolveFn) {
            // Cache the result
            processedRequests.set(key, {
                result: response.analysis,
                timestamp: Date.now()
            });
            
            // Clean up old cached entries (older than 5 minutes)
            const now = Date.now();
            for (const [cacheKey, cacheEntry] of processedRequests.entries()) {
                if (now - cacheEntry.timestamp > 300000) {
                    processedRequests.delete(cacheKey);
                }
            }
            
            resolveFn(response.analysis);
            resolveFunctions.delete(key);
            pendingRequests.delete(key);
            console.log('Resolved pending request for:', response.analysis.species);
        }
    } else if (response.error) {
        console.error('LLM error:', response.error);
        
        // Resolve all pending requests with error
        resolveFunctions.forEach((resolveFn) => {
            resolveFn(null);
        });
        resolveFunctions.clear();
        pendingRequests.clear();
    }
}

// Function to save LLM analysis to database
async function saveLLMAnalysisToDatabase(llmAnalysis) {
    try {
        console.log('Saving LLM analysis to database for species:', llmAnalysis.species);
        
        // Only save if we have a valid sighting ID
        if (!llmAnalysis.sightingId) {
            console.log('No sighting ID provided - LLM analysis not saved to database');
            return;
        }
        
        // Find the detection in database and update it with LLM analysis
        const updateData = {
            llmAnalysis: llmAnalysis,
            'management.llmAnalysis': llmAnalysis
        };
        
        const dbResponse = await fetch(`${DB_API_BASE}/api/sightings/${llmAnalysis.sightingId}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(updateData)
        });
        
        if (dbResponse.ok) {
            console.log('LLM analysis saved to database successfully');
        } else {
            console.error('Failed to save LLM analysis to database:', await dbResponse.text());
        }
    } catch (error) {
        console.error('Error saving LLM analysis to database:', error);
    }
}

// Manual trigger function for testing (now just returns status)
function triggerLLMProcessing() {
    console.log('LLM processing is now automatic - no manual trigger needed');
    return {
        success: true,
        message: 'LLM processing is automatic',
        isProcessing: pendingRequests.size > 0
    };
}

// LLM Processing Function
async function processWithLLM(species, confidence, imagePath) {
    if (!llmProcess || llmProcess.killed || !llmProcess.pid) {
        console.log('LLM process not available, initializing...');
        initializeLLMService();
        
        // Wait for LLM to be ready
        let attempts = 0;
        while (!llmReady && attempts < 10) {
            await new Promise(resolve => setTimeout(resolve, 1000));
            attempts++;
        }
        
        if (!llmReady) {
            console.log('LLM service failed to initialize');
            return null;
        }
    }

    const requestKey = `${species}_${confidence}`;
    
    // Check if request is already pending
    if (pendingRequests.has(requestKey)) {
        console.log('Request already pending, returning existing promise');
        return pendingRequests.get(requestKey);
    }
    
    // Check if request was recently processed
    const cachedResult = processedRequests.get(requestKey);
    if (cachedResult && (Date.now() - cachedResult.timestamp) < 30000) {
        console.log('Returning cached result for:', species);
        return cachedResult.result;
    }

    console.log(`Processing LLM analysis for ${species} (confidence: ${confidence})`);
    
    // Create new promise for this request
    const requestPromise = new Promise((resolve) => {
        resolveFunctions.set(requestKey, resolve);
        
        // Set timeout for request - increased to 60 seconds for LLM processing
        setTimeout(() => {
            if (resolveFunctions.has(requestKey)) {
                console.log(`LLM request timeout for: ${species} after 60 seconds`);
                resolve(null);
                resolveFunctions.delete(requestKey);
                pendingRequests.delete(requestKey);
            }
        }, 60000); // 60 second timeout for LLM processing
    });
    
    pendingRequests.set(requestKey, requestPromise);
    
    try {
        // Check if LLM process is still healthy
        if (!isLLMProcessHealthy()) {
            console.log('LLM process is not healthy, cannot send request');
            resolveFunctions.delete(requestKey);
            pendingRequests.delete(requestKey);
            return null;
        }
        
        // Send request to LLM process (keep simple format LLM expects)
        const request = JSON.stringify({
            species: species,
            confidence: confidence,
            image_path: imagePath
        });
        
        console.log('Sent LLM request:', species);
        llmProcess.stdin.write(request + '\n');
        
        return await requestPromise;
        
    } catch (error) {
        console.error('Error sending LLM request:', error);
        resolveFunctions.delete(requestKey);
        pendingRequests.delete(requestKey);
        return null;
    }
}

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const uploadDir = path.join(__dirname, 'uploads');
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: function (req, file, cb) {
        // Use sightingId and type if provided, otherwise use timestamp
        const sightingId = req.body.sightingId || 'unknown';
        const type = req.body.type || 'detection';
        const ext = path.extname(file.originalname);
        cb(null, `${sightingId}_${type}${ext}`);
    }
});

// Configure multer for database storage (images and videos)
const databaseStorage = multer.diskStorage({
    destination: function (req, file, cb) {
        const databaseId = req.body.databaseId || 'unknown';
        const type = req.body.type || 'detection';
        const fileType = file.mimetype.startsWith('image/') ? 'images' : 'videos';
        
        const storageDir = path.join(__dirname, '..', 'database', 'storage', fileType);
        if (!fs.existsSync(storageDir)) {
            fs.mkdirSync(storageDir, { recursive: true });
        }
        cb(null, storageDir);
    },
    filename: function (req, file, cb) {
        const databaseId = req.body.databaseId || 'unknown';
        const type = req.body.type || 'detection';
        const ext = path.extname(file.originalname);
        cb(null, `${databaseId}_${type}${ext}`);
    }
});

const upload = multer({ 
    storage: storage,
    limits: {
        fileSize: 10 * 1024 * 1024 // 10MB limit
    }
});

const databaseUpload = multer({ 
    storage: databaseStorage,
    limits: {
        fileSize: 50 * 1024 * 1024 // 50MB limit for videos
    }
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Serve uploaded images
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Upload endpoint
app.post('/upload', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        const filePath = req.file.path;
        const description = req.body.description || '';
        console.log('Processing uploaded image:', filePath);
        console.log('Description:', description);

        // Call the plant recognition model
        const modelProcess = spawn('python', [
            path.join(__dirname, '..', 'backend', 'ml_model.py'),
            filePath
        ]);

        let modelOutput = '';
        let modelError = '';

        modelProcess.stdout.on('data', (data) => {
            modelOutput += data.toString();
        });

        modelProcess.stderr.on('data', (data) => {
            modelError += data.toString();
        });

        modelProcess.on('close', async (code) => {
            if (code === 0 && modelOutput.trim()) {
                try {
                    const result = JSON.parse(modelOutput.trim());
                    console.log('Model prediction:', result);
                    
                    // Store full ML result globally for LLM access
                    global.lastMLResult = result;

                    // Generate a unique filename for the stored image
                    const timestamp = Date.now();
                    const randomId = Math.floor(Math.random() * 1000000);
                    const fileName = `detection-${timestamp}-${randomId}.jpg`;
                    const storedImagePath = path.join(__dirname, 'uploads', fileName);
                    
                    // Copy the uploaded file to our storage location
                    fs.copyFileSync(filePath, storedImagePath);
                    
                    // Create a relative URL for the image
                    const imageUrl = `/uploads/${fileName}`;

                    // Send immediate response with model results
                    const response = {
                        success: true,
                        predicted_species: result.predicted_species,
                        confidence: result.confidence,
                        top5_predictions: result.top5_predictions || [],
                        processing_time: result.processing_time || 0,
                        llmProcessing: false, // LLM processing handled by frontend
                        frameImage: imageUrl, // Send image URL instead of base64
                        description: description, // Include description in response
                        imageUrl: imageUrl // Add explicit imageUrl for frontend display
                    };
                    
                    // Don't save to database here - let the frontend handle it
                    console.log('Upload processed successfully - database saving handled by frontend');

                    res.json(response);

                } catch (parseError) {
                    console.error('Error parsing model output:', parseError);
                    console.error('Raw output:', modelOutput);
                    res.status(500).json({ error: 'Error processing model output' });
                }
            } else {
                console.error('Model process failed:', modelError);
                res.status(500).json({ error: 'Error running plant recognition model' });
            }
        });

        modelProcess.on('error', (error) => {
            console.error('Model process error:', error);
            res.status(500).json({ error: 'Error starting plant recognition model' });
        });

    } catch (error) {
        console.error('Upload error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Clear pending requests endpoint (for testing)
app.post('/clear-llm-queue', (req, res) => {
    pendingRequests.clear();
    resolveFunctions.clear();
    console.log('Pending requests cleared manually');
    res.json({ success: true, message: 'Pending requests cleared' });
});

// Manual trigger LLM processing endpoint (now just returns status)
app.post('/trigger-llm-processing', (req, res) => {
    const result = triggerLLMProcessing();
    res.json(result);
});

// LLM Status endpoint
app.get('/llm-status', (req, res) => {
    res.json({
        processing: pendingRequests.size > 0,
        ready: llmReady,
        queueLength: pendingRequests.size,
        serviceActive: llmProcess !== null
    });
});

// Get LLM analysis for a specific species (for background processing results)
app.get('/llm-analysis/:species', async (req, res) => {
    try {
        const { species } = req.params;
        const { confidence } = req.query;
        
        console.log(`LLM analysis request for species: ${species}, confidence: ${confidence}`);
        
        if (!confidence) {
            console.log('Missing confidence parameter');
            return res.status(400).json({ error: 'Confidence parameter required' });
        }

        console.log('Calling processWithLLM...');
        const llmAnalysis = await processWithLLM(species, parseFloat(confidence), null);
        console.log('processWithLLM result:', llmAnalysis);
        
        if (llmAnalysis) {
            console.log('LLM analysis successful, returning:', llmAnalysis);
            res.json({ success: true, analysis: llmAnalysis });
        } else {
            console.log('LLM analysis failed or not available');
            res.json({ success: false, message: 'LLM processing failed or not available' });
        }
    } catch (error) {
        console.error('LLM analysis endpoint error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Storage API endpoints
app.use('/api/storage', express.static(path.join(__dirname, '..', 'database', 'storage')));

// Save image to database storage endpoint
app.post('/api/storage/save-image', databaseUpload.single('image'), (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        const databaseId = req.body.databaseId || 'unknown';
        const type = req.body.type || 'detection';
        const localPath = `/api/storage/images/${databaseId}_${type}.jpg`;
        
        console.log('Image saved to database storage:', localPath);
        
        res.json({ 
            success: true, 
            localPath: localPath,
            filename: req.file.filename 
        });
    } catch (error) {
        console.error('Error saving image to database:', error);
        res.status(500).json({ error: 'Failed to save image' });
    }
});

// Save video to database storage endpoint
app.post('/api/storage/save-video', databaseUpload.single('video'), (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No video file provided' });
        }

        const databaseId = req.body.databaseId || 'unknown';
        const type = req.body.type || 'streaming';
        const localPath = `/api/storage/videos/${databaseId}_${type}.mp4`;
        
        console.log('Video saved to database storage:', localPath);
        
        res.json({ 
            success: true, 
            localPath: localPath,
            filename: req.file.filename 
        });
    } catch (error) {
        console.error('Error saving video to database:', error);
        res.status(500).json({ error: 'Failed to save video' });
    }
});

// Delete file from database storage endpoint
app.delete('/api/storage/delete', (req, res) => {
    try {
        const { databaseId, type, fileType } = req.body;
        
        if (!databaseId || !type || !fileType) {
            return res.status(400).json({ error: 'Missing required parameters' });
        }

        const storageDir = path.join(__dirname, '..', 'database', 'storage', fileType === 'image' ? 'images' : 'videos');
        const filename = `${databaseId}_${type}.${fileType === 'image' ? 'jpg' : 'mp4'}`;
        const filePath = path.join(storageDir, filename);

        if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
            console.log('File deleted from database storage:', filePath);
            res.json({ success: true, message: 'File deleted successfully' });
        } else {
            res.status(404).json({ error: 'File not found' });
        }
    } catch (error) {
        console.error('Error deleting file from database:', error);
        res.status(500).json({ error: 'Failed to delete file' });
    }
});

// Legacy endpoints for backward compatibility
app.post('/api/save-image', upload.single('image'), (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        const localPath = `/uploads/${req.file.filename}`;
        console.log('Image saved locally:', localPath);
        
        res.json({ 
            success: true, 
            localPath: localPath,
            filename: req.file.filename 
        });
    } catch (error) {
        console.error('Error saving image:', error);
        res.status(500).json({ error: 'Failed to save image' });
    }
});

app.post('/api/save-video', upload.single('video'), (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No video file provided' });
        }

        const localPath = `/uploads/${req.file.filename}`;
        console.log('Video saved locally:', localPath);
        
        res.json({ 
            success: true, 
            localPath: localPath,
            filename: req.file.filename 
        });
    } catch (error) {
        console.error('Error saving video:', error);
        res.status(500).json({ error: 'Failed to save video' });
    }
});

// Start server
app.listen(PORT, () => {
    console.log(`Plant Recognition Frontend server running on port ${PORT}`);
    console.log(`Web interface: http://localhost:${PORT}`);
    console.log(`Database API: ${DB_API_BASE}`);
    console.log(`LLM Integration: Initializing persistent service...`);
    
    // Initialize LLM service immediately when server starts
    initializeLLMService();
});