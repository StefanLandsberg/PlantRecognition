// Video Review Tab Manager
import { api } from '../api.js';
import { sanitizeHtml } from '../../utils/sanitize.js';
import { handleError } from '../utils.js';
import { imageStorage } from '../imageStorage.js';

class VideoManager {
    constructor() {
        this.sightingsData = [];
        this.currentFilters = {};
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        try {
            console.log('Initializing Video Manager...');
            await this.loadSightingsData();
            this.initialized = true;
            console.log('Video Manager initialized successfully');
        } catch (error) {
            console.error('Error initializing Video Manager:', error);
            handleError(error, 'video_initialization');
        }
    }

    async loadSightingsData() {
        try {
            const response = await api.getSightings();
            if (response.success && response.data) {
                this.sightingsData = response.data;
                console.log(`Loaded ${this.sightingsData.length} sightings for video review`);
            }
        } catch (error) {
            console.error('Error loading sightings data for video review:', error);
        }
    }

    async refreshData() {
        await this.loadSightingsData();
    }

    onTabActivated() {
        this.updateVideoReview();
    }

    updateVideoReview() {
        const videoGrid = document.getElementById('video-grid');
        if (!videoGrid) return;
        
        const videoSightings = this.getVideoSightings();
        
        if (videoSightings.length === 0) {
            videoGrid.innerHTML = `
                <div class="video-empty">
                    <i class="fas fa-video"></i>
                    <h3>No Videos Available</h3>
                    <p>Upload video files or use camera streaming to see videos here</p>
                </div>
            `;
            return;
        }
        
        const videoHTML = videoSightings.map(sighting => this.createVideoItem(sighting)).join('');
        videoGrid.innerHTML = videoHTML;
        
        // Setup video player functionality
        this.setupVideoPlayers();
    }

    getVideoSightings() {
        // Filter sightings that have video content
        return this.sightingsData.filter(s => 
            s.isVideo || s.videoFile || (s.imageUrl && s.imageUrl.includes('video'))
        );
    }

    createVideoItem(sighting) {
        const status = sighting.llmAnalysis?.invasive_status === true ? 'Invasive' : 
                      sighting.llmAnalysis?.invasive_status === false ? 'Native' : 'Unknown';
        const riskLevel = sighting.llmAnalysis?.risk_level || 'Unknown';
        const managementStatus = sighting.management?.status || 'pending';
        
        // Get local video URL if available
        let videoUrl = sighting.videoFile || sighting.imageUrl;
        let imageUrl = sighting.imageUrl || 'placeholder.jpg';
        
        // Try to get local video URL if we have a database ID
        if (sighting._id) {
            const localVideoUrl = imageStorage.getLocalVideoUrl(sighting._id, 'streaming');
            const localImageUrl = imageStorage.getLocalImageUrl(sighting._id, 'detection');
            
            // Check if local files exist
            imageStorage.fileExists(sighting._id, 'streaming', 'video').then(exists => {
                if (exists) {
                    videoUrl = localVideoUrl;
                }
            });
            
            imageStorage.fileExists(sighting._id, 'detection', 'image').then(exists => {
                if (exists) {
                    imageUrl = localImageUrl;
                }
            });
        }
        
        return `
            <div class="video-item" data-sighting-id="${sighting._id}">
                <div class="video-preview">
                    <img src="${imageUrl}" alt="Video Preview" class="video-thumbnail">
                    <div class="video-overlay">
                        <button class="play-btn" data-video="${videoUrl}">
                            <i class="fas fa-play"></i>
                        </button>
                    </div>
                    <div class="video-duration">
                        <span class="duration-badge">00:30</span>
                    </div>
                </div>
                <div class="video-info">
                    <div class="video-header">
                        <h4>${sanitizeHtml(sighting.species)}</h4>
                        <div class="video-badges">
                            <span class="status-badge ${status.toLowerCase()}">${status}</span>
                            <span class="risk-badge ${riskLevel.toLowerCase()}">${riskLevel}</span>
                        </div>
                    </div>
                    <div class="video-details">
                        <p><i class="fas fa-calendar"></i> ${this.formatDate(sighting.timestamp)}</p>
                        <p><i class="fas fa-map-marker-alt"></i> ${this.formatLocation(sighting.latitude, sighting.longitude)}</p>
                        <p><i class="fas fa-percentage"></i> Confidence: ${(sighting.confidence * 100).toFixed(1)}%</p>
                        <p><i class="fas fa-tasks"></i> Status: ${managementStatus}</p>
                    </div>
                    <div class="video-actions">
                        <button class="video-action-btn review-btn" onclick="videoManager.reviewVideo('${sighting._id}')">
                            <i class="fas fa-eye"></i> Review
                        </button>
                        <button class="video-action-btn validate-btn" onclick="videoManager.validateDetection('${sighting._id}')">
                            <i class="fas fa-check"></i> Validate
                        </button>
                        <button class="video-action-btn reject-btn" onclick="videoManager.rejectDetection('${sighting._id}')">
                            <i class="fas fa-times"></i> Reject
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    setupVideoPlayers() {
        // Setup play button functionality
        document.querySelectorAll('.play-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const videoUrl = btn.getAttribute('data-video');
                this.playVideo(videoUrl, btn.closest('.video-item'));
            });
        });

        // Setup filter functionality
        this.setupVideoFilters();
    }

    setupVideoFilters() {
        const statusFilter = document.getElementById('video-status-filter');
        const speciesFilter = document.getElementById('video-species-filter');
        const dateFilter = document.getElementById('video-date-filter');

        if (statusFilter) {
            statusFilter.addEventListener('change', () => this.applyVideoFilters());
        }
        if (speciesFilter) {
            speciesFilter.addEventListener('change', () => this.applyVideoFilters());
        }
        if (dateFilter) {
            dateFilter.addEventListener('change', () => this.applyVideoFilters());
        }

        // Clear filters button
        const clearVideoFiltersBtn = document.getElementById('clear-video-filters');
        if (clearVideoFiltersBtn) {
            clearVideoFiltersBtn.addEventListener('click', () => this.clearVideoFilters());
        }
    }

    applyVideoFilters() {
        this.currentFilters = {
            status: document.getElementById('video-status-filter')?.value,
            species: document.getElementById('video-species-filter')?.value,
            date: document.getElementById('video-date-filter')?.value
        };

        this.updateVideoReview();
    }

    clearVideoFilters() {
        const statusFilter = document.getElementById('video-status-filter');
        const speciesFilter = document.getElementById('video-species-filter');
        const dateFilter = document.getElementById('video-date-filter');

        if (statusFilter) statusFilter.value = '';
        if (speciesFilter) speciesFilter.value = '';
        if (dateFilter) dateFilter.value = '';

        this.currentFilters = {};
        this.updateVideoReview();
    }

    playVideo(videoUrl, videoItem) {
        // Create video player modal
        const modalHTML = `
            <div class="modal fade" id="videoPlayerModal" tabindex="-1">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Video Review</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="video-player-container">
                                <video id="videoPlayer" controls class="video-player">
                                    <source src="${videoUrl}" type="video/mp4">
                                    Your browser does not support the video tag.
                                </video>
                            </div>
                            <div class="video-controls mt-3">
                                <button class="btn btn-success" onclick="videoManager.validateCurrentVideo()">
                                    <i class="fas fa-check"></i> Validate Detection
                                </button>
                                <button class="btn btn-danger" onclick="videoManager.rejectCurrentVideo()">
                                    <i class="fas fa-times"></i> Reject Detection
                                </button>
                                <button class="btn btn-secondary" onclick="videoManager.flagForReview()">
                                    <i class="fas fa-flag"></i> Flag for Review
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal if any
        const existingModal = document.getElementById('videoPlayerModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add modal to page
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('videoPlayerModal'));
        modal.show();

        // Store current video item for actions
        this.currentVideoItem = videoItem;
    }

    async reviewVideo(sightingId) {
        const sighting = this.sightingsData.find(s => s._id === sightingId);
        if (!sighting) {
            alert('Sighting not found');
            return;
        }

        // Show detailed review modal
        this.showReviewModal(sighting);
    }

    showReviewModal(sighting) {
        const modalHTML = `
            <div class="modal fade" id="reviewModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Video Review - ${sighting.species}</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Detection Information</h6>
                                    <p><strong>Species:</strong> ${sanitizeHtml(sighting.species)}</p>
                                    <p><strong>Date:</strong> ${this.formatDate(sighting.timestamp)}</p>
                                    <p><strong>Location:</strong> ${this.formatLocation(sighting.latitude, sighting.longitude)}</p>
                                    <p><strong>Confidence:</strong> ${(sighting.confidence * 100).toFixed(1)}%</p>
                                    <p><strong>Status:</strong> ${sighting.llmAnalysis?.invasive_status === true ? 'Invasive' : 'Native'}</p>
                                    <p><strong>Risk Level:</strong> ${sighting.llmAnalysis?.risk_level || 'Unknown'}</p>
                                </div>
                                <div class="col-md-6">
                                    <h6>Review Actions</h6>
                                    <div class="review-actions">
                                        <button class="btn btn-success w-100 mb-2" onclick="videoManager.validateDetection('${sighting._id}')">
                                            <i class="fas fa-check"></i> Validate Detection
                                        </button>
                                        <button class="btn btn-danger w-100 mb-2" onclick="videoManager.rejectDetection('${sighting._id}')">
                                            <i class="fas fa-times"></i> Reject Detection
                                        </button>
                                        <button class="btn btn-warning w-100 mb-2" onclick="videoManager.flagForReview('${sighting._id}')">
                                            <i class="fas fa-flag"></i> Flag for Manual Review
                                        </button>
                                        <button class="btn btn-info w-100" onclick="videoManager.addNotes('${sighting._id}')">
                                            <i class="fas fa-edit"></i> Add Notes
                                        </button>
                                    </div>
                                </div>
                            </div>
                            ${sighting.llmAnalysis?.description ? `
                                <div class="mt-3">
                                    <h6>LLM Analysis</h6>
                                    <p>${sanitizeHtml(sighting.llmAnalysis.description)}</p>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal if any
        const existingModal = document.getElementById('reviewModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add modal to page
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        
        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('reviewModal'));
        modal.show();
    }

    async validateDetection(sightingId) {
        try {
            const response = await api.updateSighting(sightingId, {
                'management.status': 'validated',
                'management.validatedAt': new Date().toISOString(),
                'management.validatedBy': 'admin'
            });

            if (response.success) {
                alert('Detection validated successfully');
                this.refreshData();
                this.updateVideoReview();
            } else {
                alert('Failed to validate detection');
            }
        } catch (error) {
            console.error('Error validating detection:', error);
            alert('Error validating detection');
        }
    }

    async rejectDetection(sightingId) {
        try {
            const response = await api.updateSighting(sightingId, {
                'management.status': 'rejected',
                'management.rejectedAt': new Date().toISOString(),
                'management.rejectedBy': 'admin'
            });

            if (response.success) {
                alert('Detection rejected successfully');
                this.refreshData();
                this.updateVideoReview();
            } else {
                alert('Failed to reject detection');
            }
        } catch (error) {
            console.error('Error rejecting detection:', error);
            alert('Error rejecting detection');
        }
    }

    async flagForReview(sightingId) {
        try {
            const response = await api.updateSighting(sightingId, {
                'management.status': 'flagged',
                'management.flaggedAt': new Date().toISOString(),
                'management.flaggedBy': 'admin'
            });

            if (response.success) {
                alert('Detection flagged for manual review');
                this.refreshData();
                this.updateVideoReview();
            } else {
                alert('Failed to flag detection');
            }
        } catch (error) {
            console.error('Error flagging detection:', error);
            alert('Error flagging detection');
        }
    }

    addNotes(sightingId) {
        const notes = prompt('Enter notes for this detection:');
        if (notes !== null) {
            this.saveNotes(sightingId, notes);
        }
    }

    async saveNotes(sightingId, notes) {
        try {
            const response = await api.updateSighting(sightingId, {
                'management.notes': notes,
                'management.notesUpdatedAt': new Date().toISOString()
            });

            if (response.success) {
                alert('Notes saved successfully');
            } else {
                alert('Failed to save notes');
            }
        } catch (error) {
            console.error('Error saving notes:', error);
            alert('Error saving notes');
        }
    }

    validateCurrentVideo() {
        if (this.currentVideoItem) {
            const sightingId = this.currentVideoItem.getAttribute('data-sighting-id');
            this.validateDetection(sightingId);
        }
    }

    rejectCurrentVideo() {
        if (this.currentVideoItem) {
            const sightingId = this.currentVideoItem.getAttribute('data-sighting-id');
            this.rejectDetection(sightingId);
        }
    }

    flagForReview() {
        if (this.currentVideoItem) {
            const sightingId = this.currentVideoItem.getAttribute('data-sighting-id');
            this.flagForReview(sightingId);
        }
    }

    formatDate(dateString) {
        if (!dateString) return 'Unknown';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    }

    formatLocation(lat, lng) {
        if (!lat || !lng) return 'Unknown';
        return `${lat.toFixed(4)}, ${lng.toFixed(4)}`;
    }
}

// Create singleton instance
export const videoManager = new VideoManager(); 