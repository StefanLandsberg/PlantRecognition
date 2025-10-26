import { VideoSessionAPI } from '../api.js';

let currentSession = null;
let currentFrame = 0;

export async function loadVideoSessions() {
  try {
    const response = await VideoSessionAPI.list();
    const sessions = response.sessions || [];
    
    const container = document.getElementById('video-sessions-list');
    if (!container) return;

    if (sessions.length === 0) {
      container.innerHTML = '<div class="empty-state">No video sessions recorded yet.</div>';
      return;
    }

    container.innerHTML = sessions.map(session => `
      <div class="video-session-card" onclick="openVideoPlayer('${session._id}')">
        <div class="session-thumbnail">
          ${session.thumbnailUrl ? 
            `<img src="${session.thumbnailUrl}" alt="Session thumbnail" />` :
            session.videoUrl ? 
              `<video class="video-thumbnail" preload="metadata" muted>
                <source src="${session.videoUrl}#t=0.5" type="video/webm">
                <source src="${session.videoUrl}#t=0.5" type="video/mp4">
              </video>` :
              session.storageType === 'local' ?
                '<div class="no-thumbnail local-video"><i class="fas fa-mobile-alt"></i><span>Companion Video</span></div>' :
                '<div class="no-thumbnail"><i class="fas fa-video"></i><span>No Preview</span></div>'
          }
        </div>
        <div class="session-info">
          <div class="session-type">${session.sessionType === 'live_video' ? 'Live Video' : 'Uploaded Video'}</div>
          <div class="session-date">${new Date(session.startTime).toLocaleDateString()}</div>
          <div class="session-stats">
            <span>${session.duration || 0}s</span>
            <span>${session.detectionCount || 0} detections</span>
          </div>
        </div>
      </div>
    `).join('');
  } catch (error) {
    console.error('Failed to load video sessions:', error);
    const container = document.getElementById('video-sessions-list');
    if (container) {
      container.innerHTML = '<div class="error-state">Failed to load video sessions.</div>';
    }
  }
}

export async function openVideoPlayer(sessionId) {
  try {
    const response = await VideoSessionAPI.get(sessionId);
    currentSession = response.session;
    currentFrame = 0;

    // Set global reference for SSE updates
    window.currentVideoSessionId = sessionId;
    window.refreshVideoReview = refreshCurrentSession;

    const modal = document.getElementById('video-review-modal');
    const title = document.getElementById('video-review-title');
    
    if (!modal || !title) return;

    title.textContent = `${currentSession.sessionType === 'live_video' ? 'Live Video' : 'Uploaded Video'} - ${new Date(currentSession.startTime).toLocaleDateString()}`;
    
    renderVideoPlayer();
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden'; // Prevent background scrolling
  } catch (error) {
    console.error('Failed to load video session:', error);
    alert('Failed to load video session');
  }
}

async function refreshCurrentSession() {
  if (!window.currentVideoSessionId) return;
  
  try {
    console.log('Refreshing video session data...');
    const response = await VideoSessionAPI.get(window.currentVideoSessionId);
    currentSession = response.session;
    renderVideoPlayer(); // Re-render with updated data
  } catch (error) {
    console.error('Failed to refresh video session:', error);
  }
}

function renderVideoPlayer() {
  if (!currentSession) return;

  const videoDisplay = document.getElementById('video-display');
  const timelineContainer = document.getElementById('video-timeline-container');
  const videoInfo = document.getElementById('video-info');
  
  if (!videoDisplay || !timelineContainer) return;

  // Debug: Log all detections and their statuses
  console.log('All detections in session:', currentSession.detections);
  currentSession.detections.forEach((d, i) => {
    console.log(`Detection ${i}: status=${d.status}, timestamp=${d.timestamp}, sightingId=${d.sightingId}`);
  });

  // Filter detections for timeline (show invasive and unknown, but not duplicates)
  const visibleDetections = currentSession.detections.filter(d => d.status !== 'duplicate');
  console.log('Visible detections found:', visibleDetections.length);

  if (visibleDetections.length === 0) {
    // Show debug info about what detections we do have
    const allStatuses = currentSession.detections.map(d => d.status).join(', ');
    videoDisplay.innerHTML = `
      <div class="no-detections">
        <i class="fas fa-video"></i>
        <h3>No plant detections found in this session</h3>
        <p>Debug: Found ${currentSession.detections.length} total detections with statuses: ${allStatuses}</p>
      </div>
    `;
    timelineContainer.innerHTML = '';
    if (videoInfo) videoInfo.innerHTML = '';
    return;
  }

  // Show video player (without controls to restrict user interaction)
  const currentDetection = visibleDetections[currentFrame];
  console.log('Current session videoUrl:', currentSession.videoUrl);
  console.log('Current detection frameUrl:', currentDetection?.frameUrl);
  
  if (currentSession.videoUrl) {
    console.log('Creating video element with URL:', currentSession.videoUrl);
    videoDisplay.innerHTML = `
      <div class="video-container">
        <video id="session-video" class="video-frame" preload="metadata" muted disablePictureInPicture controlsList="nodownload nofullscreen noremoteplayback">
          <source src="${currentSession.videoUrl}" type="video/webm">
          <source src="${currentSession.videoUrl}" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <div id="detection-circle" class="detection-circle hidden"></div>
      </div>
    `;
    
    // Ensure video loads and shows first frame
    setTimeout(() => {
      const video = document.getElementById('session-video');
      if (video) {
        console.log('Video element created, loading first frame...');
        video.load(); // Force load
        video.currentTime = 0.1; // Show first frame
        console.log('Video element properties:', {
          src: video.src,
          readyState: video.readyState,
          networkState: video.networkState
        });
      }
    }, 100);
    
  } else if (currentDetection && currentDetection.frameUrl) {
    // Fallback to frame image if no video URL
    console.log('No video URL, using frame image:', currentDetection.frameUrl);
    videoDisplay.innerHTML = `
      <img src="${currentDetection.frameUrl}" alt="Detection frame" class="video-frame" />
    `;
  } else {
    console.log('No video URL or frame URL available');
    videoDisplay.innerHTML = `
      <div class="no-video">
        <i class="fas fa-video-slash"></i>
        <h3>No video available</h3>
        <p>Video URL: ${currentSession.videoUrl || 'Not available'}</p>
      </div>
    `;
  }

  // Render timeline with 2-second segment positioning (1 before + 1 after detection)
  const sessionDuration = currentSession.duration || 60;
  const timelineSegments = visibleDetections.map((detection, index) => {
    const detectionTime = detection.timestamp;
    const segmentStart = Math.max(0, detectionTime - 1); // Start 1 second before detection
    const segmentEnd = Math.min(sessionDuration, segmentStart + 2); // 2-second segment (1 before + 1 after)
    
    // Position and width represent the 2-second playback segment
    const startPosition = (segmentStart / sessionDuration) * 100;
    const segmentWidth = ((segmentEnd - segmentStart) / sessionDuration) * 100;
    
    const statusClass = detection.status === 'invasive' ? 'invasive' : detection.status === 'unknown' ? 'unknown' : 'pending';
    return `
      <div class="timeline-segment ${statusClass} ${index === currentFrame ? 'active' : ''}" 
           style="left: ${startPosition}%; width: ${segmentWidth}%" 
           onclick="jumpToDetection(${index})"
           title="2-second clip: ${formatTime(segmentStart)} - ${formatTime(segmentEnd)} (Detection: ${formatTime(detectionTime)})">
      </div>
    `;
  }).join('');

  // Add detection indicator lines
  const detectionIndicators = visibleDetections.map((detection, index) => {
    const detectionPosition = (detection.timestamp / sessionDuration) * 100;
    const statusClass = detection.status === 'invasive' ? 'invasive' : detection.status === 'unknown' ? 'unknown' : 'pending';
    return `
      <div class="detection-indicator ${statusClass} ${index === currentFrame ? 'active' : ''}" 
           style="left: ${detectionPosition}%" 
           title="Detection moment: ${formatTime(detection.timestamp)}">
      </div>
    `;
  }).join('');

  // Create time labels for the timeline
  const timeLabels = [];
  const labelCount = 6; // Show 6 time labels across the timeline
  for (let i = 0; i <= labelCount; i++) {
    const timePoint = (sessionDuration * i) / labelCount;
    timeLabels.push(`<span>${formatTime(Math.floor(timePoint))}</span>`);
  }

  timelineContainer.innerHTML = `
    <div class="video-timeline">
      <div class="timeline-track"></div>
      <div class="timeline-markers">
        ${timelineSegments}
      </div>
      <div class="detection-indicators">
        ${detectionIndicators}
      </div>
    </div>
    <div class="timeline-time-labels">
      ${timeLabels.join('')}
    </div>
    <div class="timeline-controls">
      <div class="detection-info">
        <div class="detection-details">
          <span class="detection-count">Detection ${currentFrame + 1} of ${visibleDetections.length}</span>
          <span class="detection-time">Time: ${formatTime(currentDetection.timestamp)}</span>
          <span class="detection-status">Status: ${currentDetection.status}</span>
          ${currentDetection.sightingId ? `
            <span class="detection-species">Species: ${currentDetection.sightingId.analysis?.predictedSpecies || 'Unknown'}</span>
            <span class="detection-confidence">Confidence: ${((currentDetection.sightingId.analysis?.confidence || 0) * 100).toFixed(1)}%</span>
          ` : ''}
        </div>
      </div>
      <div class="playback-info">
        <span class="frame-counter">${currentFrame + 1} / ${visibleDetections.length}</span>
        <span class="playback-instruction">${currentSession.videoUrl ? 'Click segments to play 2-second clips (1s before + 1s after detection)' : 'Click segments to view detection frames'}</span>
      </div>
    </div>
  `;

  // Update video info
  if (videoInfo) {
    const invasiveCount = visibleDetections.filter(d => d.status === 'invasive').length;
    const unknownCount = visibleDetections.filter(d => d.status === 'unknown').length;
    
    videoInfo.innerHTML = `
      <div class="video-stats">
        <div class="video-stat">
          <i class="fas fa-clock"></i>
          <span>Duration: ${formatDuration(sessionDuration)}</span>
        </div>
        <div class="video-stat">
          <i class="fas fa-eye"></i>
          <span>Detections: ${visibleDetections.length}</span>
        </div>
        <div class="video-stat">
          <i class="fas fa-exclamation-triangle"></i>
          <span>Invasive: ${invasiveCount}</span>
        </div>
        <div class="video-stat">
          <i class="fas fa-question-circle"></i>
          <span>Unknown: ${unknownCount}</span>
        </div>
      </div>
    `;
  }
}

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function formatDuration(seconds) {
  if (seconds < 60) {
    return `${seconds}s`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
  }
}

// Global functions for timeline controls
window.jumpToDetection = function(index) {
  if (!currentSession) return;
  const visibleDetections = currentSession.detections.filter(d => d.status !== 'duplicate');
  if (index >= 0 && index < visibleDetections.length) {
    currentFrame = index;
    renderVideoPlayer();
    
    // Play video segment for this detection
    playDetectionSegment(visibleDetections[index]);
  }
};

// Play video segment or show frame image for detection
function playDetectionSegment(detection) {
  console.log('playDetectionSegment called for detection:', detection);
  
  // Check if we have a video URL
  if (!currentSession?.videoUrl) {
    console.log('No video URL available, showing frame image only');
    // Just show the detection frame - no video playback needed
    showDetectionFrame(detection);
    return;
  }
  
  // Add a small delay to ensure video element is ready after renderVideoPlayer()
  setTimeout(() => {
    const video = document.getElementById('session-video');
    console.log('Video element found:', !!video, 'Video URL:', currentSession?.videoUrl);
    
    if (!video) {
      console.error('Video element not found');
      return;
    }
    
    console.log('Video readyState:', video.readyState, 'Video duration:', video.duration);
    
    // Try to load the video first
    if (video.readyState < 1) {
      console.log('Video not loaded, waiting for loadstart...');
      video.load(); // Force load
      video.addEventListener('loadedmetadata', () => {
        console.log('Video metadata loaded, starting playback');
        startPlayback(video, detection);
      }, { once: true });
      return;
    }
    
    startPlayback(video, detection);
  }, 200); // Increased delay
}

// Show detection frame when no video is available
function showDetectionFrame(detection) {
  const videoDisplay = document.getElementById('video-display');
  if (!videoDisplay) return;
  
  console.log('Showing detection frame:', detection.frameUrl);
  
  if (detection.frameUrl) {
    videoDisplay.innerHTML = `
      <div class="frame-container">
        <img src="${detection.frameUrl}" alt="Detection frame" class="video-frame" />
        <div class="frame-overlay">
          <div class="detection-info-overlay">
            <span>Detection at ${formatTime(detection.timestamp)}</span>
            <span>Status: ${detection.status}</span>
          </div>
        </div>
      </div>
    `;
  } else {
    videoDisplay.innerHTML = `
      <div class="no-frame">
        <i class="fas fa-image"></i>
        <h3>No frame available</h3>
        <p>Detection at ${formatTime(detection.timestamp)}</p>
      </div>
    `;
  }
}

function startPlayback(video, detection) {
  const detectionTime = detection.timestamp;
  const startTime = Math.max(0, detectionTime - 1); // Start 1 second before detection
  const endTime = Math.min(currentSession.duration || 60, detectionTime + 1); // End 1 second after detection
  const playDuration = (endTime - startTime) * 1000; // Convert to milliseconds (2 seconds total)
  
  console.log(`Playing 2-second video segment (1s before + 1s after): start=${startTime}s, detection=${detectionTime}s, end=${endTime}s`);
  
  // Remove any existing event listeners to prevent conflicts
  const existingRestrictPlayback = video.restrictPlayback;
  const existingPreventSeeking = video.preventSeeking;
  
  if (existingRestrictPlayback) {
    video.removeEventListener('timeupdate', existingRestrictPlayback);
  }
  if (existingPreventSeeking) {
    video.removeEventListener('seeking', existingPreventSeeking);
    video.removeEventListener('seeked', existingPreventSeeking);
  }
  
  // Set video to start time
  video.currentTime = startTime;
  
  // Add event listeners to restrict playback
  const restrictPlayback = () => {
    if (video.currentTime < startTime || video.currentTime > endTime) {
      video.pause();
      video.currentTime = startTime;
      console.log('Playback restricted to allowed segment');
    }
  };
  
  const preventSeeking = (e) => {
    e.preventDefault();
    if (video.currentTime < startTime || video.currentTime > endTime) {
      video.currentTime = startTime;
      console.log('Seeking prevented - restricted to detection segment');
    }
  };
  
  // Store references for cleanup
  video.restrictPlayback = restrictPlayback;
  video.preventSeeking = preventSeeking;
  
  video.addEventListener('timeupdate', restrictPlayback);
  video.addEventListener('seeking', preventSeeking);
  video.addEventListener('seeked', preventSeeking);
  
  // Play the video
  console.log('Attempting to play video from time:', startTime);
  
  const playPromise = video.play();
  
  if (playPromise !== undefined) {
    playPromise.then(() => {
      console.log('Restricted video playback started successfully');
    
    // Show detection circle during detection moment (0.3s before to 0.3s after)
    const detectionCircle = document.getElementById('detection-circle');
    const showCircleAt = (detectionTime - startTime - 0.3) * 1000; // Time to show circle (in ms from start)
    const hideCircleAt = (detectionTime - startTime + 0.3) * 1000; // Time to hide circle (in ms from start)
    
    if (detectionCircle && showCircleAt >= 0 && showCircleAt < playDuration) {
      setTimeout(() => {
        detectionCircle.classList.remove('hidden');
        console.log('Detection circle shown');
      }, Math.max(0, showCircleAt));
      
      setTimeout(() => {
        detectionCircle.classList.add('hidden');
        console.log('Detection circle hidden');
      }, Math.min(playDuration, hideCircleAt));
    }
    
    // Stop after the calculated duration
    setTimeout(() => {
      video.pause();
      video.removeEventListener('timeupdate', restrictPlayback);
      video.removeEventListener('seeking', preventSeeking);
      video.removeEventListener('seeked', preventSeeking);
      
      // Clear stored references
      video.restrictPlayback = null;
      video.preventSeeking = null;
      
      if (detectionCircle) {
        detectionCircle.classList.add('hidden'); // Ensure circle is hidden
      }
      console.log('Restricted video playback stopped');
    }, playDuration);
    
    }).catch(error => {
      console.error('Failed to play video:', error);
      // Try to handle autoplay restrictions
      if (error.name === 'NotAllowedError') {
        console.log('Autoplay blocked - user interaction required');
        // You could show a play button here or handle this differently
      }
    });
  } else {
    console.error('Video play() method did not return a promise');
  }
}

window.openVideoPlayer = openVideoPlayer;

window.refreshVideoReview = async function() {
  if (currentSession && currentSession._id) {
    console.log('Refreshing video review for session:', currentSession._id);
    await openVideoPlayer(currentSession._id);
  }
};

window.closeVideoPlayer = function() {
  const modal = document.getElementById('video-review-modal');
  if (modal) {
    modal.classList.add('hidden');
    document.body.style.overflow = ''; // Restore scrolling
  }
  
  // Clean up global references
  window.currentVideoSessionId = null;
  window.refreshVideoReview = null;
  
  currentSession = null;
  currentFrame = 0;
};