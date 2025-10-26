let stream = null, timer = null, sessionStartTime = null, mediaRecorder = null, recordedChunks = [];

export async function startVideo(onFrame) {
  const video = document.getElementById('video');
  const canvas = document.getElementById('frame');
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
  video.srcObject = stream;
  await video.play();
  
  sessionStartTime = Date.now();

  // Start video recording
  startVideoRecording();

  const ctx = canvas.getContext('2d');
  timer = setInterval(async () => {
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.85));
    
    // Calculate timestamp from session start
    const timestamp = Math.floor((Date.now() - sessionStartTime) / 1000);
    
    onFrame(blob, timestamp);
  }, 3000);
}

function startVideoRecording() {
  if (!stream) return;
  
  recordedChunks = [];
  
  try {
    mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'video/webm;codecs=vp9' // Try VP9 first
    });
  } catch (e) {
    try {
      mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp8' // Fallback to VP8
      });
    } catch (e) {
      try {
        mediaRecorder = new MediaRecorder(stream, {
          mimeType: 'video/webm' // Basic WebM
        });
      } catch (e) {
        mediaRecorder = new MediaRecorder(stream); // Default
      }
    }
  }
  
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };
  
  mediaRecorder.start(1000); // Collect data every second
  console.log('Video recording started');
}

export async function stopVideo() {
  if (timer) clearInterval(timer);
  timer = null;
  
  // Stop video recording and get the recorded video
  const videoBlob = await stopVideoRecording();
  
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  sessionStartTime = null;
  
  return videoBlob;
}

function stopVideoRecording() {
  return new Promise((resolve) => {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
      resolve(null);
      return;
    }
    
    mediaRecorder.onstop = () => {
      const videoBlob = new Blob(recordedChunks, { type: 'video/webm' });
      console.log('Video recording stopped, blob size:', videoBlob.size);
      resolve(videoBlob);
    };
    
    mediaRecorder.stop();
  });
}

export function getSessionDuration() {
  return sessionStartTime ? Math.floor((Date.now() - sessionStartTime) / 1000) : 0;
}

export async function uploadVideoFile(videoBlob, sessionId) {
  if (!videoBlob || !sessionId) {
    console.log('No video blob or session ID provided');
    return null;
  }
  
  try {
    const formData = new FormData();
    formData.append('video', videoBlob, `session-${sessionId}.webm`);
    formData.append('sessionId', sessionId);
    formData.append('storageType', 'server');

    const response = await fetch('/api/video-sessions/upload-video', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    if (result.success) {
      console.log('Video uploaded to server successfully:', result.videoUrl);
      return result.videoUrl;
    } else {
      console.error('Video upload failed:', result.error);
      return null;
    }
  } catch (error) {
    console.error('Error uploading video:', error);
    return null;
  }
}
