let stream = null, timer = null;

export async function startVideo(onFrame) {
  const video = document.getElementById('video');
  const canvas = document.getElementById('frame');
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
  video.srcObject = stream;
  await video.play();

  const ctx = canvas.getContext('2d');
  timer = setInterval(async () => {
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.85));
    onFrame(blob);
  }, 3000);
}

export function stopVideo() {
  if (timer) clearInterval(timer);
  timer = null;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
}
