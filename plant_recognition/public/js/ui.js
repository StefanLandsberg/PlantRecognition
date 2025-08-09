import { sanitizeHtml } from './sanitize.js';

export function addDetectionCard(container, det) {
  const el = document.createElement('div');
  el.className = 'card';
  el.id = `det-${det.sightingId}`;
  el.innerHTML = `
    <div class="badges">
      <div>Species: ${sanitizeHtml(det.predictedSpecies || 'Unknown')}</div>
      <div>Conf: ${(det.confidence*100).toFixed(1)}%</div>
    </div>
    ${det.imageUrl ? `<img src="${det.imageUrl}" alt="detection" />` : ''}
    <div class="llm">LLM: <span class="llm-status">pending</span></div>
  `;
  container.prepend(el);
}

export function setLLMCompleted(sightingId, llm) {
  const el = document.getElementById(`det-${sightingId}`);
  if (!el) return;
  const span = el.querySelector('.llm-status');
  if (span) span.textContent = 'completed';
}
