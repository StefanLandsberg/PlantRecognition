import { sanitizeHtml } from './sanitize.js';

export function addDetectionCard(container, det) {
  const el = document.createElement('div');
  el.className = 'card';
  el.id = `det-${det.sightingId}`;
  el.innerHTML = `
    <div class="badges">
      <div class="detection-species" onclick="toggleMainDetectionLLM('${det.sightingId}')" style="cursor: pointer; color: var(--accent);">
        Species: ${sanitizeHtml(det.predictedSpecies || 'Unknown')}
      </div>
      <div>Conf: ${(det.confidence*100).toFixed(1)}%</div>
    </div>
    ${det.imageUrl ? `<img src="${det.imageUrl}" alt="detection" />` : ''}
    <div class="classification-loading" style="display: none;">
      <div class="loading-spinner classification"></div>
      <span>Classifying...</span>
    </div>
    <div class="llm-loading" style="display: none;">
      <div class="loading-spinner llm"></div>
      <span>Analyzing with AI...</span>
    </div>
    <div class="llm" style="display: none;">
      <div class="llm-header" onclick="toggleMainDetectionLLM('${det.sightingId}')" style="cursor: pointer;">
        <span>AI Analysis</span>
        <span class="llm-arrow">▼</span>
      </div>
      <div class="llm-content" style="display: none;">
        <span class="llm-status">pending</span>
      </div>
    </div>
  `;
  container.prepend(el);
}

// Show classification loading
export function showClassificationLoading(sightingId) {
  const el = document.getElementById(`det-${sightingId}`);
  if (!el) return;

  const loadingDiv = el.querySelector('.classification-loading');
  if (loadingDiv) {
    loadingDiv.style.display = 'flex';
  }
}

// Hide classification loading, show LLM loading
export function showLLMLoading(sightingId) {
  const el = document.getElementById(`det-${sightingId}`);
  if (!el) return;

  const classificationLoading = el.querySelector('.classification-loading');
  const llmLoading = el.querySelector('.llm-loading');

  if (classificationLoading) classificationLoading.style.display = 'none';
  if (llmLoading) llmLoading.style.display = 'flex';
}

// Hide all loading, show LLM section
export function hideLLMLoading(sightingId) {
  const el = document.getElementById(`det-${sightingId}`);
  if (!el) return;

  const llmLoading = el.querySelector('.llm-loading');
  const llmDiv = el.querySelector('.llm');

  if (llmLoading) llmLoading.style.display = 'none';
  if (llmDiv) llmDiv.style.display = 'block';
}

// Global function to toggle LLM analysis in main detection cards
window.toggleMainDetectionLLM = function(sightingId) {
  const el = document.getElementById(`det-${sightingId}`);
  if (!el) return;

  const llmDiv = el.querySelector('.llm');
  const llmContent = el.querySelector('.llm-content');
  const llmArrow = el.querySelector('.llm-arrow');

  if (llmDiv && llmContent) {
    const isVisible = llmContent.style.display !== 'none';
    llmContent.style.display = isVisible ? 'none' : 'block';
    if (llmArrow) {
      llmArrow.textContent = isVisible ? '▼' : '▲';
    }
  }
};

export function setLLMCompleted(sightingId, llm) {
  const el = document.getElementById(`det-${sightingId}`);
  if (!el) return;

  // Hide LLM loading
  hideLLMLoading(sightingId);

  // Make sure LLM section is visible when completed
  const llmDiv = el.querySelector('.llm');
  if (llmDiv) {
    llmDiv.style.display = 'block';
  }

  // Update species name to show it's clickable and LLM is ready
  const speciesDiv = el.querySelector('.detection-species');
  if (speciesDiv && llm) {
    speciesDiv.style.color = 'var(--accent)';
    speciesDiv.style.fontWeight = '600';
    speciesDiv.title = 'Click to view AI analysis';
  }

  const llmContent = el.querySelector('.llm-content');
  if (llmContent && llm) {
    // Get the analysis data (should be an object now)
    const analysisData = llm.details;

    // Create a nicely formatted LLM display
    let formattedContent = '';

    if (analysisData && typeof analysisData === 'object') {
      // Species Information
      if (analysisData.advisory_content?.species_identification) {
        const speciesInfo = analysisData.advisory_content.species_identification;
        formattedContent += `
          <div class="analysis-section">
            <h4>Species Information</h4>
            <p><strong>Scientific Name:</strong> ${speciesInfo.scientific_name || 'Unknown'}</p>
            <p><strong>Common Names:</strong> ${speciesInfo.common_names || 'Unknown'}</p>
            <p><strong>Family:</strong> ${speciesInfo.family || 'Unknown'}</p>
          </div>
        `;
      }

      // Legal Status & Risk
      if (analysisData.advisory_content?.legal_status) {
        const legalInfo = analysisData.advisory_content.legal_status;
        formattedContent += `
          <div class="analysis-section">
            <h4>Legal Status</h4>
            <p><strong>NEMBA Category:</strong> ${legalInfo.nemba_category || 'Unknown'}</p>
            <p><strong>Requirements:</strong> ${legalInfo.legal_requirements || 'Unknown'}</p>
            <p><strong>Risk Level:</strong> ${analysisData.risk_level || 'Unknown'}</p>
          </div>
        `;
      }

      // Description
      if (analysisData.description) {
        formattedContent += `
          <div class="analysis-section">
            <h4>Description</h4>
            <p>${analysisData.description}</p>
          </div>
        `;
      }

      // Distribution
      if (analysisData.where_found) {
        formattedContent += `
          <div class="analysis-section">
            <h4>Where Found</h4>
            <p>${analysisData.where_found}</p>
          </div>
        `;
      }

      // Control Methods
      if (analysisData.treatment && analysisData.treatment !== 'Not found') {
        formattedContent += `
          <div class="analysis-section">
            <h4>Control Methods</h4>
            <p>${analysisData.treatment}</p>
          </div>
        `;
      }

      // Action Required
      if (analysisData.action_required) {
        formattedContent += `
          <div class="analysis-section">
            <h4>Action Required</h4>
            <p>${analysisData.action_required}</p>
          </div>
        `;
      }

      // Disclaimer
      if (analysisData.disclaimer) {
        formattedContent += `
          <div class="analysis-section disclaimer">
            <h4>Disclaimer</h4>
            <p><em>${analysisData.disclaimer}</em></p>
          </div>
        `;
      }
    } else {
      // Fallback for string content
      formattedContent = `<div class="analysis-section"><p>${analysisData || 'Analysis completed successfully.'}</p></div>`;
    }

    llmContent.innerHTML = `
      <div class="llm-details">
        ${formattedContent}
      </div>
    `;
  } else {
    const span = el.querySelector('.llm-status');
    if (span) span.textContent = 'completed';
  }
}
