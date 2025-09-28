import { sanitizeHtml } from './sanitize.js';

export function addDetectionCard(container, det) {
  const el = document.createElement('div');
  el.className = 'card';
  el.id = `det-${det.sightingId}`;
  el.innerHTML = `
    <div class="badges">
      <div class="detection-species" title="Species identified">
        Species: ${sanitizeHtml(det.predictedSpecies || 'Unknown')}
      </div>
      <div>Conf: ${(det.confidence*100).toFixed(1)}%</div>
    </div>
    ${det.imageUrl ?
      `<img src="${det.imageUrl}" alt="detection" onerror="this.style.display='none'" />` :
      `<div class="no-image-placeholder"></div>`
    }

    <div class="classification-loading" style="display: none;">
      <div class="skeleton-content">
        <div class="skeleton skeleton-text large"></div>
        <div class="skeleton skeleton-text"></div>
        <div class="skeleton skeleton-text small"></div>
      </div>
      <span style="margin-top: 0.5rem; color: var(--text-secondary); font-size: 0.875rem;">Classifying...</span>
    </div>

    <div class="llm-loading" style="display: none;">
      <div class="skeleton-content">
        <div class="skeleton skeleton-text large"></div>
        <div class="skeleton skeleton-text"></div>
        <div class="skeleton skeleton-text"></div>
        <div class="skeleton skeleton-text small"></div>
      </div>
      <span style="margin-top: 0.5rem; color: var(--text-secondary); font-size: 0.875rem;">Analyzing with AI...</span>
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
  if (llmLoading) llmLoading.style.display = 'none';
}

// Clean detection card dropdown toggle function
window.toggleMainDetectionLLM = function(sightingId) {
  const card = document.getElementById(`det-${sightingId}`);
  if (!card) return;

  const llmSection = card.querySelector('.llm-section');
  const llmContent = card.querySelector('.llm-content');
  const speciesArrow = card.querySelector('.species-dropdown-arrow');

  // If no LLM section exists yet, do nothing
  if (!llmSection || !llmContent) return;

  // Toggle content visibility
  const isCurrentlyHidden = llmContent.style.display === 'none' || !llmContent.style.display;

  if (isCurrentlyHidden) {
    llmContent.style.display = 'block';
    llmSection.classList.add('expanded');
  } else {
    llmContent.style.display = 'none';
    llmSection.classList.remove('expanded');
  }

  // Update arrow
  if (speciesArrow) {
    speciesArrow.textContent = isCurrentlyHidden ? '▲' : '▼';
  }
};

export function setLLMCompleted(sightingId, llm) {
  const card = document.getElementById(`det-${sightingId}`);
  if (!card) {
    // Retry logic for race conditions
    const tempCards = document.querySelectorAll('[id^="det-temp-"]');
    if (tempCards.length > 0) {
      setTimeout(() => {
        const delayedCard = document.getElementById(`det-${sightingId}`);
        if (delayedCard) {
          createLLMDropdown(delayedCard, sightingId, llm);
        }
      }, 150);
      return;
    }
    return;
  }

  createDetectionLLMDropdown(card, sightingId, llm);
}

function createDetectionLLMDropdown(card, sightingId, llm) {
  // Hide LLM loading
  hideLLMLoading(sightingId);

  // Remove existing LLM section if any
  const existingLLMSection = card.querySelector('.llm-section');
  if (existingLLMSection) {
    existingLLMSection.remove();
  }

  // Create the complete LLM dropdown section
  const llmSection = document.createElement('div');
  llmSection.className = 'llm-section';

  // Format the LLM content
  const formattedContent = formatLLMAnalysis(llm);

  llmSection.innerHTML = `
    <div class="species-dropdown-header" onclick="toggleMainDetectionLLM('${sightingId}')" style="cursor: pointer;">
      <span class="species-text">AI Analysis Available</span>
      <span class="species-dropdown-arrow">▼</span>
    </div>
    <div class="llm-content" style="display: none;">
      ${formattedContent}
    </div>
  `;

  // Append to card
  card.appendChild(llmSection);

  // Update species name to show analysis is available
  const speciesDiv = card.querySelector('.detection-species');
  if (speciesDiv) {
    speciesDiv.style.color = 'var(--accent)';
    speciesDiv.style.fontWeight = '600';
    speciesDiv.title = 'AI analysis completed - see dropdown below';
  }
}

function formatLLMAnalysis(llm) {
  if (!llm || !llm.details) {
    return '<div class="analysis-section"><p style="color: var(--text);">Analysis completed but no details available.</p></div>';
  }

  const analysisData = llm.details;
  let formattedContent = '';

  if (analysisData && typeof analysisData === 'object') {
    // Species Information
    if (analysisData.advisory_content?.species_identification) {
      const speciesInfo = analysisData.advisory_content.species_identification;
      formattedContent += `
        <div class="analysis-section">
          <h4 style="color: var(--accent);">Species Information</h4>
          <p style="color: var(--text);"><strong>Scientific Name:</strong> ${speciesInfo.scientific_name || 'Unknown'}</p>
          <p style="color: var(--text);"><strong>Common Names:</strong> ${speciesInfo.common_names || 'Unknown'}</p>
          <p style="color: var(--text);"><strong>Family:</strong> ${speciesInfo.family || 'Unknown'}</p>
        </div>
      `;
    }

    // Legal Status & Risk
    if (analysisData.advisory_content?.legal_status) {
      const legalInfo = analysisData.advisory_content.legal_status;
      formattedContent += `
        <div class="analysis-section">
          <h4 style="color: var(--accent);">Legal Status</h4>
          <p style="color: var(--text);"><strong>NEMBA Category:</strong> ${legalInfo.nemba_category || 'Unknown'}</p>
          <p style="color: var(--text);"><strong>Requirements:</strong> ${legalInfo.legal_requirements || 'Unknown'}</p>
          <p style="color: var(--text);"><strong>Risk Level:</strong> ${analysisData.risk_level || 'Unknown'}</p>
        </div>
      `;
    }

    // Description
    if (analysisData.advisory_content?.physical_description || analysisData.description) {
      formattedContent += `
        <div class="analysis-section">
          <h4 style="color: var(--accent);">Description</h4>
          <p style="color: var(--text);">${analysisData.advisory_content?.physical_description || analysisData.description}</p>
        </div>
      `;
    }

    // Control Methods
    if (analysisData.advisory_content?.control_methods || analysisData.treatment) {
      formattedContent += `
        <div class="analysis-section">
          <h4 style="color: var(--accent);">Control Methods</h4>
          <p style="color: var(--text);">${analysisData.advisory_content?.control_methods || analysisData.treatment}</p>
        </div>
      `;
    }

    // Distribution
    if (analysisData.advisory_content?.distribution || analysisData.where_found) {
      formattedContent += `
        <div class="analysis-section">
          <h4 style="color: var(--accent);">Distribution</h4>
          <p style="color: var(--text);">${analysisData.advisory_content?.distribution || analysisData.where_found}</p>
        </div>
      `;
    }

    // Disclaimer
    if (analysisData.disclaimer) {
      formattedContent += `
        <div class="analysis-section disclaimer">
          <h4 style="color: var(--accent);">Important Note</h4>
          <p style="color: var(--text);">${analysisData.disclaimer}</p>
        </div>
      `;
    }
  } else {
    // Fallback for string content
    formattedContent = `<div class="analysis-section"><p style="color: var(--text);">${analysisData || 'Analysis completed successfully.'}</p></div>`;
  }

  return formattedContent || '<div class="analysis-section"><p style="color: var(--text);">Analysis data not available.</p></div>';
}