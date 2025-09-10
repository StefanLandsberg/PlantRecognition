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
  
  const llmDiv = el.querySelector('.llm');
  if (llmDiv && llm) {
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
    
    llmDiv.innerHTML = `
      <details open>
        <summary>${llm.summary || 'AI Analysis Complete'}</summary>
        <div class="llm-details">
          ${formattedContent}
        </div>
      </details>
    `;
  } else {
    const span = el.querySelector('.llm-status');
    if (span) span.textContent = 'completed';
  }
}
