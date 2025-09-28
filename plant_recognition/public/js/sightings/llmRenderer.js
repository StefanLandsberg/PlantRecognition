// js/sightings/llmRenderer.js

/**
 * Formats and returns the HTML for a specific section of the LLM analysis data.
 * @param {object} llmData - The LLM analysis data object.
 * @param {string} section - The name of the section to format.
 * @returns {string} The generated HTML string.
 */
export const formatLLMSection = (llmData, section) => {
  if (!llmData || !llmData.details) return '<p>No analysis data available.</p>';

  const analysisData = llmData.details;

  switch (section) {
    case 'species':
      const speciesInfo = analysisData.advisory_content?.species_identification;
      return `
        <h4>Species Information</h4>
        <p><strong>Scientific Name:</strong> ${speciesInfo?.scientific_name || analysisData.species || 'Unknown'}</p>
        <p><strong>Common Names:</strong> ${speciesInfo?.common_names || analysisData.common_name || 'Unknown'}</p>
        <p><strong>Family:</strong> ${speciesInfo?.family || analysisData.family || 'Unknown'}</p>
      `;

    case 'legal':
      const legalInfo = analysisData.advisory_content?.legal_status;
      return `
        <h4>Legal Status</h4>
        <p><strong>NEMBA Category:</strong> ${legalInfo?.nemba_category || 'Unknown'}</p>
        <p><strong>Legal Requirements:</strong> ${legalInfo?.legal_requirements || 'Unknown'}</p>
        <p><strong>Risk Level:</strong> ${analysisData.risk_level || 'Unknown'}</p>
        <p><strong>Invasive Status:</strong> ${analysisData.invasive_status ? 'Yes' : 'No'}</p>
      `;

    case 'description':
      const physicalDesc = analysisData.advisory_content?.physical_description;
      return `
        <h4>Physical Description</h4>
        <p>${physicalDesc || analysisData.description || 'No description available.'}</p>
        <p><strong>Origin:</strong> ${analysisData.origin || 'Unknown'}</p>
      `;

    case 'distribution':
      const distributionInfo = analysisData.advisory_content?.distribution || analysisData.where_found || analysisData.distribution;
      if (!distributionInfo || distributionInfo === 'Not found') {
        return '<p>No distribution information available.</p>';
      }
      return `
        <h4>Where Found</h4>
        <p>${distributionInfo}</p>
      `;

    case 'control':
      const controlInfo = analysisData.advisory_content?.control_methods || analysisData.treatment || analysisData.control_methods;
      if (!controlInfo || controlInfo === 'Not found') {
        return '<p>No control methods available.</p>';
      }
      return `
        <h4>Control Methods</h4>
        <p>${controlInfo}</p>
      `;

    case 'action':
      const actionInfo = analysisData.action_required || analysisData.advisory_content?.action_required;
      if (!actionInfo || actionInfo === 'Not found') {
        return '<p>No action required.</p>';
      }
      return `
        <h4>Action Required</h4>
        <p>${actionInfo}</p>
      `;

    default:
      return '<p>Select a section to view details.</p>';
  }
};

/**
 * Creates the HTML for the collapsible AI Analysis dropdown.
 * @param {object} sighting - The sighting data object.
 * @returns {string} The generated HTML string for the dropdown.
 */
export const createLLMDropdown = (sighting) => {
  const hasLLM = sighting.analysis?.llm && sighting.analysis.llm.details;

  if (!hasLLM) {
    const llmStatus = sighting.analysis?.llm?.status;
    const statusText = llmStatus === 'pending' ? 'Processing...' :
                     llmStatus === 'failed' ? 'Analysis failed' :
                     'No analysis available';

    return `
      <div class="llm-dropdown">
        <div class="llm-dropdown-header">
          <span>AI Analysis</span>
          <span>${statusText}</span>
        </div>
      </div>
    `;
  }

  return `
    <div class="llm-dropdown">
      <div class="llm-dropdown-header" onclick="toggleLLMDropdown('${sighting._id}')">
        <span>AI Analysis</span>
        <span class="llm-dropdown-arrow">▼</span>
      </div>
      <div class="llm-dropdown-content llm-dropdown-content-hidden">
        <div class="llm-section-selector">
          <button class="llm-section-btn active" onclick="showLLMSection('${sighting._id}', 'species')">Species Info</button>
          <button class="llm-section-btn" onclick="showLLMSection('${sighting._id}', 'legal')">Legal Status</button>
          <button class="llm-section-btn" onclick="showLLMSection('${sighting._id}', 'description')">Description</button>
          <button class="llm-section-btn" onclick="showLLMSection('${sighting._id}', 'distribution')">Distribution</button>
          <button class="llm-section-btn" onclick="showLLMSection('${sighting._id}', 'control')">Control</button>
          <button class="llm-section-btn" onclick="showLLMSection('${sighting._id}', 'action')">Action</button>
        </div>
        <div class="llm-section-content" id="llm-content-${sighting._id}">
          ${formatLLMSection(sighting.analysis.llm, 'species')}
        </div>
      </div>
    </div>
  `;
};