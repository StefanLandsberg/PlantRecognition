import { PythonShell } from 'python-shell';

export async function kickLLM(sightingId, species, confidence) {
  try {
    // Fast parameter validation
    if (!species || species === 'undefined' || species === 'null') {
      species = 'Unknown species';
    }
    if (!confidence || isNaN(confidence) || confidence === 'undefined' || confidence === 'null') {
      confidence = 0.0;
    }
    
    // Normalize species name - replace underscores with spaces for LLM lookup
    const normalizedSpecies = species.replace(/_/g, ' ');
    
    const options = {
      mode: 'text',
      scriptPath: '../python/',
      args: ['analyze', normalizedSpecies, String(confidence)]
    };

    const results = await PythonShell.run('llm_integration.py', options);
    
    if (results && results.length > 0) {
      // Fast JSON extraction
      let jsonData = null;
      
      // Method 1: Look for single line JSON (most common case)
      for (const line of results) {
        const trimmed = line.trim();
        if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
          try {
            jsonData = JSON.parse(trimmed);
            break;
          } catch (e) {
            // Continue to next line
          }
        }
      }
      
      // Method 2: Look for multi-line JSON block (if needed)
      if (!jsonData) {
        const fullOutput = results.join('\n');
        const jsonMatch = fullOutput.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          try {
            jsonData = JSON.parse(jsonMatch[0]);
          } catch (e) {
            // Fall through to basic analysis
          }
        }
      }
      
      if (jsonData) {
        return {
          summary: `${jsonData.species || normalizedSpecies} - ${jsonData.confidence_level || 'Analysis complete'}`,
          details: jsonData,
          sources: jsonData.data_sources || []
        };
      }
    }
    
    return createBasicAnalysis(species, confidence);
  } catch (error) {
    console.error('LLM processing error:', error.message);
    return createBasicAnalysis(species, confidence);
  }
}

function createBasicAnalysis(species, confidence) {
  // Ensure confidence is a valid number
  if (!confidence || isNaN(confidence)) {
    confidence = 0.0;
  }
  if (!species || species === 'undefined' || species === 'null') {
    species = 'Unknown species';
  }
  
  const confidencePercent = (parseFloat(confidence) * 100).toFixed(1);
  
  const normalizedSpecies = species.replace(/_/g, ' ');
  
  // Enhanced fallback with more detailed analysis matching frontend structure
  const analysisDetails = {
    advisory_content: {
      species_identification: {
        scientific_name: normalizedSpecies,
        common_names: "Database lookup failed - common names not available",
        family: "Family information not available"
      },
      legal_status: {
        nemba_category: "Unknown",
        legal_requirements: "Botanical database lookup failed. Consult local botanical experts for proper classification and legal status."
      }
    },
    description: `${normalizedSpecies} was identified by our AI system with ${confidencePercent}% confidence. However, detailed botanical information could not be retrieved from our invasive species database.`,
    risk_level: confidence > 0.8 ? "Medium" : confidence > 0.6 ? "Low" : "Unknown",
    action_required: confidence > 0.8 ? 
      "High confidence detection - consider consulting local botanical experts for detailed species information." :
      confidence > 0.6 ?
      "Moderate confidence - verification recommended before taking action." :
      "Low confidence - manual identification required.",
    treatment: "Cannot provide control recommendations without proper species verification. Consult local botanical authorities.",
    disclaimer: "AI classification only. Detailed species information unavailable. For management decisions, consult qualified botanists."
  };

  return {
    summary: `${species} (${confidencePercent}% confidence)`,
    details: analysisDetails,
    sources: ["AI Classification System"]
  };
}
