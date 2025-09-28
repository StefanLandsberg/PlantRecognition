import { PythonShell } from 'python-shell';

// Simple in-memory cache for LLM results to avoid repeated computations
const llmCache = new Map();
const CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24 hours in milliseconds
const MAX_CACHE_SIZE = 1000; // Maximum number of cached entries

// LLM Queue System
class LLMQueue {
  constructor() {
    this.queue = [];
    this.processing = false;
    this.maxRetries = 3;
  }

  async add(sightingId, species, confidence, callback) {
    return new Promise((resolve, reject) => {
      const queueItem = {
        sightingId,
        species,
        confidence,
        callback,
        resolve,
        reject,
        retries: 0,
        timestamp: Date.now()
      };

      this.queue.push(queueItem);
      console.log(`[LLM Queue] Added sighting ${sightingId} to queue. Queue length: ${this.queue.length}`);

      // Start processing if not already running
      if (!this.processing) {
        this.processQueue();
      }
    });
  }

  async processQueue() {
    if (this.processing || this.queue.length === 0) {
      return;
    }

    this.processing = true;
    console.log(`[LLM Queue] Starting queue processing. Queue length: ${this.queue.length}`);

    while (this.queue.length > 0) {
      const item = this.queue.shift();

      try {
        console.log(`[LLM Queue] Processing sighting ${item.sightingId} (${item.species})`);
        const result = await this.processLLMAnalysis(item.sightingId, item.species, item.confidence);

        // Execute callback for database update and SSE notification
        if (item.callback) {
          await item.callback(result);
        }

        item.resolve(result);
        console.log(`[LLM Queue] Completed sighting ${item.sightingId}`);

      } catch (error) {
        console.error(`[LLM Queue] Error processing sighting ${item.sightingId}:`, error.message);

        // Retry logic
        if (item.retries < this.maxRetries) {
          item.retries++;
          this.queue.unshift(item); // Put back at front for immediate retry
          console.log(`[LLM Queue] Retrying sighting ${item.sightingId} (attempt ${item.retries}/${this.maxRetries})`);
        } else {
          // Max retries reached, create fallback result
          const fallbackResult = createBasicAnalysis(item.species, item.confidence);
          if (item.callback) {
            await item.callback(fallbackResult);
          }
          item.resolve(fallbackResult);
          console.log(`[LLM Queue] Max retries reached for sighting ${item.sightingId}, using fallback`);
        }
      }
    }

    this.processing = false;
    console.log(`[LLM Queue] Queue processing completed`);
  }

  async processLLMAnalysis(sightingId, species, confidence) {
    // This is the core LLM processing logic (extracted from kickLLM)
    if (!species || species === 'undefined' || species === 'null') {
      species = 'Unknown species';
    }
    if (!confidence || isNaN(confidence) || confidence === 'undefined' || confidence === 'null') {
      confidence = 0.0;
    }

    const normalizedSpecies = species.replace(/_/g, ' ');

    // Check cache first
    const cachedResult = getCachedResult(normalizedSpecies, confidence);
    if (cachedResult) {
      console.log(`[LLM Queue] Cache hit for ${normalizedSpecies}`);
      return cachedResult;
    }

    const options = {
      mode: 'text',
      scriptPath: '../python/',
      args: ['analyze', normalizedSpecies, String(confidence)],
      pythonOptions: ['-u', '-O']
    };

    const results = await PythonShell.run('llm_integration.py', options);

    if (results && results.length > 0) {
      let jsonData = null;

      // Look for JSON in results
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
        const result = {
          summary: `${jsonData.species || normalizedSpecies} - ${jsonData.confidence_level || 'Analysis complete'}`,
          details: jsonData,
          sources: jsonData.data_sources || []
        };

        setCachedResult(normalizedSpecies, confidence, result);
        return result;
      }
    }

    const fallbackResult = createBasicAnalysis(species, confidence);
    setCachedResult(normalizedSpecies, confidence, fallbackResult);
    return fallbackResult;
  }

  getQueueStatus() {
    return {
      queueLength: this.queue.length,
      processing: this.processing,
      oldestItem: this.queue.length > 0 ? this.queue[0].timestamp : null
    };
  }
}

// Create global queue instance
const llmQueue = new LLMQueue();

function getCacheKey(species, confidence) {
  // Normalize the species name and round confidence to avoid cache misses on minor differences
  const normalizedSpecies = species.toLowerCase().replace(/[^a-z0-9]/g, '');
  const roundedConfidence = Math.round(confidence * 100) / 100; // Round to 2 decimal places
  return `${normalizedSpecies}_${roundedConfidence}`;
}

function getCachedResult(species, confidence) {
  const key = getCacheKey(species, confidence);
  const cached = llmCache.get(key);

  if (cached && (Date.now() - cached.timestamp) < CACHE_EXPIRY) {
    return cached.result;
  }

  if (cached) {
    llmCache.delete(key); // Remove expired entry
  }

  return null;
}

function setCachedResult(species, confidence, result) {
  const key = getCacheKey(species, confidence);

  // Implement LRU-style cache size management
  if (llmCache.size >= MAX_CACHE_SIZE) {
    const firstKey = llmCache.keys().next().value;
    llmCache.delete(firstKey);
  }

  llmCache.set(key, {
    result: result,
    timestamp: Date.now()
  });
}

// Legacy function for backwards compatibility
export async function kickLLM(sightingId, species, confidence) {
  // Direct processing without queue (for backward compatibility)
  return await llmQueue.processLLMAnalysis(sightingId, species, confidence);
}

// New queue-based function
export async function queueLLMAnalysis(sightingId, species, confidence, callback) {
  return await llmQueue.add(sightingId, species, confidence, callback);
}

// Get queue status for monitoring
export function getLLMQueueStatus() {
  return llmQueue.getQueueStatus();
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
