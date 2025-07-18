# RAG LLM Module Technical Documentation

## Overview

The RAG LLM module serves as an invasive plant advisory system that integrates with the existing plant recognition pipeline. 
The module receives plant species names from the image recognition system and determines whether the identified plant is invasive or 
non-invasive within the South African context. The system provides comprehensive information including invasiveness assessment, 
removal guidance, and general botanical knowledge for each identified invasive species. 

When the specific plant name provided by the recognition model is not present in the RAG database, the system must generalise based 
on genus-level information or other taxonomic relationships derived from name patterns. For example, if the model identifies 
"Acacia_melanoxylon" but only "Acacia_dealbata" exists in the database, the system should extract genus-level information about 
Acacia species invasiveness patterns and apply appropriate generalisations with clear disclaimers about the inference basis.

The module operates as a post-processing layer that transforms basic species identification into actionable conservation guidance, 
combining machine learning classification with ecological expertise to support field-based invasive species management.

## System Integration

### Input Interface

The LLM module receives input from the plant recognition system through the following data structure:

```
Input Format:
{
    'predicted_species': string,        # e.g., "Acacia_baileyana"
    'confidence': float,                # 0.0 to 1.0
    'top5_predictions': list of tuples, # [(species_name, confidence), ...]
    'processing_details': dict          # metadata from recognition system
}
```

The predicted_species field contains plant names in underscore format (genus_species) as classified by the existing neural network models. 
This input comes directly from the plant_recognition_app.py identify_plant() function which processes uploaded images through a 
6-modal descriptor analysis system. The confidence score reflects the image recognition system's certainty about species identification, 
not invasiveness classification. The top5_predictions provide alternative species possibilities that may be considered if the primary 
prediction lacks database coverage.

### Output Specification

The module outputs a classification decision following a specific binary logic:
- "unknown" indicates the plant is non-invasive or benign (safe to leave in environment)
- The actual plant name (e.g., "Acacia_baileyana") indicates the plant is invasive and requires management action

This binary classification system enables simple decision-making for field workers: if the output is "unknown", no action is required; 
if the output is a species name, immediate assessment and potential removal should be considered based on the accompanying advisory content.

We can change this in future but this will be the  the current system.

Output format:
```
{
    'classification': string,           # "unknown" or species name
    'invasive_status': boolean,         # true if invasive, false if non-invasive
    'advisory_content': dict,           # structured advisory information including removal guidance, 
                                        or any other general information needed for the user to be well informed
    'confidence_score': float,          # classification confidence
    'data_sources': list,              # references used for classification
    'generalisation_level': string,    # "exact", "genus", "family", or "none"
    'disclaimer': string               # included when generalisation applied
}
```

## Knowledge Base Requirements

### Data Collection Targets

The system requires a comprehensive knowledge base of South African invasive plant species.

1. Official invasive species registries containing confirmed invasive species lists with legal classifications
2. NEMBA (National Environmental Management: Biodiversity Act) classifications providing legal framework for species management
3. Scientific literature on plant invasiveness including peer-reviewed research on ecological impacts and spread patterns
4. Regional ecological impact studies documenting specific effects on South African ecosystems and native species
5. Management and removal guidelines including mechanical, chemical, and biological control methods proven effective in South African conditions
6. Field guides and identification resources to support accurate species recognition and management planning

The knowledge base must prioritise South African-specific information over global data, as invasiveness patterns and management 
strategies often vary significantly between different climatic and ecological contexts.

### Document Processing Pipeline

Raw documents require processing through the following stages:

1. Content extraction: Remove navigation, advertisements, and formatting
2. Text cleaning: Eliminate non-content artefacts
3. Contextual chunking: Divide documents into semantic passages
4. Metadata indexing: Preserve source URLs, timestamps, and categorisation
5. Embedding generation: Create vector representations using semantic models

## RAG Architecture

### Vector Store Implementation

The system requires a vector database to store and retrieve document embeddings. Recommended implementations:

- FAISS for high-performance similarity search
- Chroma for development and testing
- Qdrant for production deployments

Vector dimensions should match the chosen embedding model .

### Retrieval Process

When querying the knowledge base:

1. Convert species name to embedding vector
2. Perform similarity search against document store
3. Retrieve top-k relevant passages (k=5-10)
4. Rank results by relevance score
5. Filter results by confidence threshold

### Generation Pipeline

The language model component must:

1. Accept retrieved context passages as input
2. Generate structured advisory content
3. Maintain factual grounding to source documents
4. Indicate uncertainty when data is insufficient
5. Provide source attribution for claims

## Classification Logic

### Invasive Species Determination

The system must implement the following classification logic:

1. Search knowledge base for exact species match
2. If direct match found with invasive classification, return species name and full advisory content
3. If no direct match, search for genus-level information and taxonomic relationships
4. Generate advisory content based on genus characteristics when specific species data unavailable
5. Extract genus from species name (e.g., "Acacia" from "Acacia_baileyana") for hierarchical matching
6. Default to "unknown" (non-invasive) when insufficient data exists for confident classification

### Taxonomic Generalisation Process

When specific species information is unavailable, the system implements hierarchical fallback logic:

1. Extract genus name from underscore-formatted species identifier (e.g., "Acacia" from "Acacia_unknown_species")
2. Search knowledge base for genus-level invasiveness patterns and documented invasive species within that genus
3. Identify related species within the same genus with known invasive status in South Africa
4. Apply genus-level removal and management strategies, adapting techniques used for documented invasive relatives
5. Include clear disclaimer indicating information is generalised from related species (e.g., "This advisory is based on Acacia dealbata characteristics as no specific data exists for Acacia melanoxylon")
6. Provide confidence indicator reflecting level of taxonomic generalisation applied (exact match = 100%, genus level = 60-80%, family level = 30-50%)

This approach ensures that novel invasive species or data gaps do not result in complete information absence, whilst maintaining 
scientific transparency about the inference basis.

### NEMBA Categories

South African invasive species fall into specific legal categories:

- Category 1a: Prohibited species requiring immediate removal
- Category 1b: Invasive species requiring control measures
- Category 2: Commercially important species with restrictions
- Category 3: Ornamental plants with invasive potential

The system can identify and report these classifications, [1]if available.

## Advisory Content Structure

### Required Output Sections

Each invasive species advisory must include:

1. Species identification: Scientific and common names
2. Legal status: NEMBA category and legal requirements[1]
3. Ecological impact: Effects on native biodiversity[1]
4. Physical description: Morphological characteristics[1]
5. Distribution: Current range and spread patterns[1]
6. Control methods: Mechanical, chemical, and biological options
7. Monitoring requirements: Post-removal surveillance needs

### Content Quality Standards

All advisory content must:

- Reference authoritative sources
- Avoid speculation or LLM model hallucination
- Include appropriate disclaimers for inferred information
- Maintain scientific accuracy
- Provide actionable guidance

## Performance Requirements

### Response Time

The system must generate advisory content within 1 second of receiving a species name. This requires: (this will most likely change in future for video classification so enable time scalable architecture)

- Optimised vector search indices
- Efficient embedding models
- Quantised language models for edge deployment
- Local inference capabilities for field use

### Memory Constraints

Total system memory usage should not exceed 4GB, requiring: (our current classification model won't exceed 100MB)

- Model quantisation (4-bit or 8-bit)
- Efficient vector storage
- Memory-mapped databases
- Streaming processing for large documents

## Error Handling

### Data Uncertainty

When facing incomplete or contradictory information:

1. Clearly indicate uncertainty levels
2. Provide basis for inferences
3. Suggest verification with authorities
4. Default to conservative classification

### System Failures

The module must handle:

- Missing vector database connections
- Corrupted embedding indices
- Network failures during web scraping
- Invalid input formats from recognition system

## Deployment Considerations

### Offline Capability

The system must function without internet connectivity for field deployment:

- All models stored locally
- Complete knowledge base cached on device
- No external API dependencies for core functionality

### Update Mechanisms

Knowledge base updates require:

- Scheduled scraping of target websites
- Incremental index updates
- Version control for knowledge base states
- Rollback capabilities for corrupted data

## Integration Testing

### Validation Requirements

System testing must verify:

- Correct classification of known invasive species
- Proper handling of non-invasive species
- Advisory content accuracy and completeness
- Performance within specified time constraints
- Offline functionality in disconnected environments

### Test Data Requirements

Testing requires validated datasets containing:

- Confirmed invasive species with official classifications
- Non-invasive native species
- Edge cases with uncertain invasive status
- Species not present in knowledge base
