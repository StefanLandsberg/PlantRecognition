#!/usr/bin/env python3
"""
RAG LLM integration for plant recognition system.
Based on the working query_rag.py script, adapted for backend server use.
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
import time
import chromadb
from sentence_transformers import SentenceTransformer
import re
import torch
import numpy as np
from functools import lru_cache

# === CONFIG ===
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(SCRIPT_DIR, "vector_db")  # Absolute path to vector_db
COLLECTION_NAME = "plants"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
SOURCE = "Source: Invasive Alien & Problem Plants on The Witwatersrand & Magaliesberg. Field Guide by Karin Spottiswoode"

# GPU Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 if torch.cuda.is_available() else 1

# === INIT ===
client = None
collection = None
model = None
_initialized = False  # Global flag to prevent multiple initializations

def initialize_components():
    """Initialize ChromaDB and model components."""
    global client, collection, model, _initialized
    
    # Prevent multiple initializations
    if _initialized:
        print("RAG system already initialized, skipping...")
        return True
    
    try:
        # Fast directory check - no verbose logging
        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError(f"ChromaDB directory not found: {CHROMA_DIR}")
        
        # Initialize ChromaDB client (minimal operations)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # Initialize Sentence Transformer model with maximum speed optimizations
        model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        if torch.cuda.is_available():
            model.half()  # Use FP16 for 2x speed
            model.eval()  # Set to eval mode
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Warm up the model with a dummy embedding for faster first query
            model.encode(["dummy"], convert_to_tensor=True, device=DEVICE)
        else:
            # CPU optimizations
            model.eval()
            # Warm up CPU model
            model.encode(["dummy"], convert_to_numpy=True)
        _initialized = True
        return True
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return False

# === QUERY ===
# Cache for embeddings to avoid recomputation
@lru_cache(maxsize=1000)
def _cached_embedding(plant_name_tuple):
    """Cached embedding computation for plant names."""
    plant_name = plant_name_tuple[0]  # Extract from tuple for caching
    with torch.inference_mode():  # Disable gradients for faster inference
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():  # Mixed precision
                embedding = model.encode([plant_name], convert_to_tensor=True, device=DEVICE)
                return embedding.cpu().numpy().tolist()
        else:
            embedding = model.encode([plant_name], convert_to_numpy=True)
            return embedding.tolist()

def query_plants(plant_name):
    """Ultra-fast RAG query with minimal overhead."""
    if not collection or not model:
        return None
    
    if not plant_name or not isinstance(plant_name, str):
        return None
    
    try:
        # Minimal processing - just strip and lowercase
        plant_name = plant_name.strip().lower()
        if not plant_name:
            return None
        
        # Use cached embedding computation with optimizations
        query_embedding = _cached_embedding((plant_name,))
        
        # Fast ChromaDB query with minimal results
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=1  # Reduced from TOP_K for speed
        )
        return results
    except Exception as e:
        return None

# Compile regex patterns for faster processing
_ID_PATTERN = re.compile(r"(?:Not to be confused with|Identification)[\s:]*\n*(.*?)(?=\n(?:Family|Common names|Origin|Where found|Treatment|Uses|Notes|Leaf|Habitat|Description)[\s:]*|\Z)", re.IGNORECASE | re.DOTALL | re.MULTILINE)
_SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+')
_POISON_KEYWORDS = {
    "poison", "toxic", "irritant", "irritation", "rash", "allergic", "reaction",
    "not edible", "noxious", "harmful", "respiratory tract"
}

@lru_cache(maxsize=100)
def extract_identification(text):
    """Extract identification information from text."""
    match = _ID_PATTERN.search(text)
    return match.group(1).strip() if match else "Not found"

@lru_cache(maxsize=100)
def extract_poisonous(text):
    """Extract poisonous information from text."""
    sentences = _SENTENCE_PATTERN.split(text)
    poison_sentences = [
        s.strip() for s in sentences
        if any(k in s.lower() for k in _POISON_KEYWORDS)
    ]
    return " ".join(poison_sentences) if poison_sentences else "Not found"

# Compile patterns for field extraction
_WHITESPACE_PATTERN = re.compile(r"(\n\s*){2,}")
_ORIGIN_PATTERN = re.compile(
    r"^\s*Origin[\s:]*\n*(.*?)(?=\n(?:^\s*(Where found|Family|Common names|Treatment)[\s:]*|\Z))",
    re.IGNORECASE | re.DOTALL | re.MULTILINE
)

@lru_cache(maxsize=50)
def extract_fields(text):
    """Extract structured information from plant text."""
    text = _WHITESPACE_PATTERN.sub("\n", text.strip())

    # Pre-compiled patterns for common field extractions
    _stop_labels = [
        "Family", "Common names", "Origin", "Where found", "Treatment",
        "Uses", "Identification", "Notes", "Leaf", "Habitat", "Description"
    ]
    
    def multiline_extract(label):
        stop_pattern = "|".join([rf"^\s*{l}[\s:]*" for l in _stop_labels if l.lower() != label.lower()])
        pattern = rf"^\s*{label}[\s:]*\n*(.*?)(?=\n(?:{stop_pattern})|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        return re.sub(r"\s+", " ", match.group(1)).strip() if match else "Not found"

    origin_match = _ORIGIN_PATTERN.search(text)

    if origin_match:
        origin_block = origin_match.group(1).strip()
        split = re.split(r"[.\n]", origin_block, maxsplit=1)
        origin = split[0].strip()
        description = split[1].strip() if len(split) > 1 else "Not found"
    else:
        origin = "Not found"
        description = "Not found"

    return {
        "Family": multiline_extract("Family"),
        "Common Names": multiline_extract("Common names"),
        "Origin": origin,
        "Description": description,
        "Where Found": multiline_extract("Where found"),
        "Treatment": multiline_extract("Treatment"),
        "Identification": extract_identification(text),
        "Poisonous": extract_poisonous(text),
    }

def safe_extract(fields, field_name, max_len=350):
    """Safely extract and clean field values."""
    val = fields.get(field_name, "Not found")
    val = re.sub(r"\s+", " ", val.strip())
    val = re.sub(r"\(\d+\)", "", val)
    val = re.sub(r"\b\d+\b(?![a-zA-Z])", "", val)
    val = re.sub(r"\s{2,}", " ", val)
    if field_name == "Where Found":
        val = re.sub(r"^\?\s*", "", val)
    return val if len(val) <= max_len else val[:max_len].strip() + "..."

def determine_nemba_category(fields, plant_name):
    """Determine NEMBA category based on plant data."""
    # Look for category indicators in the text
    treatment = fields.get("Treatment", "").lower()
    family = fields.get("Family", "").lower()
    
    # Category 1a: Prohibited species requiring immediate removal
    if any(keyword in treatment for keyword in ["immediate", "prohibited", "eradicate"]):
        return "1a"
    
    # Category 1b: Invasive species requiring control measures
    if any(keyword in treatment for keyword in ["control", "remove", "treatment"]):
        return "1b"
    
    # Category 2: Commercially important species with restrictions
    if any(keyword in treatment for keyword in ["commercial", "restricted", "permit"]):
        return "2"
    
    # Category 3: Ornamental plants with invasive potential
    if any(keyword in treatment for keyword in ["ornamental", "garden", "decorative"]):
        return "3"
    
    # Default based on presence of treatment information
    if fields.get("Treatment") != "Not found":
        return "1b"  # Assume invasive if treatment info exists
    
    return "Unknown"

def determine_invasive_status(fields, plant_name):
    """Determine if plant is invasive based on data."""
    # Check if treatment information exists (indicates invasive)
    if fields.get("Treatment") != "Not found":
        return True
    
    # Check for poisonous/toxic indicators
    if fields.get("Poisonous") != "Not found":
        return True
    
    # Check family for known invasive families
    family = fields.get("Family", "").lower()
    invasive_families = ["fabaceae", "poaceae", "asteraceae", "solanaceae"]
    if any(inv_fam in family for inv_fam in invasive_families):
        return True
    
    return False

# Result cache for analyze_plant function
_analysis_cache = {}

def analyze_plant(species_name, confidence, image_path=None):
    """Ultra-fast plant analysis with aggressive caching."""
    start_time = time.time()  # Track analysis time

    # Minimal validation
    if not species_name:
        raise ValueError("Invalid species name provided")

    # Clean species name for querying (keep underscores, they're faster)
    query_name = species_name.lower()
    
    # Aggressive caching - round confidence to nearest 0.2 for more cache hits
    confidence_bucket = round(confidence, 1) if confidence else 0.0
    cache_key = f"{query_name}_{confidence_bucket}"
    
    if cache_key in _analysis_cache:
        return _analysis_cache[cache_key]
    
    # Query the RAG system
    results = query_plants(query_name)
    
    if results and results['documents'] and results['documents'][0]:
        # Found information in RAG system
        plant_text = results['documents'][0][0]  # Get the first document
        fields = extract_fields(plant_text)
        
        # Determine invasive status and NEMBA category
        is_invasive = determine_invasive_status(fields, species_name)
        nemba_category = determine_nemba_category(fields, species_name)
        
        # Generate classification (species name if invasive, "unknown" if not)
        classification = species_name if is_invasive else "unknown"
        
        # Generate risk assessment
        if confidence > 0.8:
            confidence_level = "High confidence detection"
            action_required = "Immediate action recommended" if is_invasive else "Monitor for changes"
        elif confidence > 0.6:
            confidence_level = "Moderate confidence detection"
            action_required = "Further verification recommended"
        else:
            confidence_level = "Low confidence detection"
            action_required = "Manual verification required"
        
        # Create advisory content
        advisory_content = {
            "species_identification": {
                "scientific_name": species_name,
                "common_names": safe_extract(fields, "Common Names"),
                "family": safe_extract(fields, "Family")
            },
            "legal_status": {
                "nemba_category": nemba_category,
                "legal_requirements": f"Category {nemba_category} species - {'Immediate removal required' if nemba_category == '1a' else 'Control measures required' if nemba_category == '1b' else 'Restrictions apply' if nemba_category == '2' else 'Monitor for spread' if nemba_category == '3' else 'Unknown status'}"
            },
            "physical_description": safe_extract(fields, "Description", max_len=500),
            "distribution": safe_extract(fields, "Where Found"),
            "control_methods": safe_extract(fields, "Treatment", max_len=500),
            "identification_guide": safe_extract(fields, "Identification"),
            "poisonous_properties": safe_extract(fields, "Poisonous"),
            "monitoring_requirements": "Regular follow-up required for effective control" if is_invasive else "No monitoring required"
        }
        
        analysis = {
            "classification": classification,
            "invasive_status": is_invasive,
            "advisory_content": advisory_content,
            "confidence_score": confidence,
            "data_sources": [SOURCE],
            "generalisation_level": "exact",  # Exact match found
            "disclaimer": "Based on exact species match in database",
            "species": species_name,
            "common_name": safe_extract(fields, "Common Names"),
            "confidence_level": confidence_level,
            "risk_level": "High" if is_invasive else "Low",
            "description": safe_extract(fields, "Description"),
            "family": safe_extract(fields, "Family"),
            "origin": safe_extract(fields, "Origin"),
            "where_found": safe_extract(fields, "Where Found"),
            "treatment": safe_extract(fields, "Treatment"),
            "identification": safe_extract(fields, "Identification"),
            "poisonous": safe_extract(fields, "Poisonous"),
            "action_required": action_required,
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "data_source": SOURCE
        }
        
    else:
        # Not found in RAG system - return unknown species analysis
        analysis = {
            "classification": "unknown",
            "invasive_status": False,
            "advisory_content": {
                "species_identification": {
                    "scientific_name": species_name,
                    "common_names": "Unknown species",
                    "family": "Unknown"
                },
                "legal_status": {
                    "nemba_category": "Unknown",
                    "legal_requirements": "Manual identification required"
                },
                "physical_description": "This species was not found in our database.",
                "distribution": "Unknown",
                "control_methods": "Consult with local experts for proper identification and management.",
                "identification_guide": "Manual identification required",
                "poisonous_properties": "Unknown",
                "monitoring_requirements": "Manual identification and assessment required"
            },
            "confidence_score": confidence,
            "data_sources": [],
            "generalisation_level": "none",
            "disclaimer": "Species not found in database - manual identification required",
            "species": species_name,
            "common_name": "Unknown species",
            "confidence_level": "Unknown species detected",
            "is_invasive": False,
            "risk_level": "Unknown",
            "description": "This species was not found in our database. Manual identification required.",
            "family": "Unknown",
            "origin": "Unknown",
            "where_found": "Unknown",
            "treatment": "Consult with local experts for proper identification and management.",
            "identification": "Manual identification required",
            "poisonous": "Unknown",
            "action_required": "Manual identification and assessment required",
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "data_source": "Not found in database"
        }
    
    total_time = time.time() - start_time
    print(f"LLM analysis completed in {total_time:.3f}s")
    
    # Cache the result for future use
    _analysis_cache[cache_key] = analysis.copy()  # Store a copy to avoid mutations
    
    # Limit cache size to prevent memory issues
    if len(_analysis_cache) > 500:  # Keep most recent 500 results
        # Remove oldest entries (simple FIFO)
        oldest_keys = list(_analysis_cache.keys())[:100]
        for key in oldest_keys:
            del _analysis_cache[key]
    
    return analysis

# Global LLM instance for server mode
_llm_instance = None

def get_llm_instance():
    """Get or create the global LLM instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = initialize_components()
    return _llm_instance

def main():
    """Main function for command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python llm_integration.py <mode> [species_name] [confidence] [image_path]")
        print("Modes: analyze, server")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "server":
        # Server mode - persistent service
        run_server_mode()
    elif mode == "analyze":
        # Analyze mode - single analysis
        if len(sys.argv) < 4:
            print("Usage: python llm_integration.py analyze <species_name> <confidence> [image_path]")
            sys.exit(1)
        
        species_name = sys.argv[2]
        confidence = float(sys.argv[3])
        image_path = sys.argv[4] if len(sys.argv) > 4 else None
        
        try:
            # Initialize components
            if not initialize_components():
                print(json.dumps({"error": "Failed to initialize RAG system"}))
                sys.exit(1)
            
            # Analyze the plant
            analysis = analyze_plant(species_name, confidence, image_path)
            
            if analysis:
                # Output JSON result
                print(json.dumps(analysis, indent=2))
            else:
                print(json.dumps({"error": "Analysis failed"}, indent=2))
                
        except Exception as e:
            print(json.dumps({"error": f"Analysis error: {str(e)}"}, indent=2))
            sys.exit(1)
    else:
        print("Invalid mode. Use 'analyze' or 'server'")
        sys.exit(1)

def run_server_mode():
    """Run LLM as a persistent server that processes requests from stdin."""
    try:
        # Initialize components only once
        if not initialize_components():
            print("Failed to initialize RAG system", file=sys.stderr)
            sys.exit(1)
        
        print("LLM server initialized and ready", file=sys.stderr)
        
        # Process requests from stdin
        for line in sys.stdin:
            try:
                # Parse request
                request = json.loads(line.strip())
                request_id = request.get('id')
                species_name = request.get('species')
                confidence = request.get('confidence')
                image_path = request.get('image_path')

                if not species_name or confidence is None:
                    response = {"id": request_id, "error": "Missing required fields: species, confidence"}
                else:
                    # Analyze the plant
                    analysis = analyze_plant(species_name, confidence, image_path)
                    if analysis:
                        response = {"id": request_id, "success": True, "analysis": analysis}
                    else:
                        response = {"id": request_id, "error": "Analysis failed"}

                # Send response with immediate flush for faster communication
                print(json.dumps(response))
                sys.stdout.flush()
                
            except json.JSONDecodeError:
                response = {"error": "Invalid JSON request"}
                print(json.dumps(response))
                sys.stdout.flush()
            except Exception as e:
                response = {"error": f"Processing error: {str(e)}"}
                print(json.dumps(response))
                sys.stdout.flush()
                
    except KeyboardInterrupt:
        print("LLM server shutting down", file=sys.stderr)
    except Exception as e:
        print(f"LLM server error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Cleanup on exit
        try:
            if client:
                client.reset()
        except:
            pass

if __name__ == "__main__":
    main() 