#!/usr/bin/env python3
"""
LLM/RAG-like plant advisory using MongoDB (no ChromaDB).
- Server mode:  python llm_integration.py server
- Analyze mode: python llm_integration.py analyze <species> <confidence> [image_path]

Environment variables:
  MONGODB_URI                (required) e.g. "mongodb+srv://user:pass@cluster/db"
  MONGODB_DB                 (default: "botanica")
  MONGODB_PLANTS_COLLECTION  (default: "plants")
"""

import os
import sys
import json
import re
import time
from datetime import datetime
from typing import Optional, Dict, Any

from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.errors import PyMongoError

# ========= CONFIG =========
SOURCE = "Source: Invasive Alien & Problem Plants – Witwatersrand & Magaliesberg (Karin Spottiswoode)"
DEFAULT_DB = os.getenv("MONGODB_DB", "botanica")
DEFAULT_COLL = os.getenv("MONGODB_PLANTS_COLLECTION", "plants")
MONGO_URI = os.getenv("MONGODB_URI")  # <- must be set

# ========= GLOBALS =========
_client: Optional[MongoClient] = None
_db = None
_coll = None
_initialized = False


# ========= LOGGING (stderr) =========
def _log(msg: str):
    print(msg, file=sys.stderr)


# ========= INIT =========
def initialize_components() -> bool:
    """Initialize MongoDB connection and ensure helpful indexes."""
    global _client, _db, _coll, _initialized

    if _initialized:
        _log("LLM: already initialized")
        return True

    try:
        start = time.time()

        if not MONGO_URI:
            _log("LLM ERROR: MONGODB_URI not set")
            return False

        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
        _db = _client[DEFAULT_DB]
        _coll = _db[DEFAULT_COLL]

        # Try a quick ping
        _client.admin.command("ping")

        # Ensure basic indexes (best-effort)
        # 1) case-insensitive name search via normalized fields
        try:
            _coll.create_index([("scientific_name_lc", ASCENDING)], name="scientific_name_lc_idx", background=True)
            _coll.create_index([("common_names_lc", ASCENDING)], name="common_names_lc_idx", background=True)
        except Exception as ie:
            _log(f"LLM WARN: index creation skipped/failed: {ie}")

        elapsed = time.time() - start
        _initialized = True
        _log(f"LLM: Mongo connected (db={DEFAULT_DB}, coll={DEFAULT_COLL}) in {elapsed:.2f}s")
        return True

    except PyMongoError as e:
        _log(f"LLM ERROR: Mongo connect failed: {e}")
        return False


# ========= HELPERS =========
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _make_lc_fields(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return lowercase helper fields from a plant doc."""
    sci = _norm(doc.get("scientific_name") or doc.get("Species") or doc.get("species"))
    commons = doc.get("common_names") or doc.get("Common names") or doc.get("common_name") or ""
    if isinstance(commons, list):
        commons_lc = [c.lower().strip() for c in commons]
    else:
        # split on comma/semicolon if needed
        commons_lc = [c.strip().lower() for c in re.split(r"[,;]", commons)] if isinstance(commons, str) else []
    return {"scientific_name_lc": sci, "common_names_lc": commons_lc}


def _extract_identification(text: str) -> str:
    pattern = r"(?:Not to be confused with|Identification)[\s:]*\n*(.*?)(?=\n(?:Family|Common names|Origin|Where found|Treatment|Uses|Notes|Leaf|Habitat|Description)[\s:]*|\Z)"
    m = re.search(pattern, text or "", re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else "Not found"


def _extract_poisonous(text: str) -> str:
    poison_keywords = [
        "poison", "toxic", "irritant", "irritation", "rash", "allergic", "reaction",
        "not edible", "noxious", "harmful", "respiratory tract"
    ]
    sentences = re.split(r'(?<=[.!?])\s+', text or "")
    hits = [s.strip() for s in sentences if any(k in s.lower() for k in poison_keywords)]
    return " ".join(hits) if hits else "Not found"


def _multiline_extract(label: str, text: str) -> str:
    stop_labels = [
        "Family", "Common names", "Origin", "Where found", "Treatment",
        "Uses", "Identification", "Notes", "Leaf", "Habitat", "Description"
    ]
    stop_pattern = "|".join([rf"^\s*{l}[\s:]*" for l in stop_labels if l.lower() != label.lower()])
    pattern = rf"^\s*{label}[\s:]*\n*(.*?)(?=\n(?:{stop_pattern})|\Z)"
    m = re.search(pattern, text or "", re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else "Not found"


def _extract_fields_from_text(text: str) -> Dict[str, str]:
    text = re.sub(r"(\n\s*){2,}", "\n", (text or "").strip())

    # origin / description split
    origin_match = re.search(
        r"^\s*Origin[\s:]*\n*(.*?)(?=\n(?:^\s*(Where found|Family|Common names|Treatment)[\s:]*|\Z))",
        text, re.IGNORECASE | re.DOTALL | re.MULTILINE
    )
    if origin_match:
        origin_block = origin_match.group(1).strip()
        parts = re.split(r"[.\n]", origin_block, maxsplit=1)
        origin = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else "Not found"
    else:
        origin = "Not found"
        description = "Not found"

    return {
        "Family": _multiline_extract("Family", text),
        "Common Names": _multiline_extract("Common names", text),
        "Origin": origin,
        "Description": description,
        "Where Found": _multiline_extract("Where found", text),
        "Treatment": _multiline_extract("Treatment", text),
        "Identification": _extract_identification(text),
        "Poisonous": _extract_poisonous(text),
    }


def _safe_extract(fields: Dict[str, str], k: str, max_len: int = 350) -> str:
    val = fields.get(k, "Not found")
    val = re.sub(r"\s+", " ", (val or "").strip())
    val = re.sub(r"\(\d+\)", "", val)
    val = re.sub(r"\b\d+\b(?![a-zA-Z])", "", val)
    val = re.sub(r"\s{2,}", " ", val)
    if k == "Where Found":
        val = re.sub(r"^\?\s*", "", val)
    return val if len(val) <= max_len else val[:max_len].strip() + "..."


def _determine_nemba_category(fields: Dict[str, str]) -> str:
    t = (fields.get("Treatment") or "").lower()
    if any(x in t for x in ["immediate", "prohibited", "eradicate"]):
        return "1a"
    if any(x in t for x in ["control", "remove", "treatment"]):
        return "1b"
    if any(x in t for x in ["commercial", "restricted", "permit"]):
        return "2"
    if any(x in t for x in ["ornamental", "garden", "decorative"]):
        return "3"
    return "1b" if fields.get("Treatment") != "Not found" else "Unknown"


def _determine_invasive(fields: Dict[str, str]) -> bool:
    if fields.get("Treatment") != "Not found":
        return True
    if fields.get("Poisonous") != "Not found":
        return True
    fam = (fields.get("Family") or "").lower()
    invasive_fams = ["fabaceae", "poaceae", "asteraceae", "solanaceae"]
    return any(f in fam for f in invasive_fams)


# ========= MONGO QUERY =========
def _get_concat_text(doc: Dict[str, Any]) -> str:
    """
    Build a single text blob from doc fields (compatible with your current data).
    This lets the same extraction functions work whether content is in a single
    'text' field or split across fields.
    """
    parts = []
    for key in [
        "Family", "Common names", "Origin", "Where found", "Treatment",
        "Uses", "Notes", "Leaf", "Habitat", "Description", "Identification", "Poisonous",
        "text", "content", "body"
    ]:
        v = doc.get(key) or doc.get(key.lower().replace(" ", "_"))
        if isinstance(v, list):
            v = " ".join([str(x) for x in v])
        if isinstance(v, str) and v.strip():
            parts.append(f"{key}:\n{v.strip()}")
    return "\n".join(parts)


def _find_plant(species_name: str) -> Optional[Dict[str, Any]]:
    """
    Try in order:
      1) exact scientific_name (case-insensitive)
      2) exact in known alt keys (Species/species)
      3) any common name hit
      4) relaxed regex match on scientific_name
    """
    name_lc = _norm(species_name)
    if not _coll:
        return None

    try:
        # 1) exact scientific_name
        doc = _coll.find_one({
            "$or": [
                {"scientific_name_lc": name_lc},
                {"scientific_name": {"$regex": f"^{re.escape(species_name)}$", "$options": "i"}},
                {"Species": {"$regex": f"^{re.escape(species_name)}$", "$options": "i"}},
                {"species": {"$regex": f"^{re.escape(species_name)}$", "$options": "i"}},
            ]
        })
        if doc:
            return doc

        # 2) common name exact/contains
        doc = _coll.find_one({
            "common_names_lc": name_lc
        })
        if doc:
            return doc

        doc = _coll.find_one({
            "$or": [
                {"common_names": {"$regex": name_lc, "$options": "i"}},
                {"Common names": {"$regex": name_lc, "$options": "i"}},
                {"common_name": {"$regex": name_lc, "$options": "i"}},
            ]
        })
        if doc:
            return doc

        # 3) relaxed scientific name partial (e.g., underscores vs spaces)
        loose = name_lc.replace("_", " ").replace("-", " ")
        doc = _coll.find_one({
            "$or": [
                {"scientific_name": {"$regex": loose, "$options": "i"}},
                {"Species": {"$regex": loose, "$options": "i"}},
                {"species": {"$regex": loose, "$options": "i"}},
            ]
        })
        return doc

    except PyMongoError as e:
        _log(f"LLM ERROR: Mongo query failed: {e}")
        return None


# ========= CORE ANALYSIS =========
def analyze_plant(species_name: str, confidence: float, image_path: Optional[str] = None) -> Dict[str, Any]:
    start = time.time()

    if not species_name or not isinstance(species_name, str):
        raise ValueError("Invalid species name provided")
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        raise ValueError("Confidence must be between 0 and 1")

    # Try to find the plant doc
    doc = _find_plant(species_name)

    if doc:
        # normalize fields on the fly for old docs
        fields_blob = _get_concat_text(doc)
        fields = _extract_fields_from_text(fields_blob)

        is_inv = _determine_invasive(fields)
        nemba = _determine_nemba_category(fields)

        classification = species_name if is_inv else "unknown"

        if confidence > 0.8:
            conf_level = "High confidence detection"
            action = "Immediate action recommended" if is_inv else "Monitor for changes"
        elif confidence > 0.6:
            conf_level = "Moderate confidence detection"
            action = "Further verification recommended"
        else:
            conf_level = "Low confidence detection"
            action = "Manual verification required"

        analysis = {
            "classification": classification,
            "invasive_status": is_inv,
            "advisory_content": {
                "species_identification": {
                    "scientific_name": doc.get("scientific_name") or doc.get("Species") or species_name,
                    "common_names": _safe_extract(fields, "Common Names"),
                    "family": _safe_extract(fields, "Family"),
                },
                "legal_status": {
                    "nemba_category": nemba,
                    "legal_requirements": (
                        f"Category {nemba} species - "
                        f"{'Immediate removal required' if nemba == '1a' else 'Control measures required' if nemba == '1b' else 'Restrictions apply' if nemba == '2' else 'Monitor for spread' if nemba == '3' else 'Unknown status'}"
                    )
                },
                "physical_description": _safe_extract(fields, "Description", 500),
                "distribution": _safe_extract(fields, "Where Found"),
                "control_methods": _safe_extract(fields, "Treatment", 500),
                "identification_guide": _safe_extract(fields, "Identification"),
                "poisonous_properties": _safe_extract(fields, "Poisonous"),
                "monitoring_requirements": "Regular follow-up required for effective control" if is_inv else "No monitoring required"
            },
            "confidence_score": float(confidence),
            "data_sources": [SOURCE],
            "generalisation_level": "exact",
            "disclaimer": "Based on species record in MongoDB",
            "species": species_name,
            "common_name": _safe_extract(fields, "Common Names"),
            "confidence_level": conf_level,
            "risk_level": "High" if is_inv else "Low",
            "description": _safe_extract(fields, "Description"),
            "family": _safe_extract(fields, "Family"),
            "origin": _safe_extract(fields, "Origin"),
            "where_found": _safe_extract(fields, "Where Found"),
            "treatment": _safe_extract(fields, "Treatment"),
            "identification": _safe_extract(fields, "Identification"),
            "poisonous": _safe_extract(fields, "Poisonous"),
            "action_required": action,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "image_path": image_path,
            "data_source": SOURCE
        }
    else:
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
            "confidence_score": float(confidence),
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
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "image_path": image_path,
            "data_source": "Not found in database"
        }

    _log(f"LLM: analysis complete in {time.time()-start:.3f}s (found={bool(doc)})")
    return analysis


# ========= SERVER MODE =========
def run_server_mode():
    if not initialize_components():
        _log("LLM: init failed")
        sys.exit(1)

    _log("LLM server initialized and ready")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            species_name = req.get("species")
            confidence = req.get("confidence")
            image_path = req.get("image_path")
            if not species_name or confidence is None:
                resp = {"error": "Missing required fields: species, confidence"}
            else:
                resp = {"success": True, "analysis": analyze_plant(species_name, float(confidence), image_path)}
            print(json.dumps(resp))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": f"Processing error: {str(e)}"}))
            sys.stdout.flush()


# ========= CLI =========
def main():
    if len(sys.argv) < 2:
        print("Usage: python llm_integration.py <mode> [species] [confidence] [image_path]", file=sys.stderr)
        print("Modes: analyze, server", file=sys.stderr)
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "server":
        run_server_mode()
    elif mode == "analyze":
        if len(sys.argv) < 4:
            print("Usage: python llm_integration.py analyze <species> <confidence> [image_path]", file=sys.stderr)
            sys.exit(1)
        if not initialize_components():
            print(json.dumps({"error": "Failed to initialize MongoDB"}))
            sys.exit(1)
        species = sys.argv[2]
        confidence = float(sys.argv[3])
        image_path = sys.argv[4] if len(sys.argv) > 4 else None
        try:
            analysis = analyze_plant(species, confidence, image_path)
            print(json.dumps(analysis))
        except Exception as e:
            print(json.dumps({"error": f"Analysis error: {str(e)}"}))
            sys.exit(1)
    else:
        print("Invalid mode. Use 'analyze' or 'server'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
