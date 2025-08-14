#!/usr/bin/env python3
"""
Read LLM/chunks/plant_chunks.json and upsert into MongoDB collection "plants".
Adds helper lowercase fields + extracted sections so llm_integration.py can answer
fully without any vector DB.

ENV:
  MONGODB_URI  (required)
  MONGODB_DB   (default: botanica)
  MONGODB_PLANTS_COLLECTION (default: plants)
"""

import os, json, re, sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError

SOURCE = "Invasive Alien & Problem Plants – Witwatersrand & Magaliesberg (Karin Spottiswoode)"

# ---------------- helpers copied from your llm_integration.py ----------------
def _multiline_extract(label: str, text: str) -> str:
    stop_labels = ["Family","Common names","Origin","Where found","Treatment","Uses","Identification","Notes","Leaf","Habitat","Description"]
    stop_pattern = "|".join([rf"^\s*{l}[\s:]*" for l in stop_labels if l.lower() != label.lower()])
    pattern = rf"^\s*{label}[\s:]*\n*(.*?)(?=\n(?:{stop_pattern})|\Z)"
    m = re.search(pattern, text or "", re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else "Not found"

def _extract_identification(text: str) -> str:
    pattern = r"(?:Not to be confused with|Identification)[\s:]*\n*(.*?)(?=\n(?:Family|Common names|Origin|Where found|Treatment|Uses|Notes|Leaf|Habitat|Description)[\s:]*|\Z)"
    m = re.search(pattern, text or "", re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else "Not found"

def _extract_poisonous(text: str) -> str:
    poison_keywords = ["poison","toxic","irritant","irritation","rash","allergic","reaction","not edible","noxious","harmful","respiratory tract"]
    sentences = re.split(r'(?<=[.!?])\s+', text or "")
    hits = [s.strip() for s in sentences if any(k in s.lower() for k in poison_keywords)]
    return " ".join(hits) if hits else "Not found"

def _extract_fields_from_text(text: str) -> Dict[str,str]:
    text = re.sub(r"(\n\s*){2,}", "\n", (text or "").strip())
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

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _make_lc_fields(sci: str, common_names_value: Any) -> Dict[str, Any]:
    sci_lc = _norm(sci)
    if isinstance(common_names_value, list):
        commons_lc = [c.strip().lower() for c in common_names_value if c and isinstance(c, str)]
    elif isinstance(common_names_value, str):
        commons_lc = [c.strip().lower() for c in re.split(r"[,;]", common_names_value) if c.strip()]
    else:
        commons_lc = []
    return {"scientific_name_lc": sci_lc, "common_names_lc": commons_lc}

def _looks_like_species(name: str) -> bool:
    # skip tables/headers/chapters; require at least two tokens: "Genus species"
    if not name or len(name.split()) < 2:
        return False
    bad = ["table of", "table of content", "acknowledgements", "introduction",
           "functions of this book", "what is the problem", "nemba", "regulations"]
    nl = name.lower()
    if any(b in nl for b in bad):
        return False
    # simple pattern: Capitalized genus + lowercase species
    t1, *rest = name.split()
    return t1[:1].isupper() and (rest[0][:1].islower() if rest else False)

# ---------------- ingest ----------------
def main():
    uri = os.getenv("MONGODB_URI")
    dbname = os.getenv("MONGODB_DB", "botanica")
    collname = os.getenv("MONGODB_PLANTS_COLLECTION", "plants")
    if not uri:
        print("ERROR: MONGODB_URI not set", file=sys.stderr); sys.exit(1)

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    db = client[dbname]
    coll = db[collname]

    # indexes (best-effort)
    try:
        coll.create_index([("scientific_name_lc", ASCENDING)], name="scientific_name_lc_idx", background=True)
        coll.create_index([("common_names_lc", ASCENDING)], name="common_names_lc_idx", background=True)
    except Exception as ie:
        print(f"Index warn: {ie}", file=sys.stderr)

    chunk_path = Path("LLM/chunks/plant_chunks.json")
    chunks = json.loads(chunk_path.read_text(encoding="utf-8"))

    inserted = updated = skipped = 0
    for entry in chunks:
        raw_name = (entry.get("plant") or "").strip()
        text = entry.get("text") or ""

        if not _looks_like_species(raw_name):
            skipped += 1
            continue

        fields = _extract_fields_from_text(text)

        # Save "scientific_name" exactly as in source (e.g., "Acacia dealbata")
        sci = raw_name
        common_raw = fields.get("Common Names", "Not found")
        common_list: List[str] = []
        if isinstance(common_raw, str) and common_raw != "Not found":
            common_list = [c.strip() for c in re.split(r"[,;]", common_raw) if c.strip()]

        doc = {
            "scientific_name": sci,
            "common_names": common_list,
            "source": SOURCE,
            "text": text,                # keep full chunk text
            "fields": fields,            # structured sections
            **_make_lc_fields(sci, common_list),
        }

        # Upsert by scientific_name_lc
        res = coll.update_one(
            {"scientific_name_lc": doc["scientific_name_lc"]},
            {"$set": doc},
            upsert=True
        )
        if res.matched_count == 0 and res.upserted_id is not None:
            inserted += 1
        else:
            updated += 1

    print(f"✅ Done. inserted={inserted}, updated={updated}, skipped(non-species)={skipped}")
    print(f"DB: {dbname}, Collection: {collname}")

if __name__ == "__main__":
    main()
