import re, json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

PAGES_DIR = Path("LLM/extracted_text/pages")
OUT_JSON = Path("LLM/chunks/plant_chunks.json")
OUT_DBG = Path("LLM/extracted_text/parse_debug.json")

# ---------------- constants / regex ----------------
COLBREAK = "<<COLBREAK>>"

LABEL_START = re.compile(r"^(Family|Common names?|Origin|Where found|Treatment|Identification|Not to be confused with)\b", re.I)
STOP_LABELS = (
    "Family", "Common names", "Common name", "Origin", "Where found", "Treatment", "Identification",
    "Not to be confused with", "Tree", "Shrub", "Grass", "Herb", "Climber", "Creeper", "Vine", "Reed", "Aquatic", "Palm"
)

HEADER_LABEL_RE = re.compile(r"Family\s*:", re.I)

# binomial patterns
# Relaxed "line" matcher: allows leading number/ND, trailing category, optional "(cont.)"
BINOMIAL_LINE = re.compile(
    r"^\s*(?:\d+|ND)?\s*([A-Z][a-z]{2,})\s+(?:x\s+)?([a-z][a-z-]{3,})\s*(?:\((?:cont|cont\.)\))?\s*(?:1[abc]|2|3|ND)?\s*$",
    re.I
)
BINOMIAL_ANY = re.compile(r"\b([A-Z][a-z]{2,})\s+(?:x\s+)?([a-z][a-z-]{3,})\b")

# things that must NOT be the species epithet
BAD_SECOND = {
    "and","or","but","not","nature","reserve","river","highveld","south","north","east","west","province","weeks",
    "meters","metres","garden","labels","figure","treatment","where","found","names","common","methodology","removal",
    "individual","species","how","are","described","origin","family","grass","reed","tree","trees","shrub","vine","aquatic",
    "pods","flowers","flower","leaf","leaves","bark","young","adult","old","long","over","suckering","all"
}
BAD_CONTEXT = re.compile(r"Not to be confused|Similar species", re.I)

# ---------------- helpers ----------------
def normalize(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text, flags=re.I)          # join hyphen breaks
    text = re.sub(r"\(([a-z]|[0-9]+)\)", "", text)                # drop (a)/(9) etc.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def header_region(full: str) -> str:
    """
    Return only the portion between the LAST <<COLBREAK>> and the first 'Family:'.
    This isolates the right-hand header band and avoids left-column prose.
    """
    m = HEADER_LABEL_RE.search(full)
    if not m:
        return ""
    start = 0
    last_col = full.rfind(COLBREAK, 0, m.start())
    if last_col != -1:
        start = last_col + len(COLBREAK)
    region = full[start:m.start()]
    # keep it small-ish
    lines = [ln for ln in region.splitlines() if ln.strip()]
    return "\n".join(lines[-20:])  # just in case of huge regions

def _clean_line(s: str) -> str:
    return s.strip()

def _looks_like_binomial(name: str) -> bool:
    parts = name.split()
    if len(parts) != 2: return False
    g, sp = parts
    if not (g[:1].isupper() and g[1:].islower()): return False
    if not re.fullmatch(r"[a-z-]{3,}", sp): return False
    if sp in BAD_SECOND: return False
    return True

def header_genus_hint(text: str) -> Optional[str]:
    reg = header_region(text)
    mm = re.search(r"\b([A-Z][a-z]{2,})\b", reg)
    return mm.group(1) if mm else None

def grab_block(label: str, text: str) -> Optional[str]:
    """
    Grab everything after a label up to the next major label, a column break,
    a blank gap, or EOF. Accepts '(h) Where found?' etc.
    """
    stop = "|".join(re.escape(s) for s in STOP_LABELS)
    pat = re.compile(
        rf"(?mis)^\s*(?:\([a-z]\)\s*)?{re.escape(label)}\s*\??\s*[:\-]?\s*(.+?)"
        rf"(?=\n\s*(?:{stop})\s*[:\-]?|\n\s*{re.escape(COLBREAK)}|\n\n|\Z)"
    )
    m = pat.search(text)
    return m.group(1).strip() if m else None

def description_from_origin(text: str) -> Optional[str]:
    """
    Prefer the block after 'Origin' up to Where found / Not to be confused / Treatment / blank gap.
    """
    pat = re.compile(
        r"(?mis)^\s*Origin\s*[:\-]?\s*(?:.*?\n)?(.*?)(?=\n\s*(?:\(h\)\s*)?Where found\??\s*[:\-]?|\n\s*Not to be confused|\n\s*Treatment|\n\n|\Z)"
    )
    m = pat.search(text)
    if m:
        block = m.group(1).strip()
        # trim stray label echo at the start
        block = re.sub(r"^(Family|Common names?|Tree|Shrub|Grass)\s*:.*\n", "", block, flags=re.I)
        return block if block else None
    # fallback: explicit "Description:" label
    d2 = grab_block("Description", text)
    return d2

PLANT_TYPES = {"tree","grass","shrub","herb","climber","creeper","vine","succulent","aquatic","palm","reed"}

def split_common_names(s: Optional[str]) -> Optional[List[str]]:
    if not s: return None
    s = s.replace(";", ",").replace("/", ",")
    s = re.sub(r"\s*\n\s*", ", ", s)
    parts = [p.strip(" .,-") for p in re.split(r",|\band\b", s) if p.strip()]
    parts = [p for p in parts if p.lower() not in PLANT_TYPES]
    # dedupe, keep order
    seen, out = set(), []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower()); out.append(p)
    return out or None

# ---------------- scientific name finder ----------------
def find_scientific_name(text: str) -> Optional[str]:
    reg = header_region(text)
    if not reg:
        return None

    # 1) scan header region line-by-line (strictest)
    for ln in [l for l in reg.splitlines() if l.strip()]:
        ln = _clean_line(ln)
        m = BINOMIAL_LINE.match(ln)
        if m:
            cand = f"{m.group(1)} {m.group(2)}"
            if _looks_like_binomial(cand):
                return cand

    # 2) last-resort: find first binomial *inside header region only*
    for mm in BINOMIAL_ANY.finditer(reg):
        g, sp = mm.group(1), mm.group(2)
        if sp.lower() in BAD_SECOND:
            continue
        # don't accept if sitting on a label line
        left_line = reg.rfind("\n", 0, mm.start()) + 1
        if LABEL_START.match(reg[left_line:mm.start()] or ""):
            continue
        cand = f"{g} {sp}"
        if _looks_like_binomial(cand):
            return cand

    return None

# ---------------- other fields ----------------
def find_category(text: str) -> Optional[str]:
    reg = header_region(text)
    m = re.search(r"\b(1a|1b|1c|2|3)\b", reg, re.I)
    return m.group(1).lower() if m else None

def identification_block(full_text: str) -> Optional[str]:
    pat = re.compile(
        r"(?mis)^\s*(?:Not to be confused with|Identification)\s*[:\-]?\s*(.+?)"
        r"(?=\n\s*(?:Family|Common names?|Origin|Where found|Treatment|Uses|Notes|Leaf|Habitat|Description)\s*[:\-]?|\n\s*<<COLBREAK>>|\n\n|\Z)"
    )
    m = pat.search(full_text)
    return m.group(1).strip() if m else None


def treatment_block(full_text: str) -> Optional[str]:
    pat = re.compile(
        r"(?mis)^\s*Treatment\s*[:\-]?\s*(.+?)"
        r"(?=\n\s*(?:Family|Common names?|Origin|Where found|Identification|Not to be confused with|Uses|Notes|Leaf|Habitat|Description)\s*[:\-]?|\n\s*<<COLBREAK>>|\n\n|\Z)"
    )
    m = pat.search(full_text)
    return m.group(1).strip() if m else None

def plant_type_guess(full_text: str) -> Optional[str]:
    for t in ["Grass","Tree","Shrub","Herb","Climber","Creeper","Vine","Succulent","Aquatic","Palm","Reed"]:
        if re.search(rf"\b{t}\b", full_text, re.I):
            return t
    return None

_POISON_RE = re.compile(
    r"\b(poison|poisonous|toxic|toxicity|irritant|irritation|rash|allergic|respiratory)\b",
    re.I
)

def extract_poisonous_sentences(text: str) -> Optional[str]:
    # drop column markers & collapse whitespace
    t = (text or "").replace(COLBREAK, " ")
    t = re.sub(r"\s+", " ", t)

    # split into sentences conservatively
    parts = re.split(r"(?<=[.!?])\s+", t)
    hits = [p.strip() for p in parts if _POISON_RE.search(p)]

    if not hits:
        return None

    # de-dup while preserving order
    seen, out = set(), []
    for h in hits:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
    return " ".join(out) or None

def _clean_noise(s: Optional[str]) -> Optional[str]:
    if not s: return None
    lines = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t: 
            continue
        # tiny figure/letter markers like (f), (9), a)
        if re.fullmatch(r"\(?[a-z]\)?", t, flags=re.I) or re.fullmatch(r"\(?\d{1,3}\)?", t):
            continue
        # shouty headings (ALL CAPS etc)
        if len(t) >= 12 and re.fullmatch(r"[A-Z0-9\s'&\-\(\)]+", t):
            continue
        lines.append(t)
    return "\n".join(lines).strip() or None

def _origin_head_and_rest(origin_block: str) -> tuple[Optional[str], Optional[str]]:
    """
    Return (origin_value, description_tail).

    We only take the FIRST LINE as the origin value
    (trimmed at a ' - ' dash if present). Everything else becomes description tail.
    """
    if not origin_block:
        return None, None
    lines = [ln.strip() for ln in origin_block.splitlines() if ln.strip()]
    if not lines:
        return None, None

    head = re.split(r"\s[-–]\s", lines[0], maxsplit=1)[0].strip(" .")
    tail = _clean_noise("\n".join(lines[1:]).strip())
    return (head or None), (tail or None)


# ---------------- parse one page ----------------
def parse_page(text: str, pageno: int):
    cleaned = normalize(text)

    if not HEADER_LABEL_RE.search(cleaned):
        return None

    sci = find_scientific_name(cleaned)
    if not sci or not _looks_like_binomial(sci):
        # debug & skip
        return None

    fam = grab_block("Family", cleaned)
    common_raw = grab_block("Common names", cleaned) or grab_block("Common name", cleaned)
    origin_block = grab_block("Origin", cleaned)
    if origin_block:
        origin, _ = _origin_head_and_rest(origin_block)
    else:
        origin = None

    #descr = descr_from_origin or grab_block("Description", cleaned)
    # origin = grab_block("Origin", cleaned)
    # descr = description_from_origin(cleaned)
    where = grab_block("Where found", cleaned)  # handles optional '?'
    ident = identification_block(cleaned)
    treat = treatment_block(cleaned)
    poisonous = extract_poisonous_sentences(cleaned)
    #category = find_category(cleaned)
    #ptype = plant_type_guess(cleaned)

    entry = {
        "scientific_name": sci,
        "common_names": split_common_names(common_raw),
        "family": fam,
        #"plant_type": ptype,
        #"nemba_category": category,
        "origin": origin,
        #"description": descr,
        "where_found": where,
        "identification": ident,
        "treatment": treat,
        "poisonous": poisonous,
        "page_hint": pageno
    }
    return entry

# ---------------- merge duplicates across pages ----------------
def merge_by_species(entries: List[Dict]) -> List[Dict]:
    out: Dict[str, Dict] = {}
    order: List[str] = []
    def _merge_text(a: Optional[str], b: Optional[str]) -> Optional[str]:
        if not a: return b
        if not b: return a
        if b in a: return a
        return a + "\n\n" + b

    for e in sorted(entries, key=lambda d: d["page_hint"]):
        name = e["scientific_name"]
        if name not in out:
            out[name] = {**e}
            if out[name].get("common_names") is None:
                out[name]["common_names"] = []
            order.append(name)
        else:
            cur = out[name]
            # union common names
            new_common = e.get("common_names") or []
            have = cur.get("common_names") or []
            seen = {c.lower() for c in have}
            for c in new_common:
                if c and c.lower() not in seen:
                    have.append(c)
                    seen.add(c.lower())
            cur["common_names"] = have
            # new_common = (e.get("common_names") or [])
            # have = set(cur.get("common_names") or [])
            # cur["common_names"] = list(have.union(new_common))
            # prefer first non-empty for simple fields
            # for k in ["family","origin","where_found","identification","treatment", "poisonous"]:
            #     cur[k] = _merge_text(cur.get(k), e.get(k))
            for k in ["family", "origin"]:
                cur[k] = cur.get(k) or e.get(k)

            # merge/append for these narrative fields
            for k in ["where_found", "identification", "treatment", "poisonous"]:
                cur[k] = _merge_text(cur.get(k), e.get(k))
            cur["page_hint"] = min(cur["page_hint"], e["page_hint"])
    return [out[n] for n in order]

# ---------------- runner ----------------
def main():
    pages = sorted(PAGES_DIR.glob("page_*.txt"))
    results, skipped = [], []

    for p in pages:
        txt = p.read_text(encoding="utf-8")
        page_no = int(p.stem.split("_")[1])
        entry = parse_page(txt, page_no)
        if entry:
            results.append(entry)
        else:
            prev = txt[:220].replace("\n", " ")
            skipped.append({"page": page_no, "reason": "no_species_or_bad_name", "preview": prev + ("…" if len(txt) > 220 else "")})

    merged = merge_by_species(results)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_DBG.write_text(json.dumps({"skipped_pages": skipped, "total": len(pages), "ok": len(merged)}, indent=2), encoding="utf-8")

    print(f"✅ Extracted {len(merged)} plant entries → {OUT_JSON}")
    if skipped:
        print(f"⚠️ Skipped {len(skipped)} pages → {OUT_DBG}")

if __name__ == "__main__":
    main()