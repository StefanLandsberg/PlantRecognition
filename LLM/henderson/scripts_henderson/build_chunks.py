import json, re 
from pathlib import Path 
from typing import Optional, Dict, List, Tuple 
from collections import defaultdict, Counter

PAGES_DIR = Path("LLM/henderson/extracted_text_henderson/pages") 
OUT_JSON = Path("LLM/henderson/chunks_henderson/plant_chunks_henderson.json") 
OUT_DBG = Path("LLM/henderson/extracted_text_henderson/parse_debug_henderson.json") 

BINOMIAL_RE = re.compile(
    r"""\b(
        [A-Z][a-z-]{2,}
        \s+(?:[×x]\s+)?                       
        (?:
            (?=[A-Za-z-]*[a-z])[A-Za-z-]{2,}
            (?:\s+(?:subsp\.|ssp\.|var\.)\s+(?=[A-Za-z-]*[a-z])[A-Za-z-]{2,})?
          | s?pp?\.?
        )
        (?:\s+[‘’']\s*[^‘’']+?\s*[’']\s*)?
    )\b""",
    re.VERBOSE
)

INVISIBLES_RE = re.compile(r"[\u00AD\u200B\u200C\u200D\u2060\uFEFF]") 
NBSP_RE = re.compile(r"\u00A0")                                  
DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014]")   

def _canon_name(name: str) -> str:
    """Normalize scientific names for dedupe keys."""
    n = name.strip()
    n = n.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    n = re.sub(r"\s*[×x]\s*", " x ", n)
    n = re.sub(r"\s+", " ", n)
    return n.lower()

#  Skipping these two specifically because they are just photos, not infomation blocks
SKIP_SPECIES_CANON = {
    _canon_name("Oenothera biennis"),
    _canon_name("Oenothera glazioviana"),
}

# Adding this one is ONLY becasue it is the only common name with a 'human' name
HUMAN_LIKE_COMMON_ALLOW = {
    "Port Jackson",
}

# Adding in for that ONE plant that refuses to listen
HEADER_CONT_TOKENS = re.compile(
    r"(?i)\b("
    r"and other|seed[- ]producing|species|hybrids?|non[- ]indigenous|non[- ]native|"
    r"nonindigenous|nonnative|complex|aggregate|agg\.|group|sensu|to south africa"
    r")\b"
)
GENUS_FAMILY_OVERRIDES = {
    "Lolium": "Poaceae",
    "Cabomba": "Cabombaceae",
    "Nymphaea": "Nymphaeaceae",
    "Pistia": "Araceae",
    "Sagittaria": "Alismataceae",
    "Colocasia": "Araceae",
    "Echinodorus": "Alismataceae",
    "Ludwigia": "Onagraceae",
    "Marsilea": "Marsileaceae",
    "Nymphoides": "Menyanthaceae",
    "Callisia": "Commelinaceae",
    "Houttuynia": "Saururaceae",
    "Asphodelus": "Asphodelaceae",
    "Centranthus": "Caprifoliaceae",
    "Equisetum": "Equisetaceae",
    "Flaveria": "Asteraceae",
    "Lepidium": "Brassicaceae",
    "Malva": "Malvaceae",
    "Nicandra": "Solanaceae",
    "Salvia": "Lamiaceae",
    "Solanum": "Solanaceae",
    "Stachytarpheta": "Verbenaceae",
    "Verbascum": "Scrophulariaceae",
    "Convolvulus": "Convolvulaceae",
    "Epipremnum": "Araceae",
    "Tropaeolum": "Tropaeolaceae",
    "Furcraea": "Agavaceae",
    "Ulex": "Fabaceae",
    "Rosa": "Rosaceae",
    "Ficus": "Moraceae",
    "Quercus": "Fagaceae",
    "Ardisia": "Primulaceae",
    "Buddleja": "Buddlejaceae",
    "Cotoneaster": "Rosaceae",
    "Kunzea": "Myrtaceae",
    "Pittosporum": "Pittosporaceae",
    "Cinnamomum": "Lauraceae",
    "Clusia": "Clusiaceae",
    "Melaleuca": "Myrtaceae",
    "Reynoutria": "Polygonaceae",
    "Wigandia": "Boraginaceae",
    "Tecoma": "Bignoniaceae",
    "Fraxinus": "Oleaceae",
    "Mahonia": "Berberidaceae",
    "Rhus": "Anacardiaceae",
    "Styphnolobium": "Fabaceae",
    "Asparagopsis": "Florideophyceae",
    "Phlebodium": "Polypodiaceae",
}

# ---- normalization ---- 
def _normalize_keep_lines(text: str) -> str: 
    t = text
    t = NBSP_RE.sub(" ", t)
    t = INVISIBLES_RE.sub("", t)
    t = DASH_RE.sub("-", t) 
    t = t.replace("<<COLBREAK>>", "\n\n")
    t = re.sub(r"-\s*\n\s*", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    return t.strip()

def _normalize_flat(text: str) -> str: 
    t = _normalize_keep_lines(text) 
    t = t.replace("\n", " ") 
    t = re.sub(r"\s{2,}", " ", t) 
    t = re.sub(r"(?i)\b(Description|Leaves?|Inflorescen[cs]e|Fruits?|Origin|Invades)\s*:\s*(?=\S)", r"\1: ", t) 
    return t 

# ----------------------- COMMON NAMES -----------------------
COMMON_NAME_SEP_RE = re.compile(r"\s*(?:,|/| or )\s+", re.IGNORECASE)
HAS_MEASURE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m)\b", re.IGNORECASE)

def extract_common_names(sect_text: str, sci_name: Optional[str]) -> List[str]:
    names, _ = extract_common_names_debug(sect_text, sci_name)
    return names

def extract_common_names_debug(sect_text: str, sci_name: Optional[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    dbg: List[Dict[str, str]] = []
    if not sect_text:
        return [], dbg

    txt = _normalize_keep_lines(sect_text)
    lines = [ln.strip() for ln in txt.splitlines()]

    header_i = None
    if sci_name:
        sci_pat = re.escape(_clean_scientific(sci_name))
        header_re = re.compile(rf"^\s*{sci_pat}\b", re.IGNORECASE)
        for i, ln in enumerate(lines):
            if header_re.search(ln):
                header_i = i
                break
    if header_i is None:
        for i, ln in enumerate(lines):
            if ln:
                header_i = i
                break
    if header_i is None:
        dbg.append({"line": "", "decision": "abort", "reason": "no_header_found"})
        return [], dbg
    
    for j in range(header_i + 1, min(header_i + 10, len(lines))):
        cand = lines[j]
        if HEADER_CONT_TOKENS.search(cand):
            dbg.append({"line": cand, "decision": "skip", "reason": "header_continuation"})
            continue
        if not cand:
            dbg.append({"line": "", "decision": "skip", "reason": "blank"})
            continue

        cand_clean = _clean_scientific(cand)
        if _looks_like_scientific(cand_clean):
            dbg.append({"line": cand, "decision": "skip", "reason": "scientific_name"})
            continue

        if (
            ":" in cand
            or FAMILY_BANNER_RE.search(cand)
            or ICON_LINE_RE.match(cand)
            or SKIP_LINE_RE.match(cand)
            or ">>" in cand
        ):
            dbg.append({"line": cand, "decision": "break", "reason": "label_or_banner_or_icon"})
            break

        if _looks_like_human_name(cand) and cand not in HUMAN_LIKE_COMMON_ALLOW:
            dbg.append({"line": cand, "decision": "skip", "reason": "human_name"})
            continue
        if cand.endswith("."):
            dbg.append({"line": cand, "decision": "skip", "reason": "ends_with_period"})
            continue
        if not re.search(r"[a-z]", cand):
            dbg.append({"line": cand, "decision": "skip", "reason": "no_lowercase"})
            continue
        if ";" in cand or HAS_MEASURE.search(cand) or len(cand) > 80:
            dbg.append({"line": cand, "decision": "skip", "reason": "too_measurey_or_long"})
            continue

        parts = [p.strip(" \t;") for p in COMMON_NAME_SEP_RE.split(cand) if p.strip(" \t;")]
        seen = set()
        out = []
        for p in parts:
            k = p.lower()
            if k not in seen:
                seen.add(k)
                out.append(p)
        dbg.append({"line": cand, "decision": "accept", "reason": "first_viable"})
        return out, dbg

    dbg.append({"line": "", "decision": "none_found", "reason": "no_viable_line_before_labels"})
    return [], dbg

PAGE_PATCHES = {
    110: [
        ("regex", r"(?m)^\s*Solidago altissima\s*&\s*S\.\s*gigantea\s*$", "Solidago altissima"),
        ("drop",  r"(?m)^\s*Jessi\s+Kalwij\s*$"),
    ],
}

def _apply_page_patches(text: str, page_hint: Optional[int]) -> str:
    ops = PAGE_PATCHES.get(page_hint)
    if not ops:
        return text
    
    for op in ops:
        kind = op[0]
        if kind == "regex":
            pattern, repl = op[1], op[2]
            text = re.sub(pattern, repl, text)
        elif kind == "drop":
            pattern = op[1]
            text = re.sub(pattern, "", text)
        elif kind == "keep":
            continue
        else:
            raise ValueError(f"Unknown patch kind: {kind}")
    return text
# ------------------------------------------------------------

# ----------------------- FAMILY NAMES -----------------------
FAMILY_INLINE_RE = re.compile(r"(?i)\b([A-Z][a-z-]*aceae)\b")

def _normalize_family_name(fam: str) -> str:
    s = fam.strip().lower()
    return s.capitalize() if s.endswith("aceae") else fam.strip()

def extract_family(sect_text: str, sci_name: Optional[str]) -> Optional[str]:
    fam, _ = extract_family_debug(sect_text, sci_name)
    return fam

def extract_family_debug(sect_text: str, sci_name: Optional[str]) -> Tuple[Optional[str], List[Dict[str, str]]]:
    dbg: List[Dict[str, str]] = []
    if not sect_text:
        return None, dbg

    txt = _normalize_keep_lines(sect_text)
    lines = [ln.strip() for ln in txt.splitlines()]

    header_i = None
    if sci_name:
        sci_pat = re.escape(_clean_scientific(sci_name))
        header_re = re.compile(rf"^\s*{sci_pat}\b", re.IGNORECASE)
        for i, ln in enumerate(lines):
            if header_re.search(ln):
                header_i = i
                break
    if header_i is None:
        for i, ln in enumerate(lines):
            if ln:
                header_i = i
                break
    if header_i is None:
        dbg.append({"line": "", "decision": "abort", "reason": "no_header_found"})
        return None, dbg

    for j in range(header_i + 1, min(header_i + 12, len(lines))):
        cand = lines[j]
        if not cand:
            dbg.append({"line": "", "decision": "skip", "reason": "blank"})
            continue

        if (":" in cand or ICON_LINE_RE.match(cand) or SKIP_LINE_RE.match(cand)):
            dbg.append({"line": cand, "decision": "break", "reason": "label_or_heading"})
            break

        m = FAMILY_BANNER_RE.fullmatch(cand)
        if m:
            fam = _normalize_family_name(m.group(0))
            dbg.append({"line": cand, "decision": "accept", "reason": "banner_fullmatch"})
            return fam, dbg

        m2 = FAMILY_INLINE_RE.search(cand)
        if m2:
            fam = _normalize_family_name(m2.group(1))
            dbg.append({"line": cand, "decision": "accept", "reason": "inline_match"})
            return fam, dbg

    post = "\n".join(lines[header_i + 1 :])
    m3 = FAMILY_BANNER_RE.search(post)
    if m3:
        fam = _normalize_family_name(m3.group(0))
        dbg.append({"line": m3.group(0), "decision": "accept", "reason": "banner_fallback_anywhere"})
        return fam, dbg

    m4 = FAMILY_INLINE_RE.search(post)
    if m4:
        fam = _normalize_family_name(m4.group(1))
        dbg.append({"line": m4.group(1), "decision": "accept", "reason": "inline_fallback_anywhere"})
        return fam, dbg

    dbg.append({"line": "", "decision": "none_found", "reason": "no_family_seen"})
    return None, dbg
# ------------------------------------------------------------

# ----------------------- POISONUS OR IRRITANT NAMES -----------------------
_SENT_END = r"\.(?=\s|$)"
_VAL_SENT_RE = r":\s*(.+?)" + _SENT_END

IRR_POIS_COMBINED_RE = re.compile(rf"(?im)^\s*(?:poisonous|irritant)\s*&\s*(?:poisonous|irritant)\s*{_VAL_SENT_RE}")
IRR_LINE_RE = re.compile(rf"(?im)^\s*irritant\s*{_VAL_SENT_RE}")
POIS_LINE_RE = re.compile(rf"(?im)^\s*poisonous\s*{_VAL_SENT_RE}")
POIS_Q_RE = re.compile(r"(?im)^\s*poisonous\?\s*$")

IRR_POIS_COMBINED_ANY_RE = re.compile(rf"(?is)\b(?:poisonous|irritant)\s*&\s*(?:poisonous|irritant)\s*{_VAL_SENT_RE}")
IRR_ANY_RE = re.compile(rf"(?is)\birritant\s*{_VAL_SENT_RE}")
POIS_ANY_RE = re.compile(rf"(?is)\bpoisonous\s*{_VAL_SENT_RE}")
POIS_Q_ANY_RE = re.compile(r"(?is)\bpoisonous\?(?!\w)")

IRR_BARE_LINE_RE = re.compile(r"(?im)^\s*irritant\.\s*$")
POIS_BARE_LINE_RE = re.compile(r"(?im)^\s*poisonous\.\s*$")

IRR_BARE_ANY_RE = re.compile(r"(?is)\birritant\.(?=\s|$)")
POIS_BARE_ANY_RE = re.compile(r"(?is)\bpoisonous\.(?=\s|$)")

def _clean_toxic_val(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.split(r"(?i)\b(?:nb:|compare with|see(?: also)?:|photos?:)", s)[0].strip()
    s = re.sub(r"\b[A-Z]{3,}ACEAE\b.*$", "", s).strip()
    if ICON_LINE_RE.match(s):
        s = ""
    return s

def extract_irritant_poisonous(sect_text: str) -> Tuple[Optional[str], Optional[str]]:
    irr, pois, _ = extract_irritant_poisonous_debug(sect_text)
    return irr, pois

def extract_irritant_poisonous_debug(sect_text: str) -> Tuple[Optional[str], Optional[str], List[Dict[str, str]]]:
    dbg: List[Dict[str, str]] = []
    if not sect_text:
        return None, None, dbg

    txt = _normalize_keep_lines(sect_text)
    lines = [ln.strip() for ln in txt.splitlines()]

    # 1) Combined label on its own line
    for ln in lines:
        m = IRR_POIS_COMBINED_RE.match(ln)
        if m:
            raw = _clean_toxic_val(m.group(1))
            val = (raw + ".") if raw and not raw.endswith(".") else raw
            dbg.append({"line": ln, "decision": "accept_both", "reason": "combined_label"})
            return val, val, dbg

    # 2) Individual labels on their own lines
    irritant: Optional[str] = None
    for ln in lines:
        mi = IRR_LINE_RE.match(ln)
        if mi:
            raw = _clean_toxic_val(mi.group(1))
            if raw:
                irritant = (raw + ".") if not raw.endswith(".") else raw
                dbg.append({"line": ln, "decision": "accept", "reason": "irritant_line"})
                break
        elif 'IRR_BARE_LINE_RE' in globals() and IRR_BARE_LINE_RE.match(ln):
            irritant = "Irritant."
            dbg.append({"line": ln, "decision": "accept", "reason": "irritant_bare_line"})
            break

    # 2b) Poisonous on its own line — scan bottom-up so *last* wins
    poisonous: Optional[str] = None
    for ln in reversed(lines):
        mp = POIS_LINE_RE.match(ln)
        if mp:
            raw = _clean_toxic_val(mp.group(1))
            if raw:
                poisonous = (raw + ".") if not raw.endswith(".") else raw
                dbg.append({"line": ln, "decision": "accept", "reason": "poisonous_line_last"})
                break
        if 'POIS_BARE_LINE_RE' in globals() and POIS_BARE_LINE_RE.match(ln):
            poisonous = "Poisonous."
            dbg.append({"line": ln, "decision": "accept", "reason": "poisonous_bare_line_last"})
            break
        mq = POIS_Q_RE.match(ln)
        if mq:
            poisonous = "Poisonous?"
            dbg.append({"line": ln, "decision": "accept", "reason": "poisonous_question_line_last"})
            break

    # 3) Fallback: anywhere in the section — use the LAST match found
    if irritant is None or poisonous is None:
        full = "\n".join(lines)

        if irritant is None and poisonous is None:
            m = None
            for _m in IRR_POIS_COMBINED_ANY_RE.finditer(full):
                m = _m
            if m:
                raw = _clean_toxic_val(m.group(1))
                val = (raw + ".") if raw and not raw.endswith(".") else raw
                dbg.append({"line": m.group(0)[:120], "decision": "accept_both", "reason": "combined_fallback_anywhere"})
                return val, val, dbg

        if irritant is None:
            mi = None
            for _m in IRR_ANY_RE.finditer(full):
                mi = _m
            if mi:
                raw = _clean_toxic_val(mi.group(1))
                if raw:
                    irritant = (raw + ".") if not raw.endswith(".") else raw
                    dbg.append({"line": mi.group(0)[:120], "decision": "accept", "reason": "irritant_fallback_anywhere"})
            elif 'IRR_BARE_ANY_RE' in globals():
                last_bare_i = None
                for _m in IRR_BARE_ANY_RE.finditer(full):
                    last_bare_i = _m
                if last_bare_i:
                    irritant = "Irritant."
                    dbg.append({"line": last_bare_i.group(0)[:120], "decision": "accept", "reason": "irritant_bare_anywhere"})

        if poisonous is None:
            last_pos = -1
            last_val = None

            # with-value
            for _m in POIS_ANY_RE.finditer(full):
                raw = _clean_toxic_val(_m.group(1))
                if raw and _m.start() >= last_pos:
                    last_pos = _m.start()
                    last_val = (raw + ".") if not raw.endswith(".") else raw

            # bare
            for _m in POIS_BARE_ANY_RE.finditer(full):
                if _m.start() >= last_pos:
                    last_pos = _m.start()
                    last_val = "Poisonous."

            # question
            for _m in POIS_Q_ANY_RE.finditer(full):
                if _m.start() >= last_pos:
                    last_pos = _m.start()
                    last_val = "Poisonous?"

            if last_val:
                poisonous = last_val
                dbg.append({"line": full[max(0, last_pos-40): last_pos+40],
                            "decision": "accept", "reason": "poisonous_last_anywhere"})
        
    if irritant is None and poisonous is None:
        dbg.append({"line": "", "decision": "none_found", "reason": "no_irritant_or_poisonous"})
    return irritant, poisonous, dbg


TOX_LABEL_VAL_ANY_RE = re.compile(rf"(?is)\b(irritant|poisonous)\s*{_VAL_SENT_RE}")
COMBINED_ANY_RE      = re.compile(rf"(?is)\b(?:poisonous|irritant)\s*&\s*(?:poisonous|irritant)\s*{_VAL_SENT_RE}")

def _extract_toxic_tokens_anywhere(text: str) -> List[Tuple[str, str]]:
    """
    Return a flat, ordered list of ('irritant'|'poisonous', value) tokens.
    Combined 'Irritant & Poisonous:' emits *two* tokens with the same value.
    """
    events: List[Tuple[int, Tuple[str, str]]] = []
    tnorm = _normalize_keep_lines(text)

    for m in COMBINED_ANY_RE.finditer(tnorm):
        raw = _clean_toxic_val(m.group(1))
        val = (raw + ".") if raw and not raw.endswith(".") else raw
        pos = m.start()
        events.append((pos, ("irritant",  val)))
        events.append((pos, ("poisonous", val)))

    for m in TOX_LABEL_VAL_ANY_RE.finditer(tnorm):
        label = m.group(1).lower()
        raw = _clean_toxic_val(m.group(2))
        val = (raw + ".") if raw and not raw.endswith(".") else raw
        events.append((m.start(), (label, val)))

    for m in POIS_Q_ANY_RE.finditer(tnorm):
        events.append((m.start(), ("poisonous", "Poisonous?")))

    for m in POIS_BARE_ANY_RE.finditer(tnorm):
        events.append((m.start(), ("poisonous", "Poisonous.")))
    for m in IRR_BARE_ANY_RE.finditer(tnorm):
        events.append((m.start(), ("irritant", "Irritant.")))

    events.sort(key=lambda x: x[0])
    return [ev for _, ev in events]
# --------------------------------------------------------------------------

# ----------------------- ORIGIN -------------------------------------------
ORIGIN_OVERRIDES: Dict[Tuple[str, int], str] = {
    (_canon_name("Austrocylindropuntia subulata"), 144): "Peru.",
    (_canon_name("Harrisia pomanensis"), 152): "Argentina, Paraguay, Bolivia.",
    (_canon_name("Opuntia engelmannii"), 156): "S USA, Mexico.",
    (_canon_name("Casuarina equisetifolia"), 170): "Australasia and Pacific Isles.",
    (_canon_name("Parkinsonia aculeata"), 188): "S USA and Mexico to Argentina.",
    (_canon_name("Pyracantha crenulata"), 192): "Himalayas to SW China.",
    (_canon_name("Rubus immixtus"), 200): "Possibly European or a hybrid with the indigenous R. rigidus.",
    (_canon_name("Solanum sisymbriifolium"), 202): "S America.",
    (_canon_name("Salix fragilis"), 230): "W Europe, Asia.",
    (_canon_name("Cinnamomum camphora"), 246): "China, Taiwan and Japan.",
    (_canon_name("Cotoneaster glaucophyllus"), 248): "S China, Vietnam.",

}

NATIVE_ANY_RE  = re.compile(r"(?is)\bnative\s+to\s+([^.]+)\.(?=\s|$)")
LABELS_FOR_KV = r"(?:origin|invades|where\s+found|potentially\s+invasive|invasive\s+status|cultivated\s+for|fruits?|irritant|poisonous|uses|notes|identification)"
KV_ANY_RE = re.compile(rf"(?is)\b({LABELS_FOR_KV})\s*:\s*(.+?)(?=(?:\s*\b{LABELS_FOR_KV}\s*:)|$)")

FAM_BANNER_TAIL_RE = re.compile(r"(?i)\b[A-Z]{3,}ACEAE\b.*$")
ICON_TAIL_RE = re.compile(r"(?:\s*\b(?:POACEAE|[A-Z]{1,2}|Pt)\b)+\s*$")
def _clean_kv_val(s: str) -> str:
    s = s.strip()
    s = _strip_photo_credits(s)
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.split(r"(?i)\b(?:nb:|compare with|see(?: also)?:|photos?:)", s)[0].strip()
    return s

def _clean_origin_val(s: str) -> str:
    s = _clean_kv_val(s)
    # Hard stop at the first sentence; origin is always 1 sentence in this book so far i see.
    m = re.search(r"\.(?=\s|$)", s)
    if m:
        s = s[: m.end()]
    s = FAM_BANNER_TAIL_RE.sub("", s).strip()
    s = ICON_TAIL_RE.sub("", s).strip()
    if s and not s.endswith("."):
        s += "."
    return s

ORIGIN_SENT_ANY_RE = re.compile(r"(?is)\borigin\s*:\s*(.+?)\.(?=\s|$)")

def _extract_origin_tokens_anywhere(text: str) -> List[str]:
    t = _normalize_keep_lines(text)
    events = []
    for m in ORIGIN_SENT_ANY_RE.finditer(t):
        val = _clean_origin_val(m.group(1))
        if val:
            events.append((m.start(), val))
    events.sort(key=lambda x: x[0])
    return [v for _, v in events]

def _extract_kv_pairs_anywhere(text: str) -> List[Tuple[str, str]]:
    t = _normalize_keep_lines(text)
    pairs: List[Tuple[str, str]] = []
    for m in KV_ANY_RE.finditer(t):
        key = m.group(1).lower()
        val = _clean_kv_val(m.group(2))
        if val:
            pairs.append((key, val))
    return pairs

def extract_origin(sect_text: str) -> Optional[str]:
    v, _ = extract_origin_debug(sect_text)
    return v

def extract_origin_debug(sect_text: str) -> Tuple[Optional[str], List[Dict[str, str]]]:
    dbg: List[Dict[str, str]] = []
    if not sect_text:
        return None, dbg

    # Prefer KV chunker; take the FIRST 'Origin' in THIS section
    pairs = _extract_kv_pairs_anywhere(sect_text)
    for k, v in pairs:
        if k == "origin":
            out = _clean_origin_val(v)
            dbg.append({"line": out[:120], "decision": "accept", "reason": "kv_first_sentence"})
            return out, dbg

    # Fallback: sentence match anywhere in the section
    txt = _normalize_keep_lines(sect_text)
    first = None
    for m in ORIGIN_SENT_ANY_RE.finditer(txt):
        first = m
        break
    if first:
        out = _clean_origin_val(first.group(1))
        dbg.append({"line": out[:120], "decision": "accept", "reason": "origin_sentence_anywhere"})
        return out, dbg

    dbg.append({"line": "", "decision": "none_found", "reason": "no_origin"})
    return None, dbg
# --------------------------------------------------------------------------

# ----------------------- WHERE FOUND -------------------------------------------
WHERE_FOUND_OVERRIDES: Dict[Tuple[str, int], str] = {
    (_canon_name("Cryptostegia madagascariensis"), 118): "Savanna, riverbanks.",
    (_canon_name("Passiflora subpeltata"), 126): "Woodland, bush clumps, roadsides, riverbanks.",
    (_canon_name("Furcraea foetida"), 150): "Coastal bush, rocky sites, ravines.",
    (_canon_name("Harrisia pomanensis"), 152): "Dry savanna, pastoral land.",
    (_canon_name("Trichocereus spachianus"), 164): "Dry savanna, karoo.",
    (_canon_name("Cylindropuntia fulgida var. fulgida"), 148): "Dry savanna, karoo.",
    (_canon_name("Asparagopsis armata"), 56): "Subtidal zones to depths of 30 m, occasionally in deeper pools; attaching to rocky substrates or other algae or floating freely (EC, WC).",
    (_canon_name("Asparagopsis taxiformis"), 56): "Subtidal zones to depths of 30 m, favouring reef edges with constant water motion; attaching to rocky substrates or other algae or floating freely (KZN, WC).",
}
def _clean_where_val(s: str) -> str:
    s = _clean_kv_val(s)
    s = FAM_BANNER_TAIL_RE.sub("", s).strip()
    s = ICON_TAIL_RE.sub("", s).strip()
    s = re.sub(r"\s{2,}", " ", s)
    if s and not re.search(r"[.!?]$", s):
        s += "."
    return s

def extract_where_found(sect_text: str) -> Optional[str]:
    v, _ = extract_where_found_debug(sect_text)
    return v

def extract_where_found_debug(sect_text: str) -> Tuple[Optional[str], List[Dict[str, str]]]:
    dbg: List[Dict[str, str]] = []
    if not sect_text:
        return None, dbg

    # --- First pass: KV pairs from the section ---
    pairs = _extract_kv_pairs_anywhere(sect_text)
    kv = {k: v for k, v in pairs}

    # 1) Prefer Invades:
    if "invades" in kv:
        out = _clean_where_val(kv["invades"])
        dbg.append({"line": out[:120], "decision": "accept", "reason": "kv_invades"})
        return out, dbg

    # 2) Invasive status:
    if "invasive status" in kv:
        if re.search(r"(?i)\bundetermined\b", kv["invasive status"]):
            dbg.append({"line": "Undetermined", "decision": "accept", "reason": "kv_invasive_status_undetermined"})
            return "Undetermined", dbg

    # 3) Potentially invasive (but skip NB species lists):
    if "potentially invasive" in kv:
        base = _clean_where_val(kv["potentially invasive"])
        if len(GENUS_ABBR_RE.findall(base)) < 2:
            out = f"(Potentially found) {base}"
            dbg.append({"line": out[:120], "decision": "accept", "reason": "kv_potentially_invasive"})
            return out, dbg

    # --- Second pass:direct fallbacks (robust to KV misses / line breaks) ---
    txt = _normalize_keep_lines(sect_text)

    m = INVADES_SENT_ANY_RE.search(txt)
    if m:
        out = _clean_where_val(m.group(1))
        dbg.append({"line": out[:120], "decision": "accept", "reason": "fallback_invades"})
        return out, dbg

    m = INV_STATUS_ANY_RE.search(txt)
    if m and re.search(r"(?i)\bundetermined\b", m.group(1)):
        dbg.append({"line": "Undetermined", "decision": "accept", "reason": "fallback_invasive_status_undetermined"})
        return "Undetermined", dbg

    m = POT_INV_SENT_ANY_RE.search(txt)
    if m:
        base = _clean_where_val(m.group(1))
        if len(GENUS_ABBR_RE.findall(base)) < 2:
            out = f"(Potentially found) {base}"
            dbg.append({"line": out[:120], "decision": "accept", "reason": "fallback_potentially_invasive"})
            return out, dbg

    dbg.append({"line": "", "decision": "none_found", "reason": "no_where_found"})
    return None, dbg

def _extract_where_tokens_anywhere(text: str) -> List[str]:
    t = _normalize_keep_lines(text)
    vals: List[str] = []
    for k, v in _extract_kv_pairs_anywhere(t):
        if k in ("invades", "where found"):
            cleaned = _clean_where_val(v)
            if cleaned:
                vals.append(cleaned)
        elif k == "potentially invasive":
            cleaned = _clean_where_val(v)
            if cleaned:
                vals.append(f"(Potentially found) {cleaned}")
        elif k == "invasive status":
            if re.search(r"(?i)\bundetermined\b", v):
                vals.append("Undetermined")
    return vals

def _extract_where_tokens_anywhere(text: str) -> List[Tuple[str, str]]:
    t = _normalize_keep_lines(text)
    tokens: List[Tuple[str, str]] = []
    for k, v in _extract_kv_pairs_anywhere(t):
        if k in ("invades", "where found"):
            cleaned = _clean_where_val(v)
            if cleaned:
                tokens.append(("invades", cleaned))
        elif k == "potentially invasive":
            cleaned = _clean_where_val(v)
            if cleaned and len(GENUS_ABBR_RE.findall(cleaned)) < 2:
                tokens.append(("potential", f"(Potentially found) {cleaned}"))
        elif k == "invasive status":
            if re.search(r"(?i)\bundetermined\b", v):
                tokens.append(("status", "Undetermined"))
    return tokens

POTENTIALLY_INV_RE = re.compile(r"(?i)^\s*potentially\s*invasive\s*[:.-]?\s*(.*)$")
GENUS_ABBR_RE = re.compile(r"\b[A-Z]\.\s*[a-z-]{3,}\b")

INVADES_SENT_ANY_RE = re.compile(rf"(?is)\binvades\s*:\s*(.+?)(?=(?:\s*\b{LABELS_FOR_KV}\s*:)|$)")
POT_INV_SENT_ANY_RE = re.compile(rf"(?is)\bpotentially\s+invasive\s*:\s*(.+?)(?=(?:\s*\b{LABELS_FOR_KV}\s*:)|$)")
INV_STATUS_ANY_RE = re.compile(rf"(?is)\binvasive\s+status\s*:\s*(.+?)(?=(?:\s*\b{LABELS_FOR_KV}\s*:)|$)")

def _ensure_terminal_punct(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s{2,}", " ", s)
    if s and not re.search(r'[.!?]"?$', s):
        s += "."
    return s

def apply_where_found_special_cases(records: List[Dict], debug: Optional[List[Dict]] = None) -> None:
    """
    - If where_found begins with 'Potentially invasive:', rewrite to:
      '(Potentially found) {rest...}'
    - If scientific_name == 'Nephrolepis exaltata', force where_found = 'Undetermined'
      (book uses 'Invasive status: Undetermined.').
    """
    changed = 0
    for r in records:
        # 1) Potentially invasive -> (Potentially found) ...
        wf = r.get("where_found")
        if isinstance(wf, str) and wf.strip():
            m = POTENTIALLY_INV_RE.match(wf.strip())
            if m:
                body = _clean_kv_val(m.group(1))
                body = FAM_BANNER_TAIL_RE.sub("", body).strip()
                body = ICON_TAIL_RE.sub("", body).strip()
                body = _ensure_terminal_punct(body)
                new_wf = f"(Potentially found) {body}"
                if new_wf != wf:
                    r["where_found"] = new_wf
                    changed += 1
                    if debug is not None:
                        debug.append({
                            "file": "POST",
                            "page_hint": r.get("page_hint"),
                            "scientific_name_extracted": r.get("scientific_name"),
                            "note": "WHERE_FOUND_REWRITE_POTENTIALLY",
                            "old": wf,
                            "new": new_wf
                        })

        # 2) Special one-off: Nephrolepis exaltata -> 'Undetermined'
        sci = (r.get("scientific_name") or "").strip()
        if sci and _canon_name(sci) == _canon_name("Nephrolepis exaltata"):
            if r.get("where_found") != "Undetermined":
                r["where_found"] = "Undetermined"
                changed += 1
                if debug is not None:
                    debug.append({
                        "file": "POST",
                        "page_hint": r.get("page_hint"),
                        "scientific_name_extracted": r.get("scientific_name"),
                        "note": "WHERE_FOUND_FORCE_UNDETERMINED"
                    })
    if debug is not None:
        debug.append({"file": "POST", "note": "WHERE_FOUND_POSTPASS_DONE", "changed": changed})
# -------------------------------------------------------------------------------

# ----------------------- IDENTIFICATION-------------------------------------------
ID_LABELS_FOR_STOP = rf"(?:{LABELS_FOR_KV}|description|leaves?|flowers?|fruits?|inflorescen[cs]e)"
ID_ANY_RE = re.compile(rf"(?is)\b(description|leaves?|flowers?|fruits?|inflorescen[cs]e)\s*:\s*(.+?)(?=(?:\s*\b{ID_LABELS_FOR_STOP}\s*:)|$)")

CREDIT_NAMES = [
    r"Geoff\s+Nichols",
    r"Dan[’']sile\s+Cindi",
    r"Peter\s+Shisani",
    r"Stefan\s+Neser",
]
PHOTO_CREDIT_LINE_RE = re.compile(rf"(?im)^\s*(?:{'|'.join(CREDIT_NAMES)})\s*\.{{0,3}}\s*$")
PHOTO_CREDIT_TAIL_RE = re.compile(rf"(?i)\s*(?:[–—-]\s*)?(?:by\s+)?(?:{'|'.join(CREDIT_NAMES)})\s*\.{{0,3}}\s*$")

FRUITS_SENT_ANY_RE = re.compile(r"(?is)\bfruits\s*:\s*(.+?)\.(?=\s|$)")
FRUITS_CLAUSE_IN_ID_RE = re.compile(r"(?is)\bFruits\s*:\s*.+?(?:\.|$)")

def _squash_newlines(s: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s+([,;:.!?])", r"\1", s)
    return s.strip()

def _extract_fruits_tokens_anywhere(text: str) -> List[str]:
    t = _normalize_keep_lines(text)
    tokens: List[str] = []
    for m in FRUITS_SENT_ANY_RE.finditer(t):
        val = _clean_kv_val(m.group(1))
        clause = _ensure_terminal_punct(f"Fruits: {val}")
        tokens.append(_squash_newlines(clause))  
    return tokens

def _strip_photo_credits(s: str) -> str:
    s = re.sub(PHOTO_CREDIT_LINE_RE, "", s)
    return s

def _clean_ident_val(s: str) -> str:
    s = _clean_kv_val(s)
    s = FAM_BANNER_TAIL_RE.sub("", s).strip()
    s = ICON_TAIL_RE.sub("", s).strip()
    s = re.sub(r"\s{2,}", " ", s)
    if s and not re.search(r"[.!?]$", s):
        s += "."
    return s

_ID_LABEL_CANON = {
    "description": "Description",
    "leaf": "Leaves",
    "leaves": "Leaves",
    "flowers": "Flowers",
    "inflorescence": "Inflorescence",
    "inflorescense": "Inflorescence",  # just in case
    "fruits": "Fruits",
    "fruit": "Fruits",
}

def extract_identification(sect_text: str) -> Optional[str]:
    ident, _ = extract_identification_debug(sect_text)
    return ident

def extract_identification_debug(sect_text: str) -> Tuple[Optional[str], List[Dict[str, str]]]:
    dbg: List[Dict[str, str]] = []
    if not sect_text:
        return None, dbg

    txt = _normalize_keep_lines(sect_text)
    chunks: Dict[str, str] = {}
    order_seen: List[str] = []

    for m in ID_ANY_RE.finditer(txt):
        raw_label = m.group(1).lower()
        norm_label = _ID_LABEL_CANON.get(raw_label, raw_label.capitalize())
        val = _clean_ident_val(m.group(2))
        if not val:
            continue

        if norm_label not in chunks:
            chunks[norm_label] = val
            order_seen.append(norm_label)
            dbg.append({"line": f"{norm_label}: {val[:120]}", "decision": "accept", "reason": "id_label_slice"})

    if not chunks:
        dbg.append({"line": "", "decision": "none_found", "reason": "no_id_labels"})
        return None, dbg

    preferred = ["Description", "Leaves", "Flowers", "Inflorescence", "Fruits"]
    final_order = [lbl for lbl in preferred if lbl in chunks] or order_seen

    parts = [f"{lbl}: {chunks[lbl]}" for lbl in final_order]
    out = " ".join(parts).strip()
    return (out if out else None), dbg
# -------------------------------------------------------------------------------

# ----------------------- TREATMENT/ BIOCONTROL -------------------------------------------
BIO_PAGES = range(340, 360)  # inclusive start, exclusive end
BIO_DEGREE_RE = re.compile(
    r"""(?mx)^\s*
        ([A-Z][a-z-]{2,}\s+[a-z-]{3,}
           (?:\s+(?:subsp\.|ssp\.|var\.)\s+[a-z-]{3,})?
           (?:\s+\([^)]+\))?
        )\s*,\s*
        (Complete|Substantial|Moderate|Negligible|Not\s+determined|Unknown)\b
    """
)

def _canon_species_key(s: str) -> str:
    s = _clean_scientific(s)
    toks = s.split()
    return " ".join(toks[:2]).lower() if len(toks) >= 2 else s.lower()

def build_biocontrol_index(pages_dir: Path) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for p, text, hint in _iter_pages(pages_dir):
        if hint is None or hint not in BIO_PAGES:
            continue
        t = _normalize_keep_lines(text)
        for m in BIO_DEGREE_RE.finditer(t):
            sci_raw, degree = m.group(1), m.group(2)
            sci = re.sub(r"\s+\([^)]+\)\s*$", "", _clean_scientific(sci_raw))
            idx[_canon_species_key(sci)] = degree.title().replace("Not Determined", "Not determined")

    return idx

BIO_TREATMENT_TEXT = {
    "Complete": "Biological control: Complete — effective agents established; control is complete and the species is no longer invasive.",
    "Substantial": "Biological control: Substantial — effective agents established; species still invasive in places, other methods often still needed.",
    "Moderate": "Biological control: Moderate — agents present; additional control required.",
    "Negligible": "Biological control: Negligible — agents released but ineffective; rely on mechanical/chemical control.",
    "Not determined": "Biological control: Not determined — programme under evaluation; use mechanical/chemical control as appropriate.",
    "Unknown": "Biological control: Not determined — programme under evaluation; use mechanical/chemical control as appropriate.",
}


GENERIC_CONTROL_SUFFIX = (" Use integrated control (mechanical + chemical where registered), with follow-up and proper herbicide stewardship.")
GENERIC_CONTROL_ONLY = GENERIC_CONTROL_SUFFIX.strip()
# -----------------------------------------------------------------------------------------

AMP = re.compile(r"\s*(?:&|and)\s*", re.IGNORECASE)
ABBREV_GENUS = re.compile(r"^[A-Z]\.$")

def _split_ampersand_names(name: str) -> List[str]:
    """
    'Cuscuta campestris & C. suaveolens' ->
        ['Cuscuta campestris', 'Cuscuta suaveolens']
    'Passiflora tarminiana & P. tripartita var. mollissima' ->
        ['Passiflora tarminiana', 'Passiflora tripartita var. mollissima']
    Otherwise returns [name].
    """
    parts = AMP.split(name)
    if len(parts) == 1:
        return [name]

    first = parts[0].strip()
    tokens = first.split()
    if not tokens:
        return [name]
    genus = tokens[0]

    expanded = [first]
    for p in parts[1:]:
        p = p.strip()
        if not p:
            continue
        ptoks = p.split()
        if ptoks and ABBREV_GENUS.match(ptoks[0]):
            p = " ".join([genus] + ptoks[1:])
        expanded.append(p)
    return expanded

# ---- single-name extractor (fallback) ---- 
COMMON_NOUN_2ND = {
    "grass","lily","vine","dodder","pear","guava","raspberry","rubber","poplar",
    "ginger","cactus","bramble","pumpkin","berry","bramble","pear","prickly","pear",
    "hollygrape", "grass", "bluegrass", "grass", "fish", "swordplant", "primrosebush",
    "tumbleweed", "daisy", "shot", "ageratum", "periwinkle", "loosestrife", "pokeweed",
    "cherry", "ivy", "agave", "hemp", "broom", "pine", "from", "tamarisk", "usneoides",
    "gorse", "thorn", "firethorn", "dewberry", "blackberry", "barberry", "sunflower",
    "maple", "oak", "blackwood", "cestrum", "cheesewood", "sagewood", "cotoneaster", "myrtle",
    "privet", "laurel", "brush-cherry", "elm", "elder", "senna",
}

FUNCTION_WORDS = {
    "de","da","del","van","von","la","le","du",
    "and","with","for","of","to","in","on","by","from","into",
    "without","under","over","between","within"
}

def _is_common_noun_header(name: str) -> bool:
    toks = _clean_scientific(name).split()
    if len(toks) < 2:
        return False
    return toks[1].lower() in COMMON_NOUN_2ND

def _looks_like_scientific(name: str) -> bool:
    n = _clean_scientific(name)
    m = BINOMIAL_RE.fullmatch(n)
    if not m:
        return False

    toks = n.split()
    if len(toks) < 2:
        return False

    second_orig = toks[1]
    second = second_orig.lower()

    if (len(second) < 3 or second in FUNCTION_WORDS or second in COMMON_NOUN_2ND or
        second_orig.isupper() or TITLE_WORD.match(second_orig)):  # TitleCase
        return False

    if not re.fullmatch(r"[A-Za-z-]{3,}", second_orig):
        return False

    return True

def extract_scientific_name(page_text: str) -> Optional[str]: 
    t = _normalize_flat(page_text) 
    header = t.split("Description:", 1)[0] if "Description:" in t else t 
    for s in ("Grasses, Reeds & Grass-like Plants", "Trees & Shrubs", "Herbs", "POACEAE","FABACEAE","ASTERACEAE"): 
        header = header.replace(s, " ") 
    m = BINOMIAL_RE.search(header) 
    if m:
        cand = _clean_scientific(m.group(1))
        if _looks_like_scientific(cand):
            return cand
    return None

def _clean_scientific(name: str) -> str: 
    name = re.sub(r"</?i>", "", name).strip() 
    name = re.sub(r"\s+", " ", name) 
    name = re.sub(r"\bspp\.\b", "spp", name) 
    name = re.sub(r"\bsp\.\b", "sp", name) 
    return name.strip(" ,.;:") 

def make_plant_record(
    page_text: str,
    page_hint: Optional[int] = None,
    scientific_name: Optional[str] = None,
) -> Dict:
    sci = _clean_scientific(scientific_name) if scientific_name else extract_scientific_name(page_text)
    source = 'https://invasives.org.za/wp-content/uploads/2022/05/Alien-Weeds-and-Invasive-Plants-A-Complete-guide-to-declared-weeds-and-invaders-in-South-Africa-By-Lesley-Henderson.pdf'
    return {
        "scientific_name": sci,
        "common_names": [],
        "family": None,
        "origin": None,
        "where_found": None,
        "identification": None,
        "treatment": None,
        "poisonous": None,
        "irritant": None,
        "source": source,
        "page_hint": page_hint,
    }

# ---- NEW: split a page into sections by plant headers ---- # 
HEADER_LINE_RE = re.compile(
    r"""(?mx)
    ^\s*
    (?:(?:\d+|ND)\s+)?                 
    (?:\*|<i>)?
    (?P<name>
        [A-Z][a-z-]{2,}
        \s+(?:[×x]\s+)?                 
        (?:
            [a-z-]{3,}
            (?:\s+(?:subsp\.|ssp\.|var\.)\s+[a-z-]{3,})?
          | s?pp?\.?
        )
        (?:\s+[‘’']\s*[^‘’']+?\s*[’']\s*)?
    )
    (?:\s+(?:[A-Z][a-z]?\.|[a-z]{1,3}\.))*    
    (?:\s+\(.*?\))?                           
    (?:</i>|\*)?
    \s*(?:\(cont\.\))?
    \s*$
    """
)

SKIP_LINE_RE = re.compile(r"""(?x)^(?:(?-i:[A-Z]{3,})  | (?i:[A-Za-z ,&\-]+Plants))$""")
ICON_LINE_RE = re.compile(r"^(?:PE|Pt|T|S|Bt|Bc|Gc|L|N|R|V|Pb|Pc)\b.*$", re.IGNORECASE)
FAMILY_BANNER_RE = re.compile(r"\b[A-Z]{3,}ACEAE\b")

TITLE_WORD = re.compile(r"^[A-Z][a-z]+$")

def _looks_like_human_name(line: str) -> bool:
    toks = _clean_scientific(line).split()
    if 2 <= len(toks) <= 3 and all(TITLE_WORD.match(t) for t in toks):
        return True
    return False

HEADINGS_TO_SKIP = {
    "names", "photographs", "photo", "description", "uses", "notes", "leaves",
    "leaf", "inflorescence", "fruits", "origin", "invades", "where found"
}

def _heal_truncated_name(sci: Optional[str], sect_text: str) -> Optional[str]:
    if not sci:
        return None

    s = _clean_scientific(sci)
    toks = s.split()
    if len(toks) < 2:
        return s

    genus = toks[0]
    rest_tokens = toks[1:]

    lines = sect_text.splitlines()
    window = "\n".join(lines[:5])
    w = NBSP_RE.sub(" ", window)
    w = INVISIBLES_RE.sub("", w)
    w = DASH_RE.sub("-", w)
    w = re.sub(r"-\s*\n\s*", "", w)
    w = w.replace("\n", " ")

    # ---------- species healing ----------
    species = rest_tokens[0]
    if len(species) <= 8:
        pat_species = rf"\b{re.escape(genus)}\s+({re.escape(species)}[A-Za-z]{{2,}})\b"
        m = re.search(pat_species, w)
        if m:
            species_full = m.group(1)
            tail = " ".join(rest_tokens[1:])
            candidate = f"{genus} {species_full}{(' ' + tail) if tail else ''}"
            candidate = _clean_scientific(candidate)
            if _looks_like_scientific(candidate):
                return candidate

    # ---------- infraspecific healing (var./ssp./subsp.) ----------
    markers = {"var.", "ssp.", "subsp."}
    for i, tok in enumerate(rest_tokens):
        if tok in markers:
            marker = tok
            infra_prefix = rest_tokens[i+1] if i + 1 < len(rest_tokens) else ""
            if infra_prefix and len(infra_prefix) <= 8:
                species_full = rest_tokens[0]
                pat_infra = (
                    rf"\b{re.escape(genus)}\s+{re.escape(species_full)}\s+"
                    rf"{marker}\s+({re.escape(infra_prefix)}[A-Za-z]{{2,}})\b"
                )
                m2 = re.search(pat_infra, w)
                if m2:
                    infra_full = m2.group(1)
                    new_rest = " ".join(rest_tokens[:i+1] + [infra_full] + rest_tokens[i+2:])
                    candidate = _clean_scientific(f"{genus} {new_rest}")
                    if _looks_like_scientific(candidate):
                        return candidate
            break

    return s

def split_page_into_plant_sections(raw_text: str) -> List[Tuple[str, str]]:
    txt = _normalize_keep_lines(raw_text)

    headers: List[Tuple[int, int, str]] = []
    seen_starts = set()

    for m in HEADER_LINE_RE.finditer(txt):
        line = m.group(0).strip()
        low = line.lower()
        if (SKIP_LINE_RE.match(line) or ICON_LINE_RE.match(line) or
            FAMILY_BANNER_RE.search(line) or any(low.startswith(h) for h in HEADINGS_TO_SKIP)
            or _looks_like_human_name(line)):
            continue

        raw = _clean_scientific(m.group("name"))
        tail = _clean_scientific(line[m.end("name"):])
        m_tail = re.match(r"^\s*(?:&|and)\s+(.+?)\s*$", tail)
        if m_tail:
            raw = f"{raw} & {m_tail.group(1)}"

        if _canon_name(raw) in SKIP_SPECIES_CANON:
            continue
        if _is_common_noun_header(raw):
            continue

        start = m.start()
        if start in seen_starts:
            continue
        seen_starts.add(start)

        first_only = True
        for sci in _split_ampersand_names(raw):
            sci = _clean_scientific(sci)
            if first_only:
                headers.append((start, m.end(), sci))
                first_only = False

    if not headers:
        sci = extract_scientific_name(raw_text)
        return [(txt, sci)]

    headers.sort(key=lambda x: x[0])

    n = len(headers)
    group_last_idx = list(range(n))
    i = 0
    while i < n:
        j = i
        while j + 1 < n:
            between = txt[headers[j][1]:headers[j+1][0]]
            if between.strip() == "":
                j += 1
            else:
                break
        for k in range(i, j + 1):
            group_last_idx[k] = j
        i = j + 1

    # build sections
    sections: List[Tuple[str, str]] = []
    for idx, (start, _endline, sci) in enumerate(headers):
        last = group_last_idx[idx]
        end = headers[last + 1][0] if (last + 1) < n else len(txt)
        section_text = txt[start:end].strip()
        sections.append((section_text, sci))
    return sections

# ---- file IO helpers ----
def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1") 
    
def _page_hint_from_name(path: Path) -> Optional[int]: 
    m = re.findall(r"(\d+)", path.stem) 
    return int(m[-1]) if m else None 

def _iter_pages(pages_dir: Path) -> List[Tuple[Path, str, Optional[int]]]: 
    files = sorted([p for p in pages_dir.rglob("*") if p.suffix.lower() in {".txt", ".md"}]) 
    return [(p, _read_text(p), _page_hint_from_name(p)) for p in files] 

def main():
    pages = _iter_pages(PAGES_DIR)
    records: List[Dict] = []
    debug: List[Dict] = []
    skipped_null = 0
    bio_index = build_biocontrol_index(PAGES_DIR)
    
    START_PAGE = 22
    END_PAGE = 306

    for path, text, hint in pages:
        if hint is None:
            continue
        if hint < START_PAGE:
            continue
        if hint > END_PAGE:
            continue
        if hint == 93:
            continue
        if hint == 194:
            continue

        text = _apply_page_patches(text, hint)

        sections = split_page_into_plant_sections(text)
        if not isinstance(sections, list) or not sections:
            sections = [(_normalize_keep_lines(text), extract_scientific_name(text))]

        seen_page = set()
        page_bucket = []
        bruhhh = []
        for idx, (sect_text, sci_guess) in enumerate(sections):
            rec = make_plant_record(sect_text, page_hint=hint, scientific_name=sci_guess)

            # 1) validate / alt-extract
            if not rec["scientific_name"] or not _looks_like_scientific(rec["scientific_name"]):
                alt = extract_scientific_name(sect_text)
                rec["scientific_name"] = alt if (alt and _looks_like_scientific(alt)) else rec["scientific_name"]

            # 2) heal truncated epithets
            if rec["scientific_name"]:
                healed = _heal_truncated_name(rec["scientific_name"], sect_text)
                if healed and _looks_like_scientific(healed):
                    rec["scientific_name"] = healed

            # 3) still invalid? log + skip
            if not rec["scientific_name"] or not _looks_like_scientific(rec["scientific_name"]):
                skipped_null += 1
                debug.append({
                    "file": str(path),
                    "page_hint": hint,
                    "section_index": idx,
                    "scientific_name_header": sci_guess,
                    "scientific_name_extracted": rec["scientific_name"],
                    "section_preview": sect_text[:220] + ("..." if len(sect_text) > 220 else ""),
                    "note": "NO_NAME_IN_SECTION_OR_TRUNCATED"
                })
                continue

            # 4) DEDUPE
            key = (hint, _canon_name(rec["scientific_name"]))
            if key in seen_page:
                debug.append({
                    "file": str(path),
                    "page_hint": hint,
                    "section_index": idx,
                    "scientific_name_header": sci_guess,
                    "scientific_name_extracted": rec["scientific_name"],
                    "section_preview": sect_text[:220] + ("..." if len(sect_text) > 220 else ""),
                    "note": "DUPLICATE_ON_PAGE_SKIPPED"
                })
                continue
            seen_page.add(key)

            names, cn_dbg = extract_common_names_debug(sect_text, rec["scientific_name"])
            if names:
                rec["common_names"] = names

            fam, fam_dbg = extract_family_debug(sect_text, rec["scientific_name"])
            if fam:
                rec["family"] = fam

            orig, orig_dbg = extract_origin_debug(sect_text)
            if orig:
                rec["origin"] = orig

            wf, wf_dbg = extract_where_found_debug(sect_text)
            if wf:
                rec["where_found"] = wf

            ident, ident_dbg = extract_identification_debug(sect_text)
            if ident:
                rec["identification"] = ident

            irr, pois, tox_dbg = extract_irritant_poisonous_debug(sect_text)
            if irr:
                rec["irritant"] = irr
            if pois:
                rec["poisonous"] = pois

            debug.append({
                "file": str(path),
                "page_hint": hint,
                "section_index": idx,
                "scientific_name_header": sci_guess,
                "scientific_name_extracted": rec["scientific_name"],
                "section_preview": sect_text[:220] + ("..." if len(sect_text) > 220 else ""),
                "note": None,
                "common_name_debug": cn_dbg,
                "common_names_found": names,
                "family_debug": fam_dbg,
                "family_found": fam,
                "origin_debug": orig_dbg,
                "origin_found": orig,
                "where_found_debug": wf_dbg,
                "where_found_found": wf,
                "identification_debug": ident_dbg,
                "identification_found": ident,
                "toxic_debug": tox_dbg,
                "irritant_found": irr,
                "poisonous_found": pois
            })

            records.append(rec)
            debug.append({
                "file": str(path),
                "page_hint": hint,
                "section_index": idx,
                "scientific_name_header": sci_guess,
                "scientific_name_extracted": rec["scientific_name"],
                "section_preview": sect_text[:220] + ("..." if len(sect_text) > 220 else ""),
                "note": None
            })
            page_bucket.append(rec)
            bruhhh.append(sect_text)

        non_null_fams = [r.get("family") for r in page_bucket if r.get("family")]
        unique_fams = {f for f in non_null_fams if f}
        if len(unique_fams) == 1:
            fam = next(iter(unique_fams))
            for r in page_bucket:
                if not r.get("family"):
                    r["family"] = fam
        else:
            for i, r in enumerate(page_bucket):
                if r.get("family"):
                    continue
                neighbours = []
                if i > 0 and page_bucket[i-1].get("family"):
                    neighbours.append(page_bucket[i-1]["family"])
                if i + 1 < len(page_bucket) and page_bucket[i+1].get("family"):
                    neighbours.append(page_bucket[i+1]["family"])
                neighbours = [f for f in neighbours if f]
                if neighbours and all(f == neighbours[0] for f in neighbours):
                    r["family"] = neighbours[0]

        if any((not r.get("irritant") or not r.get("poisonous")) for r in page_bucket):
            if "<<COLBREAK>>" in text:
                tail_text = text.split("<<COLBREAK>>")[-1]
            else:
                n = max(1, int(0.25 * len(text)))
                tail_text = text[-n:]

            tokens = _extract_toxic_tokens_anywhere(tail_text)
            if tokens:
                idx_cursor = 0
                for label, value in tokens:
                    if not value:
                        continue
                    i = idx_cursor
                    while i < len(page_bucket) and page_bucket[i].get(label):
                        i += 1

                    while i < len(page_bucket):
                        if page_bucket[i].get(label):
                            i += 1
                            continue

                        if (label == "irritant" and "glochids" in value.lower() and re.search(r"glochids[^.\n]{0,80}\babsent\b", _normalize_keep_lines(bruhhh[i]), re.I)):
                            i += 1
                            continue

                        page_bucket[i][label] = value
                        idx_cursor = i
                        break

        if any(not r.get("origin") for r in page_bucket):
            if "<<COLBREAK>>" in text:
                tail_text = text.split("<<COLBREAK>>")[-1]
            else:
                n = max(1, int(0.25 * len(text)))
                tail_text = text[-n:]

            origin_vals = _extract_origin_tokens_anywhere(tail_text)
            if origin_vals:
                idx_cursor = 0
                for val in origin_vals:
                    i = idx_cursor
                    while i < len(page_bucket) and page_bucket[i].get("origin"):
                        i += 1
                    if i < len(page_bucket):
                        page_bucket[i]["origin"] = val
                        idx_cursor = i
        
        for r in page_bucket:
            sci_key = _canon_name(r.get("scientific_name") or "")
            override = ORIGIN_OVERRIDES.get((sci_key, hint))
            if override:
                r["origin"] = override

        for r in page_bucket:
            sci_key = _canon_name(r.get("scientific_name") or "")
            wf_override = WHERE_FOUND_OVERRIDES.get((sci_key, hint))
            if wf_override:
                r["where_found"] = wf_override
                debug.append({
                    "file": str(path),
                    "page_hint": hint,
                    "section_index": None,
                    "scientific_name_extracted": r.get("scientific_name"),
                    "note": "WHERE_FOUND_OVERRIDE_APPLIED",
                    "where_found": wf_override,
                })

        if any(not r.get("where_found") for r in page_bucket):
            if "<<COLBREAK>>" in text:
                tail_text = text.split("<<COLBREAK>>")[-1]
            else:
                n = max(1, int(0.25 * len(text)))
                tail_text = text[-n:]

            tokens = _extract_where_tokens_anywhere(tail_text)
            if tokens:
                pri = {"invades": 0, "status": 1, "potential": 2}
                tokens.sort(key=lambda kv: pri.get(kv[0], 9))
                idx_cursor = 0
                for kind, val in tokens:
                    if not val:
                        continue
                    i = idx_cursor
                    while i < len(page_bucket) and page_bucket[i].get("where_found"):
                        i += 1
                    if i < len(page_bucket):
                        page_bucket[i]["where_found"] = val
                        idx_cursor = i

        try:
            if "<<COLBREAK>>" in text:
                tail_text = text.split("<<COLBREAK>>")[-1]
            else:
                n = max(1, int(0.25 * len(text)))
                tail_text = text[-n:]

            fruit_tokens = _extract_fruits_tokens_anywhere(tail_text)
            if fruit_tokens:
                j = 0
                for r in page_bucket:
                    if j >= len(fruit_tokens):
                        break
                    ident = (r.get("identification") or "").strip()
                    if ident:
                        if FRUITS_CLAUSE_IN_ID_RE.search(ident):
                            ident = FRUITS_CLAUSE_IN_ID_RE.sub(fruit_tokens[j], ident)
                        else:
                            ident = _ensure_terminal_punct(ident) + " " + fruit_tokens[j]
                    else:
                        ident = fruit_tokens[j]
                    r["identification"] = ident.strip()
                    j += 1
        except Exception as _e:
            debug.append({
                "file": str(path),
                "page_hint": hint,
                "note": f"FRUITS_TAIL_POSTPASS_ERROR: {_e}"
            })

    # ---------- MOVE GLOBAL DEDUPE HERE (AFTER the loops) ----------
    kept_by_name: Dict[str, Dict] = {}
    kept_from_page: Dict[str, int] = {}
    globally_dropped = 0
    new_debug = []

    for r in records:
        gkey = _canon_name(r["scientific_name"])
        pg = r.get("page_hint")

        if gkey not in kept_by_name:
            kept_by_name[gkey] = r
            kept_from_page[gkey] = (pg if isinstance(pg, int) else 10**9)
            continue

        existing_page = kept_from_page[gkey]
        current_page = (pg if isinstance(pg, int) else 10**9)

        if current_page < existing_page:
            new_debug.append({
                "file": "GLOBAL",
                "page_hint": current_page,
                "section_index": None,
                "scientific_name_header": r["scientific_name"],
                "scientific_name_extracted": r["scientific_name"],
                "section_preview": "",
                "note": f"DUPLICATE_GLOBAL_REPLACED; kept earlier page {current_page} over {existing_page}"
            })
            kept_by_name[gkey] = r
            kept_from_page[gkey] = current_page
            globally_dropped += 1
        else:
            new_debug.append({
                "file": "GLOBAL",
                "page_hint": current_page,
                "section_index": None,
                "scientific_name_header": r["scientific_name"],
                "scientific_name_extracted": r["scientific_name"],
                "section_preview": "",
                "note": f"DUPLICATE_GLOBAL_SKIPPED; kept page {existing_page}"
            })
            globally_dropped += 1

    records = list(kept_by_name.values())
    debug.extend(new_debug)
    print(f"Global dedupe removed {globally_dropped} duplicates; final unique taxa: {len(records)}")
    # ---------------------------------------------------------------

    # ---- apply biocontrol-based treatment text ----
    applied = 0
    for r in records:
        sci = r.get("scientific_name") or ""
        key = _canon_species_key(sci)
        degree = bio_index.get(key)
        if degree:
            base = BIO_TREATMENT_TEXT.get(degree)
            if base:
                r["treatment"] = base + GENERIC_CONTROL_SUFFIX
                applied += 1
        elif not r.get("treatment"):
            # fallback: generic control guidance even when the book has no degree
            r["treatment"] = GENERIC_CONTROL_ONLY
        # optional: print(f"Treatment filled from biocontrol table for {applied} taxa")
    
    TEXT_FIELDS = ("identification", "where_found", "origin", "poisonous", "irritant", "treatment")
    for r in records:
        for f in TEXT_FIELDS:
            if isinstance(r.get(f), str):
                r[f] = _squash_newlines(r[f])

    def _majority_family(c: Counter, min_count=2, margin=1):
        if not c:
            return None
        mc = c.most_common()
        top_fam, n1 = mc[0]
        n2 = mc[1][1] if len(mc) > 1 else 0
        # need enough votes AND lead over #2
        if n1 >= min_count and n1 >= n2 + margin:
            return top_fam
        return None

    # 1) learn genus -> family from already-set families
    genus_counts = defaultdict(Counter)
    for r in records:
        fam = r.get("family")
        sci = r.get("scientific_name") or ""
        genus = _clean_scientific(sci).split()[0] if sci else None
        if fam and genus:
            genus_counts[genus][_normalize_family_name(fam)] += 1

    auto_overrides = {}
    for g, cnt in genus_counts.items():
        mfam = _majority_family(cnt, min_count=2, margin=1)
        if mfam:
            auto_overrides[g] = mfam

    # 2) apply: manual override wins; otherwise auto-fill/repair
    for r in records:
        sci = r.get("scientific_name") or ""
        genus = _clean_scientific(sci).split()[0] if sci else None
        if not genus:
            continue
        # manual always wins
        if genus in GENUS_FAMILY_OVERRIDES:
            r["family"] = GENUS_FAMILY_OVERRIDES[genus]
            continue
        # otherwise fill/repair using learned majority
        if genus in auto_overrides:
            inferred = auto_overrides[genus]
            if (not r.get("family")) or (r["family"] and r["family"] not in genus_counts[genus]):
                r["family"] = inferred

    apply_where_found_special_cases(records, debug)
    TEXT_FIELDS = ("identification", "where_found", "origin", "poisonous", "irritant", "treatment")
    for r in records:
        for f in TEXT_FIELDS:
            if isinstance(r.get(f), str):
                r[f] = _squash_newlines(r[f])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_DBG.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_DBG.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

    have = sum(1 for r in records if r["scientific_name"])
    print(f"Sections: {len(records) + skipped_null}; with names: {have}; null skipped: {skipped_null}")

if __name__ == "__main__": 
    main()