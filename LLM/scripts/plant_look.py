import argparse, json, difflib, re
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_GLOBS = (
    "**/plant_chunks*.json",
    "**/chunks*/plant_chunks*.json",
)

def discover_jsons(dirs: List[str] | None) -> List[Path]:
    roots = [Path(d) for d in (dirs or ["."])]
    found: List[Path] = []
    for r in roots:
        for pat in DEFAULT_GLOBS:
            found.extend(r.glob(pat))
    seen, uniq = set(), []
    for p in found:
        if p.exists() and p.suffix == ".json" and p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq

def infer_source_label(p: Path) -> str:
    for part in reversed(p.parts):
        if part.startswith("chunks_"):
            return part.replace("chunks_", "")
    return p.parent.name

def load_records(paths: List[Path]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            data = json.loads(p.read_text(encoding="latin-1"))
        label = infer_source_label(p)
        for rec in data:
            rec = dict(rec)
            rec["_source_label"] = label
            rec["_source_path"]  = str(p)
            out.append(rec)
    return out

SPACE_RE = re.compile(r"\s+")
def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = SPACE_RE.sub(" ", s).strip()
    return s

def candidate_strings(rec: Dict[str, Any]) -> List[str]:
    strs = []
    sci = rec.get("scientific_name") or ""
    strs.append(norm(sci))
    for cn in rec.get("common_names") or []:
        strs.append(norm(cn))
    return strs

def score_one(query: str, candidate: str) -> float:
    if not candidate:
        return 0.0
    base = difflib.SequenceMatcher(a=query, b=candidate).ratio() * 100.0
    bonus = 0.0
    if candidate.startswith(query):
        bonus += 20.0
    if query in candidate:
        bonus += 10.0
    q_tokens = set(query.split())
    c_tokens = set(candidate.split())
    if q_tokens:
        overlap = len(q_tokens & c_tokens) / len(q_tokens) * 100.0
        base = max(base, overlap)
    return min(base + bonus, 100.0)

def best_score_for_record(query: str, rec: Dict[str, Any]) -> float:
    cand = candidate_strings(rec)
    return max((score_one(query, c) for c in cand), default=0.0)

def search_grouped_by_source(records, query, k_sources=3, min_score=30.0):
    """Return the single best record per source (book)."""
    q = norm(query)
    best_per_src = {}
    for r in records:
        src = r.get("_source_label") or r.get("_source_path") or "?"
        s = best_score_for_record(q, r)
        if s < min_score:
            continue
        prev = best_per_src.get(src)
        if (prev is None) or (s > prev[0]):
            best_per_src[src] = (s, r)
    ranked = sorted(best_per_src.values(), key=lambda t: (-t[0], (t[1].get("scientific_name") or "")))
    return [r for _, r in ranked][:k_sources]

def search(records: List[Dict[str, Any]], query: str, k: int = 3) -> List[Dict[str, Any]]:
    q = norm(query)
    scored = [(best_score_for_record(q, r), r) for r in records]
    scored.sort(key=lambda t: (-t[0], (r := t[1]).get("scientific_name") or ""))
    top = [r for s, r in scored[:k] if s >= 40.0]
    return top

# -------------- pretty print --------------
def preview(txt: str | None, n: int = 180) -> str:
    if not isinstance(txt, str) or not txt.strip():
        return "—"
    t = re.sub(r"\s+", " ", txt.strip())
    return t if len(t) <= n else (t[:n-1] + "…")

def print_card(rec: Dict[str, Any], i: int) -> None:
    sci = rec.get("scientific_name") or "?"
    cns = ", ".join(rec.get("common_names") or []) or "—"
    fam = rec.get("family") or "—"
    src = rec.get("_source_label") or "?"
    page = rec.get("page_hint")
    print(f"\n[{i}] {sci}  [{src}{f' · p.{page}' if page else ''}]")
    print(f"    Form         : {rec.get('form') or '—'}")
    print(f"    Common names : {cns}")
    print(f"    Family       : {fam}")
    print(f"    Origin       : {preview(rec.get('origin'))}")
    print(f"    Where found  : {preview(rec.get('where_found'))}")
    print(f"    Identification: {preview(rec.get('identification'))}")
    if rec.get("treatment"):
        print(f"    Treatment    : {preview(rec.get('treatment'), 140)}")
    print(f"    Poisonous    : {preview(rec.get('poisonous'), 140)}")
    print(f"    Irritant     : {preview(rec.get('irritant'), 140)}")

# -------------- cli --------------
def main():
    ap = argparse.ArgumentParser(description="Quick plant lookup across multiple JSONs.")
    ap.add_argument("--json", nargs="*", help="Explicit JSON paths.")
    ap.add_argument("--dirs", nargs="*", help="Folders to search (default: current tree).")
    ap.add_argument("--once", help="Run a single query and exit.")
    ap.add_argument("-k", type=int, default=3, help="How many matches to show (default 3).")
    args = ap.parse_args()

    paths: List[Path] = []
    if args.json:
        paths = [Path(p) for p in args.json]
    else:
        paths = discover_jsons(args.dirs)

    if not paths:
        print("No JSON files found. Pass --json or point --dirs at your folders.")
        return

    records = load_records(paths)
    print(f"Loaded {len(records)} records from {len(paths)} files.")
    print("Files:")
    for p in paths:
        print(f"  - {p}")

    def run_query(q: str):
        hits = search_grouped_by_source(records, q, k_sources=args.k)
        if not hits:
            print("No good matches.")
        else:
            for i, r in enumerate(hits, 1):
                print_card(r, i)

    if args.once:
        run_query(args.once)
        return

    print("\nType a plant name (scientific or common). Enter to repeat, or 'q' to quit.")
    last = ""
    while True:
        try:
            q = input("\nsearch> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            q = last
        if not q:
            continue
        last = q
        run_query(q)

if __name__ == "__main__":
    main()