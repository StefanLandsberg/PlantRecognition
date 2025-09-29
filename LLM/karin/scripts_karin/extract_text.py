import re
from pathlib import Path
import fitz  # PyMuPDF

PDF_PATH   = Path("LLM/karin/pdf_karin/AlP3Feb25_KarinSpottiswoode.pdf")
OUT_RAW_DIR = Path("LLM/karin/extracted_text_karin/pages")
OUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DEBUG  = Path("LLM/karin/extracted_text_karin/all_text_debug.txt")

# ----------------- helpers -----------------
def normalize_ligatures(s: str) -> str:
    return (s.replace("ﬁ", "fi")
             .replace("ﬂ", "fl")
             .replace("’", "'")
             .replace("“", '"')
             .replace("”", '"')
             .replace("\xa0", " "))

def clean_text(s: str) -> str:
    s = normalize_ligatures(s)
    s = re.sub(r"-\n(?=[a-z])", "", s, flags=re.I)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    s = "\n".join(
        ln for ln in s.splitlines()
        if not re.fullmatch(r"\d{1,3}", ln.strip())
    )
    return s.strip()

def block_is_text(b):
    return (len(b) >= 7 and b[6] == 0) or (b[4] and b[4].strip())

def get_blocks(page):
    blocks = [b for b in page.get_text("blocks") if block_is_text(b)]
    h = page.rect.height
    kept = []
    for (x0, y0, x1, y1, text, *_rest) in blocks:
        t = (text or "").strip()
        if re.fullmatch(r"\d{1,3}", t):
            if y1 < 0.12 * h or y0 > 0.88 * h:
                continue
        kept.append((x0, y0, x1, y1, text))
    return kept

def detect_split_x(blocks, page_width):
    """
    Find a per-page gutter by looking at x-centers of blocks and
    choosing the largest mid-range gap. Fall back to mid-page if unimodal.
    """
    if not blocks:
        return page_width * 0.5

    xs = sorted(((b[0] + b[2]) / 2.0) for b in blocks)
    if len(xs) > 10:
        lo = int(len(xs) * 0.10)
        hi = int(len(xs) * 0.90)
        xs = xs[lo:hi]

    xs = sorted(set(xs))
    if len(xs) < 2:
        return page_width * 0.5

    gaps = [(xs[i+1] - xs[i], (xs[i] + xs[i+1]) / 2.0) for i in range(len(xs)-1)]
    mid_lo, mid_hi = page_width * 0.33, page_width * 0.67
    gaps.sort(key=lambda g: ( -(g[0]), 0 if (mid_lo <= g[1] <= mid_hi) else 1 ))
    split = gaps[0][1]
    split = max(page_width * 0.30, min(page_width * 0.70, split))
    return split

def assemble_columns(blocks, page_width):
    """
    Assign blocks to left/right columns by split_x; sort each column by y0,x0;
    return left_text + '\n\n' + right_text
    """
    if not blocks:
        return ""

    split_x = detect_split_x(blocks, page_width)

    left, right = [], []
    for (x0, y0, x1, y1, text) in blocks:
        xm = (x0 + x1) / 2.0
        (left if xm <= split_x else right).append((y0, x0, x1, y1, text))

    left.sort(key=lambda t: (round(t[0], 1), t[1]))
    right.sort(key=lambda t: (round(t[0], 1), t[1]))

    left_text = clean_text("\n".join(b[4] for b in left if b[4]))
    right_text = clean_text("\n".join(b[4] for b in right if b[4]))

    combined = (left_text + "\n<<COLBREAK>>\n" + right_text).strip()
    combined = re.sub(r"\n{3,}", "\n\n", combined)
    return combined

# ----------------- main -----------------
def main():
    all_debug = []
    with fitz.open(str(PDF_PATH)) as doc:
        for i, page in enumerate(doc):
            blocks = get_blocks(page)
            combined = assemble_columns(blocks, page.rect.width)

            if len(combined) < 100:
                fallback = page.get_text("text")
                combined = clean_text(fallback)

            out_path = OUT_RAW_DIR / f"page_{i+1:03d}.txt"
            out_path.write_text(combined, encoding="utf-8")
            all_debug.append(f"\n\n===== PAGE {i+1} =====\n{combined}")

    OUT_DEBUG.write_text("".join(all_debug), encoding="utf-8")
    print(f"✅ Wrote raw page texts to {OUT_RAW_DIR} and debug to {OUT_DEBUG}")

if __name__ == "__main__":
    main()