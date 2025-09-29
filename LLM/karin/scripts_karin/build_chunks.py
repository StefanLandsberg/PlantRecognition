import re, json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

PAGES_DIR = Path("LLM/karin/extracted_text_karin/pages")
OUT_JSON = Path("LLM/karin/chunks_karin/plant_chunks.json")
OUT_DBG = Path("LLM/karin/extracted_text_karin/parse_debug.json")

# ---------------- constants / regex ----------------
COLBREAK = "<<COLBREAK>>"

LABEL_START = re.compile(r"^(Family|Common names?|Origin|Where found|Treatment|Identification|Not to be confused with)\b", re.I)
STOP_LABELS = (
    "Family", "Common names", "Common name", "Origin", "Where found", "Treatment", "Identification",
    "Not to be confused with", "Tree", "Shrub", "Grass", "Herb", "Climber", "Creeper", "Vine", 
    "Reed", "Aquatic", "Palm" 
)

HEADER_LABEL_RE = re.compile(r"Family\s*:", re.I)

BAD_SECOND = {
    "and","or","but","not","nature","reserve","river","highveld","south","north","east","west","province","weeks",
    "meters","metres","garden","labels","figure","treatment","where","found","names","common","methodology","removal",
    "individual","species","how","are","described","origin","family","grass","reed","tree","trees","shrub","vine","aquatic",
    "pods","flowers","flower","leaf","leaves","bark","young","adult","old","long","over","suckering","all", "herb"
}
BAD_CONTEXT = re.compile(r"Not to be confused|Similar species", re.I)

BINOMIAL_LINE = re.compile(r"""
    ^\s*(?:\d+|ND)?\s*
    ([A-Z][a-z]{2,})
    \s+(?:[×x]\s+)?
    ([A-Za-z][a-z-]{2,})
    (?:\s+(?:subsp\.|ssp\.|var\.)\s+[a-z-]{2,})?
    \s*(?:\((?:cont|cont\.)\))?
    \s*(?:1[abc]|2|3|ND)?\s*$
""", re.I | re.X)

BINOMIAL_ANY = re.compile(r"""
    \b([A-Z][a-z]{2,})
    \s+(?:[×x]\s+)?([A-Za-z][a-z-]{2,})
    (?:\s+(?:subsp\.|ssp\.|var\.)\s+[a-z-]{2,})?
    \b
""", re.I | re.X)

NUMBERED_TITLE_RE = re.compile(
    r"(?m)^\s*(?:\d+|ND)\s+[A-Z][a-z]{2,}\s+(?:[×x]\s+)?[a-z][a-z-]{2,}\b"
)

FULL_LABEL_LINE = re.compile(r"""
    ^\s*(?:\d+|ND)?\s*
    (
      [A-Z][a-z]{2,}                   
      \s+(?:[×x]\s+)?              
      [A-Za-z][a-z-]{2,}        
      (?:\s+(?:subsp\.|ssp\.|var\.)\s*[A-Za-z-]{2,})?
      (?:\s*/\s*[A-Z][a-z]{2,}\s+(?:[A-Za-z][a-z-]{2,})(?:\s+(?:subsp\.|ssp\.|var\.)\s*[A-Za-z-]{2,})?)?
    )
    (?:\s*\((?:cont|cont\.)\))?
    \s*$
""", re.I | re.X)

_WS_AROUND_PUNCT = re.compile(r"\s+([,.;:!?])")

def _flatten_text(s: Optional[str], preserve_paragraphs: bool = True) -> Optional[str]:
    if not s:
        return s
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"(?i)\bwhere found\?\s*", "", s)
    if preserve_paragraphs:
        s = re.sub(r"\n{2,}", "¶¶", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = _WS_AROUND_PUNCT.sub(r"\1", s)
    if preserve_paragraphs:
        s = s.replace("¶¶", "\n\n")
    return s.strip(" -.;,")


def _fix_rank_spacing(name: str) -> str:
    name = re.sub(r"\b(var\.)(?=[A-Za-z])", r"\1 ", name, flags=re.I)
    name = re.sub(r"\b(subsp\.)(?=[A-Za-z])", r"\1 ", name, flags=re.I)
    name = re.sub(r"\b(ssp\.)(?=[A-Za-z])", r"\1 ", name, flags=re.I)
    name = re.sub(r"\s*\((?:cont|cont\.)\)\s*$", "", name, flags=re.I)
    return name.strip()

def _first_title_before_labels(region: str) -> Optional[str]:
    for ln in [l.strip() for l in region.splitlines() if l.strip()]:
        if LABEL_START.match(ln):
            break
        m = FULL_LABEL_LINE.match(ln)
        if m:
            fixed = _fix_rank_spacing(m.group(1))
            mm = re.search(r"\b([A-Z][a-z-]{2,})\s+(?:[×x]\s+)?([A-Za-z][a-z-]{2,})\b", fixed)
            if mm and _looks_like_binomial(f"{mm.group(1)} {mm.group(2).lower()}"):
                return fixed
    return None

def find_full_label(full_text: str) -> Optional[str]:
    # 1) header between last COLBREAK and family:
    reg = header_region(full_text)
    val = _first_title_before_labels(reg)
    if val: return val

    # 2) top of right column
    reg2 = right_header_region(full_text)
    val2 = _first_title_before_labels(reg2)
    if val2: return val2

    # 3) special
    mu = UNKNOWN_SPECIES_RE.search(reg or reg2 or "")
    if mu:
        return f"{mu.group(1)} sp."
    return None

def canonical_from_label(label: str) -> Optional[str]:
    if not label: return None
    left = label.split("/", 1)[0]
    m = re.search(r"\b([A-Z][a-z-]{2,})\s+(?:[×x]\s+)?([A-Za-z][a-z-]{2,})\b", left)
    if not m:
        return None
    return f"{m.group(1)} {m.group(2).lower()}"

def has_numbered_title(text: str) -> bool:
    reg = header_region(text)
    return bool(NUMBERED_TITLE_RE.search(reg))

def normalize(text: str) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text, flags=re.I)
    text = re.sub(r"\(([a-z]|[0-9]+)\)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def header_region(full: str) -> str:
    m = HEADER_LABEL_RE.search(full)
    if not m:
        return ""
    start = 0
    last_col = full.rfind(COLBREAK, 0, m.start())
    if last_col != -1:
        start = last_col + len(COLBREAK)
    region = full[start:m.start()]

    lines = [ln for ln in region.splitlines() if ln.strip()]
    return "\n".join(lines[-20:])


def _looks_like_binomial(name: str) -> bool:
    parts = name.split()
    if len(parts) != 2: return False
    g, sp = parts
    if not (g[:1].isupper() and g[1:].islower()): return False
    if not re.fullmatch(r"[a-z-]{2,}", sp): return False   # was {3,}
    if sp in BAD_SECOND: return False
    return True

def header_genus_hint(text: str) -> Optional[str]:
    reg = header_region(text)
    mm = re.search(r"\b([A-Z][a-z]{2,})\b", reg)
    return mm.group(1) if mm else None

# Hard overrides for tricky "Where found" lines
WHERE_OVERRIDES = {
    ("Pinus roxburghii", 55): "Invades forest margins & grassland.",
    ("Campuloclinium macrocephalum", 91): "All over Kloofendal amongst grasses in sunny areas in rocky ground.",
    ("Iris pseudacorus", 152): "In the Kloofendal wetland at a freshwater perennial spring.",
}

def grab_block(label: str, text: str) -> Optional[str]:
    stop = "|".join(re.escape(s) for s in STOP_LABELS)
    pat = re.compile(
        rf"(?mis)^\s*(?:\([a-z]\)\s*)?{re.escape(label)}\s*\??\s*[:\-]?\s*(.+?)"
        rf"(?=\n\s*(?:{stop})\s*[:\-]?|\n\s*{re.escape(COLBREAK)}|\n\n|\Z)"
    )
    m = pat.search(text)
    return m.group(1).strip() if m else None

def description_from_origin(text: str) -> Optional[str]:
    pat = re.compile(
        r"(?mis)^\s*Origin\s*[:\-]?\s*(?:.*?\n)?(.*?)(?=\n\s*(?:\(h\)\s*)?Where found\??\s*[:\-]?|\n\s*Not to be confused|\n\s*Treatment|\n\n|\Z)"
    )
    m = pat.search(text)
    if m:
        block = m.group(1).strip()
        block = re.sub(r"^(Family|Common names?|Tree|Shrub|Grass)\s*:.*\n", "", block, flags=re.I)
        return block if block else None
    d2 = grab_block("Description", text)
    return d2

PLANT_TYPES = {"tree","grass","shrub","herb","climber","creeper","vine","succulent","aquatic","palm","reed", 
               "waterweed", "flat-growing herb"}

def find_form(full_text: str) -> Optional[str]:
    if COLBREAK in full_text:
        right = full_text.split(COLBREAK, 1)[1]
        lines = [ln.strip() for ln in right.splitlines() if ln.strip()]
        for ln in lines[:6]:
            low = ln.lower()
            if low in PLANT_TYPES:
                return ln.title()
    m = re.search(r"(?mi)^(Tree|Shrub|Grass|Herb|Climber|Creeper|Vine|Succulent|Aquatic|Palm|Reed)\s*$", full_text)
    if m:
        return m.group(1).title()
    t = plant_type_guess(full_text)
    return t.title() if t else None

_BADGE_TAIL_RE = re.compile(r"(?i)\s+(?:Waterweed|Succulent|Flat[-\s]*growing herb)\.?$")

def split_common_names(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None

    s = s.replace(";", ",").replace("/", ",")
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()

    chunks = [c.strip(" .;-") for c in s.split(",") if c.strip(" .;-")]
    items: List[str] = []
    for c in chunks:
        if re.search(r"\(.*\band\b.*\)", c, re.I):
            parts = [c]
        else:
            parts = [p.strip() for p in re.split(r"\band\b", c, flags=re.I) if p.strip()]
        items.extend(parts)

    cleaned = []
    for p in items:
        p2 = _BADGE_TAIL_RE.sub("", p).strip()
        if p2:
            cleaned.append(p2)

    seen, out = set(), []
    for p in cleaned:
        k = p.lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out or None

# ---------------- scientific name finder ----------------
def right_header_region(full: str) -> str:
    if COLBREAK in full:
        right = full.split(COLBREAK, 1)[1]
        lines = [ln for ln in right.splitlines() if ln.strip()]
        return "\n".join(lines[:12])
    return ""

UNKNOWN_SPECIES_RE = re.compile(
    r"(?mi)^\s*(?:\d+|ND)?\s*([A-Z][a-z]{2,})\s*(?:–|-|—)\s*species\s+unknown\b"
)

def _find_in_region(reg: str) -> Optional[str]:
    if not reg:
        return None

    mu = UNKNOWN_SPECIES_RE.search(reg)
    if mu:
        return f"{mu.group(1)} sp."

    for ln in [l for l in reg.splitlines() if l.strip()]:
        m = BINOMIAL_LINE.match(ln.strip())
        if m:
            cand = f"{m.group(1)} {m.group(2).lower()}"
            if _looks_like_binomial(cand):
                return cand

    for mm in BINOMIAL_ANY.finditer(reg):
        g, sp = mm.group(1), mm.group(2).lower()
        if sp in BAD_SECOND:
            continue
        left_line = reg.rfind("\n", 0, mm.start()) + 1
        if LABEL_START.match(reg[left_line:mm.start()] or ""):
            continue
        cand = f"{g} {sp}"
        if _looks_like_binomial(cand):
            return cand
    return None

def find_scientific_name(text: str) -> Optional[str]:
    reg = header_region(text)
    name = _find_in_region(reg)
    if name:
        return name
    reg2 = right_header_region(text)
    return _find_in_region(reg2)

# ---------------- other fields ----------------
def find_category(text: str) -> Optional[str]:
    reg = header_region(text)
    m = re.search(r"\b(1a|1b|1c|2|3)\b", reg, re.I)
    return m.group(1).lower() if m else None

LABEL_START = re.compile(
    r"""^\s*(
        Origin|Treatment|Family|Common\s+names?|Where\s+found\??|Not\s+to\s+be\s+confused\s+with
    )\b""",
    re.I | re.X,
)

TYPE_HEADINGS = {"tree","shrub","grass","herb","climber","creeper","vine","reed","aquatic","palm"}

BINOMIAL_HEAD = re.compile(
    r"^[A-Z][a-z-]{2,}(?:\s*/\s*[A-Z][a-z-]{2,})?\s+(?:[x×]\s+)?[a-z-]{2,}\b"
)

IDENTIFICATION_OVERRIDES = {
    ("Nasturtium officinale", 151): "Perennial, aquatic herb with erect, creeping or floating, hollow stems up to 1 m long, rooting at the nodes, and matforming. Bright green, pinnate with 3–11 leaflets, terminal leaflet often larger than the laterals; margins entire, somewhat fleshy. Flowers are white 5 mm long, in terminal, elongated racemes, flowering September to March. Fruits are brown, linearcylindrical, 2-valved, curved upwards.",
    ("Solanum seaforthianum", 142): "A slender herbaceous or woody climber growing 2-3m high It creeps over indigenous bush and smothers them. Bright green leaves that are thinly textured and deeply lobed into leaflets. Sprays of nice smelling lilac flower with yellow stamens sticking out, flowering from December to March. Small, shiny berries that go from green to bright red as they mature. Plant reproduces only by these berries.",
    ("Rumex saggitatus", 141): "Rumex sagittatus is a soft-stemmed herbaceous scrambling and climbing herb which scrambles quickly over plants smothering them. It has prominent triangular arrow-shaped leaves. The grooved green stem may be distinctly tinted red at times. The small pinkish flowers grow on panicles up to 15 cm long. These are followed by a 3-sided greenish 1 cm diameter pod. The leaves and young stems can be cooked and eaten and are sometimes cultivated as a vegetable in Java. It has a tuberous woody rootstock with extensive rhizomes. It spreads by seed and by resprouting from the tuber.",
    ("Podranea ricosaliana", 140): "A vigorous, woody, rambling, evergreen It sends up many tall strong stems — 3 to 5 m up to 10+ m high if left unchecked. It has many underground stems with newly sprouting plants coming up (89), like in Kloofendal, from gardens of neighbouring houses. The roots grow into the cement between bricks of brick walls, seriously damaging these walls. Podranea ricasoliana has glossy foliage and large bunches of fragrant lilac-pink, trumpet-shaped flowers.",
    ("Jasminum mesnyi", 136): "Primrose jasmine is a rambling, open evergreen shrub with long, slender, arching stems that will climb like a sprawling vine if given support. The stems are square in cross section, and green, becoming woody with age. The glossy dark green leaves are opposite and divided into three leaflets. The fragrant, yellow trumpet shaped flowers are borne in early spring and sporadically into summer. They are semi-double with 6-10 petals, and sweetly fragrant.",
    ("Hedera canariensis", 134): "This evergreen perennial can climb up to 10m high by means of rootlets on the stems, or it can spread over the ground to form a carpet. Leaves are bright to dark green, sometimes with broad silvery- grey or white edges, glossy when new, becoming leathery with age. They are mostly wider than they are long, unlobed to shallowly three-lobed. The flowers are green in terminal, globular umbels flowering from March to July, but seldom appear. Fruits are drupes, which are black when ripe.",
    ("Cuscuta campestris", 132): "Slender, leafless, parasitic herb with yellow or whitish, twining stems up to 2m high and forming dense patches up to 6m across. It smothers and parasitizes other plants of economic importance in agricultural crop lands. Dodder has no leaves and no chlorophyll in its stems either, so is totally parasitic on its host plant. Leafless annual herb, looks like entwined yellow string creeping plentifully over other vegetation from which it parasitizes by suckers (hausteria).",
    ("Tradescantia fluminensis", 128): "Wandering jew is a trailing perennial. It forms a mat which smothers low growing plants and prevents the natural regeneration of taller forest species. Grows in shade, under trees, forming extensive mats, that smother all other vegetation and eventually replace it . Flowers bloom intermittently throughout the year.",
    ("Persicaria capitata", 126): "Knotweed is a mat-like, perennial, herbaceous plant with a slender, woody rootstock and long, creeping, rooting stems. It also spreads with seeds. It is found growing on roadsides, dry banks, slopes and in open areas, preferring full sun. It negatively impacts on the environment by forming a dense carpet, replacing most indigenous vegetation. The hairy leaves are 1-6cm long and 0.7-3cm wide with a reddish midrib and a distinctive dark red V pattern. The leaves turn red when the plant is under stress. The tiny pink flowers are clustered in ball-shaped flower spikes above the foliage. These ball-like inflorescences are 7-20mm across with numerous pink flowers 2.5mm long. Flowering occurs from October to March.",
    ("Duchesnea indica", 125): "Creeping perennial herb spreading by slender stolons, grows in damp, shady places. It tends to spread rampantly, blocking drainage lines and replacing other species. Leaves trifoliate, fruit red small strawberry, which is edible but has hardly any taste. The flowers are yellow",
    ("Xanthium strumarium", 121): "This herbaceous shrub grows up to 1,2m high. The erect stems are brownish or reddish-brown, often with red spots, ribbed and roughly downy Leaves are blue-green, sparsely hairy above and densely white- woolly below, entire or three- lobed, ± lanceolate, up to 60 mm long x 20 mm wide. Three-lobed leaf and very prickly burrs. Flowers are greenish, inconspicuous in axils of leaves, flowering from October to April Green burrs which will turn brown. Brownish burrs up to 2cm long crowned with two stout horns and covered with hooked spines up to 4mm long.The burrs are an irritant to the skin, the prickles are sore",
    ("Pteridium aquilinum", 112): "A perennial, deciduous fern, growing into dense stands. Bracken has large, roughly triangular fronds (large, divided leaf) produced singly from an underground rhizome, and grows to 0,3 to 1 meter tall. The plant dies back to ground level in autumn. The rhizome grows up to 3.5 meters deep, 5 cm in diameter, and up to 15 meters long It regrows in the spring from an underground rhizome, new growth presents as vertical stalks, coiled and covered with silver grey hairs, unfurling into fronds. It also reproduces from its copious spores.",
    ("Pseudognaphalium luteo-album", 111): "Leaves long and narrow, folded along the midriff longitudinally. Grows in dense stands. Flowers in summer.",
    ("Polygonum lapathifolium", 110): "Widespread naturalized weed in South Africa, common on riverbanks, dam walls, ditches, even in water. ",
    ("Nephrolepis cordifolia", 105): "Evergreen fern with ~ erect, stiff fronds (leaves) up to 1 m high; terrestrial or epiphytic (grows on other plants). It forms extensive colonies by means of stolons and produces tubers Fronds tend to be dull green in shaded areas and lighter green or yellowish-green when growing in a sunny position. Both fertile and sterile fronds are pinnately compound (once divided) and 7cm wide. Numerous spore containing structures are produced between the leaflet midvein and margin on the undersurface of the leaflets. (9) Thousands of spores can be produced by one plant, which, enables this fern to spread aggressively, making it a threat to indigenous plant species. ",
    ("Galinsoga parviflora", 102): "Troublesome in wide range of crops in South Africa, but also in gardens. Small white and yellow flowers. Flowers in summer",
    ("Cynoglossum lanceolatum", 99): "Many little blue flowers which become green fruits which dry out to become burr seeds which stick onto anything that comes past. They grow in the same disturbed areas where the declared AIPs grow, obstructing the removal of these plants. Leaves bluish–green with distinct central and lateral veins. Taproot, which can get pretty big!",
    ("Conyza albida", 95): "Annual major weed over 2 meters high. Growing in gardens, along the roadside, fallow land, forests and can infest perennial crops and is known to host the tomato spotted wilt virus, known as “Kromnek” in tomatoes, potatoes, tobacco and peas",
    ("Pyracantha angustifolia", 79): "Evergreen shrub 2-4m high with stiff, spiny branches. Young shoots are covered in thick, yellowish down and woody spines which bear leaves. P. angustifolia leaves are dull dark green above, densely grey-downy beneath, narrowly elongate, margins entire, rolled under, apex rounded, often notched. Flowers are white with a downy calyx. Spines are woody, sharp-pointed and bear leaves. Orange-red or orange-yellow berries.",
    ("Ligustrum lucidum", 73): "Evergreen shrub or small tree 3-10m high. Used hedges and ornamentally. Leaves dark green, glossy, thick and leathery, large (6–12 cm long), tapered at the base, long-tapering at the apex, sometimes variegated in green and yellow Leaves are opposite, margins entire. Heavily, scented white flowers in large terminal clusters appearing from October to February, tighter than L.Japonicum Shiny black berries Some birds prefer the Privet fruit to fruit from indigenous plants, thereby dispersing the seeds. Privets are used as hedges and ornamentally in gardens",
    ("Prunus serotina", 57): "Black cherry is a deciduous tree, growing up to 15-30 meters tall, with a trunk diameter of up to 70-120cm. Very broken, dark grey to black bark . Finely serrated leaf, red petiole. Leaf arrangement is alternate. Black cherry is a leading cause of livestock illness. The flowers are small (10-15 in diameter), with five white petals and about 20 stamens, and are fragrant. There are around 40 flowers on each raceme. Fruit/seeds are drupes, 1cm in diameter, green to red at first, ripening to black.",
    ("Pinus roxburghii", 55): "General description: Coniferous tree up to 20m high or more; with a conical or oval crown; branches distinctly ascending, secondary shoots absent from trunk; bark very thick and fissured. Leaves: Needles, light to bright green, in bundles of three, 15-30 cm long. The pine tree does not produce any flowers.",
    ("Celtis sinensis", 43): "Leaf of Celtis sinensis – smooth and shiny, quite long compared to C. Africana. Fruit are drupes, green turning dark orange, globose",
}

def _split_paragraphs_for_id(text: str) -> list[str]:
    t = text.replace(COLBREAK, "\n\n")
    return [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]

def _is_heading_or_title(para: str) -> bool:
    p = para.strip()
    if p.lower() in TYPE_HEADINGS:
        return True
    first_line = p.splitlines()[0]
    if BINOMIAL_HEAD.match(first_line) and len(p) <= 80 and ":" not in p:
        return True
    if len(p) <= 10 and re.fullmatch(r"[A-Za-z0-9.'/-]+", p):
        return True
    return False

def extract_identification_paragraphs(page_text: str) -> list[str]:
    id_paras: list[str] = []
    for para in _split_paragraphs_for_id(page_text):
        # exclude labeled sections entirely
        if LABEL_START.match(para):
            continue
        # exclude obvious headings/titles
        if _is_heading_or_title(para):
            continue
        id_paras.append(para)
    return id_paras

def identification_text(full_text: str) -> Optional[str]:
    paras = extract_identification_paragraphs(full_text)
    txt = "\n\n".join(paras).strip()
    return txt or None

ABBREV_SPECIES_RE = re.compile(r"(?im)^[ \t]*([A-Z])\.\s*([a-z-]{3,})\b")  # only at line start
ND_HEADING_RE     = re.compile(r"(?im)^[ \t]*ND\s+[A-Z][a-z-]{2,}\s+(?:[×x]\s+)?[a-z-]{2,}.*$")
REM_CONTEXT_RE    = re.compile(r"(?im)^[ \t]*(?:Not\s+to\s+be\s+confused\s+with|Similar\s+species)\b.*$")

def _canon_parts_from_label(canon: str) -> tuple[Optional[str], Optional[str]]:
    m = re.search(r"\b([A-Z][a-z-]{2,})\s+(?:[×x]\s+)?([a-z-]{2,})\b", canon)
    return (m.group(1), m.group(2).lower()) if m else (None, None)

def _starts_with_other_species(line: str, g: Optional[str], sp: Optional[str]) -> bool:
    if not g or not sp:
        return False
    m = re.match(r"^[ \t]*([A-Z][a-z-]{2,})\s+(?:[×x]\s+)?([a-z-]{2,})\b", line)
    if m:
        g2, sp2 = m.group(1), m.group(2).lower()
        if _looks_like_binomial(f"{g2} {sp2}") and not (g2 == g and sp2 == sp):
            return True
    m2 = ABBREV_SPECIES_RE.match(line)
    if m2:
        gi, sp2 = m2.group(1), m2.group(2).lower()
        if sp2 not in BAD_SECOND and re.fullmatch(r"[a-z-]{3,}", sp2):
            if not (gi == g[0] and sp2 == sp):
                return True
    return False

def clean_identification_light(ident_text: Optional[str], canon: str) -> Optional[str]:
    if not ident_text:
        return ident_text

    ident_text = ND_HEADING_RE.sub("", ident_text)
    ident_text = REM_CONTEXT_RE.sub("", ident_text)
    g, sp = _canon_parts_from_label(canon)
    paras = [p for p in re.split(r"\n\s*\n", ident_text) if p.strip()]
    cleaned_paras: list[str] = []
    for para in paras:
        units = _split_units(_normalize_for_hazards(para))
        kept: list[str] = []
        prev_dropped = False
        for u in units:
            us = u.strip()
            if _POISON_TOKEN_RE.search(us) or _IRRITANT_RE.search(us):
                continue
            if re.fullmatch(r"[\(\)\[\]\s.,;:'\"-]*", us):
                continue
            if us.endswith("/") and len(us) <= 40:
                continue
            if re.fullmatch(r"[()a-z\s]*1[abc]?\.*", us, flags=re.I):
                continue
            if _starts_with_other_species(us, g, sp):
                prev_dropped = True
                continue
            if prev_dropped and len(us) <= 120 and re.match(r"^(is|are|has|have|with)\b", us.lower()):
                prev_dropped = False
                continue
            prev_dropped = False
            kept.append(us)
        para_txt = _post_trim(" ".join(kept)).strip()
        if para_txt:
            cleaned_paras.append(para_txt)
    out = "\n\n".join(cleaned_paras).strip()
    return out or None

def treatment_block(full_text: str) -> Optional[str]:
    pat_newline = re.compile(
        r"(?mis)^\s*Treatment\s*[:\-]?\s*\n"
        r"(.+?)"
        r"(?=\n\s*(?:Family|Common names?|Origin|Where found|Identification|Not to be confused with|Uses|Notes|Leaf|Habitat|Description)\s*[:\-]?|\n\s*<<COLBREAK>>|\Z)"  # …until next label/colbreak/EOF
    )
    m = pat_newline.search(full_text)
    if m:
        return m.group(1).strip()

    pat_inline = re.compile(
        r"(?mis)^\s*Treatment\s*[:\-]\s*"
        r"(.+?)"
        r"(?=\n\s*(?:Family|Common names?|Origin|Where found|Identification|Not to be confused with|Uses|Notes|Leaf|Habitat|Description)\s*[:\-]?|\n\s*<<COLBREAK>>|\Z)"
    )
    m = pat_inline.search(full_text)
    return m.group(1).strip() if m else None

def plant_type_guess(full_text: str) -> Optional[str]:
    for t in ["Grass","Tree","Shrub","Herb","Climber","Creeper","Vine","Succulent","Aquatic","Palm","Reed"]:
        if re.search(rf"\b{t}\b", full_text, re.I):
            return t
    return None

POISONOUS_PAGES = {
    38, 42, 48, 50, 51, 52, 53, 60, 64, 65, 
    68, 69, 71, 72, 73, 74, 75, 77, 79, 82, 
    83, 85, 88, 89, 100, 101, 103, 104, 107, 114, 
    115, 119, 121, 129, 131, 134, 135, 140, 142, 145, 
    150, 152
}

_POISON_TOKEN_RE = re.compile(
    r"\b(poison(?:ous|ing)?|toxic|toxicity|toxin[s]?|hallucinogen(?:ic)?)\b", re.I
)

_IRRITANT_RE = re.compile(r"\b(irritant|irritation|rash|dermatitis|allergic|itch)\b", re.I)

_HEADING_CUTOFF_RE = re.compile(
    r"(?mi)^(Where found\??|Treatment|Not to be confused with|Family|Common names?|Origin)\b"
)
_BADGE_LINE_RE = re.compile(
    r"(?mi)^(Tree|Shrub|Herb|Grass|Creeper|Vine|Succulent|Reed|Waterweed|Flat[-\s]*growing herb)\s*$"
)
def _normalize_for_hazards(text: str) -> str:
    t = text.replace("<<COLBREAK>>", ". ")
    t = re.sub(r"(?<![.!?])\s*\n+(?=[A-Z])", ". ", t)
    t = re.sub(r"(?<=[.!?])(?=[A-Z])", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _split_units(t: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|(?<=[.!?])(?=[A-Z])", t)
    parts = [p.strip() for p in parts if p.strip()]
    parts = [p for p in parts if not (_HEADING_CUTOFF_RE.match(p) or _BADGE_LINE_RE.match(p))]
    return parts

def _post_trim(h: str) -> str:
    h = re.sub(r"^(Origin|Where found\??|Treatment|Family|Common names?):\s*", "", h, flags=re.I)
    m = _HEADING_CUTOFF_RE.search(h)
    if m:
        h = h[:m.start()]
    h = re.sub(r"\s+", " ", h).strip(" .;,-")
    return h

def _crop_to_hazard(u: str) -> str:
    m = _POISON_TOKEN_RE.search(u) or _IRRITANT_RE.search(u)
    if not m:
        return u
    start = u.rfind(". ", 0, m.start())
    start = 0 if start == -1 else start + 2
    end = u.find(". ", m.end())
    end = len(u) if end == -1 else end
    return u[start:end].strip()

def extract_hazards(full_text: str, page_no: int) -> tuple[Optional[str], Optional[str]]:
    t = _normalize_for_hazards(full_text)
    units = _split_units(t)
    pois_units: list[str] = []
    irr_units: list[str] = []

    def _uniq(xs: list[str]) -> list[str]:
        seen = set(); out = []
        for x in xs:
            k = x.lower()
            if k not in seen:
                seen.add(k); out.append(x)
        return out

    if page_no in POISONOUS_PAGES:
        for u in units:
            has_p = bool(_POISON_TOKEN_RE.search(u))
            has_i = bool(_IRRITANT_RE.search(u))
            if has_p:
                pois_units.append(_post_trim(_crop_to_hazard(u)))
            elif has_i:
                irr_units.append(_post_trim(_crop_to_hazard(u)))
        if not pois_units and re.search(r"\bpoisonous\b", t, re.I):
            pois_units.append("Poisonous")
    else:
        for u in units:
            if _IRRITANT_RE.search(u) and not _POISON_TOKEN_RE.search(u):
                irr_units.append(_post_trim(_crop_to_hazard(u)))

    pois = " ".join(_uniq(pois_units)) or None
    irr  = " ".join(_uniq(irr_units)) or None
    return pois, irr

POISON_OVERRIDES = {
    ("Celtis sinensis", 43): "All Celtis species are poisonous.",
    ("Robinia pseudoacacia", 58): "Seeds, leaves and inner bark are poisonous.",
    ("Cestrum parqui", 65): "All parts of the plant are reported to be highly toxic. It is described as significant hazard to livestock (especially cattle) which may eat green Cestrum inadvertently or during shortages of other foods, often resulting in death. Death is usually rapid and painful.",
}


def _clean_noise(s: Optional[str]) -> Optional[str]:
    if not s: return None
    lines = []
    for ln in s.splitlines():
        t = ln.strip()
        if not t: 
            continue

        if re.fullmatch(r"\(?[a-z]\)?", t, flags=re.I) or re.fullmatch(r"\(?\d{1,3}\)?", t):
            continue
        if len(t) >= 12 and re.fullmatch(r"[A-Z0-9\s'&\-\(\)]+", t):
            continue
        lines.append(t)
    return "\n".join(lines).strip() or None

_MORPH_LINE_START = re.compile(
    r"(?i)^(an|a|the|evergreen|deciduous|perennial|annual|succulent|shrub|tree|herb|grass|climber|creeper|leaves?|flowers?|fruit|seed|bark|stems?|roots?|where found\??|not to be|treatment|uses?|notes?)\b"
)
_MORPH_INLINE = re.compile(
    r"(?i)\b(evergreen|deciduous|perennial|annual|succulent|shrub|tree|herb|grass|climber|creeper|erect|spreading|scrambling)\b"
)

# Hard overrides for the three that wont budge
ORIGIN_OVERRIDES = {
    ("Nephrolepis cordifolia", 105): "Tropical forests all over the world, especially Central America & the West Indies.",
    ("Cynoglossum lanceolatum", 99): "Yemen, Pakistan, India, Mediterranean, Asia and Madagascar, growing in disturbed habitats throughout parts of Africa widely in all S. African provinces (Wiki).",
    ("Achyranthes aspera", 87): "Pantropical – no clear origin, found everywhere.",
}

def _origin_head_and_rest(origin_block: str) -> tuple[Optional[str], Optional[str]]:
    if not origin_block:
        return None, None

    raw_lines = [ln.strip() for ln in origin_block.splitlines()]
    lines = [ln for ln in raw_lines if ln and not re.fullmatch(r"\(\d+\)", ln)]
    if not lines:
        return None, None

    kept = []
    for i, ln in enumerate(lines):
        if i > 0 and _MORPH_LINE_START.match(ln):
            break
        kept.append(ln)
        if ln.endswith("."):
            break

    s = " ".join(kept)
    m_inline = _MORPH_INLINE.search(s)
    if m_inline:
        s = s[:m_inline.start()].rstrip()

    s = re.sub(r"\s*\(\d+\)\s*$", "", s).strip()
    s = re.sub(r"[ ,;:–—-]+$", "", s).strip()
    s = re.sub(r"\b(South|North|East|West)(Eastern|Western|Northern|Southern)\b", r"\1 \2", s)
    s = re.sub(r"\b([NSEW])\.\s*(Africa|America|China|Asia|Europe)\b", r"\1.\2", s)
    s = re.sub(r"\b([NSEW])\s+(Africa|America|China|Asia|Europe)\b", r"\1.\2", s)
    s = re.sub(r"\s{2,}", " ", s).strip()

    head = s or None
    tail_lines = lines[len(kept):]
    tail = _clean_noise("\n".join(tail_lines).strip()) or None
    return head, tail

# ---------------- parse one page ----------------
start_pg = 24
def parse_page(text: str, pageno: int):
    if pageno < start_pg:
        return None
    cleaned = normalize(text)
    source = "https://invasives.org.za/wp-content/uploads/2025/02/AlP3Feb25_KarinSpottiswoode.pdf"

    if not HEADER_LABEL_RE.search(cleaned):
        return None

    sci = find_scientific_name(cleaned)
    if not sci or not _looks_like_binomial(sci):
        return None
    
    full_label = find_full_label(cleaned) or sci
    canon = canonical_from_label(full_label) if full_label else find_scientific_name(cleaned)
    if not canon or not _looks_like_binomial(canon):
        return None

    fam = grab_block("Family", cleaned)
    common_raw = grab_block("Common names", cleaned) or grab_block("Common name", cleaned)
    origin_block = grab_block("Origin", cleaned)
    if origin_block:
        origin, _ = _origin_head_and_rest(origin_block)
    else:
        origin = None

    override_origin = (
        ORIGIN_OVERRIDES.get((canon, pageno)) or
        ORIGIN_OVERRIDES.get((sci, pageno)) or
        ORIGIN_OVERRIDES.get(canon) or
        ORIGIN_OVERRIDES.get(sci)
    )
    if override_origin:
        origin = override_origin

    where = grab_block("Where found", cleaned)
    override_where = (
        WHERE_OVERRIDES.get((canon, pageno)) or
        WHERE_OVERRIDES.get((sci, pageno)) or
        WHERE_OVERRIDES.get(canon) or
        WHERE_OVERRIDES.get(sci)
    )
    if override_where:
        where = override_where
    ident = identification_text(cleaned)
    if ident:
        ident = clean_identification_light(ident, canon) or None
    override_identification = (
        IDENTIFICATION_OVERRIDES.get((canon, pageno)) or
        IDENTIFICATION_OVERRIDES.get((sci, pageno)) or
        IDENTIFICATION_OVERRIDES.get(canon) or
        IDENTIFICATION_OVERRIDES.get(sci)
    )
    if override_identification:
        ident = override_identification
    treat = treatment_block(cleaned)
    poisonous, irritant = extract_hazards(cleaned, pageno)
    override_poison = (
        POISON_OVERRIDES.get((canon, pageno)) or
        POISON_OVERRIDES.get((sci, pageno)) or
        POISON_OVERRIDES.get(canon) or
        POISON_OVERRIDES.get(sci)
    )
    if override_poison:
        poisonous = override_poison
    form = find_form(cleaned)
    entry = {
        "scientific_name": full_label or canon,
        "form": form,
        "common_names": split_common_names(common_raw),
        "family": fam,
        "origin": origin,
        "where_found": where,
        "identification": ident,
        "treatment": treat,
        "poisonous": poisonous,
        "irritant": irritant,
        "source": source,
        "page_hint": pageno
    }
    entry["_canon"] = canon
    return entry

# ---------------- merge duplicates across pages ----------------
def merge_by_species(entries: List[Dict]) -> List[Dict]:
    out: Dict[Tuple[str, str], Dict] = {}
    order: List[Tuple[str, str]] = []

    def _merge_text(a: Optional[str], b: Optional[str]) -> Optional[str]:
        if not a: return b
        if not b: return a
        if b in a: return a
        return a + "\n\n" + b

    for e in sorted(entries, key=lambda d: d["page_hint"]):
        canon = e.get("_canon") or canonical_from_label(e.get("scientific_name","")) or e["scientific_name"]
        key = (canon, (e.get("form") or "").lower())
        if key not in out:
            out[key] = {**e}
            out[key].pop("_canon", None)
            if out[key].get("common_names") is None:
                out[key]["common_names"] = []
            order.append(key)
        else:
            cur = out[key]
            new_common = e.get("common_names") or []
            have = cur.get("common_names") or []
            seen = {c.lower() for c in have}
            for c in new_common:
                if c and c.lower() not in seen:
                    have.append(c); seen.add(c.lower())
            cur["common_names"] = have
            for k in ["family", "origin", "form"]:
                cur[k] = cur.get(k) or e.get(k)
            for k in ["where_found", "identification", "treatment", "poisonous", "irritant"]:
                cur[k] = _merge_text(cur.get(k), e.get(k))
            cur["page_hint"] = min(cur["page_hint"], e["page_hint"])

    result = [out[k] for k in order]
    for r in result:
        r.pop("_canon", None)
    return result
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

    FLATTEN_FIELDS = ("where_found", "identification", "treatment", "poisonous", "origin")
    for e in merged:
        for k in FLATTEN_FIELDS:
            if e.get(k):
                e[k] = _flatten_text(e[k], preserve_paragraphs= True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_DBG.write_text(json.dumps({"skipped_pages": skipped, "total": len(pages), "ok": len(merged)}, indent=2), encoding="utf-8")

    print(f"Extracted {len(merged)} plant entries → {OUT_JSON}")
    if skipped:
        print(f"Skipped {len(skipped)} pages → {OUT_DBG}")

if __name__ == "__main__":
    main()