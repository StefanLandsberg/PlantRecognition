import chromadb
from sentence_transformers import SentenceTransformer
import re

# === CONFIG ===
CHROMA_DIR = "LLM/vector_db"
COLLECTION_NAME = "plants"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 1
SOURCE = "Source: Invasive Alien & Problem Plants on The Witwatersrand & Magaliesberg. Field Guide by Karin Spottiswoode (https://invasives.org.za/wp-content/uploads/2025/02/AlP3Feb25_KarinSpottiswoode.pdf)"


# === INIT ===
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME)
model = SentenceTransformer(EMBED_MODEL)

# === QUERY ===
def query_plants(plant_name):
    query_embedding = model.encode([plant_name], convert_to_numpy=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K
    )
    return results

def extract_identification(text):
    # Capture everything after 'Not to be confused with' or 'Identification' up to the next section or end
    pattern = r"(?:Not to be confused with|Identification)[\s:]*\n*(.*?)(?=\n(?:Family|Common names|Origin|Where found|Treatment|Uses|Notes|Leaf|Habitat|Description)[\s:]*|\Z)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else "❌ Not found"

def extract_poisonous(text):
    poison_keywords = [
        "poison", "toxic", "irritant", "irritation", "rash", "allergic", "reaction",
        "not edible", "noxious", "harmful", "respiratory tract"
    ]
    # Split text into sentences (handles . ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    poison_sentences = [
        s.strip() for s in sentences
        if any(k in s.lower() for k in poison_keywords)
    ]
    return " ".join(poison_sentences) if poison_sentences else "❌ Not found"

def extract_fields(text):
    text = re.sub(r"(\n\s*){2,}", "\n", text.strip())  # Collapse extra newlines

    def multiline_extract(label):
        stop_labels = [
            "Family", "Common names", "Origin", "Where found", "Treatment",
            "Uses", "Identification", "Notes", "Leaf", "Habitat", "Description"
        ]
        stop_pattern = "|".join([rf"^\s*{l}[\s:]*" for l in stop_labels if l.lower() != label.lower()])
        pattern = rf"^\s*{label}[\s:]*\n*(.*?)(?=\n(?:{stop_pattern})|\Z)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        return re.sub(r"\s+", " ", match.group(1)).strip() if match else "❌ Not found"

    origin_match = re.search(
        r"^\s*Origin[\s:]*\n*(.*?)(?=\n(?:^\s*(Where found|Family|Common names|Treatment)[\s:]*|\Z))",
        text, re.IGNORECASE | re.DOTALL | re.MULTILINE
    )

    if origin_match:
        origin_block = origin_match.group(1).strip()
        # Split at first period or newline
        split = re.split(r"[.\n]", origin_block, maxsplit=1)
        origin = split[0].strip()
        description = split[1].strip() if len(split) > 1 else "❌ Not found"
    else:
        origin = "❌ Not found"
        description = "❌ Not found"

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
    val = fields.get(field_name, "❌ Not found")
    val = re.sub(r"\s+", " ", val.strip())  # Collapse all whitespace
    val = re.sub(r"\(\d+\)", "", val)  # Remove numbers in brackets, e.g., (9)
    val = re.sub(r"\b\d+\b(?![a-zA-Z])", "", val)  # Remove standalone numbers (not part of measurements like 3m)
    val = re.sub(r"\s{2,}", " ", val)  # Remove extra spaces left after removals
    # Remove leading question mark and any following spaces for "Where Found"
    if field_name == "Where Found":
        val = re.sub(r"^\?\s*", "", val)
    return val if len(val) <= max_len else val[:max_len].strip() + "..."

# === CLI ===
if __name__ == "__main__":
    while True:
        query = input("🔍 Enter plant name to search (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        result = query_plants(query)
        for doc, meta, score in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            fields = extract_fields(doc)
            print("\n--- Match ---")
            print(f"🌿 Plant: {meta['plant']}")
            print(f"📊 Score: {score:.4f}")
            print(f"🏷️ Family: {safe_extract(fields, 'Family')}")
            print(f"🗣️ Common Names: {safe_extract(fields, 'Common Names')}")
            print(f"🌍 Origin: {safe_extract(fields, 'Origin')}")
            print(f"📖 Description: {safe_extract(fields, 'Description', max_len=1000)}")
            print(f"📍 Where Found: {safe_extract(fields, 'Where Found')}")
            print(f"🧪 Treatment: {safe_extract(fields, 'Treatment')}")
            print(f"🆔 Identification: {safe_extract(fields, 'Identification')}")
            print(f"\033[91m☠️ Poisonous: {safe_extract(fields, 'Poisonous')}\033[0m")
            print(f"🔗 {SOURCE}")

    print("\nGoodbye! 👋\n")