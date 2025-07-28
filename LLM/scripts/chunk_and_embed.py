import json
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# === CONFIG ===
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_FILE = "LLM/chunks/plant_chunks.json"
CHROMA_DIR = "LLM/vector_db"

# === INIT ===
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
model = SentenceTransformer(MODEL_NAME)

# === CREATE COLLECTION (allow if already exists)
collection = chroma.get_or_create_collection(name="plants")

# === LOAD CHUNKS ===
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# === BATCH EMBEDDING ===
texts = [entry["text"] for entry in chunks]
embeddings = model.encode(texts, convert_to_numpy=True)

# === ADD to COLLECTION with embeddings
collection.add(
    embeddings=embeddings,
    metadatas=[{"plant": entry["plant"]} for entry in chunks],
    ids=[f"plant-{i}" for i in range(len(chunks))],
    documents=texts
)

print(f"✅ Stored {len(chunks)} plant entries using manual embeddings.")