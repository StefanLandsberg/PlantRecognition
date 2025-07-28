import re
import json
from pathlib import Path

# Path to your text file
text_path = Path("LLM/extracted_text/karin_spottiswoode.txt")
full_text = text_path.read_text(encoding="utf-8")

# Match "45 Acacia dealbata" or "ND Paspalum notatum"
pattern = re.compile(r"(?<=\n)(\d{1,3}|ND)\s+([A-Z][a-z]+(?:\s+[a-z]+)+)")

matches = list(pattern.finditer(full_text))

# Extract cleanly delimited chunks
plant_chunks = []
for i, match in enumerate(matches):
    name = match.group(2).strip()
    start = match.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
    chunk_text = full_text[start:end].strip()

    plant_chunks.append({
        "plant": name,
        "text": chunk_text
    })

# Save result
output_path = Path("LLM/chunks/plant_chunks.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(plant_chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Extracted {len(plant_chunks)} clean plant entries.")