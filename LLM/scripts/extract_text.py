import fitz
import os

# Define paths
pdf_path = "LLM/pdf/AlP3Feb25_KarinSpottiswoode.pdf"
output_path = "LLM/extracted_text/karin_spottiswoode.txt" 

# Open the PDF
doc = fitz.open(pdf_path)
text = ""

# Extract text from each page
for page in doc:
    text += page.get_text()

# Close the document
doc.close()

# Save extracted text to file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"✅ Text extracted and saved to {output_path}")