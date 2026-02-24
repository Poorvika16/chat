import os
import io
import faiss
import numpy as np
import zipfile
import json
import pickle
from docx import Document
from sentence_transformers import SentenceTransformer
from PIL import Image
import re
import pytesseract
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data for BM25
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configuration
DOC_PATH = r"C:\Users\HP\OneDrive\Documents\poorvika\documentbot\docs\AD_UserGuide_v1.4.docx"
IMAGE_FOLDER = "static/images"
INDEX_FILE = "faiss_index.bin"
META_FILE = "meta_data.npy"
BM25_FILE = "bm25_index.pkl"
HEADING_IMAGE_MAP_FILE = "heading_image_map.npy"

# Clean old files
os.makedirs(IMAGE_FOLDER, exist_ok=True)
for f in os.listdir(IMAGE_FOLDER):
    os.remove(os.path.join(IMAGE_FOLDER, f))
for f in [INDEX_FILE, META_FILE, BM25_FILE, HEADING_IMAGE_MAP_FILE]:
    if os.path.exists(f):
        os.remove(f)

print("=" * 60)
print("BUILDING INDEX WITH BAAI/bge-small-en + BM25 + OCR")
print("=" * 60)

# 1. Load BAAI model (as required)
print("\n📦 Loading BAAI/bge-small-en model...")
model = SentenceTransformer("BAAI/bge-small-en")
print(f"✓ Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

# 2. Read document
print(f"\n📄 Reading document: {os.path.basename(DOC_PATH)}")
doc = Document(DOC_PATH)

# 3. Extract sections
sections = {}
current_section = None
content = []
section_positions = []

for para in doc.paragraphs:
    text = para.text.strip()
    if not text:
        continue
    
    # Check if heading
    is_heading = False
    if para.style and para.style.name:
        if any(h in para.style.name.lower() for h in ['heading', 'title']):
            is_heading = True
    if re.match(r'^\d+(\.\d+)*\.?\s', text):
        is_heading = True
    
    if is_heading:
        if current_section and content:
            sections[current_section] = "\n".join(content)
            section_positions.append(current_section)
        current_section = text
        content = []
    else:
        content.append(text)

# Save last section
if current_section and content:
    sections[current_section] = "\n".join(content)
    section_positions.append(current_section)

print(f"✓ Found {len(sections)} sections")

# 4. Extract images with OCR (Tesseract)
section_images = {s: [] for s in sections.keys()}
image_texts = {}

print("\n🖼️ Extracting images with Tesseract OCR...")
with zipfile.ZipFile(DOC_PATH, 'r') as docx_zip:
    image_files = [f for f in docx_zip.namelist() if f.startswith("word/media/")]
    image_files.sort()
    
    image_count = 0
    for img_file in image_files:
        try:
            img_data = docx_zip.read(img_file)
            img = Image.open(io.BytesIO(img_data))
            
            if img.size[0] < 50 or img.size[1] < 50:
                continue
            
            image_count += 1
            img_name = f"img_{image_count}.png"
            img.save(os.path.join(IMAGE_FOLDER, img_name))
            
            # OCR using Tesseract
            try:
                ocr_text = pytesseract.image_to_string(img)
                if ocr_text.strip():
                    image_texts[img_name] = ocr_text.strip()
                    print(f"  ✓ OCR: {ocr_text[:50]}...")
            except Exception as e:
                print(f"  ✗ OCR failed: {e}")
            
            # Map to section
            section_index = min(image_count // 3, len(section_positions)-1)
            if section_positions:
                target_section = section_positions[section_index]
                section_images[target_section].append(img_name)
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

print(f"✓ Saved {image_count} images, OCR text from {len(image_texts)} images")

# 5. Create chunks
chunks = []
metadata = []
chunk_id = 0

print("\n📝 Creating chunks...")
for section, text in sections.items():
    if text and len(text) > 50:
        words = text.split()
        for i in range(0, len(words), 150):
            chunk = " ".join(words[i:i+150])
            chunks.append(chunk)
            metadata.append({
                "id": chunk_id,
                "text": chunk,
                "section": section,
                "type": "text"
            })
            chunk_id += 1

# Add OCR text as chunks
for img_name, img_text in image_texts.items():
    for section, imgs in section_images.items():
        if img_name in imgs:
            chunks.append(f"[IMAGE: {img_name}] {img_text}")
            metadata.append({
                "id": chunk_id,
                "text": img_text,
                "section": section,
                "type": "image",
                "image": img_name
            })
            chunk_id += 1
            break

print(f"✓ Created {len(chunks)} chunks")

# 6. Generate embeddings with BAAI model (for FAISS)
print("\n🧮 Generating embeddings with BAAI/bge-small-en...")
embeddings = model.encode(chunks, show_progress_bar=True)
faiss.normalize_L2(embeddings)

# 7. Create FAISS index (vector search)
print("\n💾 Creating FAISS index...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, INDEX_FILE)
print(f"✓ FAISS index saved to {INDEX_FILE}")

# 8. Create BM25 index (keyword search)
print("\n📊 Creating BM25 keyword index...")
tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)
with open(BM25_FILE, 'wb') as f:
    pickle.dump(bm25, f)
print(f"✓ BM25 index saved to {BM25_FILE}")

# 9. Save metadata
np.save(META_FILE, metadata)
np.save(HEADING_IMAGE_MAP_FILE, section_images)
print(f"✓ Metadata saved")

print("\n" + "=" * 60)
print("✅ BUILD COMPLETE")
print(f"   • Model: BAAI/bge-small-en")
print(f"   • Search: FAISS + BM25")
print(f"   • OCR: Tesseract")
print(f"   • Chunks: {len(chunks)}")
print(f"   • Images: {image_count}")
print("=" * 60)