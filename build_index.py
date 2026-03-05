# build_index.py
import os
import io
import faiss
import numpy as np
import json
import pickle
import re
import warnings
import traceback
import sys
import gzip

from docx import Document
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================
DOC_PATH = r"C:\Users\HP\Documents\documentbot\AD_UserGuide_v1.4.docx"
JSON_PATH = r"C:\Users\HP\Documents\documentbot\final_stackoverflow_faq.json"
CHUNK_SIZE = 250  # words per chunk
CHUNK_OVERLAP = 50  # words overlap

# REMOVED: os.makedirs("static", exist_ok=True) - NOT NEEDED

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ============================================
# HELPER FUNCTIONS
# ============================================
def clean_section_name(section):
    """Remove numbering from section names"""
    return re.sub(r'^\d+(\.\d+)*\.?\s*', '', section).strip()

def extract_sections_from_doc(doc_path):
    """Extract sections from Word document"""
    print("\n📑 Reading Word document...")
    doc = Document(doc_path)
    print(f"✓ Document loaded: {os.path.basename(doc_path)}")
    
    sections = {}
    current_section = None
    content = []
    section_order = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        # Check if this is a heading
        is_heading = (
            (para.style and any(h in para.style.name.lower() for h in ['heading', 'title'])) or
            re.match(r'^\d+(\.\d+)*\.?\s', text)
        )
        
        if is_heading:
            # Save previous section
            if current_section and content:
                sections[current_section] = "\n".join(content)
                section_order.append(current_section)
            
            # Start new section
            current_section = text
            content = []
        else:
            content.append(text)
    
    # Save last section
    if current_section and content:
        sections[current_section] = "\n".join(content)
        section_order.append(current_section)
    
    print(f"✓ {len(sections)} sections extracted")
    return sections, section_order

def load_faq_data(json_path):
    """Load FAQ data from JSON"""
    print("\n📊 Loading StackOverflow FAQ JSON...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            faq_data = json.load(f)
        print(f"✓ {len(faq_data)} FAQ entries loaded")
        return faq_data
    except Exception as e:
        print(f"✗ Error loading FAQ JSON: {e}")
        return []

def create_document_chunks(sections, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Create overlapping chunks from document sections"""
    print("\n📝 Creating document chunks...")
    chunks = []
    meta = []
    cid = 0
    
    for sec, txt in sections.items():
        if not txt or len(txt) < 50:
            continue
        
        words = txt.split()
        section_header = f"Section: {sec}\n\n"
        
        # Create overlapping chunks
        step = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk_text = section_header + " ".join(chunk_words)
            
            chunks.append(chunk_text)
            meta.append({
                "id": cid,
                "text": chunk_text,
                "section": sec,
                "type": "doc_text",
                "source": "user_guide"
            })
            cid += 1
    
    print(f"✓ {len([m for m in meta if m.get('source')=='user_guide'])} document chunks")
    return chunks, meta, cid

def create_faq_chunks(faq_data, start_id):
    """Create chunks from FAQ data"""
    print("\n📝 Creating FAQ chunks...")
    chunks = []
    meta = []
    cid = start_id
    faq_count = 0
    
    for item in faq_data:
        if 'question' in item and 'answer' in item:
            # Create searchable chunk
            qa_text = f"FAQ QUESTION: {item['question']}\n\nFAQ ANSWER: {item['answer']}"
            chunks.append(qa_text)
            meta.append({
                "id": cid,
                "text": qa_text,
                "section": f"FAQ - {item.get('source', 'StackOverflow')}",
                "type": "faq",
                "source": "stackoverflow",
                "faq_id": item.get('id'),
                "question": item.get('question'),
                "answer": item.get('answer')
            })
            cid += 1
            faq_count += 1
    
    print(f"✓ {faq_count} FAQ chunks created")
    return chunks, meta, cid

# ============================================
# MAIN BUILD PROCESS
# ============================================
print("="*50)
print("BUILDING INDEX WITH OPTIMIZED SETTINGS")
print("="*50)

# Clean old files
for f in ["faiss_index.bin", "meta_data.npy", "bm25_index.pkl", 
          "chunks.json", "chunks.json.gz"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"🗑️ Removed {f}")

# Load text model
print("\n📦 Loading text model...")
text_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("✓ Text model loaded")

# Extract sections from Word doc
sections, section_order = extract_sections_from_doc(DOC_PATH)

# Load FAQ data
faq_data = load_faq_data(JSON_PATH)

# Create all chunks
chunks = []
meta = []
cid = 0

# Document chunks
doc_chunks, doc_meta, cid = create_document_chunks(sections, CHUNK_SIZE, CHUNK_OVERLAP)
chunks.extend(doc_chunks)
meta.extend(doc_meta)

# FAQ chunks
faq_chunks, faq_meta, cid = create_faq_chunks(faq_data, cid)
chunks.extend(faq_chunks)
meta.extend(faq_meta)

print(f"\n📊 TOTAL: {len(chunks)} chunks created")
print(f"   • Document chunks: {len(doc_chunks)}")
print(f"   • FAQ chunks: {len(faq_chunks)}")

# Generate embeddings
print("\n🧮 Generating text embeddings...")
embeddings = text_model.encode(chunks, show_progress_bar=True)
faiss.normalize_L2(embeddings)
print("✓ Text embeddings generated")

# Create main FAISS index
print("\n🔍 Creating main FAISS index...")
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "faiss_index.bin")
print("✓ FAISS index saved")

# Create BM25 index
print("\n📊 Creating BM25 index...")
tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)
with open("bm25_index.pkl", 'wb') as f:
    pickle.dump(bm25, f)
print("✓ BM25 index saved")

# Save metadata
print("\n💾 Saving metadata...")
np.save("meta_data.npy", meta)
print("✓ Metadata saved")

# Save complete data with compression
print("\n💾 Saving complete data with compression...")

# Check available space before writing
import shutil
total, used, free = shutil.disk_usage(".")
print(f"   Free space: {free // (1024**2)} MB")

# Estimate file size (rough estimate)
estimated_size = len(str(chunks)) + len(str(meta)) + len(str(faq_data))
estimated_size_mb = estimated_size / (1024 * 1024)
print(f"   Estimated size: {estimated_size_mb:.1f} MB")

if free < estimated_size + 100 * 1024 * 1024:  # Need at least 100MB buffer
    print("⚠ Low disk space detected! Using compression...")
    # Save compressed version
    with gzip.open("chunks.json.gz", 'wt', encoding='utf-8') as f:
        json.dump({
            "chunks": chunks,
            "meta": meta,
            "faq_data": faq_data
        }, f, ensure_ascii=False)
    print(f"✓ Compressed data saved as chunks.json.gz ({os.path.getsize('chunks.json.gz') // (1024**2)} MB)")
else:
    # Save uncompressed
    with open("chunks.json", 'w', encoding='utf-8') as f:
        json.dump({
            "chunks": chunks,
            "meta": meta,
            "faq_data": faq_data
        }, f, indent=2, ensure_ascii=False)
    print(f"✓ Complete data saved as chunks.json ({os.path.getsize('chunks.json') // (1024**2)} MB)")

print("\n" + "="*50)
print("✅ BUILD COMPLETE")
print(f"   • Word doc sections: {len(sections)}")
print(f"   • FAQ entries: {len(faq_data)}")
print(f"   • Total chunks: {len(chunks)}")
print("="*50)