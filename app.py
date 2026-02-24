from flask import Flask, render_template, request, jsonify, send_from_directory
import faiss
import numpy as np
import os
import pickle
import json
import torch
from threading import Timer
from sentence_transformers import SentenceTransformer
import logging
import re
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import nltk
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data for BM25
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FAISS_WEIGHT = 0.6      # Weight for FAISS vector search
BM25_WEIGHT = 0.4       # Weight for BM25 keyword search
TOP_K_FAISS = 30        # Number of FAISS results to consider
TOP_K_BM25 = 20         # Number of BM25 results to consider
FINAL_TOP_K = 5         # Final number of chunks to use
SIMILARITY_THRESHOLD = 0.25
IMAGE_FOLDER = "static/images"

app = Flask(__name__, static_folder="static", static_url_path="/static")
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# ========== LOAD ALL REQUIRED COMPONENTS ==========
logger.info("=" * 60)
logger.info("LOADING ALL COMPONENTS")
logger.info("=" * 60)

# 1. Load BAAI model (for vector search)
logger.info("\n📦 Loading BAAI/bge-small-en...")
model = SentenceTransformer("BAAI/bge-small-en")
logger.info("✓ BAAI model loaded")

# 2. Load FAISS index
logger.info("\n🔍 Loading FAISS index...")
index = faiss.read_index("faiss_index.bin")
logger.info(f"✓ FAISS index loaded with {index.ntotal} vectors")

# 3. Load metadata
logger.info("\n📚 Loading metadata...")
metadata = np.load("meta_data.npy", allow_pickle=True)
section_images = np.load("heading_image_map.npy", allow_pickle=True).item()
logger.info(f"✓ Metadata loaded: {len(metadata)} entries")

# 4. Load BM25 index
logger.info("\n📊 Loading BM25 index...")
with open("bm25_index.pkl", 'rb') as f:
    bm25 = pickle.load(f)
logger.info("✓ BM25 index loaded")

# 5. Load chunks for reference
with open("chunks.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    chunks = data["chunks"]
logger.info(f"✓ Loaded {len(chunks)} chunks")

# 6. Load Mistral 7B Instruct (with 4-bit quantization to save memory)
logger.info("\n🤖 Loading Mistral 7B Instruct (this may take a few minutes)...")
try:
    # Use 4-bit quantization to fit in memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    mistral_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Create pipeline
    mistral_pipeline = pipeline(
        "text-generation",
        model=mistral_model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.15
    )
    logger.info("✓ Mistral 7B Instruct loaded successfully!")
    
except Exception as e:
    logger.error(f"✗ Failed to load Mistral: {e}")
    logger.error("Please ensure you have enough GPU memory or use CPU fallback")
    mistral_pipeline = None

logger.info("\n" + "=" * 60)
logger.info("✅ ALL COMPONENTS LOADED READY")
logger.info("=" * 60)

def open_browser():
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")

def hybrid_search(query):
    """Combine FAISS vector search + BM25 keyword search"""
    
    # 1. FAISS vector search (semantic)
    query_vec = model.encode([query])
    faiss.normalize_L2(query_vec)
    faiss_scores, faiss_indices = index.search(query_vec, TOP_K_FAISS)
    
    # 2. BM25 keyword search
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[-TOP_K_BM25:][::-1]
    
    # 3. Combine scores with weights
    combined_scores = {}
    
    # Add FAISS scores (semantic weight)
    max_faiss = max(faiss_scores[0]) if max(faiss_scores[0]) > 0 else 1
    for score, idx in zip(faiss_scores[0], faiss_indices[0]):
        if score > SIMILARITY_THRESHOLD and idx < len(metadata):
            norm_score = score / max_faiss
            combined_scores[idx] = combined_scores.get(idx, 0) + (norm_score * FAISS_WEIGHT)
    
    # Add BM25 scores (keyword weight)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    for idx in bm25_top_indices:
        if idx < len(metadata):
            norm_score = bm25_scores[idx] / max_bm25
            combined_scores[idx] = combined_scores.get(idx, 0) + (norm_score * BM25_WEIGHT)
    
    # Sort by combined score
    sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top-k results with metadata
    results = []
    seen_sections = set()
    
    for idx, score in sorted_indices[:FINAL_TOP_K]:
        if idx < len(metadata):
            meta = metadata[idx]
            section = meta.get("section", "Unknown")
            
            # Limit to 2 chunks per section for diversity
            section_count = len([r for r in results if r["section"] == section])
            if section_count >= 2:
                continue
                
            results.append({
                "text": meta.get("text", ""),
                "section": section,
                "score": float(score),
                "type": meta.get("type", "text"),
                "image": meta.get("image", None)
            })
    
    return results

def get_relevant_images(section_name):
    """Get images for a section"""
    images = []
    section_imgs = section_images.get(section_name, [])
    
    for img in section_imgs[:8]:  # Max 8 images
        img_path = f"/static/images/{img}"
        if os.path.exists(os.path.join("static/images", img)):
            images.append(img_path)
    
    return images

def generate_answer_with_mistral(query, search_results):
    """Use Mistral 7B Instruct to generate answer"""
    
    if not search_results:
        return "I couldn't find information about that in the Agent Desktop User Guide."
    
    if mistral_pipeline is None:
        # Fallback if Mistral not loaded
        return f"Based on the documentation (Section: {search_results[0]['section']}):\n\n{search_results[0]['text'][:500]}..."
    
    # Prepare context from search results
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(f"[SOURCE {i}: {result['section']}]\n{result['text']}")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt for Mistral 7B Instruct
    prompt = f"""<s>[INST] You are a helpful assistant for the Agent Desktop User Guide. Answer the user's question based ONLY on the provided documentation sources.

DOCUMENTATION SOURCES:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer directly and accurately based ONLY on the documentation provided
2. If the documentation doesn't contain the answer, say "I don't have enough information about that in the documentation"
3. Be specific and mention section names when relevant
4. Keep the answer clear, well-structured, and concise
5. If the answer involves steps, use bullet points or numbers

ANSWER: [/INST]"""
    
    try:
        # Generate with Mistral
        response = mistral_pipeline(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract answer
        full_response = response[0]['generated_text']
        answer = full_response.split("[/INST]")[-1].strip()
        
        # Clean up
        answer = re.sub(r'\s+', ' ', answer)
        
        return answer
        
    except Exception as e:
        logger.error(f"Mistral generation error: {e}")
        # Fallback
        return f"Based on the documentation (Section: {search_results[0]['section']}):\n\n{search_results[0]['text'][:500]}..."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        query = request.json["query"]
        logger.info(f"\n📝 Query: {query}")
        
        # Step 1: Hybrid search (FAISS + BM25)
        logger.info("🔍 Performing hybrid search...")
        search_results = hybrid_search(query)
        logger.info(f"✓ Found {len(search_results)} relevant chunks")
        
        if not search_results:
            return jsonify({
                "answer": "I couldn't find information about that in the Agent Desktop User Guide.",
                "images": []
            })
        
        # Step 2: Generate answer with Mistral 7B
        logger.info("🤖 Generating answer with Mistral 7B...")
        answer = generate_answer_with_mistral(query, search_results)
        
        # Step 3: Get relevant images
        top_section = search_results[0]["section"]
        images = get_relevant_images(top_section)
        logger.info(f"✓ Found {len(images)} relevant images")
        
        return jsonify({
            "answer": answer,
            "images": images,
            "section": top_section
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({
            "answer": f"Sorry, an error occurred: {str(e)}",
            "images": []
        }), 500

if __name__ == "__main__":
    Timer(1.5, open_browser).start()
    app.run(debug=True, host="127.0.0.1", port=5000)