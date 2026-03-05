from flask import Flask, render_template, request, jsonify, send_from_directory
import faiss
import numpy as np
import os
import pickle
import json
import re
import time
import logging
import warnings
import traceback
from collections import defaultdict
import gc
import threading
import hashlib
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import ollama
import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
import pytesseract

import history

warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

TOP_K_RETRIEVAL = 8
FINAL_K = 5
RRF_K = 60
UPLOAD_DIR = "static/uploads"
IMG_DIR = "static/images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

query_cache = {}
answer_cache = {}

def get_cache_key(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="/static")


text_model = None
index = None
meta = None
sec_imgs = None
bm25 = None
image_data = {}
faq_data = []
ollama_ok = False

def load_models():
    """Load all models at startup"""
    global text_model, index, meta, sec_imgs, bm25, image_data, faq_data, ollama_ok
    
    logger.info("="*60)
    logger.info("🚀 LOADING OPTIMIZED ASSISTANT")
    logger.info("="*60)
    
    
    try:
        logger.info("📦 Loading BAAI/bge-small-en-v1.5...")
        text_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        logger.info("✅ Text model loaded")
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        text_model = None
    
   
    try:
        if os.path.exists("faiss_index.bin"):
            index = faiss.read_index("faiss_index.bin")
            logger.info(f"✅ FAISS: {index.ntotal} vectors")
        else:
            index = None
    except Exception as e:
        logger.error(f"❌ FAISS failed: {e}")
        index = None
    
    try:
        if os.path.exists("meta_data.npy"):
            meta = np.load("meta_data.npy", allow_pickle=True)
            logger.info(f"✅ Metadata: {len(meta)} entries")
        else:
            meta = []
    except Exception as e:
        logger.error(f"❌ Metadata failed: {e}")
        meta = []
    
    
    try:
        if os.path.exists("heading_image_map.npy"):
            sec_imgs = np.load("heading_image_map.npy", allow_pickle=True).item()
            logger.info(f"✅ Section images: {len(sec_imgs)}")
        else:
            sec_imgs = {}
    except Exception as e:
        logger.error(f"❌ Section images failed: {e}")
        sec_imgs = {}
    
    try:
        if os.path.exists("bm25_index.pkl"):
            with open("bm25_index.pkl", 'rb') as f:
                bm25 = pickle.load(f)
            logger.info("✅ BM25 loaded")
        else:
            bm25 = None
    except Exception as e:
        logger.error(f"❌ BM25 failed: {e}")
        bm25 = None
    
    try:
        if os.path.exists("chunks.json"):
            with open("chunks.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
               
                if "faq_data" in data:
                    faq_data = data.get("faq_data", [])
              
                image_data = data.get("img_data", {})
            logger.info(f"✅ Loaded {len(image_data)} images, {len(faq_data)} FAQs")
            
            if faq_data and len(faq_data) > 0:
                logger.info(f"🔍 Sample FAQ: {faq_data[0].get('question', 'No question')[:50]}...")
    except Exception as e:
        logger.warning(f"⚠️ Chunks error: {e}")
    
    try:
        ollama.list()
        logger.info(f"✅ Ollama available")
        ollama_ok = True
    except Exception as e:
        logger.error(f"❌ Ollama not available: {e}")
        ollama_ok = False
    
    logger.info("="*60)
    logger.info("✅ LOADING COMPLETE")
    logger.info("="*60)

load_models()

@lru_cache(maxsize=128)
def get_query_embedding_cached(query):
    """Cached embedding - MUCH faster"""
    if text_model is None:
        return None
    try:
        qv = text_model.encode([query], show_progress_bar=False)
        faiss.normalize_L2(qv)
        return qv
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None

def hybrid_search(query, top_k=TOP_K_RETRIEVAL):
    """Fast hybrid search - optimized"""
    
    if index is None or bm25 is None or text_model is None:
        return []
    
    cache_key = get_cache_key(query)
    if cache_key in query_cache:
        logger.info(f"⚡ Cache hit for: {query[:30]}...")
        return query_cache[cache_key]
  
    qv = get_query_embedding_cached(query)
    if qv is None:
        return []
    
    semantic_scores, semantic_indices = index.search(qv, top_k)
    
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]
    
  
    rrf_scores = defaultdict(float)
    
    for rank, idx in enumerate(semantic_indices[0]):
        rrf_scores[idx] += 1 / (rank + RRF_K)
    
    for rank, idx in enumerate(bm25_indices[:top_k]):
        rrf_scores[idx] += 1 / (rank + RRF_K)
    
    
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:FINAL_K]
    
   
    results = []
    for idx, score in sorted_results:
        if idx < len(meta):
            m = meta[idx]
            results.append({
                "id": idx,
                "text": m.get("text", ""),
                "section": m.get("section", "Unknown"),
                "score": score,
                "source": m.get("source", "unknown"),
                "type": m.get("type", "text"),
                "image": m.get("image") if m.get("type") == "image" else None,
                "question": m.get("question") if m.get("type") == "faq" else None,
                "answer": m.get("answer") if m.get("type") == "faq" else None
            })
    
    
    query_cache[cache_key] = results
    if len(query_cache) > 200:
        for k in list(query_cache.keys())[:50]:
            del query_cache[k]
    
    return results

def get_relevant_images(results, max_images=2):
    """Get images quickly"""
    imgs = []
    seen = set()
    
    for r in results:
        if len(imgs) >= max_images:
            break
        
        if r.get("type") == "image" and r.get("image") and r.get("image") not in seen:
            img_name = r['image']
            if os.path.exists(os.path.join(IMG_DIR, img_name)):
                imgs.append({
                    "url": f"/static/images/{img_name}",
                    "caption": image_data.get(img_name, {}).get("caption", r.get("section", "Screenshot")),
                    "section": r.get("section", "Unknown")
                })
                seen.add(img_name)
    
    return imgs


def generate_strict_answer(query, results):
    """Generate answer STRICTLY from docs/FAQ - DOCS FIRST, then FAQ"""
    
    if not results:
        return "I don't have information about that in my knowledge base. Please ask about Agent Desktop features, troubleshooting, or configuration."
   
    doc_matches = [r for r in results if r.get("type") == "doc_text"]
    faq_matches = [r for r in results if r.get("type") == "faq"]
    
    logger.info(f"📊 Found {len(doc_matches)} document chunks, {len(faq_matches)} FAQs")
    
    if doc_matches:
        logger.info("📄 Using document answer first")
        
        if ollama_ok:
            
            context_parts = []
            for i, r in enumerate(doc_matches[:3], 1):
                clean_text = r['text'].replace('\n', ' ').strip()
                if len(clean_text) > 800:
                    clean_text = clean_text[:800] + "..."
                context_parts.append(f"[{r['section']}]\n{clean_text}")
            
            ctx = "\n\n---\n\n".join(context_parts)
           
            cache_key = get_cache_key(f"doc_{query}" + ctx[:100])
            if cache_key in answer_cache:
                logger.info(f"⚡ Document answer cache hit")
                return answer_cache[cache_key]
          
            prompt = f"""<s>[INST] You are a documentation assistant. Answer ONLY from the documentation below.

USER QUESTION: {query}

DOCUMENTATION:
{ctx}

INSTRUCTIONS:
1. ONLY use information from the documentation above
2. Keep answers SHORT and TO THE POINT (max 3-4 sentences)
3. If the answer is in the docs, provide it directly
4. If not in docs, say "I don't have information about that"
5. DO NOT add any explanations, examples, or information not in the docs

YOUR ANSWER: [/INST]"""

            try:
                response = ollama.generate(
                    model='mistral:latest',
                    prompt=prompt,
                    options={
                        'temperature': 0.1,
                        'max_tokens': 300,
                        'top_p': 0.5,
                    }
                )
                answer = response['response'].strip()
                
                
                if "don't have information" in answer.lower() or "not in docs" in answer.lower():
                    
                    logger.info("📄 Document search returned no info, checking FAQs...")
                else:
                    
                    answer_cache[cache_key] = answer
                    if len(answer_cache) > 200:
                        for k in list(answer_cache.keys())[:50]:
                            del answer_cache[k]
                    return answer
                    
            except Exception as e:
                logger.error(f"LLM error: {e}")
                
                return f"**{doc_matches[0]['section']}**\n\n{doc_matches[0]['text'][:300]}..."
        else:
            
            return f"**{doc_matches[0]['section']}**\n\n{doc_matches[0]['text'][:300]}..."
    
    
    if faq_matches:
        logger.info("❓ Using FAQ answer")
        best_faq = faq_matches[0]
        answer = best_faq.get("answer", "")
        
        
        if answer and len(answer) > 10:
            source = best_faq.get("source", "Knowledge Base").upper()
            
            
            sentences = answer.split('. ')
            query_words = set(query.lower().split())
            
            
            scored_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                
                matches = sum(1 for word in query_words if word in sentence_lower and len(word) > 2)
                scored_sentences.append((matches, sentence))
            
            
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            
            
            if scored_sentences and scored_sentences[0][0] > 0:
                
                best_answer = scored_sentences[0][1]
               
                if len(scored_sentences) > 1 and len(best_answer) < 100:
                    best_answer += ". " + scored_sentences[1][1]
            else:
                
                if len(sentences) >= 2:
                    best_answer = '. '.join(sentences[:2]) + '.'
                else:
                    best_answer = answer
            
            return f"**📑 FAQ - {source}**\n\n{best_answer}"
        else:
           
            text = best_faq.get("text", "")
            if text and len(text) > 10:
                if "FAQ ANSWER:" in text:
                    parts = text.split("FAQ ANSWER:")
                    if len(parts) > 1:
                        a_part = parts[1].strip()
                        return f"**📑 FAQ**\n\n{a_part}"
                return f"**📑 FAQ**\n\n{text[:500]}..."
    
    return "I don't have information about that in my knowledge base. Please ask about Agent Desktop features, troubleshooting, or configuration."

def extract_text_from_image(image_path):
    """Fast OCR"""
    try:
        image = Image.open(image_path)
        if image.size[0] > 1000:
            ratio = 1000 / image.size[0]
            new_size = (1000, int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        text = pytesseract.image_to_string(image, config='--psm 6').strip()
        return text[:1000]
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""


@app.route("/debug-faq", methods=["GET"])
def debug_faq():
    """Debug endpoint to check FAQ data"""
    try:
        test_query = request.args.get("q", "T-Widgets")
        results = hybrid_search(test_query, top_k=10)
        
        doc_results = [r for r in results if r.get("type") == "doc_text"]
        faq_results = [r for r in results if r.get("type") == "faq"]
        
        response = {
            "query": test_query,
            "total_results": len(results),
            "doc_matches": len(doc_results),
            "faq_matches": len(faq_results),
            "faqs": []
        }
        
        for r in faq_results[:3]:
            response["faqs"].append({
                "question": r.get("question", "No question"),
                "answer_length": len(r.get("answer", "")),
                "answer_preview": r.get("answer", "")[:100] + "..." if r.get("answer") else "EMPTY",
                "source": r.get("source"),
                "score": r.get("score")
            })
        
        for r in doc_results[:2]:
            if "doc_previews" not in response:
                response["doc_previews"] = []
            response["doc_previews"].append({
                "section": r.get("section"),
                "text_preview": r.get("text", "")[:150] + "..."
            })
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/static/<path:p>")
def serve(p):
    return send_from_directory("static", p)

@app.route("/ask", methods=["POST"])
def ask():
    """Fixed route - handles both JSON and form data"""
    try:
        start_time = time.time()
        
        
        if request.is_json:
            query = request.json.get("query", "")
        else:
            query = request.form.get("query", "")
        
        if not query:
            return jsonify({"answer": "Please provide a question.", "images": []})
        
        logger.info(f"📝 Query: {query[:50]}...")
        
        results = hybrid_search(query)
        
        
        answer = generate_strict_answer(query, results)
       
        imgs = get_relevant_images(results) if results else []
        
        section = results[0]["section"] if results else "General"
        history.add(query, answer, section, imgs)
        
        elapsed = time.time() - start_time
        logger.info(f"⏱️ Response: {elapsed:.2f}s")
        
        return jsonify({
            "answer": answer,
            "images": imgs,
            "section": section
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"answer": f"Sorry, an error occurred. Please try again.", "images": []}), 500

@app.route("/ask-with-image", methods=["POST"])
def ask_with_image():
    """Fast image + text queries"""
    try:
        start_time = time.time()
        query = request.form.get("query", "").strip()
        image_file = request.files.get("image")
        
        logger.info(f"📝 Image query")
        
        extracted_text = ""
        
        if image_file and image_file.filename:
            filename = f"img_{int(time.time())}.jpg"
            img_path = os.path.join(UPLOAD_DIR, filename)
            image_file.save(img_path)
            extracted_text = extract_text_from_image(img_path)
        
        search_query = query if query else (extracted_text[:200] if extracted_text else "")
        
        if not search_query:
            return jsonify({
                "answer": "I couldn't extract text from your image. Please type your question.",
                "images": []
            })
        
        results = hybrid_search(search_query)
        answer = generate_strict_answer(search_query, results)
        
        
        if extracted_text and not results:
            answer = f"I couldn't find matching documentation. The image shows:\n\n```\n{extracted_text[:300]}\n```\n\nPlease ask a specific question about this."
        
        imgs = get_relevant_images(results) if results else []
        section = results[0]["section"] if results else "General"
        history.add(query or "Image query", answer, section, imgs)
        
        elapsed = time.time() - start_time
        logger.info(f"⏱️ Response: {elapsed:.2f}s")
        
        return jsonify({
            "answer": answer,
            "images": imgs,
            "section": section
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return jsonify({"answer": "Error processing image. Please try again.", "images": []}), 500

@app.route("/history", methods=["GET"])
def get_hist():
    return jsonify({"history": history.get()})

@app.route("/history/<int:id>", methods=["DELETE"])
def del_hist(id):
    history.delete(id)
    return jsonify({"ok": True})

@app.route("/history/<int:id>/star", methods=["POST"])
def star_hist(id):
    history.star(id)
    return jsonify({"ok": True})

@app.route("/history/clear", methods=["POST"])
def clear_hist():
    history.clear()
    return jsonify({"ok": True})

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "ollama": ollama_ok,
        "docs_loaded": len(meta) > 0 if meta else False,
        "faqs": len(faq_data),
        "status": "ready"
    })

def open_browser():
    try:
        time.sleep(1.5)
        import webbrowser
        webbrowser.open("http://127.0.0.1:5000")
    except:
        pass

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 OPTIMIZED ASSISTANT - FAST & STRICT")
    print("📡 http://127.0.0.1:5000")
    print("⚡ Target response: < 30 seconds")
    print("📋 Answer Priority: 1. Docs | 2. FAQ | 3. I don't know")
    print("="*60 + "\n")
    
    threading.Timer(1.5, open_browser).start()
    
    try:
        app.run(debug=True, port=5000, use_reloader=True, host='127.0.0.1')
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("\nPress Enter to exit...")