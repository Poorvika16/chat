from flask import Flask, render_template, request, jsonify, send_from_directory
import faiss, numpy as np, os, pickle, json, re
from threading import Timer
from sentence_transformers import SentenceTransformer
import logging
from rank_bm25 import BM25Okapi
import ollama
import nltk
from nltk.tokenize import word_tokenize
import history
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FAISS_W, BM25_W, TOP_K_F, TOP_K_B, FINAL_K = 0.6, 0.4, 30, 20, 5
SIM_THRESH, IMG_DIR = 0.25, "static/images"

app = Flask(__name__, static_folder="static", static_url_path="/static")
os.makedirs(IMG_DIR, exist_ok=True)

logger.info("="*50+"\nLOADING COMPONENTS\n"+"="*50)
model = SentenceTransformer("BAAI/bge-small-en")
index = faiss.read_index("faiss_index.bin")
meta = np.load("meta_data.npy", allow_pickle=True)
sec_imgs = np.load("heading_image_map.npy", allow_pickle=True).item()
with open("bm25_index.pkl", 'rb') as f: bm25 = pickle.load(f)

# Load image data
image_data = {}
try:
    with open("chunks.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
        image_data = data.get("img_data", {})
    logger.info(f"✓ Loaded {len(image_data)} image captions")
except Exception as e:
    logger.warning(f"No image data found: {e}")

ollama_ok = False
try:
    test = ollama.generate(model='mistral:latest', prompt='OK', options={'max_tokens':5})
    logger.info(f"✓ Ollama: {test['response'].strip()}")
    ollama_ok = True
except Exception as e:
    logger.error(f"✗ Ollama not available: {e}")

logger.info("="*50+"\n✅ READY\n"+"="*50)

def search(query):
    qv = model.encode([query])
    faiss.normalize_L2(qv)
    fs, fi = index.search(qv, TOP_K_F)
    tq = word_tokenize(query.lower())
    bs = bm25.get_scores(tq)
    bt = np.argsort(bs)[-TOP_K_B:][::-1]
    
    comb = {}
    max_f = max(fs[0]) if max(fs[0])>0 else 1
    for s,i in zip(fs[0], fi[0]):
        if s>SIM_THRESH and i<len(meta):
            comb[i] = comb.get(i,0)+(s/max_f)*FAISS_W
    max_b = max(bs) if max(bs)>0 else 1
    for i in bt:
        if i<len(meta):
            comb[i] = comb.get(i,0)+(bs[i]/max_b)*BM25_W
    
    res, seen = [], set()
    for i,s in sorted(comb.items(), key=lambda x:x[1], reverse=True)[:FINAL_K]:
        if i>=len(meta): continue
        m = meta[i]
        sec = m.get("section","Unknown")
        if sec in seen and len([r for r in res if r["section"]==sec])>=2: continue
        seen.add(sec)
        res.append({"text":m.get("text",""), "section":sec, "score":float(s)})
    return res

def get_imgs(query, results):
    imgs, seen = [], set()
    for r in results[:3]:
        for img in sec_imgs.get(r["section"], [])[:4]:
            if img in seen or len(imgs)>=8: continue
            p = f"/static/images/{img}"
            if os.path.exists(os.path.join("static/images", img)):
                cap = image_data.get(img, {}).get("caption", "Screenshot")
                imgs.append({"url": p, "caption": cap})
                seen.add(img)
    return imgs

def answer(query, results):
    if not results: return "No information found."
    if not ollama_ok:
        ans = f"**{results[0]['section']}:**\n\n"
        for i,r in enumerate(results[:3],1):
            t = r['text'].replace('\n',' ').strip()[:300]
            ans += f"{i}. {t}...\n\n"
        return ans
    
    # FIXED: No backslashes in f-strings
    context_parts = []
    for i, r in enumerate(results[:3], 1):
        clean_text = r['text'].replace('\n', ' ').strip()
        if len(clean_text) > 800:
            clean_text = clean_text[:800] + "..."
        context_parts.append(f"[{i}] {r['section']}:\n{clean_text}")
    
    ctx = "\n\n".join(context_parts)
    prompt = f"<s>[INST] Based on documentation, answer: {query}\n\n{ctx}\n\nAnswer: [/INST]"
    
    try:
        resp = ollama.generate(model='mistral:latest', prompt=prompt, options={'temperature':0.3,'max_tokens':800})
        return resp['response'].strip()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return f"**{results[0]['section']}:**\n\n{results[0]['text'][:500]}..."

@app.route("/")
def home(): return render_template("index.html")

@app.route("/static/<path:p>")
def serve(p): return send_from_directory("static", p)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        q = request.json["query"]
        logger.info(f"\n📝 {q}")
        res = search(q)
        if not res: return jsonify({"answer":"None","images":[]})
        ans = answer(q, res)
        imgs = get_imgs(q, res)
        sec = res[0]["section"]
        history.add(q, ans, sec, imgs)
        return jsonify({"answer":ans,"images":imgs,"section":sec})
    except Exception as e:
        logger.error(f"❌ {e}")
        return jsonify({"answer":"Error","images":[]}),500

@app.route("/history", methods=["GET"])
def get_hist(): return jsonify({"history":history.get()})

@app.route("/history/<int:id>", methods=["DELETE"])
def del_hist(id): history.delete(id); return jsonify({"ok":True})

@app.route("/history/<int:id>/star", methods=["POST"])
def star_hist(id): history.star(id); return jsonify({"ok":True})

@app.route("/history/clear", methods=["POST"])
def clear_hist(): history.clear(); return jsonify({"ok":True})

if __name__ == "__main__":
    Timer(1.5, lambda: __import__('webbrowser').open("http://127.0.0.1:5000")).start()
    app.run(debug=True, port=5000)