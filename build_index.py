import os,io,faiss,numpy as np,json,pickle,re,warnings
from docx import Document
from sentence_transformers import SentenceTransformer
from PIL import Image
import pytesseract
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
warnings.filterwarnings('ignore')

nltk.download('punkt',quiet=True)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Your document path
DOC_PATH = r"C:\Users\HP\OneDrive\Documents\poorvika\documentbot\AD_UserGuide_v1.4.docx"
IMG_DIR = "static/images"
os.makedirs(IMG_DIR, exist_ok=True)

# Clean old files
for f in os.listdir(IMG_DIR): os.remove(os.path.join(IMG_DIR, f))
for f in ["faiss_index.bin","meta_data.npy","bm25_index.pkl","heading_image_map.npy","chunks.json"]:
    if os.path.exists(f): os.remove(f)

print("="*50+"\nBUILDING INDEX\n"+"="*50)

def clean_section_name(section):
    """Remove numbering from section names"""
    return re.sub(r'^\d+(\.\d+)*\.?\s*', '', section).strip()

# Load model
print("\n📦 Loading BAAI/bge-small-en...")
model = SentenceTransformer("BAAI/bge-small-en")

# Read document
doc = Document(DOC_PATH)

# Extract sections
sections,cur,cont,pos = {},None,[],[]
for p in doc.paragraphs:
    t=p.text.strip()
    if not t: continue
    is_head = (p.style and any(h in p.style.name.lower() for h in ['heading','title'])) or re.match(r'^\d+(\.\d+)*\.?\s',t)
    if is_head:
        if cur and cont: 
            sections[cur]="\n".join(cont)
            pos.append(cur)
        cur,cont = t,[]
    else: cont.append(t)
if cur and cont: 
    sections[cur]="\n".join(cont)
    pos.append(cur)
print(f"✓ {len(sections)} sections")

# Extract images (directly from docx using python-docx)
print("\n🖼️ Extracting images...")
img_map,img_data = {s:[] for s in sections},{}
cnt = 0
for rel in doc.part.rels.values():
    if "image" in rel.reltype:
        try:
            img_data_bytes = rel.target_part.blob
            img = Image.open(io.BytesIO(img_data_bytes))
            if img.size[0]<50 or img.size[1]<50: continue
            
            cnt+=1
            name = f"img_{cnt}.png"
            img.save(os.path.join(IMG_DIR, name))
            
            # OCR
            ocr = pytesseract.image_to_string(img).strip()
            
            # Map to section (rough mapping)
            sec_idx = min(cnt//3, len(pos)-1) if pos else 0
            sec = pos[sec_idx] if pos else list(sections.keys())[0]
            
            img_map[sec].append(name)
            clean_sec = clean_section_name(sec)
            img_data[name] = {"file":name,"caption":clean_sec,"ocr":ocr,"section":sec}
            print(f"  ✓ {name}: {clean_sec}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
print(f"✓ {cnt} images")

# Create text chunks
print("\n📝 Creating chunks...")
chunks,meta,cid = [],[],0
for sec,txt in sections.items():
    if txt and len(txt)>50:
        words = txt.split()
        for i in range(0,len(words),150):
            chunks.append(" ".join(words[i:i+150]))
            meta.append({"id":cid,"text":chunks[-1],"section":sec,"type":"text"})
            cid+=1

# Add image chunks
for n,inf in img_data.items():
    if inf['ocr']:
        chunks.append(f"[IMAGE: {n}] {inf['ocr']}")
        meta.append({"id":cid,"text":inf['ocr'],"section":inf['section'],
                    "type":"image","image":n,"caption":inf['caption']})
        cid+=1
print(f"✓ {len(chunks)} chunks")

# Generate embeddings
print("\n🧮 Generating embeddings...")
emb = model.encode(chunks, show_progress_bar=True)
faiss.normalize_L2(emb)

# Create FAISS index
idx = faiss.IndexFlatIP(emb.shape[1])
idx.add(emb)
faiss.write_index(idx, "faiss_index.bin")

# Create BM25 index
tok = [word_tokenize(c.lower()) for c in chunks]
with open("bm25_index.pkl",'wb') as f: pickle.dump(BM25Okapi(tok), f)

# Save metadata
np.save("meta_data.npy", meta)
np.save("heading_image_map.npy", img_map)
with open("chunks.json",'w',encoding='utf-8') as f:
    json.dump({"chunks":chunks,"meta":meta,"img_data":img_data}, f, indent=2)

print("\n"+"="*50+"\n✅ BUILD COMPLETE")
print(f"   • Sections: {len(sections)}")
print(f"   • Chunks: {len(chunks)}")
print(f"   • Images: {cnt}")
print("="*50)