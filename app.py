from flask import Flask, render_template, request, jsonify
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
import webbrowser
from threading import Timer
import os

app = Flask(__name__)


SIMILARITY_THRESHOLD = 0.30
TOP_K = 12
MAX_CONTEXT_LENGTH = 3000


print("Loading embedding model...")
embed_model = SentenceTransformer("BAAI/bge-small-en")

if not os.path.exists("doc_index.faiss"):
    raise FileNotFoundError("doc_index.faiss missing. Run build_index.py")

if not os.path.exists("doc_metadata.npy"):
    raise FileNotFoundError("doc_metadata.npy missing. Run build_index.py")

print("Loading FAISS index...")
index = faiss.read_index("doc_index.faiss")
metadata = np.load("doc_metadata.npy", allow_pickle=True)

print("Index loaded successfully")

def search(query, k=TOP_K):

    query_vector = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, k)

    results = []

    for i, idx in enumerate(indices[0]):

        if idx < len(metadata):

            score = scores[0][i]

            if score >= SIMILARITY_THRESHOLD:
                item = metadata[idx]

                if "content" in item and "type" in item:
                    results.append(item)

    return results


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    try:

        if request.is_json:
            user_question = request.json.get("question")
        else:
            user_question = request.form.get("question")

        if not user_question:
            return jsonify({
                "answer": "Please ask a question.",
                "images": []
            })

        print("\nQuestion:", user_question)

    
        results = search(user_question)

        print("Retrieved chunks:", len(results))

        if len(results) == 0:
            return jsonify({
                "answer": "This topic is not available in the document.",
                "images": []
            })

        text_chunks = []

        for r in results:
            if r.get("type") == "text":
                text_chunks.append(r["content"])

        if len(text_chunks) == 0:
            return jsonify({
                "answer": "This topic is not available in the document.",
                "images": []
            })

        context = "\n\n".join(text_chunks)
        context = context[:MAX_CONTEXT_LENGTH]

        try:
            response = ollama.chat(
                model="mistral:7b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a document assistant.\n"
                            #"Answer ONLY using the given context.\n"
                            "Explain clearly in paragraph format.\n"
                            "If answer not present reply exactly:\n"
                            "This topic is not available in the document."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {user_question}"
                    }
                ]
            )

            answer = response["message"]["content"].strip()

        except Exception as e:
            print("LLM error:", e)
            return jsonify({
                "answer": "Error generating answer.",
                "images": []
            })

        image_list = []

        for r in results:
            if r.get("type") == "image" and r.get("image"):

                path = "/static/images/" + r["image"]

                if path not in image_list:
                    image_list.append(path)

        print("Images found:", len(image_list))

        return jsonify({
            "answer": answer,
            "images": image_list
        })

    except Exception as e:
        print("ASK route error:", e)

        return jsonify({
            "answer": "Internal server error. Check terminal.",
            "images": []
        })


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000")
    Timer(1, open_browser).start()
    app.run(debug=True)
