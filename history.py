# history.py
import json, os
from datetime import datetime
from threading import Lock

HISTORY_FILE = "chat_history.json"
lock = Lock()

def _load():
    with lock:
        if not os.path.exists(HISTORY_FILE):
            return []
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

def _save(history):
    with lock:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

def get(limit=20):
    history = _load()
    return history[-limit:]

def add(query, answer, section, images=None):
    h = _load()
    h.append({
        "id": len(h) + 1 if h else 1,
        "query": query,
        "answer": answer,
        "section": section,
        "images": images or [],
        "time": datetime.now().isoformat(),
        "star": False
    })
    _save(h)

def delete(id):
    h = _load()
    _save([c for c in h if c.get("id") != id])

def star(id):
    h = _load()
    for c in h:
        if c.get("id") == id:
            c["star"] = not c.get("star", False)
            break
    _save(h)

def clear():
    _save([])

def get_all():
    return _load()