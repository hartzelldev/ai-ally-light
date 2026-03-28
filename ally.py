"""
ally.py — Multi-project RAG chat using Ollama embeddings + OpenRouter LLM.

Requirements:
    pip install flask chromadb requests watchdog python-dotenv

Usage:
    1. Run: python ally.py
    2. Open http://localhost:5000 in your browser.
    3. Create projects and manage everything from the UI.
"""

import json
import os
import shutil
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

import requests
from flask import Flask, request, jsonify, send_from_directory
import chromadb
from chromadb.config import Settings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rag_chat")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
CONFIG_FILE  = BASE_DIR / "config.json"
PROJECTS_DIR = BASE_DIR / "projects"
PROJECTS_META = BASE_DIR / "projects.json"

PROJECTS_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "openrouter_api_key": "",
    "openrouter_model": "openai/gpt-4o-mini",
    "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435"),
    "ollama_embed_model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k_results": 5,
    "system_prompt": (
        "You are a helpful assistant. Answer questions using the provided context. "
        "If the context doesn't contain relevant information, say so and answer "
        "from your general knowledge."
    )
}

def load_config() -> dict:
    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    with open(CONFIG_FILE) as f:
        cfg = json.load(f)
    # Ensure all default keys are present, even if not in the file
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    return cfg

def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

config = load_config()

# ── Per-project config ────────────────────────────────────────────────────────
# Keys that can be overridden per project
PROJECT_OVERRIDABLE = ["openrouter_model", "top_k_results", "system_prompt"]

def project_config_file(pid: str) -> Path:
    return PROJECTS_DIR / pid / "config.json"

def load_project_config(pid: str) -> dict:
    """Return only the keys this project explicitly overrides."""
    f = project_config_file(pid)
    if not f.exists():
        return {}
    try:
        with open(f) as fh:
            return json.load(fh)
    except Exception:
        return {}

def save_project_config(pid: str, overrides: dict):
    f = project_config_file(pid)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(overrides, indent=2))

def effective_config(pid: str) -> dict:
    """Global config merged with project overrides. Project wins on conflicts."""
    merged = dict(config)
    merged.update(load_project_config(pid))
    return merged

# ── ChromaDB — one client per project ────────────────────────────────────────
_chroma_clients: dict = {}  # { pid: chromadb.PersistentClient }

def get_chroma_client(pid: str):
    if pid not in _chroma_clients:
        chroma_dir = PROJECTS_DIR / pid / "chroma_db"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        _chroma_clients[pid] = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
    return _chroma_clients[pid]

def get_collection(pid: str):
    return get_chroma_client(pid).get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )

# ── Projects metadata ─────────────────────────────────────────────────────────
def load_projects() -> dict:
    if not PROJECTS_META.exists():
        return {}
    with open(PROJECTS_META) as f:
        return json.load(f)

def save_projects(projects: dict):
    PROJECTS_META.write_text(json.dumps(projects, indent=2))

def project_docs_dir(pid: str) -> Path:
    d = PROJECTS_DIR / pid / "docs"
    d.mkdir(parents=True, exist_ok=True)
    return d

def project_logs_dir(pid: str) -> Path:
    d = PROJECTS_DIR / pid / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d

def append_to_log(pid: str, session_id: str, role: str, content: str, sources: list = None):
    """Append a single message to the session log file."""
    logs_dir = project_logs_dir(pid)
    log_file = logs_dir / f"{session_id}.md"

    # Write header if file is new
    if not log_file.exists():
        projects = load_projects()
        proj_name = projects.get(pid, {}).get("name", pid)
        header = f"# Chat Log — {proj_name}\n**Session:** {session_id}\n\n---\n\n"
        log_file.write_text(header, encoding="utf-8")

    timestamp = datetime.now().strftime("%H:%M:%S")
    label = "**You**" if role == "user" else "**Assistant**"
    block = f"### {label} — {timestamp}\n\n{content}\n\n"

    if sources:
        src_lines = ", ".join(f"{s['filename']} ({s['score']*100:.1f}%)" for s in sources)
        block += f"*Sources: {src_lines}*\n\n"

    block += "---\n\n"

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(block)

# ── Ollama ────────────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list:
    url = f"{config['ollama_base_url']}/api/embeddings"
    r = requests.post(url, json={"model": config["ollama_embed_model"], "prompt": text}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]

def check_ollama() -> tuple:
    try:
        r = requests.get(f"{config['ollama_base_url']}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        em = config["ollama_embed_model"]
        available = any(m == em or m.startswith(em + ":") for m in models)
        if available:
            return True, f"Ollama ready. Embedding model '{em}' found."
        model_list = ", ".join(models) if models else "none"
        return False, (
            f"Ollama running but '{em}' not found. "
            f"Available: {model_list}. Run: ollama pull {em}"
        )
    except Exception:
        return False, f"Cannot reach Ollama at {config['ollama_base_url']}. Is it running?"

# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list:
    size, overlap = config["chunk_size"], config["chunk_overlap"]
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += size - overlap
    return chunks

def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

# ── Indexing ──────────────────────────────────────────────────────────────────
indexed_hashes: dict = {}  # { pid: { filepath: hash } }

def index_file(pid: str, path: Path):
    col = get_collection(pid)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception as e:
        log.warning(f"[{pid}] Cannot read {path}: {e}")
        return
    if not text:
        return

    fhash = file_hash(path)
    str_path = str(path)
    proj_hashes = indexed_hashes.setdefault(pid, {})
    if proj_hashes.get(str_path) == fhash:
        return

    try:
        existing = col.get(where={"source": str_path})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
    except Exception:
        pass

    chunks = chunk_text(text)
    if not chunks:
        return

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        try:
            emb = get_embedding(chunk)
        except Exception as e:
            log.error(f"[{pid}] Embedding error: {e}")
            return
        ids.append(f"{fhash}_{i}")
        embeddings.append(emb)
        documents.append(chunk)
        metadatas.append({
            "source": str_path,
            "filename": path.name,
            "chunk": i,
            "indexed_at": datetime.now().isoformat()
        })

    col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    proj_hashes[str_path] = fhash
    log.info(f"[{pid}] Indexed {path.name} — {len(chunks)} chunks.")

def index_project(pid: str):
    docs_dir = project_docs_dir(pid)
    files = list(docs_dir.rglob("*.txt")) + list(docs_dir.rglob("*.md"))
    if not files:
        log.info(f"[{pid}] No .txt or .md files found.")
        return
    log.info(f"[{pid}] Indexing {len(files)} file(s)…")
    for f in files:
        index_file(pid, f)
    log.info(f"[{pid}] Done.")

def get_project_index_status(pid: str) -> dict:
    col = get_collection(pid)
    count = col.count()
    files = sorted(set(
        m["filename"]
        for m in (col.get(include=["metadatas"])["metadatas"] or [])
    ))
    return {"chunk_count": count, "indexed_files": files}

# ── File Watcher ──────────────────────────────────────────────────────────────
watchers: dict = {}

class ProjectDocHandler(FileSystemEventHandler):
    def __init__(self, pid: str):
        self.pid = pid

    def _handle(self, path_str: str):
        path = Path(path_str)
        if path.suffix.lower() in (".txt", ".md"):
            ollama_ok, _ = check_ollama()
            if ollama_ok:
                index_file(self.pid, path)

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            p = str(Path(event.src_path))
            col = get_collection(self.pid)
            try:
                existing = col.get(where={"source": p})
                if existing["ids"]:
                    col.delete(ids=existing["ids"])
                indexed_hashes.get(self.pid, {}).pop(p, None)
            except Exception:
                pass

def start_watcher(pid: str):
    if pid in watchers:
        return
    docs_dir = project_docs_dir(pid)
    obs = Observer()
    obs.schedule(ProjectDocHandler(pid), str(docs_dir), recursive=True)
    obs.daemon = True
    obs.start()
    watchers[pid] = obs

def stop_watcher(pid: str):
    obs = watchers.pop(pid, None)
    if obs:
        obs.stop()

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(pid: str, query: str, cfg: dict = None) -> list:
    if cfg is None:
        cfg = config
    col = get_collection(pid)
    if col.count() == 0:
        return []
    try:
        q_emb = get_embedding(query)
    except Exception:
        return []
    n = min(cfg["top_k_results"], col.count())
    results = col.query(
        query_embeddings=[q_emb],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "text": doc,
            "filename": meta.get("filename", "unknown"),
            "score": round(1 - dist, 4)
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]

# ── OpenRouter ────────────────────────────────────────────────────────────────
def chat_with_llm(messages: list, context_chunks: list, cfg: dict) -> str:
    api_key = cfg.get("openrouter_api_key", "")
    if not api_key:
        return "Error: No OpenRouter API key set. Open the Settings panel and add your key."

    if context_chunks:
        ctx = "\n\n".join(
            f"[{i+1}] From '{c['filename']}':\n{c['text']}"
            for i, c in enumerate(context_chunks)
        )
        system = f"{cfg['system_prompt']}\n\n---\nRelevant context:\n\n{ctx}"
    else:
        system = cfg["system_prompt"] + "\n\n(No relevant documents found for this query.)"

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "RAG Chat"
            },
            json={
                "model": cfg["openrouter_model"],
                "messages": [{"role": "system", "content": system}] + messages,
            },
            timeout=120
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.HTTPError as e:
        try:
            detail = e.response.json()
            return f"OpenRouter error: {detail.get('error', {}).get('message', str(e))}"
        except Exception:
            return f"OpenRouter HTTP error: {e}"
    except Exception as e:
        return f"Request failed: {e}"

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# System status
@app.route("/api/ally/status")
def api_status():
    ollama_ok, ollama_msg = check_ollama()
    return jsonify({
        "ollama_ok": ollama_ok,
        "ollama_message": ollama_msg,
        "openrouter_key_set": bool(config.get("openrouter_api_key")),
        "model": config["openrouter_model"],
        "embed_model": config["ollama_embed_model"],
    })

# Settings
@app.route("/api/ally/settings", methods=["GET"])
def api_get_settings():
    safe = dict(config)
    key = safe.get("openrouter_api_key", "")
    safe["openrouter_api_key_masked"] = ("sk-or-…" + key[-4:]) if key else ""
    safe["openrouter_api_key"] = ""  # never send the real key to browser
    return jsonify(safe)

@app.route("/api/ally/settings", methods=["POST"])
def api_save_settings():
    data = request.json
    editable = [
        "openrouter_api_key", "openrouter_model",
        "ollama_base_url", "ollama_embed_model",
        "chunk_size", "chunk_overlap", "top_k_results", "system_prompt"
    ]
    for key in editable:
        if key in data and str(data[key]).strip() != "":
            val = data[key]
            if key in ("chunk_size", "chunk_overlap", "top_k_results"):
                try:
                    val = int(val)
                except ValueError:
                    continue
            config[key] = val
    save_config(config)
    return jsonify({"success": True})

# Project settings (overrides only)
@app.route("/api/ally/projects/<pid>/settings", methods=["GET"])
def api_get_project_settings(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    overrides = load_project_config(pid)
    eff = effective_config(pid)
    return jsonify({
        "effective": {k: eff.get(k) for k in PROJECT_OVERRIDABLE},
        "overrides": {k: overrides.get(k) for k in PROJECT_OVERRIDABLE},
        "global": {k: config.get(k) for k in PROJECT_OVERRIDABLE},
    })

@app.route("/api/ally/projects/<pid>/settings", methods=["POST"])
def api_save_project_settings(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json
    overrides = load_project_config(pid)
    for key in PROJECT_OVERRIDABLE:
        if key in data:
            val = data[key]
            if val is None or str(val).strip() == "":
                overrides.pop(key, None)
            else:
                if key == "top_k_results":
                    try:
                        val = int(val)
                    except ValueError:
                        continue
                overrides[key] = val
    save_project_config(pid, overrides)
    return jsonify({"success": True})

# Projects list
@app.route("/api/ally/projects", methods=["GET"])
def api_list_projects():
    projects = load_projects()
    return jsonify([
        {
            "id": pid,
            "name": meta["name"],
            "created_at": meta.get("created_at", ""),
            "docs_folder": str(project_docs_dir(pid))
        }
        for pid, meta in projects.items()
    ])

# Create project
@app.route("/api/ally/projects", methods=["POST"])
def api_create_project():
    data = request.json
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Project name is required."}), 400

    projects = load_projects()
    base_id = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-") or "project"
    pid = base_id
    counter = 1
    while pid in projects:
        pid = f"{base_id}-{counter}"
        counter += 1

    projects[pid] = {"name": name, "created_at": datetime.now().isoformat()}
    save_projects(projects)
    project_docs_dir(pid)
    start_watcher(pid)
    log.info(f"Created project '{name}' ({pid})")
    return jsonify({"id": pid, "name": name, "docs_folder": str(project_docs_dir(pid))})

# Rename project
@app.route("/api/ally/projects/<pid>", methods=["PUT"])
def api_rename_project(pid):
    projects = load_projects()
    if pid not in projects:
        return jsonify({"error": "Project not found."}), 404
    name = (request.json.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name is required."}), 400
    projects[pid]["name"] = name
    save_projects(projects)
    return jsonify({"success": True})

# Delete project
@app.route("/api/ally/projects/<pid>", methods=["DELETE"])
def api_delete_project(pid):
    projects = load_projects()
    if pid not in projects:
        return jsonify({"error": "Project not found."}), 404
    stop_watcher(pid)
    _chroma_clients.pop(pid, None)
    proj_folder = PROJECTS_DIR / pid
    if proj_folder.exists():
        shutil.rmtree(proj_folder)
    del projects[pid]
    save_projects(projects)
    indexed_hashes.pop(pid, None)
    return jsonify({"success": True})

# Project index status
@app.route("/api/ally/projects/<pid>/status")
def api_project_status(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    return jsonify(get_project_index_status(pid))

# Re-index project
@app.route("/api/ally/projects/<pid>/reindex", methods=["POST"])
def api_reindex(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    ollama_ok, msg = check_ollama()
    if not ollama_ok:
        return jsonify({"success": False, "message": msg}), 503
    indexed_hashes.pop(pid, None)
    threading.Thread(target=index_project, args=(pid,), daemon=True).start()
    return jsonify({"success": True, "message": "Re-indexing started."})

# List logs for a project
@app.route("/api/ally/projects/<pid>/logs")
def api_list_logs(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    logs_dir = project_logs_dir(pid)
    files = sorted(logs_dir.glob("*.md"), reverse=True)
    return jsonify([
        {"filename": f.name, "path": str(f), "size": f.stat().st_size}
        for f in files
    ])

# Chat
@app.route("/api/ally/projects/<pid>/chat", methods=["POST"])
def api_chat(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json
    messages = data.get("messages", [])
    session_id = data.get("session_id", datetime.now().strftime("%Y-%m-%d"))
    if not messages:
        return jsonify({"error": "No messages provided."}), 400
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    if last_user:
        append_to_log(pid, session_id, "user", last_user)

    cfg = effective_config(pid)
    chunks = retrieve(pid, last_user, cfg) if last_user else []
    reply = chat_with_llm(messages, chunks, cfg)
    sources = [{"filename": c["filename"], "score": c["score"]} for c in chunks]

    append_to_log(pid, session_id, "assistant", reply, sources)

    return jsonify({
        "reply": reply,
        "sources": sources,
        "session_id": session_id,
        "model": cfg["openrouter_model"]
    })

# ── Startup ───────────────────────────────────────────────────────────────────
def startup():
    for pid in load_projects():
        start_watcher(pid)
    ollama_ok, msg = check_ollama()
    log.info(msg)
    if ollama_ok:
        for pid in load_projects():
            index_project(pid)
    else:
        log.warning("Skipping indexing — fix Ollama first, then Re-index from the UI.")

import socket
import webbrowser
import time

def find_free_port(start_port, end_port):
    """Checks for the first available port in a given range."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return None

def startup(port):
    """
    Wait for the server to settle, then open the browser.
    Replace your existing startup logic with this if it handles browser opening.
    """
    time.sleep(1.5)
    url = f"http://localhost:{port}"
    print(f"--- Opening RAG Chat at {url} ---")
    webbrowser.open(url)

if __name__ == "__main__":
    target_port = find_free_port(5001, 5020) or 5000
    threading.Thread(target=startup, args=(target_port,), daemon=True).start()
    
    log.info(f"AI Ally Light running at http://localhost:{target_port}")
    app.run(host="127.0.0.1", port=target_port, debug=False)
    
