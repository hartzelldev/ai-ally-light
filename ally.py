"""
ally.py — Multi-project RAG chat using multiple LLM and embedding providers.

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
from werkzeug.utils import secure_filename
import chromadb
from chromadb.config import Settings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from providers import chat as chat_providers
from providers.embeddings import get_embedding as provider_get_embedding
from providers.embeddings import check_provider_connection

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
    "chat_provider": os.getenv("CHAT_PROVIDER", "openrouter"),
    "chat_api_key": os.getenv("CHAT_API_KEY", ""),
    "chat_model": os.getenv("CHAT_MODEL", "openai/gpt-4o-mini"),
    "chat_base_url": os.getenv("CHAT_BASE_URL", ""),
    
    "embed_provider": os.getenv("EMBED_PROVIDER", "ollama"),
    "embed_api_key": os.getenv("EMBED_API_KEY", ""),
    "embed_model": os.getenv("EMBED_MODEL", "nomic-embed-text"),
    "embed_base_url": os.getenv("EMBED_BASE_URL", "http://127.0.0.1:11434"),
    
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k_results": 5,
    "max_history_turns": 20,
    "max_threads_display": 10,
    "sound_enabled": True,
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

# ── Threads ────────────────────────────────────────────────────────────────────
def project_threads_dir(pid: str) -> Path:
    """Return the threads directory for a project, creating it if needed."""
    d = PROJECTS_DIR / pid / "threads"
    d.mkdir(parents=True, exist_ok=True)
    return d

def load_threads(pid: str) -> dict:
    """Load thread metadata for a project."""
    meta_file = PROJECTS_DIR / pid / "threads.json"
    if not meta_file.exists():
        return {}
    try:
        return json.loads(meta_file.read_text())
    except Exception:
        return {}

def save_threads(pid: str, threads: dict):
    """Save thread metadata for a project."""
    meta_file = PROJECTS_DIR / pid / "threads.json"
    meta_file.write_text(json.dumps(threads, indent=2))

def create_thread(pid: str, name: str = None) -> dict:
    """Create a new thread and return its metadata."""
    projects = load_projects()
    proj_name = projects.get(pid, {}).get("name", pid)
    
    tid = datetime.now().strftime("%Y%m%d%H%M%S")
    
    if not name:
        name = f"{proj_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    threads = load_threads(pid)
    threads[tid] = {
        "name": name,
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat()
    }
    save_threads(pid, threads)
    
    thread_file = project_threads_dir(pid) / f"{tid}.json"
    thread_file.write_text(json.dumps({"messages": []}, indent=2))
    
    log.info(f"[{pid}] Created thread '{name}' ({tid})")
    return {"id": tid, "name": name, "created": threads[tid]["created"], "updated": threads[tid]["updated"]}

def load_thread(pid: str, tid: str) -> dict:
    """Load thread conversation history."""
    thread_file = project_threads_dir(pid) / f"{tid}.json"
    if not thread_file.exists():
        return {"messages": []}
    try:
        return json.loads(thread_file.read_text())
    except Exception:
        return {"messages": []}

def save_thread(pid: str, tid: str, thread_data: dict):
    """Save thread conversation history."""
    thread_file = project_threads_dir(pid) / f"{tid}.json"
    thread_file.write_text(json.dumps(thread_data, indent=2))
    
    threads = load_threads(pid)
    if tid in threads:
        threads[tid]["updated"] = datetime.now().isoformat()
        save_threads(pid, threads)

def add_message_to_thread(pid: str, tid: str, role: str, content: str, sources: list = None):
    """Add a message to a thread."""
    thread_data = load_thread(pid, tid)
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    if sources:
        message["sources"] = sources
    thread_data["messages"].append(message)
    save_thread(pid, tid, thread_data)

def rename_thread(pid: str, tid: str, new_name: str) -> bool:
    """Rename a thread."""
    threads = load_threads(pid)
    if tid not in threads:
        return False
    threads[tid]["name"] = new_name
    threads[tid]["updated"] = datetime.now().isoformat()
    save_threads(pid, threads)
    return True

def delete_thread(pid: str, tid: str) -> bool:
    """Delete a thread and its files."""
    threads = load_threads(pid)
    if tid not in threads:
        return False
    
    del threads[tid]
    save_threads(pid, threads)
    
    thread_file = project_threads_dir(pid) / f"{tid}.json"
    if thread_file.exists():
        thread_file.unlink()
    
    log.info(f"[{pid}] Deleted thread {tid}")
    return True

def get_thread_summary(pid: str) -> list:
    """Get thread list with message counts."""
    threads = load_threads(pid)
    result = []
    for tid, meta in threads.items():
        thread_data = load_thread(pid, tid)
        message_count = len(thread_data.get("messages", []))
        result.append({
            "id": tid,
            "name": meta["name"],
            "created": meta["created"],
            "updated": meta["updated"],
            "message_count": message_count
        })
    result.sort(key=lambda x: x["updated"], reverse=True)
    return result

# ── Embeddings ─────────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list:
    provider = config.get("embed_provider", "ollama")
    model = config.get("embed_model", "nomic-embed-text")
    api_key = config.get("embed_api_key", "")
    base_url = config.get("embed_base_url", "")
    
    return provider_get_embedding(provider, text, model, api_key, base_url or None)

def check_embeddings_provider() -> tuple:
    provider = config.get("embed_provider", "ollama")
    base_url = config.get("embed_base_url", "")
    
    if provider == "ollama":
        try:
            url = (base_url or "http://127.0.0.1:11434") + "/api/tags"
            r = requests.get(url, timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            em = config.get("embed_model", "nomic-embed-text")
            available = any(m == em or m.startswith(em + ":") for m in models)
            if available:
                return True, f"Ollama ready. Embedding model '{em}' found."
            model_list = ", ".join(models) if models else "none"
            return False, (
                f"Ollama running but '{em}' not found. "
                f"Available: {model_list}. Run: ollama pull {em}"
            )
        except Exception as e:
            return False, f"Cannot reach Ollama at {base_url or 'http://127.0.0.1:11434'}: {e}"
    else:
        return check_provider_connection(provider, base_url or None)

def check_ollama() -> tuple:
    return check_embeddings_provider()

# ── Chunking ──────────────────────────────────────────────────────────────────
import re

def split_into_sentences(text: str) -> list:
    """Split text into sentences, preserving the sentence-ending punctuation."""
    sentence_pattern = r'[^.!?…\n]+(?:[.!?…]+\s*|\n+)(?=\s|$)?|[^.!?…\n]+$'
    sentences = re.findall(sentence_pattern, text, re.UNICODE)
    return [s.strip() for s in sentences if s.strip()]

def extract_title(text: str, filename: str) -> str:
    """Extract title from markdown file (first H1) or use filename."""
    h1_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    first_line = text.split('\n')[0].strip()
    if first_line and len(first_line) <= 100:
        return first_line
    return filename

def extract_sections(text: str) -> list:
    """Extract headings and their starting positions. Returns list of (position, level, title)."""
    sections = []
    for match in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        sections.append((match.start(), level, title))
    return sections

def get_section_at_position(sections: list, position: int) -> tuple:
    """Given sections and a character position, return (section_title, level).
    Returns (None, 0) if no section found."""
    if not sections:
        return None, 0
    current_section = None
    current_level = 0
    for sec_pos, sec_level, sec_title in sections:
        if sec_pos <= position:
            current_section = sec_title
            current_level = sec_level
        else:
            break
    return current_section, current_level

def get_chunk_position(total_chunks: int, chunk_index: int) -> str:
    """Return 'beginning', 'middle', or 'end' based on chunk position."""
    if total_chunks <= 2:
        return "full"
    ratio = chunk_index / total_chunks
    if ratio < 0.25:
        return "beginning"
    elif ratio > 0.75:
        return "end"
    return "middle"

def chunk_text(text: str, path: Path = None) -> list:
    """
    Split text into chunks using sentences.
    Each chunk is a dict with: text, start_pos, end_pos
    """
    size, overlap = config["chunk_size"], config["chunk_overlap"]
    
    if path and path.suffix.lower() == '.md':
        sections = extract_sections(text)
    else:
        sections = []
    
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    
    chunks = []
    current_words = []
    current_text = ""
    chunk_start = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)
        
        if not current_words:
            chunk_start = text.find(sentence)
            if chunk_start == -1:
                chunk_start = 0
        
        current_words.extend(sentence_words)
        current_text = " ".join(current_words)
        
        if len(current_words) >= size:
            end_pos = chunk_start + len(current_text)
            section, level = get_section_at_position(sections, chunk_start)
            chunks.append({
                "text": current_text,
                "start_pos": chunk_start,
                "end_pos": end_pos,
                "section": section,
                "section_level": level
            })
            
            tail_words = current_words[-overlap:] if overlap > 0 else []
            current_words = list(tail_words)
            current_text = " ".join(current_words)
            
            if tail_words:
                chunk_start = text.find(" ".join(tail_words), chunk_start + 1)
            else:
                chunk_start = end_pos
    
    if current_words:
        end_pos = chunk_start + len(current_text)
        section, level = get_section_at_position(sections, chunk_start)
        chunks.append({
            "text": current_text,
            "start_pos": chunk_start,
            "end_pos": end_pos,
            "section": section,
            "section_level": level
        })
    
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

    chunks = chunk_text(text, path)
    if not chunks:
        return

    document_title = extract_title(text, path.name)
    total_chunks = len(chunks)
    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        try:
            emb = get_embedding(chunk["text"])
        except Exception as e:
            log.error(f"[{pid}] Embedding error: {e}")
            return
        position = get_chunk_position(total_chunks, i)
        ids.append(f"{fhash}_{i}")
        embeddings.append(emb)
        documents.append(chunk["text"])
        metadatas.append({
            "source": str_path,
            "filename": path.name,
            "title": document_title,
            "chunk": i,
            "section": chunk.get("section") or "",
            "section_level": chunk.get("section_level", 0),
            "position": position,
            "indexed_at": datetime.now().isoformat()
        })

    col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    proj_hashes[str_path] = fhash
    log.info(f"[{pid}] Indexed {path.name} ({document_title}) — {len(chunks)} chunks.")

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
    
    files_with_chunks = {}
    for m in (col.get(include=["metadatas"])["metadatas"] or []):
        fname = m.get("filename", "unknown")
        if fname not in files_with_chunks:
            files_with_chunks[fname] = 0
        files_with_chunks[fname] += 1
    
    filenames = sorted(files_with_chunks.keys())
    indexed_files = [
        {"filename": fname, "chunk_count": files_with_chunks[fname]}
        for fname in filenames
    ]
    
    return {"chunk_count": count, "indexed_files": indexed_files}

def delete_file_from_index(pid: str, filename: str):
    """Remove a file's chunks from the index."""
    col = get_collection(pid)
    try:
        existing = col.get(where={"filename": filename})
        if existing["ids"]:
            col.delete(ids=existing["ids"])
            log.info(f"[{pid}] Removed '{filename}' from index.")
            return True
    except Exception as e:
        log.error(f"[{pid}] Error removing '{filename}': {e}")
    return False

def reindex_file(pid: str, filename: str):
    """Re-index a single file."""
    docs_dir = project_docs_dir(pid)
    filepath = docs_dir / filename
    if filepath.exists():
        indexed_hashes.get(pid, {}).pop(str(filepath), None)
        index_file(pid, filepath)
        return True
    return False

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
            "title": meta.get("title", meta.get("filename", "unknown")),
            "section": meta.get("section"),
            "section_level": meta.get("section_level", 0),
            "position": meta.get("position", "unknown"),
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
    provider = cfg.get("chat_provider", "openrouter")
    model = cfg.get("chat_model", "openai/gpt-4o-mini")
    api_key = cfg.get("chat_api_key", "")
    base_url = cfg.get("chat_base_url", "")
    
    if chat_providers.CHAT_PROVIDERS.get(provider, {}).get("requires_api_key", True) and not api_key:
        provider_name = chat_providers.CHAT_PROVIDERS.get(provider, {}).get("name", provider)
        return f"Error: No API key set for {provider_name}. Open the Settings panel and add your key."

    if context_chunks:
        ctx = "\n\n".join(
            f"[{i+1}] From '{c['filename']}':\n{c['text']}"
            for i, c in enumerate(context_chunks)
        )
        system = f"{cfg['system_prompt']}\n\n---\nRelevant context:\n\n{ctx}"
    else:
        system = cfg["system_prompt"] + "\n\n(No relevant documents found for this query.)"

    full_messages = [{"role": "system", "content": system}] + messages

    try:
        return chat_providers.chat_with_provider(provider, full_messages, model, api_key, base_url or None)
    except requests.HTTPError as e:
        try:
            detail = e.response.json()
            return f"{provider.capitalize()} error: {detail.get('error', {}).get('message', str(e))}"
        except Exception:
            return f"{provider.capitalize()} HTTP error: {e}"
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
        "embed_ok": ollama_ok,
        "embed_message": ollama_msg,
        "chat_key_set": bool(config.get("chat_api_key")),
        "chat_provider": config.get("chat_provider", "openrouter"),
        "chat_model": config.get("chat_model", "openai/gpt-4o-mini"),
        "embed_provider": config.get("embed_provider", "ollama"),
        "embed_model": config.get("embed_model", "nomic-embed-text"),
    })

# Settings
@app.route("/api/ally/settings", methods=["GET"])
def api_get_settings():
    safe = dict(config)
    chat_key = safe.get("chat_api_key", "")
    embed_key = safe.get("embed_api_key", "")
    safe["chat_api_key_masked"] = ("sk-or-…" + chat_key[-4:]) if chat_key else ""
    safe["chat_api_key"] = ""  # never send the real key to browser
    safe["embed_api_key"] = ""  # never send the real key to browser
    return jsonify(safe)

ENV_KEYS = {
    "chat_provider": "CHAT_PROVIDER",
    "chat_api_key": "CHAT_API_KEY",
    "chat_model": "CHAT_MODEL",
    "chat_base_url": "CHAT_BASE_URL",
    "embed_provider": "EMBED_PROVIDER",
    "embed_api_key": "EMBED_API_KEY",
    "embed_model": "EMBED_MODEL",
    "embed_base_url": "EMBED_BASE_URL",
}

def save_to_env(key: str, value: str):
    """Save a key-value pair to .env file."""
    env_path = BASE_DIR / ".env"
    lines = []
    key_written = False
    
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    key_written = True
                else:
                    lines.append(line)
    
    if not key_written:
        lines.append(f"{key}={value}\n")
    
    with open(env_path, "w") as f:
        f.writelines(lines)

@app.route("/api/ally/settings", methods=["POST"])
def api_save_settings():
    data = request.json
    
    env_keys_to_save = {}
    int_keys = ("chunk_size", "chunk_overlap", "top_k_results", "max_history_turns", "max_threads_display")
    
    for old_key, env_key in ENV_KEYS.items():
        if old_key in data and str(data[old_key]).strip() != "":
            env_keys_to_save[env_key] = data[old_key]
            config[old_key] = data[old_key]
    
    for key in env_keys_to_save:
        save_to_env(key, env_keys_to_save[key])
    
    editable = ["chunk_size", "chunk_overlap", "top_k_results", "max_history_turns", "max_threads_display", "system_prompt"]
    for key in editable:
        if key in data and str(data[key]).strip() != "":
            val = data[key]
            if key in int_keys:
                try:
                    val = int(val)
                except ValueError:
                    continue
            config[key] = val
    
    if "sound_enabled" in data:
        config["sound_enabled"] = bool(data["sound_enabled"])
    
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

ALLOWED_EXTENSIONS = {'.txt', '.md'}

@app.route("/api/ally/projects/<pid>/upload", methods=["POST"])
def api_upload_document(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400
    
    file = request.files['file']
    original_filename = file.filename or ''
    if not original_filename:
        return jsonify({"error": "No file selected."}), 400
    
    filename = secure_filename(original_filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400
    ext = Path(filename).suffix.lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"File type {ext} not allowed. Use .txt or .md"}), 400
    
    docs_dir = project_docs_dir(pid)
    filepath = docs_dir / filename
    
    try:
        file.save(str(filepath))
        log.info(f"[{pid}] Uploaded {filename}")
        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        log.error(f"[{pid}] Upload error: {e}")
        return jsonify({"error": str(e)}), 500

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

# Delete file from index
@app.route("/api/ally/projects/<pid>/files", methods=["DELETE"])
def api_delete_files(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json or {}
    filenames = data.get("filenames", [])
    if not filenames:
        return jsonify({"error": "No filenames provided."}), 400
    deleted = []
    for fname in filenames:
        if delete_file_from_index(pid, fname):
            deleted.append(fname)
    return jsonify({"success": True, "deleted": deleted})

# Re-index files
@app.route("/api/ally/projects/<pid>/files/reindex", methods=["POST"])
def api_reindex_files(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json or {}
    filenames = data.get("filenames", [])
    if not filenames:
        return jsonify({"error": "No filenames provided."}), 400
    reindexed = []
    for fname in filenames:
        if reindex_file(pid, fname):
            reindexed.append(fname)
    return jsonify({"success": True, "reindexed": reindexed})

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

# ── Threads API ────────────────────────────────────────────────────────────────

@app.route("/api/ally/projects/<pid>/threads")
def api_list_threads(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    
    max_display = config.get("max_threads_display", 10)
    all_threads = request.args.get("all", "false").lower() == "true"
    
    threads = get_thread_summary(pid)
    
    if not all_threads:
        threads = threads[:max_display]
    
    has_more = len(get_thread_summary(pid)) > max_display if not all_threads else False
    
    return jsonify({
        "threads": threads,
        "has_more": has_more,
        "max_display": max_display
    })

@app.route("/api/ally/projects/<pid>/threads", methods=["POST"])
def api_create_thread(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json or {}
    name = data.get("name")
    thread = create_thread(pid, name)
    return jsonify(thread), 201

@app.route("/api/ally/projects/<pid>/threads/<tid>")
def api_get_thread(pid, tid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    thread_data = load_thread(pid, tid)
    if not thread_data:
        return jsonify({"error": "Thread not found."}), 404
    
    max_turns = config.get("max_history_turns", 20)
    messages = thread_data.get("messages", [])
    
    return jsonify({
        "messages": messages[-max_turns:] if len(messages) > max_turns else messages,
        "total_messages": len(messages),
        "showing_messages": min(len(messages), max_turns)
    })

@app.route("/api/ally/projects/<pid>/threads/<tid>", methods=["PUT"])
def api_update_thread(pid, tid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json or {}
    
    if "name" in data:
        if rename_thread(pid, tid, data["name"]):
            return jsonify({"success": True})
        return jsonify({"error": "Thread not found."}), 404
    
    return jsonify({"error": "No updates provided."}), 400

@app.route("/api/ally/projects/<pid>/threads/<tid>", methods=["DELETE"])
def api_delete_thread(pid, tid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    if delete_thread(pid, tid):
        return jsonify({"success": True})
    return jsonify({"error": "Thread not found."}), 404

@app.route("/api/ally/projects/<pid>/threads/batch-delete", methods=["POST"])
def api_delete_threads_batch(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json or {}
    tids = data.get("thread_ids", [])
    if not tids:
        return jsonify({"error": "No thread IDs provided."}), 400
    
    deleted = []
    for tid in tids:
        if delete_thread(pid, tid):
            deleted.append(tid)
    
    return jsonify({"success": True, "deleted": deleted})

# Chat
@app.route("/api/ally/projects/<pid>/chat", methods=["POST"])
def api_chat(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json
    messages = data.get("messages", [])
    thread_id = data.get("thread_id")
    
    if not messages:
        return jsonify({"error": "No messages provided."}), 400
    
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

    cfg = effective_config(pid)
    chunks = retrieve(pid, last_user, cfg) if last_user else []
    reply = chat_with_llm(messages, chunks, cfg)
    sources = [{"filename": c["filename"], "score": c["score"]} for c in chunks]

    if thread_id:
        add_message_to_thread(pid, thread_id, "user", last_user)
        add_message_to_thread(pid, thread_id, "assistant", reply, sources)
    else:
        thread = create_thread(pid)
        thread_id = thread["id"]
        add_message_to_thread(pid, thread_id, "user", last_user)
        add_message_to_thread(pid, thread_id, "assistant", reply, sources)

    return jsonify({
        "reply": reply,
        "sources": sources,
        "thread_id": thread_id,
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
    
