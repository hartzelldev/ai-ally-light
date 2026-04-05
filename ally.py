"""
ally.py — Multi-project RAG chat using multiple LLM and embedding providers.

Requirements:
    pip install flask chromadb requests watchdog python-dotenv

Usage:
    1. Run: python ally.py
    2. Open http://localhost:5000 in your browser. (This should happen automatically)
    3. Create projects and manage everything from the UI.
"""

import json
import os
import re
import shutil
import hashlib
import logging
import socket
import threading
import time
import webbrowser
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

import certifi
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

import requests
from flask import Flask
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
BASE_DIR      = Path(__file__).parent
CONFIG_FILE   = BASE_DIR / "config.json"
PROJECTS_DIR  = BASE_DIR / "projects"
PROJECTS_META = BASE_DIR / "projects.json"

PROJECTS_DIR.mkdir(exist_ok=True)

# ── Allowed upload extensions ─────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".txt", ".md"}

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "chat_provider": os.getenv("CHAT_PROVIDER", "openrouter"),
    "chat_api_key": "",
    "chat_model": os.getenv("CHAT_MODEL", "openai/gpt-4o-mini"),
    "chat_base_url": os.getenv("CHAT_BASE_URL", "https://openrouter.ai/api/v1"),

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
    "temperature": 0.8,
    "min_p": 0.05,
    "top_p": 1.0,
    "max_tokens": 4096,
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
    for k, v in DEFAULT_CONFIG.items():
        cfg.setdefault(k, v)
    cfg["chat_api_key"] = os.getenv("CHAT_API_KEY", "")
    cfg["embed_api_key"] = os.getenv("EMBED_API_KEY", "")
    return cfg

def save_config(cfg: dict):
    save_data = cfg.copy()
    save_data.pop("chat_api_key", None)
    save_data.pop("embed_api_key", None)
    CONFIG_FILE.write_text(json.dumps(save_data, indent=2))

config = load_config()

# ── Environment key management ────────────────────────────────────────────────
ENV_KEYS = {
    "chat_api_key":   "CHAT_API_KEY",
    "embed_api_key":  "EMBED_API_KEY",
    "chat_provider":  "CHAT_PROVIDER",
    "chat_model":     "CHAT_MODEL",
    "chat_base_url":  "CHAT_BASE_URL",
    "embed_provider": "EMBED_PROVIDER",
    "embed_model":    "EMBED_MODEL",
    "embed_base_url": "EMBED_BASE_URL",
}

def save_to_env(key: str, value: str):
    """Write or update a key=value line in the .env file."""
    env_path = BASE_DIR / ".env"
    lines = env_path.read_text().splitlines() if env_path.exists() else []

    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
            lines[i] = f"{key}={value}"
            updated = True
            break

    if not updated:
        lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n")

# ── Per-project config ────────────────────────────────────────────────────────
PROJECT_OVERRIDABLE = ["chat_model", "top_k_results", "system_prompt", "temperature", "min_p", "top_p", "max_tokens"]

def project_config_file(pid: str) -> Path:
    return PROJECTS_DIR / pid / "config.json"

def load_project_config(pid: str) -> dict:
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
    merged = dict(config)
    merged.update(load_project_config(pid))
    return merged

# ── ChromaDB — one client per project ────────────────────────────────────────
_chroma_clients: dict = {}

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
    logs_dir = project_logs_dir(pid)
    log_file = logs_dir / f"{session_id}.md"

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
    d = PROJECTS_DIR / pid / "threads"
    d.mkdir(parents=True, exist_ok=True)
    return d

def load_threads(pid: str) -> dict:
    meta_file = PROJECTS_DIR / pid / "threads.json"
    if not meta_file.exists():
        return {}
    try:
        return json.loads(meta_file.read_text())
    except Exception:
        return {}

def save_threads(pid: str, threads: dict):
    meta_file = PROJECTS_DIR / pid / "threads.json"
    meta_file.write_text(json.dumps(threads, indent=2))

def create_thread(pid: str, name: str = None) -> dict:
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
    thread_file = project_threads_dir(pid) / f"{tid}.json"
    if not thread_file.exists():
        return {"messages": []}
    try:
        return json.loads(thread_file.read_text())
    except Exception:
        return {"messages": []}

def save_thread(pid: str, tid: str, thread_data: dict):
    thread_file = project_threads_dir(pid) / f"{tid}.json"
    thread_file.write_text(json.dumps(thread_data, indent=2))

    threads = load_threads(pid)
    if tid in threads:
        threads[tid]["updated"] = datetime.now().isoformat()
        save_threads(pid, threads)

def add_message_to_thread(pid: str, tid: str, role: str, content: str, sources: list = None):
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
    threads = load_threads(pid)
    if tid not in threads:
        return False
    threads[tid]["name"] = new_name
    threads[tid]["updated"] = datetime.now().isoformat()
    save_threads(pid, threads)
    return True

def delete_thread(pid: str, tid: str) -> bool:
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

# ── Embeddings ────────────────────────────────────────────────────────────────
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
def split_into_sentences(text: str) -> list:
    sentence_pattern = r'[^.!?…\n]+(?:[.!?…]+\s*|\n+)(?=\s|$)?|[^.!?…\n]+$'
    sentences = re.findall(sentence_pattern, text, re.UNICODE)
    return [s.strip() for s in sentences if s.strip()]

def extract_title(text: str, filename: str) -> str:
    h1_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    first_line = text.split('\n')[0].strip()
    if first_line and len(first_line) <= 100:
        return first_line
    return filename

def extract_sections(text: str) -> list:
    sections = []
    for match in re.finditer(r'^(#{1,6})\s+(.+)$', text, re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        sections.append((match.start(), level, title))
    return sections

def get_section_at_position(sections: list, position: int) -> tuple:
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
    if total_chunks <= 2:
        return "full"
    ratio = chunk_index / total_chunks
    if ratio < 0.25:
        return "beginning"
    elif ratio > 0.75:
        return "end"
    return "middle"

def chunk_text(text: str, path: Path = None) -> list:
    size, overlap = config["chunk_size"], config["chunk_overlap"]

    sections = extract_sections(text) if (path and path.suffix.lower() == '.md') else []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_words = []
    current_text = ""
    chunk_start = 0

    for sentence in sentences:
        sentence_words = sentence.split()

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
indexed_hashes: dict = {}

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
        files_with_chunks[fname] = files_with_chunks.get(fname, 0) + 1

    indexed_files = [
        {"filename": fname, "chunk_count": files_with_chunks[fname]}
        for fname in sorted(files_with_chunks)
    ]
    return {"chunk_count": count, "indexed_files": indexed_files}

def delete_file_from_index(pid: str, filename: str):
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

# ── LLM chat ──────────────────────────────────────────────────────────────────
def chat_with_llm(messages: list, context_chunks: list, cfg: dict) -> str:
    provider = cfg.get("chat_provider", "openrouter")
    model = cfg.get("chat_model", "openai/gpt-4o-mini")
    api_key = cfg.get("chat_api_key", "")
    base_url = cfg.get("chat_base_url", "")
    sampling_params = {
        "temperature": float(cfg.get("temperature", 0.8)),
        "min_p": float(cfg.get("min_p", 0.05)),
        "top_p": float(cfg.get("top_p", 1.0)),
        "max_tokens": int(cfg.get("max_tokens", 4096)),
    }

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
        return chat_providers.chat_with_provider(
            provider, full_messages, model, api_key, base_url or None, sampling_params
        )
    except requests.HTTPError as e:
        try:
            detail = e.response.json()
            return f"{provider.capitalize()} error: {detail.get('error', {}).get('message', str(e))}"
        except Exception:
            return f"{provider.capitalize()} HTTP error: {e}"
    except Exception as e:
        return f"Request failed: {e}"

# ── App factory ───────────────────────────────────────────────────────────────
def create_app() -> Flask:
    """Create and configure the Flask app, registering all blueprints."""
    app = Flask(__name__, static_folder="static")

    from routes.main import bp
    app.register_blueprint(bp)

    return app

# ── Port helpers ──────────────────────────────────────────────────────────────
def find_free_port(start_port, end_port):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return None

def open_browser(port):
    time.sleep(1.5)
    url = f"http://localhost:{port}"
    print(f"--- Opening AI Ally Light at {url} ---")
    webbrowser.open(url)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for pid in load_projects():
        start_watcher(pid)
    ollama_ok, msg = check_ollama()
    log.info(msg)
    if ollama_ok:
        for pid in load_projects():
            index_project(pid)
    else:
        log.warning("Skipping indexing — fix Ollama first, then Re-index from the UI.")

    app = create_app()

    target_port = find_free_port(5000, 5020) or 5000
    threading.Thread(target=open_browser, args=(target_port,), daemon=True).start()

    log.info(f"AI Ally Light running at http://localhost:{target_port}")
    app.run(host="127.0.0.1", port=target_port, debug=False)
