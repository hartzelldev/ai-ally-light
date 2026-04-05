"""
routes/main.py — Flask routes organized in a Blueprint
"""

import shutil
import threading
import logging
from flask import Blueprint, request, jsonify, send_from_directory
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename

from ally import (
    config, BASE_DIR, PROJECTS_DIR, PROJECTS_META, PROJECT_OVERRIDABLE,
    ALLOWED_EXTENSIONS, ENV_KEYS,
    load_config, save_config, save_to_env,
    load_projects, save_projects,
    load_project_config, save_project_config, effective_config,
    get_chroma_client, get_collection, _chroma_clients,
    project_docs_dir, project_logs_dir, append_to_log,
    project_threads_dir, load_threads, save_threads,
    create_thread, load_thread, save_thread, add_message_to_thread,
    rename_thread, delete_thread, get_thread_summary,
    get_embedding, check_embeddings_provider, check_ollama,
    split_into_sentences, extract_title, extract_sections,
    get_section_at_position, get_chunk_position, chunk_text,
    file_hash, indexed_hashes, index_file, index_project,
    get_project_index_status, delete_file_from_index, reindex_file,
    start_watcher, stop_watcher, watchers, ProjectDocHandler,
    retrieve, chat_with_llm,
    log,
)

bp = Blueprint('main', __name__)

# Root route
@bp.route("/")
def index():
    return send_from_directory("static", "index.html")

# System status
@bp.route("/api/ally/status")
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
@bp.route("/api/ally/settings", methods=["GET"])
def api_get_settings():
    safe = dict(config)
    chat_key = safe.get("chat_api_key", "")
    safe["chat_api_key_masked"] = ("sk-or-…" + chat_key[-4:]) if chat_key else ""
    safe["chat_api_key"] = ""
    safe["embed_api_key"] = ""
    return jsonify(safe)

@bp.route("/api/ally/settings", methods=["POST"])
def api_save_settings():
    data = request.json

    int_keys = ("chunk_size", "chunk_overlap", "top_k_results", "max_history_turns", "max_threads_display", "max_tokens")
    float_keys = ("temperature", "min_p", "top_p")

    env_keys_to_save = {}
    for old_key, env_key in ENV_KEYS.items():
        if old_key in data and str(data[old_key]).strip() != "":
            env_keys_to_save[env_key] = data[old_key]
            config[old_key] = data[old_key]

    for env_key, value in env_keys_to_save.items():
        save_to_env(env_key, value)

    editable = ["chunk_size", "chunk_overlap", "top_k_results", "max_history_turns",
                "max_threads_display", "max_tokens", "system_prompt"]
    for key in editable:
        if key in data and str(data[key]).strip() != "":
            val = data[key]
            if key in int_keys:
                try:
                    val = int(val)
                except ValueError:
                    continue
            config[key] = val

    for key in float_keys:
        if key in data and str(data[key]).strip() != "":
            try:
                config[key] = float(data[key])
            except ValueError:
                pass

    if "sound_enabled" in data:
        config["sound_enabled"] = bool(data["sound_enabled"])

    save_config(config)
    return jsonify({"success": True})

# Project settings (overrides only)
@bp.route("/api/ally/projects/<pid>/settings", methods=["GET"])
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

@bp.route("/api/ally/projects/<pid>/settings", methods=["POST"])
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
                if key in ("temperature", "min_p", "top_p"):
                    try:
                        val = float(val)
                    except ValueError:
                        continue
                elif key in ("top_k_results", "max_tokens"):
                    try:
                        val = int(val)
                    except ValueError:
                        continue
                overrides[key] = val
    save_project_config(pid, overrides)
    return jsonify({"success": True})

# Projects list
@bp.route("/api/ally/projects", methods=["GET"])
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
@bp.route("/api/ally/projects", methods=["POST"])
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

@bp.route("/api/ally/projects/<pid>/upload", methods=["POST"])
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
@bp.route("/api/ally/projects/<pid>", methods=["PUT"])
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
@bp.route("/api/ally/projects/<pid>", methods=["DELETE"])
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
@bp.route("/api/ally/projects/<pid>/status")
def api_project_status(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    return jsonify(get_project_index_status(pid))

@bp.route("/api/ally/projects/<pid>/model")
def api_project_model(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    cfg = effective_config(pid)
    return jsonify({
        "chat_provider": cfg.get("chat_provider", "openrouter"),
        "chat_model": cfg.get("chat_model", "openai/gpt-4o-mini"),
        "embed_provider": cfg.get("embed_provider", "ollama"),
        "embed_model": cfg.get("embed_model", "nomic-embed-text"),
    })

# Delete file from index
@bp.route("/api/ally/projects/<pid>/files", methods=["DELETE"])
def api_delete_files(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json or {}
    filenames = data.get("filenames", [])
    if not filenames:
        return jsonify({"error": "No filenames provided."}), 400
    deleted = [fname for fname in filenames if delete_file_from_index(pid, fname)]
    return jsonify({"success": True, "deleted": deleted})

# Re-index files
@bp.route("/api/ally/projects/<pid>/files/reindex", methods=["POST"])
def api_reindex_files(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json or {}
    filenames = data.get("filenames", [])
    if not filenames:
        return jsonify({"error": "No filenames provided."}), 400
    reindexed = [fname for fname in filenames if reindex_file(pid, fname)]
    return jsonify({"success": True, "reindexed": reindexed})

# Re-index project
@bp.route("/api/ally/projects/<pid>/reindex", methods=["POST"])
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
@bp.route("/api/ally/projects/<pid>/logs")
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

@bp.route("/api/ally/projects/<pid>/threads")
def api_list_threads(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404

    max_display = config.get("max_threads_display", 10)
    all_threads = request.args.get("all", "false").lower() == "true"
    all_summary = get_thread_summary(pid)
    threads = all_summary if all_threads else all_summary[:max_display]
    has_more = len(all_summary) > max_display if not all_threads else False

    return jsonify({"threads": threads, "has_more": has_more, "max_display": max_display})

@bp.route("/api/ally/projects/<pid>/threads", methods=["POST"])
def api_create_thread(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json or {}
    thread = create_thread(pid, data.get("name"))
    return jsonify(thread), 201

@bp.route("/api/ally/projects/<pid>/threads/<tid>")
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

@bp.route("/api/ally/projects/<pid>/threads/<tid>", methods=["PUT"])
def api_update_thread(pid, tid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json or {}
    if "name" in data:
        if rename_thread(pid, tid, data["name"]):
            return jsonify({"success": True})
        return jsonify({"error": "Thread not found."}), 404
    return jsonify({"error": "No updates provided."}), 400

@bp.route("/api/ally/projects/<pid>/threads/<tid>", methods=["DELETE"])
def api_delete_thread(pid, tid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    if delete_thread(pid, tid):
        return jsonify({"success": True})
    return jsonify({"error": "Thread not found."}), 404

@bp.route("/api/ally/projects/<pid>/threads/batch-delete", methods=["POST"])
def api_delete_threads_batch(pid):
    if pid not in load_projects():
        return jsonify({"error": "Project not found."}), 404
    data = request.json or {}
    tids = data.get("thread_ids", [])
    if not tids:
        return jsonify({"error": "No thread IDs provided."}), 400
    deleted = [tid for tid in tids if delete_thread(pid, tid)]
    return jsonify({"success": True, "deleted": deleted})

# Chat
@bp.route("/api/ally/projects/<pid>/chat", methods=["POST"])
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

    if not thread_id:
        thread = create_thread(pid)
        thread_id = thread["id"]

    add_message_to_thread(pid, thread_id, "user", last_user)
    add_message_to_thread(pid, thread_id, "assistant", reply, sources)

    return jsonify({
        "reply": reply,
        "sources": sources,
        "thread_id": thread_id,
        "model": cfg["chat_model"]
    })

# Export the Blueprint for registration
__all__ = ['bp']
