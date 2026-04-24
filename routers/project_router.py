from fastapi import APIRouter, Request, Form, Response, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import shutil
import json
from dotenv import load_dotenv
from typing import Optional
import gc
from core.config_manager import manager, get_active_config
from providers.chat import CHAT_PROVIDERS
from providers.embeddings import EMBED_PROVIDERS
from core.indexer import chunk_text, index_file_chunks, delete_file_from_index, clear_entire_index
from core.database_manager import ProjectDatabase


router = APIRouter()
templates = Jinja2Templates(directory="templates")

PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

# --- HELPERS ---

def get_project_files(project_name: str):
    """Helper to list files in the project's dedicated files directory."""
    project_files_path = PROJECTS_DIR / project_name / "files"
    files_info = []
    
    ignore_list = [".env", "project_config.json", "embeddings.db", "history.db"]
    
    if not project_files_path.exists():
        return []

    for item in project_files_path.iterdir():
        if item.is_file() and item.name not in ignore_list:
            files_info.append({
                "name": item.name,
                "size_kb": round(item.stat().st_size / 1024, 2)
            })
            
    return files_info

# --- PROJECT MANAGEMENT ---

@router.get("/list", response_class=HTMLResponse)
async def list_projects(request: Request):
    projects = [d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()]
    return templates.TemplateResponse(
        request=request,
        name="partials/project_list.html",
        context={"projects": projects}
    )

@router.post("/create")
async def create_project(project_name: str = Form(...)):
    folder_name = project_name.lower().replace(" ", "_")
    new_path = PROJECTS_DIR / folder_name
    if not new_path.exists():
        new_path.mkdir()
        # Automatically create the 'files' subfolder on creation
        (new_path / "files").mkdir(exist_ok=True)
    return Response(headers={"HX-Refresh": "true"})

@router.get("/open/{project_name}", response_class=HTMLResponse)
async def open_project(request: Request, project_name: str):
    project_path = PROJECTS_DIR / project_name
    
    db = ProjectDatabase(project_path) 
    active_config = get_active_config(project_name)    

    thread_limit = active_config.get("max_shown_threads", 10)
    threads = db.get_threads(limit=thread_limit)
    has_more = len(threads) >= thread_limit

    display_name = project_name.replace("_", " ").title()
    files = get_project_files(project_name) 
    
    return templates.TemplateResponse(
        request=request,
        name="project_dashboard.html",
        context={
            "project_name": project_name,
            "display_name": display_name,
            "config": active_config,
            "chat_providers": CHAT_PROVIDERS,
            "embed_providers": EMBED_PROVIDERS,
            "files": files,
            "threads": threads,
            "has_more": has_more 
        }
    )

@router.get("/rename-prompt/{project_name}", response_class=HTMLResponse)
async def get_rename_modal(request: Request, project_name: str):
    """Returns the modal for renaming a project."""
    # Clean up the name for the display (e.g., 'test14' -> 'Test14')
    display_name = project_name.replace("_", " ").capitalize()
    
    from app import templates
    return templates.TemplateResponse(
        request=request,
        name="partials/rename_modal.html",
        context={
            "project_name": project_name,
            "display_name": display_name
        }
    )

@router.post("/rename/{project_name}")
async def rename_project(request: Request, project_name: str, new_name: str = Form(...)):
    """Handles the actual folder renaming on the filesystem."""
    new_folder_name = new_name.lower().replace(" ", "_")
    old_path = PROJECTS_DIR / project_name
    new_path = PROJECTS_DIR / new_folder_name
    
    # Validation: Don't overwrite an existing project
    if not new_path.exists() and old_path.exists():
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            # For your screen reader, you might want to return an error status here
            return HTMLResponse(content=f"Error renaming folder: {str(e)}", status_code=500)
    
    # After renaming, we tell HTMX to refresh the whole page 
    # to update all sidebar links and header titles.
    return Response(headers={"HX-Refresh": "true"})

@router.delete("/delete/{project_name}")
async def delete_project(project_name: str):
    project_path = PROJECTS_DIR / project_name
    
    if project_path.exists():
        try:
            # 1. If you have a global DB manager, tell it to close connections
            # This is the most likely culprit.
            from core.database_manager import ProjectDatabase
            db = ProjectDatabase(project_path)
            
            # Explicitly close the connection if your class has a close method
            if hasattr(db, 'close'):
                db.close()
            
            # 2. Force Python to clear any lingering file handles
            del db
            gc.collect() 
            
            # 3. Now try to delete the folder
            shutil.rmtree(project_path)
            
        except PermissionError:
            return HTMLResponse(
                content="<script>alert('Error: The database is still in use. Try navigating to Home first to release the file lock.');</script>", 
                status_code=500
            )
        except Exception as e:
            return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

    return Response(headers={"HX-Refresh": "true"})

# --- SETTINGS MANAGEMENT ---

@router.get("/settings/{project_name}", response_class=HTMLResponse)
async def get_project_settings_modal(request: Request, project_name: str):
    project_path = PROJECTS_DIR / project_name
    config_path = project_path / "project_config.json"
    project_env = project_path / ".env"
    
    if project_env.exists():
        load_dotenv(project_env, override=True)

    active_config = {
        "chat_provider": manager.config.chat.provider,
        "chat_model": manager.config.chat.model,
        "chat_api_key": os.getenv("PROJECT_CHAT_API_KEY", ""),
        "temperature": manager.config.chat.temperature,
        "system_prompt": manager.config.chat.system_prompt,
        "embed_provider": manager.config.embeddings.provider,
        "embed_model": manager.config.embeddings.model,
        "embed_api_key": os.getenv("PROJECT_EMBED_API_KEY", ""),
        "embed_method": manager.config.embeddings.method,
        "chunk_size": manager.config.embeddings.chunk_size,
        "chunk_overlap": manager.config.embeddings.chunk_overlap
    }

    if config_path.exists():
        with open(config_path, "r") as f:
            active_config.update(json.load(f))

    return templates.TemplateResponse(
        request=request,
        name="partials/project_settings_modal.html",
        context={
            "project_name": project_name,
            "display_name": project_name.replace("_", " ").title(),
            "config": active_config,
            "chat_providers": CHAT_PROVIDERS,
            "embed_providers": EMBED_PROVIDERS
        }
    )

@router.post("/save-settings/{project_name}")
async def save_project_settings(
    project_name: str, 
    chat_provider: str = Form(...),
    chat_model: str = Form(...),
    chat_api_key: Optional[str] = Form(None),
    # Added a default for temperature to prevent validation errors
    temperature: float = Form(0.7),
    # Added a default for system_prompt
    system_prompt: str = Form(""),
    embed_provider: str = Form(...),
    # CHANGE: embed_model must be Optional since the HTML might hide/omit it
    embed_model: Optional[str] = Form(None),
    embed_api_key: Optional[str] = Form(None),
    embed_method: str = Form(...),
    chunk_size: Optional[int] = Form(None),
    chunk_overlap: Optional[int] = Form(None),
    delimiter: Optional[str] = Form(None)
):
    project_path = PROJECTS_DIR / project_name
    config_path = project_path / "project_config.json"
    project_env = project_path / ".env"
    
    # LOGIC: If the user chose the built-in model, we force the model name here
    # since the HTML form won't be sending it.
    final_embed_model = embed_model
    if embed_provider == "builtin":
        final_embed_model = "all-MiniLM-L6-v2"
    elif not final_embed_model:
        # Fallback for other providers if no model was specified
        final_embed_model = "nomic-embed-text" 

    # 1. Build the dictionary with the validated/defaulted values
    raw_overrides = {
        "chat_provider": chat_provider,
        "chat_model": chat_model,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "embed_provider": embed_provider,
        "embed_model": final_embed_model,
        "embed_method": embed_method,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "delimiter": delimiter
    }
    
    # 2. THE FIX: Filter out any keys that are empty strings or None.
    # This ensures that empty UI fields don't overwrite global defaults with blanks.
    # We use 'is not None' and '!= ""' to be explicit.
    final_overrides = {
        k: v for k, v in raw_overrides.items() 
        if v is not None and str(v).strip() != ""
    }
    
    # 3. Write the JSON config
    with open(config_path, "w") as f:
        json.dump(final_overrides, f, indent=4)

    # 4. Handle the .env file for API keys
    env_lines = []
    if chat_api_key and chat_api_key.strip() != "":
        env_lines.append(f"PROJECT_CHAT_API_KEY={chat_api_key}\n")
    if embed_api_key and embed_api_key.strip() != "":
        env_lines.append(f"PROJECT_EMBED_API_KEY={embed_api_key}\n")
        
    if env_lines:
        with open(project_env, "w") as f:
            f.writelines(env_lines)
    elif project_env.exists():
        # Clean up if keys were removed
        project_env.unlink()

    # Success: Send the HX-Redirect to close the modal and refresh
    return Response(
        headers={"HX-Redirect": f"/projects/open/{project_name}"}
    )

# --- FILE MANAGEMENT ---

@router.get("/files/list/{project_name}")
async def list_files(request: Request, project_name: str):
    files = get_project_files(project_name)
    return templates.TemplateResponse(
        request=request,
        name="partials/file_list_inner.html",
        context={"request": request, "project_name": project_name, "files": files}
    )

@router.post("/files/upload/{project_name}")
async def upload_file(request: Request, project_name: str, file: UploadFile = File(...)):
    project_path = PROJECTS_DIR / project_name
    upload_dir = project_path / "files"
    upload_dir.mkdir(parents=True, exist_ok=True)
    save_path = upload_dir / file.filename
    
    # 1. Save the file
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Get the config MANUALLY
    active_config = {}
    config_path = project_path / "project_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            active_config = json.load(f)
    
    # Set defaults if missing
    c_size = active_config.get("chunk_size") or 500
    c_overlap = active_config.get("chunk_overlap") or 50

    # 3. Process and Index
    try:
        content = save_path.read_text(encoding="utf-8", errors="ignore")
        # We use the raw variables here to be safe
        chunks = chunk_text(content, {"chunk_size": c_size, "chunk_overlap": c_overlap}) 
        num_indexed = index_file_chunks(project_name, file.filename, chunks)
        print(f"Success: {file.filename} indexed into {num_indexed} chunks.")
    except Exception as e:
        print(f"Indexing failed for {file.filename}: {e}")

    # 4. Refresh the UI
    files = get_project_files(project_name)
    return templates.TemplateResponse(
        request=request,
        name="partials/file_list_inner.html",
        context={"request": request, "project_name": project_name, "files": files}
    )

@router.delete("/files/delete/{project_name}/{filename}")
async def delete_file(request: Request, project_name: str, filename: str):
    # FIX: Look in the 'files' subfolder
    file_path = PROJECTS_DIR / project_name / "files" / filename
    if file_path.exists():
        file_path.unlink()
        delete_file_from_index(project_name, filename)
    
    files = get_project_files(project_name)
    return templates.TemplateResponse(
        request=request,
        name="partials/file_list_inner.html",
        context={"request": request, "project_name": project_name, "files": files}
    )

@router.post("/files/reindex/{project_name}/{filename}")
async def reindex_file(request: Request, project_name: str, filename: str):
    # FIX: Look in the 'files' subfolder
    file_path = PROJECTS_DIR / project_name / "files" / filename
    
    if file_path.exists():
        delete_file_from_index(project_name, filename)
        active_config = get_active_config(project_name)
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(content, active_config)
        index_file_chunks(project_name, filename, chunks)

    files = get_project_files(project_name)
    return templates.TemplateResponse(
        request=request,
        name="partials/file_list_inner.html",
        context={"request": request, "project_name": project_name, "files": files}
    )

@router.post("/reindex-all/{project_name}")
async def reindex_all(request: Request, project_name: str):
    clear_entire_index(project_name)
    active_config = get_active_config(project_name)
    
    # FIX: Iterate through the 'files' subfolder
    files_dir = PROJECTS_DIR / project_name / "files"
    
    if files_dir.exists():
        for file_path in files_dir.iterdir():
            if file_path.suffix in ['.txt', '.md', '.json', '.log']:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                chunks = chunk_text(content, active_config)
                index_file_chunks(project_name, file_path.name, chunks)

    files = get_project_files(project_name)
    return templates.TemplateResponse(
        request=request,
        name="partials/file_list_inner.html",
        context={"request": request, "project_name": project_name, "files": files}
    )

@router.get("/settings/embed-fields/{project_name}")
async def get_project_embed_fields(request: Request, project_name: str, embed_method: str):
    # Use the shared helper we moved to core/config_manager.py
    from core.config_manager import get_active_config
    
    config = get_active_config(project_name)
    
    # Manually override with the selection from the radio button 
    # so the partial template renders the correct fields
    config['embed_method'] = embed_method
    
    return templates.TemplateResponse(
        request=request,
        name="partials/project_embed_fields_inner.html",
        context={
            "request": request, 
            "config": config, 
            "project_name": project_name
        }
    )
