from fastapi import APIRouter, Request, Form, Response, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import shutil
import json
from dotenv import load_dotenv
from typing import Optional
from core.config_manager import manager
from providers.chat import CHAT_PROVIDERS
from providers.embeddings import EMBED_PROVIDERS
from core.indexer import chunk_text, index_file_chunks, delete_file_from_index, clear_entire_index

# If you include this in main.py with prefix="/projects", 
# then these routes will be /projects/list, /projects/open, etc.
router = APIRouter()
templates = Jinja2Templates(directory="templates")

PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

# --- HELPERS ---

def get_active_config(project_name: str):
    """Helper to get the merged global and local project settings."""
    project_path = PROJECTS_DIR / project_name
    config_path = project_path / "project_config.json"
    project_env = project_path / ".env"
    
    # 1. Start with Global Defaults
    active_config = {
        "chat_provider": manager.config.chat.provider,
        "chat_model": manager.config.chat.model,
        "temperature": manager.config.chat.temperature,
        "system_prompt": manager.config.chat.system_prompt,
        "embed_provider": manager.config.embeddings.provider,
        "embed_model": manager.config.embeddings.model,
        "embed_method": manager.config.embeddings.method,
        "chunk_size": manager.config.embeddings.chunk_size,
        "chunk_overlap": manager.config.embeddings.chunk_overlap,
        "delimiter": getattr(manager.config.embeddings, 'delimiter', '\n\n')
    }
    
    # 2. Load Environment Keys (if they exist)
    if project_env.exists():
        load_dotenv(project_env, override=True)
        active_config["chat_api_key"] = os.getenv("PROJECT_CHAT_API_KEY", "")
        active_config["embed_api_key"] = os.getenv("PROJECT_EMBED_API_KEY", "")

    # 3. Apply Local JSON Overrides
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                active_config.update(json.load(f))
        except Exception:
            pass
            
    return active_config

def get_project_files(project_name: str):
    """Helper to list files in the project directory."""
    project_path = PROJECTS_DIR / project_name
    files_info = []
    
    # Ignore internal configuration and database files
    ignore_list = [".env", "project_config.json", "embeddings.db"]
    
    if not project_path.exists():
        return []

    for item in project_path.iterdir():
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
    return Response(headers={"HX-Refresh": "true"})

@router.get("/open/{project_name}", response_class=HTMLResponse)
async def open_project(request: Request, project_name: str):
    project_path = PROJECTS_DIR / project_name
    config_path = project_path / "project_config.json"
    project_env = project_path / ".env"
    
    if project_env.exists():
        load_dotenv(project_env, override=True)
    
    active_config = get_active_config(project_name)    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                active_config.update(json.load(f))
        except Exception:
            pass

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
            "files": files
        }
    )

@router.post("/rename/{project_name}")
async def rename_project(project_name: str, new_name: str = Form(...)):
    new_folder_name = new_name.lower().replace(" ", "_")
    old_path = PROJECTS_DIR / project_name
    new_path = PROJECTS_DIR / new_folder_name
    
    if not new_path.exists() and old_path.exists():
        os.rename(old_path, new_path)
    
    return Response(headers={"HX-Refresh": "true"})

@router.delete("/delete/{project_name}")
async def delete_project(project_name: str):
    project_path = PROJECTS_DIR / project_name
    if project_path.exists():
        shutil.rmtree(project_path)
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
        final_embed_model = "nomic-embed-text" # Or your preferred default

    # We build the dictionary with the validated/defaulted values
    overrides = {
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
    
    # Write the JSON config
    with open(config_path, "w") as f:
        json.dump(overrides, f, indent=4)

    # Handle the .env file for API keys
    env_lines = []
    if chat_api_key:
        env_lines.append(f"PROJECT_CHAT_API_KEY={chat_api_key}\n")
    if embed_api_key:
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
        request,  # <--- Added this as a positional argument
        name="partials/file_list_inner.html", 
        context={"request": request, "project_name": project_name, "files": files}
    )

@router.post("/files/upload/{project_name}")
async def upload_file(request: Request, project_name: str, file: UploadFile = File(...)):
    project_path = PROJECTS_DIR / project_name
    save_path = project_path / file.filename
    
    # 1. Save the actual file to disk
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
# 2. Get the config
    active_config = get_active_config(project_name) 

    # 3. Process and Index
    try:
        content = save_path.read_text(encoding="utf-8", errors="ignore")
        
        # Match 'active_config' here
        chunks = chunk_text(content, active_config) 
        
        # Capture the return value so 'num_indexed' actually exists
        num_indexed = index_file_chunks(project_name, file.filename, chunks)
        
        print(f"Success: {file.filename} indexed into {num_indexed} chunks.")
    except Exception as e:
        print(f"Indexing failed for {file.filename}: {e}")
    # 4. Refresh the UI
    files = get_project_files(project_name)
    return templates.TemplateResponse(
        request, 
        name="partials/file_list_inner.html",
        context={"request": request, "project_name": project_name, "files": files}
    )

@router.delete("/files/delete/{project_name}/{filename}")
async def delete_file(request: Request, project_name: str, filename: str):
    file_path = PROJECTS_DIR / project_name / filename
    if file_path.exists():
        file_path.unlink()
    
    # After deleting, we fetch the new list and return the same partial
    files = get_project_files(project_name)
    return templates.TemplateResponse(
        request, 
        name="partials/file_list_inner.html",
        context={"request": request, "project_name": project_name, "files": files}
    )
    

@router.post("/files/reindex/{project_name}/{filename}")
async def reindex_file(request: Request, project_name: str, filename: str):
    project_path = PROJECTS_DIR / project_name
    file_path = project_path / filename
    
    if file_path.exists():
        # 1. Clear old data
        delete_file_from_index(project_name, filename)
        
        # 2. Re-index with current settings
        active_config = get_active_config(project_name)
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(content, active_config)
        index_file_chunks(project_name, filename, chunks)
        print(f"Reindexed: {filename} into {len(chunks)} chunks.")

    files = get_project_files(project_name)
    return templates.TemplateResponse(
        request, 
        name="partials/file_list_inner.html",
        context={"request": request, "project_name": project_name, "files": files}
    )

@router.post("/reindex-all/{project_name}")
async def reindex_all(request: Request, project_name: str):
    # 1. Nuke the whole DB
    clear_entire_index(project_name)
    
    # 2. Loop through all files on disk and index them
    active_config = get_active_config(project_name)
    project_path = PROJECTS_DIR / project_name
    
    for file_path in project_path.iterdir():
        # Only index common text files to avoid binary junk
        if file_path.suffix in ['.txt', '.md', '.json', '.log']:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text(content, active_config)
            index_file_chunks(project_name, file_path.name, chunks)
            print(f"Reindexed: {file_path.name}")

    files = get_project_files(project_name)
    return templates.TemplateResponse(
        request, 
        name="partials/file_list_inner.html",
        context={"request": request, "project_name": project_name, "files": files}
    )

@router.get("/settings/embed-fields/{project_name}")
async def get_project_embed_fields(request: Request, project_name: str, embed_method: str):
    # Fetch existing config
    config = get_active_config(project_name)
    
    # Update the method in the config object passed to the template
    config['embed_method'] = embed_method
    
    return templates.TemplateResponse(
        request,
        name="partials/project_embed_fields_inner.html",
        context={"request": request, "config": config}
    )
