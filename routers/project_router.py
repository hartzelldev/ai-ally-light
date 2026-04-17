from fastapi import APIRouter, Request, Form, Response, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import shutil
import json
from dotenv import load_dotenv
from core.config_manager import manager
from providers.chat import CHAT_PROVIDERS
from providers.embeddings import EMBED_PROVIDERS

# If you include this in main.py with prefix="/projects", 
# then these routes will be /projects/list, /projects/open, etc.
router = APIRouter()
templates = Jinja2Templates(directory="templates")

PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

# --- HELPERS ---

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
    chat_api_key: str = Form(None),
    temperature: float = Form(...),
    system_prompt: str = Form(...),
    embed_provider: str = Form(...),
    embed_model: str = Form(...),
    embed_api_key: str = Form(None),
    embed_method: str = Form(...),
    chunk_size: int = Form(...),
    chunk_overlap: int = Form(...)
):
    project_path = PROJECTS_DIR / project_name
    config_path = project_path / "project_config.json"
    project_env = project_path / ".env"
    
    overrides = {
        "chat_provider": chat_provider,
        "chat_model": chat_model,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "embed_provider": embed_provider,
        "embed_model": embed_model,
        "embed_method": embed_method,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }
    
    with open(config_path, "w") as f:
        json.dump(overrides, f, indent=4)

    env_lines = []
    if chat_api_key:
        env_lines.append(f"PROJECT_CHAT_API_KEY={chat_api_key}\n")
    if embed_api_key:
        env_lines.append(f"PROJECT_EMBED_API_KEY={embed_api_key}\n")
        
    if env_lines:
        with open(project_env, "w") as f:
            f.writelines(env_lines)
    
    return Response(headers={"HX-Refresh": "true"})

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
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    files = get_project_files(project_name)
    return templates.TemplateResponse(
        request,  # <--- Added this as a positional argument
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
    
    