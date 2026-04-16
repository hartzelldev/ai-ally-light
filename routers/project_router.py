from fastapi import APIRouter, Request, Form, Response
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

router = APIRouter()
templates = Jinja2Templates(directory="templates")

PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

@router.get("/list", response_class=HTMLResponse)
async def list_projects(request: Request):
    """Returns a list of folders in the projects directory."""
    projects = [d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()]
    return templates.TemplateResponse(
        request=request,
        name="partials/project_list.html",
        context={"projects": projects}
    )

@router.post("/create")
async def create_project(project_name: str = Form(...)):
    """Creates a new project folder and tells the browser to reload."""
    folder_name = project_name.lower().replace(" ", "_")
    new_path = PROJECTS_DIR / folder_name
    
    if not new_path.exists():
        new_path.mkdir()
    
    # This replaces the old 'return await list_projects'
    return Response(headers={"HX-Refresh": "true"})

@router.get("/open/{project_name}", response_class=HTMLResponse)
async def open_project(request: Request, project_name: str):
    project_path = PROJECTS_DIR / project_name
    config_path = project_path / "project_config.json"
    project_env = project_path / ".env"
    
    # 1. Load project-specific keys into the environment
    if project_env.exists():
        load_dotenv(project_env, override=True)
    
    # 2. Build the configuration dictionary from Global Defaults
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
    
    # 3. Apply Local Overrides
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                active_config.update(json.load(f))
        except Exception:
            pass

    # Clean display name for the heading
    display_name = project_name.replace("_", " ").title()

    return templates.TemplateResponse(
        request=request,
        name="project_dashboard.html",
        context={
            "project_name": project_name,
            "display_name": display_name,
            "config": active_config,
            # We pass the dictionaries so the modal can build the <select> options
            "chat_providers": CHAT_PROVIDERS,
            "embed_providers": EMBED_PROVIDERS
        }
    )

@router.get("/rename-prompt/{project_name}", response_class=HTMLResponse)
async def rename_prompt(request: Request, project_name: str):
    """Returns a small modal or inline form to rename the project."""
    return templates.TemplateResponse(
        request=request,
        name="partials/rename_modal.html",
        context={"project_name": project_name}
    )

@router.post("/rename/{project_name}")
async def rename_project(project_name: str, new_name: str = Form(...)):
    """
    Renames the project folder on disk and tells the browser to refresh.
    """
    # 1. Clean the new name for a folder path
    new_folder_name = new_name.lower().replace(" ", "_")
    
    # 2. Define the paths
    old_path = PROJECTS_DIR / project_name
    new_path = PROJECTS_DIR / new_folder_name
    
    # 3. Safety Check: If the new folder name already exists, don't do anything
    if new_path.exists():
        # You could return an error message here, but for now, 
        # we'll just refresh to clear the modal.
        return Response(headers={"HX-Refresh": "true"})

    # 4. Perform the rename
    if old_path.exists():
        os.rename(old_path, new_path)
    
    # 5. Tell HTMX to reload the whole page
    return Response(headers={"HX-Refresh": "true"})

@router.delete("/delete/{project_name}")
async def delete_project(project_name: str):
    """Deletes the project and tells HTMX to refresh the page."""
    project_path = PROJECTS_DIR / project_name
    if project_path.exists():
        shutil.rmtree(project_path)
    return Response(headers={"HX-Refresh": "true"})

@router.get("/settings/{project_name}", response_class=HTMLResponse)
async def get_project_settings_modal(request: Request, project_name: str):
    project_path = PROJECTS_DIR / project_name
    config_path = project_path / "project_config.json"
    project_env = project_path / ".env"
    
    # 1. Load project keys so the modal can show if they exist
    if project_env.exists():
        load_dotenv(project_env, override=True)

    # 2. Re-build the config (same logic as open_project)
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

    # 3. Return the PARTIAL with the required variables
    return templates.TemplateResponse(
        request=request,
        name="partials/project_settings_modal.html", # Note: using the partial directly
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
    """
    Saves project-specific settings: JSON for configuration, .env for keys.
    """
    project_path = PROJECTS_DIR / project_name
    config_path = project_path / "project_config.json"
    project_env = project_path / ".env"
    
    # 1. Prepare and save non-sensitive settings to JSON
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

    # 2. Save sensitive API keys to the project-specific .env file
    # We only write the file if keys are provided to keep it clean
    env_lines = []
    if chat_api_key:
        env_lines.append(f"PROJECT_CHAT_API_KEY={chat_api_key}\n")
    if embed_api_key:
        env_lines.append(f"PROJECT_EMBED_API_KEY={embed_api_key}\n")
        
    if env_lines:
        with open(project_env, "w") as f:
            f.writelines(env_lines)
    
    # 3. Trigger a full page refresh via HTMX header
    return Response(headers={"HX-Refresh": "true"})
    