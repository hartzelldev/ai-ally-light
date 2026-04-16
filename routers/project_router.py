from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import shutil

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

@router.post("/create", response_class=HTMLResponse)
async def create_project(request: Request, project_name: str = Form(...)):
    """Creates a new project folder and returns the updated list."""
    # Clean the name for a folder (lowercase, no spaces)
    folder_name = project_name.lower().replace(" ", "_")
    new_path = PROJECTS_DIR / folder_name
    
    if not new_path.exists():
        new_path.mkdir()
        # You could also initialize a blank config here
    
    # Trigger a refresh of the project list
    return await list_projects(request)
    
@router.get("/open/{project_name}", response_class=HTMLResponse)
async def open_project(request: Request, project_name: str):
    """
    Renders the main project dashboard.
    Swaps the 'main-content' div with the project UI.
    """
    # In the future, we'll load project-specific config here
    return templates.TemplateResponse(
        request=request,
        name="project_dashboard.html",
        context={
            "project_name": project_name,
            "display_name": project_name.replace("_", " ").capitalize()
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

@router.post("/rename/{project_name}", response_class=HTMLResponse)
async def rename_project(request: Request, project_name: str, new_name: str = Form(...)):
    """Renames the folder and returns the project list with a success message."""
    new_folder_name = new_name.lower().replace(" ", "_")
    old_path = PROJECTS_DIR / project_name
    new_path = PROJECTS_DIR / new_folder_name
    
    # 1. Check for collision
    if new_path.exists():
        return f"<span role='alert' style='color: red;'>Error: A project named {new_name} already exists.</span>"

    # 2. Perform the rename on the disk
    os.rename(old_path, new_path)
    
    # 3. Get the fresh list of folders
    projects = [d.name for d in PROJECTS_DIR.iterdir() if d.is_dir()]
    
    # 4. Return the template WITH the message context
    return templates.TemplateResponse(
        request=request,
        name="partials/project_list.html",
        context={
            "projects": projects,
            "message": f"Success: Project renamed to {new_name}"
        }
    )

@router.delete("/delete/{project_name}")
async def delete_project(project_name: str):
    """Deletes the project and tells HTMX to refresh the page."""
    project_path = PROJECTS_DIR / project_name
    if project_path.exists():
        shutil.rmtree(project_path)
    
    # Returning this header tells HTMX to refresh the entire page automatically
    return HTMLResponse(content="", headers={"HX-Refresh": "true"})

@router.get("/settings/{project_name}", response_class=HTMLResponse)
async def get_project_settings(request: Request, project_name: str):
    # For now, we'll use a simplified version of the global modal
    # We can eventually make this a dedicated partial
    return templates.TemplateResponse(
        request=request,
        name="partials/project_settings_modal.html",
        context={"project_name": project_name, "config": manager.config}
    )
    