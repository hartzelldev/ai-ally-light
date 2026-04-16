from fastapi import APIRouter, Request, Form, Response
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
async def get_project_settings(request: Request, project_name: str):
    # For now, we'll use a simplified version of the global modal
    # We can eventually make this a dedicated partial
    return templates.TemplateResponse(
        request=request,
        name="partials/project_settings_modal.html",
        context={"project_name": project_name, "config": manager.config}
    )
    