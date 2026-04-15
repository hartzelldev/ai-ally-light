from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os

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
    