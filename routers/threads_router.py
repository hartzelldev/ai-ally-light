from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse
from core.database_manager import ProjectDatabase
from pathlib import Path
import os

router = APIRouter(tags=["threads"])

# Helper to get the absolute path to your projects folder
PROJECTS_DIR = Path("projects") 

@router.post("/create/{project_name}", response_class=HTMLResponse)
async def create_thread(request: Request, project_name: str):
    """Creates a new thread and returns the partial for the sidebar."""
    project_path = PROJECTS_DIR / project_name
    db = ProjectDatabase(project_path)
    
    # Create thread in DB
    thread_id = db.create_thread(name="New Conversation")
    
    # Fetch thread object to ensure ID and Name are correct
    thread = {"id": thread_id, "name": "New Conversation"}
    
    from app import templates
    return templates.TemplateResponse(
        request=request,
        name="partials/thread_item.html",
        context={
            "project_name": project_name, 
            "thread": thread
        }
    )

@router.get("/rename-form/{project_name}/{thread_id}", response_class=HTMLResponse)
async def get_rename_form(request: Request, project_name: str, thread_id: int):
    """Returns the dedicated partial template for renaming."""
    project_path = PROJECTS_DIR / project_name
    db = ProjectDatabase(project_path)
    
    # Fetch the thread so we have the current name for the input value
    thread = db.get_thread_by_id(thread_id)
    
    from app import templates
    return templates.TemplateResponse(
        request=request,
        name="partials/thread_rename_form.html",
        context={
            "project_name": project_name,
            "thread": thread
        }
    )

@router.post("/rename/{project_name}/{thread_id}", response_class=HTMLResponse)
async def rename_thread(request: Request, project_name: str, thread_id: int, new_name: str = Form(...)):
    """Updates the thread name and returns the standard thread_item partial."""
    project_path = PROJECTS_DIR / project_name
    db = ProjectDatabase(project_path)
    
    # Update the database
    db.rename_thread(thread_id, new_name)
    
    # Prepare updated thread object for the response
    thread = {"id": thread_id, "name": new_name}
    
    from app import templates
    return templates.TemplateResponse(
        request=request, 
        name="partials/thread_item.html",
        context={"project_name": project_name, "thread": thread}
    )

@router.delete("/delete/{project_name}/{thread_id}")
async def delete_thread(request: Request, project_name: str, thread_id: int):
    """Deletes the thread. Empty response allows HTMX to remove the <li>."""
    project_path = PROJECTS_DIR / project_name
    db = ProjectDatabase(project_path)
    db.delete_thread(thread_id)
    return HTMLResponse(content="")
    
@router.get("/item/{project_name}/{thread_id}", response_class=HTMLResponse)
async def get_thread_item(request: Request, project_name: str, thread_id: int):
    """Returns the read-only thread item partial. Used for 'Cancel' actions."""
    project_path = PROJECTS_DIR / project_name
    db = ProjectDatabase(project_path)
    
    thread = db.get_thread_by_id(thread_id)
    
    if not thread:
        return HTMLResponse(content="Thread not found", status_code=404)

    from app import templates 
    return templates.TemplateResponse(
        request=request,
        name="partials/thread_item.html", 
        context={
            "project_name": project_name, 
            "thread": thread
        }
    )
    