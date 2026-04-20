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
    
    # Default name for a brand new thread
    thread_id = db.create_thread(name="New Conversation")
    
    # Fetch the thread object back to pass to the template
    # We grab the first one because it will be the newest (by ID)
    thread = {"id": thread_id, "name": "New Conversation"}
    
    # Return just the single thread item to be swapped into the list via HTMX
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
async def get_rename_form(request: Request, project_name: str, thread_id: int, current_name: str = ""):
    """Returns a small inline form to rename the thread."""
    # This replaces the thread link with an input field for the screen reader
    return f"""
    <form hx-post="/threads/rename/{project_name}/{thread_id}" hx-target="closest .thread-item" hx-swap="outerHTML">
        <label for="new_name" class="visually-hidden">New Thread Name</label>
        <input type="text" id="new_name" name="new_name" value="{current_name}" autofocus onfocus="this.select()">
        <button type="submit" aria-label="Confirm Rename">✅</button>
        <button type="button" hx-get="/threads/item/{project_name}/{thread_id}" hx-target="closest .thread-item" hx-swap="outerHTML" aria-label="Cancel">❌</button>
    </form>
    """

@router.post("/rename/{project_name}/{thread_id}", response_class=HTMLResponse)
async def rename_thread(request: Request, project_name: str, thread_id: int, new_name: str = Form(...)):
    """Updates the thread name and returns the updated partial."""
    project_path = PROJECTS_DIR / project_name
    db = ProjectDatabase(project_path)
    db.rename_thread(thread_id, new_name)
    
    thread = {"id": thread_id, "name": new_name}
    from app import templates
    return templates.TemplateResponse(
        request=request, 
        name="partials/thread_item.html",
        context={"project_name": project_name, "thread": thread}
    )

@router.delete("/delete/{project_name}/{thread_id}")
async def delete_thread(request: Request, project_name: str, thread_id: int):
    """Deletes the thread. Returns empty string so HTMX removes the element."""
    project_path = PROJECTS_DIR / project_name
    db = ProjectDatabase(project_path)
    db.delete_thread(thread_id)
    return HTMLResponse(content="")
    
@router.get("/item/{project_name}/{thread_id}", response_class=HTMLResponse)
async def get_thread_item(request: Request, project_name: str, thread_id: int):
    project_path = PROJECTS_DIR / project_name
    db = ProjectDatabase(project_path)
    
    # Fetch the thread details
    thread = db.get_thread_by_id(thread_id)
    
    if not thread:
        return HTMLResponse(content="Thread not found", status_code=404)

    # Re-import templates if needed, or use your global instance
    from main import templates 
    return templates.TemplateResponse(
        "partials/thread_item.html", 
        {
            "request": request, 
            "project_name": project_name, 
            "thread": thread
        }
    )
    