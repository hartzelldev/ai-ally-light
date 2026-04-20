from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from core.database_manager import ProjectDatabase
from pathlib import Path
import os

router = APIRouter(tags=["chat"])
PROJECTS_DIR = Path("projects")

@router.get("/{project_name}/{thread_id}", response_class=HTMLResponse)
async def open_thread(request: Request, project_name: str, thread_id: int):
    """Loads the chat history for a specific thread into the dashboard."""
    project_path = PROJECTS_DIR / project_name
    db = ProjectDatabase(project_path)
    
    # 1. Fetch messages from SQLite
    messages = db.get_messages(thread_id)
    
    # 2. Return a partial that updates the chat-log and the input form
    from app import templates
    return templates.TemplateResponse(
        request=request,
        name="partials/chat_window.html",
        context={
            "project_name": project_name,
            "thread_id": thread_id,
            "messages": messages
        }
    )

@router.post("/send/{project_name}/{thread_id}", response_class=HTMLResponse)
async def send_message(
    request: Request, 
    project_name: str, 
    thread_id: int, 
    user_input: str = Form(...)
):
    """Processes a new message, saves to DB, and returns AI response."""
    project_path = PROJECTS_DIR / project_name
    db = ProjectDatabase(project_path)
    
    # 1. Save User Message to DB
    db.save_message(thread_id, role="user", content=user_input)
    
    # 2. (Placeholder for AI Logic) 
    # Eventually, this is where your LLM call goes.
    ai_response = f"I received your message for project {project_name}: {user_input}"
    
    # 3. Save AI Message to DB
    db.save_message(thread_id, role="assistant", content=ai_response)
    
    # 4. Return the new message snippets to append to the chat log
    from app import templates
    return templates.TemplateResponse(
        request=request,
        name="partials/message_pair.html",
        context={"user_msg": user_input, "ai_msg": ai_response}
    )
    