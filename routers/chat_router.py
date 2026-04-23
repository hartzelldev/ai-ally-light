from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from core.database_manager import ProjectDatabase
from pathlib import Path
import os
from core.indexer import query_vector_db
from core.config_manager import PROJECTS_DIR 
import openai

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
    """
    Processes a new message, saves to DB, queries RAG context, 
    calls the appropriate LLM provider, and returns the AI response.
    """
    # 1. Setup paths and database
    # Assuming PROJECTS_DIR is imported at the top of your file
    project_path = PROJECTS_DIR / project_name
    db = ProjectDatabase(project_path)
    
    # 2. Save User Message to Database
    db.save_message(thread_id, role="user", content=user_input)
    
    # 3. Load Project Configuration (Merged Global + Local)
    from core.config_manager import get_active_config
    config = get_active_config(project_name)
    
    # 4. Retrieval (RAG): Get context from the vector database
    from core.indexer import query_vector_db
    try:
        context_text = query_vector_db(project_name, user_input)
    except Exception as e:
        context_text = f"Error retrieving context: {str(e)}"
    
    # 5. Initialize AI Logic using AIEngine
    from core.ai_logic import AIEngine
    provider_name = config.get("chat_provider", "openrouter").lower()
    ai_engine = AIEngine(config=config)
    
    # 6. Construct the messages for the LLM
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    
    augmented_user_input = (
        f"Use the following context from the project files to answer the user's question.\n"
        f"Context:\n{context_text}\n\n"
        f"User Question: {user_input}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": augmented_user_input}
    ]

    # 7. Get AI Response
    try:
        ai_response = ai_engine.get_response(messages)
    except Exception as e:
        ai_response = f"AI Error ({provider_name}): {str(e)}"
    
    # 8. Save AI Response to Database
    db.save_message(thread_id, role="assistant", content=ai_response)
    
    # 9. Return the HTMX partial
    from app import templates
    return templates.TemplateResponse(
        request=request,
        name="partials/message_pair.html",
        context={
            "user_msg": user_input, 
            "ai_msg": ai_response,
            "project_name": project_name,
            "thread_id": thread_id
        }
    )
    