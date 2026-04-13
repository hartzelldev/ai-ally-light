from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from core.config_manager import manager
from schemas.settings import ChatSettings

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# --- 1. THE HELPER ROUTE (For Dynamic Updates) ---
@router.get("/chat-fields", response_class=HTMLResponse)
async def get_chat_fields(request: Request, chat_provider: str):
    """
    Returns the HTML for API Key and Base URL based on the provider.
    HTMX calls this whenever you change the dropdown.
    """
    # Define your defaults for the "Smart" auto-fill
    defaults = {
        "ollama": ("http://localhost:11434", "https://ollama.com/library"),
        "groq": ("https://api.groq.com/openai/v1", "https://console.groq.com/docs/models"),
        "openrouter": ("https://openrouter.ai/api/v1", "https://openrouter.ai/models"),
        "together": ("https://api.together.xyz/v1", "https://docs.together.ai/docs/inference-models"),
    }
    
    base_url, model_link = defaults.get(chat_provider, ("", "#"))
    current_key = manager.get_api_key(chat_provider)

    return templates.TemplateResponse(
        request=request,
        name="partials/chat_fields_inner.html",
        context={
            "provider": chat_provider,
            "base_url": base_url,
            "model_link": model_link,
            "api_key": current_key
        }
    )

# --- 2. THE MODAL ROUTE ---
@router.get("/modal", response_class=HTMLResponse)
async def get_global_settings_modal(request: Request):
    config = manager.config
    return templates.TemplateResponse(
        request=request,
        name="partials/settings_modal.html",
        context={
            "config": config,
            "providers": ["ollama", "groq", "openrouter", "together", "other"]
        }
    )

# --- 3. THE SAVE ROUTE ---
@router.post("/save", response_class=HTMLResponse)
async def save_global_settings(
    request: Request,
    chat_provider: str = Form(...),
    chat_model: str = Form(...),
    chat_base_url: str = Form(...),
    temperature: float = Form(...),
    chat_api_key: str = Form(None)  # We added this to capture the key
):
    # Update the Chat Settings object
    new_chat = ChatSettings(
        provider=chat_provider,
        model=chat_model,
        base_url=chat_base_url,
        temperature=temperature
    )
    
    # Update master config
    updated_config = manager.config
    updated_config.chat = new_chat
    manager.save_config(updated_config)
    
    # Save the API key to the .env file if it was provided
    if chat_api_key:
        manager.save_api_key(chat_provider, chat_api_key)
    
    return "<span class='success' role='alert' style='color: green;'>Settings and API Key Saved!</span>"