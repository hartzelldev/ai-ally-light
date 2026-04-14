from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from core.config_manager import manager
from schemas.settings import ChatSettings, EmbeddingSettings

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# --- 1. DYNAMIC FIELD HELPERS ---

@router.get("/chat-fields", response_class=HTMLResponse)
async def get_chat_fields(request: Request, chat_provider: str):
    """
    Returns the HTML for API Key and Base URL based on the provider.
    HTMX calls this whenever you change the chat provider dropdown.
    """
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

@router.get("/embed-fields", response_class=HTMLResponse)
async def get_embed_fields(request: Request, embed_method: str):
    """
    Returns the specific fields for the chosen chunking method.
    HTMX calls this whenever a different embedding radio button is selected.
    """
    return templates.TemplateResponse(
        request=request,
        name="partials/embed_fields_inner.html",
        context={
            "method": embed_method,
            "config": manager.config
        }
    )


# --- 2. MODAL RENDERING ---

@router.get("/modal", response_class=HTMLResponse)
async def get_global_settings_modal(request: Request):
    """Renders the full settings modal with current config values."""
    config = manager.config
    return templates.TemplateResponse(
        request=request,
        name="partials/settings_modal.html",
        context={
            "config": config,
            "providers": ["ollama", "groq", "openrouter", "together", "other"],
            "method": config.embeddings.method  # Ensure the initial partial knows the method
        }
    )


# --- 3. SAVE LOGIC ---

@router.post("/save", response_class=HTMLResponse)
async def save_global_settings(
    request: Request,
    # Chat Fields
    chat_provider: str = Form(...),
    chat_model: str = Form(...),
    chat_base_url: str = Form(...),
    temperature: float = Form(...),
    chat_api_key: str = Form(None),
    # Embedding Fields
    embed_method: str = Form("size"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    top_k: int = Form(5),
    delimiter: str = Form("\n\n")
):
    """
    Validates and saves all settings.
    Updates config.json for general settings and .env for sensitive keys.
    """
    # 1. Update Chat Settings
    new_chat = ChatSettings(
        provider=chat_provider,
        model=chat_model,
        base_url=chat_base_url,
        temperature=temperature
    )
    
    # 2. Update Embedding Settings
    new_embed = EmbeddingSettings(
        method=embed_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        delimiter=delimiter
    )
    
    # 3. Update master config object
    current_config = manager.config
    current_config.chat = new_chat
    current_config.embeddings = new_embed
    
    # 4. Save to config.json
    manager.save_config(current_config)
    
    # 5. Save API Key to .env if provided
    if chat_api_key:
        manager.save_api_key(chat_provider, chat_api_key)
    
    return "<span class='success' role='alert' style='color: green; font-weight: bold;'>Settings and API Keys Saved Successfully!</span>"