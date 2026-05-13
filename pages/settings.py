from nicegui import ui
from utils.navigation import home_button
from core.config_manager import manager

# --- DATA ---
CHAT_DEFAULTS = {
    "ollama": {"url": "http://localhost:11434", "link": "https://ollama.com/library"},
    "groq": {"url": "https://api.groq.com/openai/v1", "link": "https://console.groq.com/docs/models"},
    "openrouter": {"url": "https://openrouter.ai/api/v1", "link": "https://openrouter.ai/models"},
    "together": {"url": "https://api.together.xyz/v1", "link": "https://docs.together.ai/docs/inference-models"},
    "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta", "link": "https://ai.google.dev/gemini-api/docs/models"},
}

# --- STATE HELPERS ---
# Buffer for API keys to be saved to .env
key_buffer = {p: manager.get_api_key(p) for p in CHAT_DEFAULTS.keys()}

@ui.refreshable
def chat_fields_container(provider):
    data = CHAT_DEFAULTS.get(provider, {"url": "", "link": "#"})
    with ui.column().classes('w-full gap-4'):
        ui.input(label=f'{provider.title()} API Key', password=True) \
            .classes('w-full').props('outlined') \
            .set_enabled(provider != 'ollama') \
            .bind_value(key_buffer, provider)
        
        ui.input(label='Base URL') \
            .classes('w-full').props('outlined') \
            .bind_value(manager.config.chat, 'base_url')

@ui.refreshable
def embed_fields_container(method):
    with ui.column().classes('w-full gap-4 q-mt-md'):
        if method == 'size':
            with ui.row().classes('w-full gap-4'):
                ui.number(label='Chunk Size').classes('col').props('outlined') \
                    .bind_value(manager.config.embeddings, 'chunk_size')
                ui.number(label='Overlap').classes('col').props('outlined') \
                    .bind_value(manager.config.embeddings, 'chunk_overlap')
        elif method == 'delimiter':
            ui.input(label='Delimiter String').classes('w-full').props('outlined') \
                .bind_value(manager.config.embeddings, 'delimiter')

def global_settings_page():
    home_button()
    
    # Nested references to make binding lines shorter
    c_cfg = manager.config.chat
    e_cfg = manager.config.embeddings
    g_cfg = manager.config

    def save_all_settings():
        try:
            # 1. Save JSON (GlobalConfig dumps both sub-models automatically)
            manager.save_config(g_cfg)
            
            # 2. Save currently selected provider's key to .env
            current_provider = c_cfg.provider
            current_key = key_buffer.get(current_provider, "")
            manager.save_api_key(current_provider, current_key)
            
            ui.notify('Settings and API Keys saved!', color='positive', icon='save')
        except Exception as e:
            ui.notify(f'Save failed: {e}', color='negative')

    with ui.column().classes('w-full max-w-4xl mx-auto q-pa-md'):
        ui.markdown('# Global Settings')
        
        # --- AI Personality & Logic ---
        with ui.card().classes('w-full q-pa-md q-mb-lg'):
            ui.markdown('## AI Personality & Logic')
            
            ui.select(
                options=list(CHAT_DEFAULTS.keys()), 
                label='Chat Provider',
                on_change=lambda e: chat_fields_container.refresh(e.value)
            ).classes('w-full q-mb-md').props('outlined').bind_value(c_cfg, 'provider')
            
            chat_fields_container(c_cfg.provider)
            
            ui.textarea(label='Global System Prompt').classes('w-full q-mt-md').props('outlined autogrow') \
                .bind_value(c_cfg, 'system_prompt')
            
            with ui.row().classes('w-full gap-4 q-mt-md'):
                ui.number(label='Max Tokens', step=256).classes('col').props('outlined') \
                    .bind_value(c_cfg, 'max_tokens')
                ui.number(label='Temperature', step=0.1).classes('col').props('outlined') \
                    .bind_value(c_cfg, 'temperature')

        # --- Knowledge Base (RAG) Settings ---
        with ui.card().classes('w-full q-pa-md q-mb-lg'):
            ui.markdown('## Knowledge Base (RAG) Settings')
            ui.radio(
                options={'size': 'Size', 'delimiter': 'Delimiter', 'full': 'Full'},
                on_change=lambda e: embed_fields_container.refresh(e.value)
            ).props('inline').bind_value(e_cfg, 'method')
            
            embed_fields_container(e_cfg.method)

        # --- UI & Accessibility ---
        with ui.card().classes('w-full q-pa-md q-mb-lg'):
            ui.markdown('## UI & Accessibility')
            
            with ui.row().classes('w-full items-center gap-8'):
                ui.number(label='Max Thread History (Memory)') \
                    .classes('col').props('outlined') \
                    .bind_value(g_cfg, 'max_history_turns')
                
                ui.checkbox('Enable Audio Cues') \
                    .bind_value(g_cfg, 'enable_sounds')

        # --- Actions ---
        with ui.row().classes('w-full justify-end gap-4 q-mt-md'):
            ui.button('Cancel', on_click=lambda: ui.navigate.to('/')).props('flat')
            ui.button('Save All Settings', icon='save', on_click=save_all_settings).props('color=positive')