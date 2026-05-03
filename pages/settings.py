from nicegui import ui
from utils.navigation import home_button


# --- UPDATED PROVIDER DATA ---
CHAT_DEFAULTS = {
    "ollama": {"url": "http://localhost:11434", "link": "https://ollama.com/library"},
    "groq": {"url": "https://api.groq.com/openai/v1", "link": "https://console.groq.com/docs/models"},
    "openrouter": {"url": "https://openrouter.ai/api/v1", "link": "https://openrouter.ai/models"},
    "together": {"url": "https://api.together.xyz/v1", "link": "https://docs.together.ai/docs/inference-models"},
    "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta", "link": "https://ai.google.dev/gemini-api/docs/models"},
}

@ui.refreshable
def chat_fields_container(provider):
    data = CHAT_DEFAULTS.get(provider, {"url": "", "link": "#"})
    with ui.column().classes('w-full gap-4'):
        ui.input(label='API Key', password=True, placeholder='Enter your API key') \
            .classes('w-full').props('outlined') \
            .set_enabled(provider != 'ollama')
        
        ui.input(label='Base URL', value=data['url']) \
            .classes('w-full').props('outlined')

@ui.refreshable
def embed_fields_container(method):
    with ui.column().classes('w-full gap-4 q-mt-md'):
        if method == 'size':
            with ui.row().classes('w-full gap-4'):
                ui.number(label='Chunk Size', value=500).classes('col').props('outlined')
                ui.number(label='Overlap', value=50).classes('col').props('outlined')
        elif method == 'delimiter':
            ui.input(label='Delimiter String', value='\\n\\n').classes('w-full').props('outlined')

def global_settings_page():
    home_button()
    with ui.column().classes('w-full max-w-4xl mx-auto q-pa-md'):
        ui.markdown('# Global Settings')
        
        # Chat & Model Logic
        with ui.card().classes('w-full q-pa-md q-mb-lg'):
            ui.markdown('## AI Personality & Logic')
            
            provider_select = ui.select(
                options=list(CHAT_DEFAULTS.keys()), 
                label='Chat Provider',
                value='ollama',
                on_change=lambda e: chat_fields_container.refresh(e.value)
            ).classes('w-full q-mb-md').props('outlined')
            
            chat_fields_container(provider_select.value)
            
            ui.textarea(label='Global System Prompt', 
                        placeholder='You are a helpful assistant...') \
                .classes('w-full q-mt-md').props('outlined autogrow')
            
            with ui.row().classes('w-full gap-4 q-mt-md'):
                ui.number(label='Max Tokens', value=2048, step=256).classes('col').props('outlined')
                ui.number(label='Temperature', value=0.7, step=0.1).classes('col').props('outlined')

        # Embedding Strategy
        with ui.card().classes('w-full q-pa-md q-mb-lg'):
            ui.markdown('## Knowledge Base (RAG) Settings')
            method_radio = ui.radio(
                options={'size': 'Size', 'delimiter': 'Delimiter', 'full': 'Full'},
                value='size',
                on_change=lambda e: embed_fields_container.refresh(e.value)
            ).props('inline')
            embed_fields_container(method_radio.value)

        # Application Preferences
        with ui.card().classes('w-full q-pa-md q-mb-lg'):
            ui.markdown('## UI & Accessibility')
            
            with ui.row().classes('w-full items-center gap-8'):
                # Essential for your screen reader flow
                ui.number(label='Messages to Display in Thread', value=20) \
                    .classes('col').props('outlined')
                
                # Sound notification toggle
                ui.checkbox('Play Sound on AI Response Complete') \
                    .classes('col')

            ui.checkbox('Auto-scroll to bottom on new message', value=True)

        # Actions
        with ui.row().classes('w-full justify-end gap-4 q-mt-md'):
            ui.button('Cancel', on_click=lambda: ui.navigate.to('/')).props('flat')
            ui.button('Save All Settings', icon='save').props('color=positive')
            