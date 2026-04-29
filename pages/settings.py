from nicegui import ui
from core.config_manager import manager
from providers.chat import CHAT_PROVIDERS

def settings_page():
    # We'll use a local dict to hold keys during the session
    # so we don't spam the disk on every keystroke.
    keys = {
        'OpenRouter': manager.get_api_key('OpenRouter'),
        'Groq': manager.get_api_key('Groq'),
        'Anthropic': manager.get_api_key('Anthropic'),
        'OpenAI': manager.get_api_key('OpenAI'),
    }

    def save_global_settings():
        # Save the API keys to .env
        for provider, key in keys.items():
            if key:
                manager.save_api_key(provider, key)
        
        ui.notify('Global settings and API keys saved!', type='positive')

    with ui.column().classes('w-full max-w-2xl mx-auto q-pa-md'):
        with ui.row().classes('w-full items-center justify-between'):
            ui.button(icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat')
            ui.label('Global Settings').classes('text-h4')
            ui.label('') # Spacer

        ui.markdown('---')

        # --- API KEYS SECTION ---
        with ui.card().classes('w-full q-pa-md'):
            ui.label('API Keys').classes('text-h6 q-mb-md')
            ui.label('These are stored in your local .env file.').classes('text-caption q-mb-md')

            # We loop through providers to create input fields
            for provider in keys.keys():
                ui.input(label=f'{provider} API Key', password=True, password_toggle_button=True) \
                    .bind_value(keys, provider) \
                    .classes('w-full q-mb-sm')

        # --- SAVE BUTTON ---
        ui.button('Save All Settings', on_click=save_global_settings) \
            .classes('w-full q-mt-lg').props('color=primary icon=save')

        # --- UTILITY SECTION ---
        with ui.expansion('Advanced / Debug Info', icon='bug_report').classes('w-full q-mt-xl'):
            ui.label(f'Config Path: {manager.config}')
            ui.button('Reload .env', on_click=lambda: manager.load_config()) \
                .props('outline size=sm')