from nicegui import ui
from pathlib import Path
import json
from utils.navigation import project_navigation_header, home_button

PROJECTS_DIR = Path("projects")

def get_project_config(project_name: str):
    path = PROJECTS_DIR / project_name / "project_config.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}

@ui.refreshable
def chunking_fields_container(method: str, config: dict):
    with ui.column().classes('w-full gap-4 q-mt-md'):
        if method == 'size':
            with ui.row().classes('w-full gap-4'):
                ui.number(label='Chunk Size Override', value=config.get('chunk_size')).classes('col').props('outlined')
                ui.number(label='Overlap Override', value=config.get('chunk_overlap')).classes('col').props('outlined')
        elif method == 'delimiter':
            ui.input(label='Delimiter Override', value=config.get('delimiter', '\\n\\n')).classes('w-full').props('outlined')

def project_settings_page(project_name: str):
    home_button()
    project_navigation_header(project_name, current_page='settings')
    config = get_project_config(project_name)
    display_name = project_name.replace("_", " ").title()

    with ui.column().classes('w-full max-w-4xl mx-auto q-pa-md'):
        ui.markdown(f'# Settings: {display_name}')

        with ui.card().classes('w-full q-pa-md q-mb-lg'):
            ui.markdown('## 1. Personality Overrides')
            ui.select(options=['ollama', 'groq', 'openrouter', 'gemini'], label='Provider', value=config.get('chat_provider')).classes('w-full').props('outlined')
            ui.input(label='Model Override', value=config.get('chat_model', '')).classes('w-full q-mt-sm').props('outlined')
            ui.textarea(label='System Prompt', value=config.get('system_prompt', '')).classes('w-full q-mt-sm').props('outlined autogrow')

        with ui.card().classes('w-full q-pa-md q-mb-lg'):
            ui.markdown('## 2. Extraction & Chunking')
            method = ui.radio(options={'size': 'Size', 'delimiter': 'Delimiter', 'full': 'Full'}, 
                              value=config.get('embed_method', 'size'),
                              on_change=lambda e: chunking_fields_container.refresh(e.value, config)).props('inline')
            chunking_fields_container(method.value, config)

        with ui.row().classes('w-full justify-end q-mt-md'):
            ui.button('Save All', icon='save', color='positive').props('unelevated')
            