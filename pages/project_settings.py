import json
import os
import shutil
import re
from pathlib import Path
from nicegui import ui
from utils.navigation import project_navigation_header, home_button
from utils.audio import event_beep
# IMPORT the absolute path from your config manager
from core.config_manager import PROJECTS_DIR 

def get_project_config(project_name: str):
    """Loads existing config or returns standard defaults."""
    path = PROJECTS_DIR / project_name / "project_config.json"
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading project config: {e}")
    
    return {
        'override_provider': False,
        'chat_provider': 'ollama',
        'chat_model': 'llama3',
        'override_tuning': False,
        'system_prompt': 'You are a helpful assistant.',
        'temperature': 0.7,
        'max_tokens': 2048,
        'embed_method': 'size',
        'chunk_size': 512,
        'chunk_overlap': 50,
        'delimiter': '\\n\\n'
    }

def save_project_config(project_name: str, config_data: dict):
    """Saves the configuration to the project folder."""
    project_path = PROJECTS_DIR / project_name
    config_file = project_path / "project_config.json"
    try:
        project_path.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
        
        event_beep('info')
        ui.notify(f'Settings saved for {project_name.title()}', color='positive')
    except Exception as e:
        event_beep('error')
        ui.notify(f'Error saving settings: {e}', color='negative')

@ui.refreshable
def chunking_fields_container(method: str, config: dict):
    """Dynamically renders RAG fields based on selected method."""
    with ui.column().classes('w-full gap-2'):
        if method == 'size':
            with ui.row().classes('w-full gap-4'):
                ui.number(label='Chunk Size', format='%d').classes('col').props('outlined') \
                    .bind_value(config, 'chunk_size')
                ui.number(label='Chunk Overlap', format='%d').classes('col').props('outlined') \
                    .bind_value(config, 'chunk_overlap')
        elif method == 'delimiter':
            ui.input(label='Custom Delimiter').classes('w-full').props('outlined') \
                .bind_value(config, 'delimiter')
        elif method == 'full':
            ui.label('Full Context: Entire file processed as one block.').classes('text-italic text-grey-7 q-pa-sm')

def project_settings_page(project_name: str):
    home_button()
    project_navigation_header(project_name, current_page='settings')
    
    # Load the config once at the page level
    config = get_project_config(project_name)

    with ui.column().classes('w-full max-w-4xl mx-auto q-pa-md'):
        ui.markdown(f'# Project Settings: {project_name.replace("_", " ").title()}')

        # --- Section 1: Provider Override ---
        with ui.card().classes('w-full q-pa-md q-mb-md shadow-2'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.markdown('## Model & Provider')
                ui.checkbox('Override Global Provider').bind_value(config, 'override_provider') \
                    .on_value_change(lambda: provider_container.refresh())

            @ui.refreshable
            def provider_container():
                if config.get('override_provider'):
                    with ui.column().classes('w-full q-mt-md gap-2'):
                        ui.select(options=['ollama', 'groq', 'openrouter', 'together', 'gemini'], label='Provider') \
                            .classes('w-full').props('outlined').bind_value(config, 'chat_provider')
                        ui.input(label='Model Name').classes('w-full').props('outlined').bind_value(config, 'chat_model')
                else:
                    ui.label('Using global provider settings.').classes('text-grey-7')
            provider_container()

        # --- Section 2: Model Tuning ---
        with ui.card().classes('w-full q-pa-md q-mb-md shadow-2'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.markdown('## Model Tuning')
                ui.checkbox('Override Global Tuning').bind_value(config, 'override_tuning') \
                    .on_value_change(lambda: tuning_container.refresh())

            @ui.refreshable
            def tuning_container():
                if config.get('override_tuning'):
                    with ui.column().classes('w-full q-mt-md gap-4'):
                        ui.textarea(label='System Prompt').classes('w-full').props('autogrow outlined') \
                            .bind_value(config, 'system_prompt')
                        with ui.row().classes('w-full gap-4'):
                            ui.number(label='Temperature', step=0.1).classes('col').props('outlined') \
                                .bind_value(config, 'temperature')
                            ui.number(label='Max Tokens', step=256).classes('col').props('outlined') \
                                .bind_value(config, 'max_tokens')
                else:
                    ui.label('Using global tuning settings.').classes('text-grey-7')
            tuning_container()

        # --- Section 3: RAG Strategy ---
        with ui.card().classes('w-full q-pa-md shadow-2 q-mb-md'):
            ui.markdown('## Knowledge Base (RAG)')
            # Important: Ensure the method is refreshed when the selection changes
            ui.select(options={'size': 'By Size', 'delimiter': 'By Delimiter', 'full': 'Full'}, 
                      label='Method') \
                .classes('w-full').props('outlined') \
                .bind_value(config, 'embed_method') \
                .on_value_change(lambda e: chunking_fields_container.refresh(e.value, config))
            
            chunking_fields_container(config.get('embed_method', 'size'), config)

        # --- Section 4: Danger Zone ---
        with ui.card().classes('w-full q-pa-md shadow-2 border-red-200'):
            ui.markdown('## Danger Zone')
            with ui.row().classes('w-full items-center gap-4 q-mb-md'):
                r_input = ui.input(label='New Folder Name', value=project_name).classes('grow').props('outlined dense')
                def do_rename():
                    new_n = re.sub(r'[^\w\-]', '_', r_input.value.strip().replace(' ', '_')).lower()
                    if not new_n or new_n == project_name: return
                    try:
                        os.rename(PROJECTS_DIR / project_name, PROJECTS_DIR / new_n)
                        ui.navigate.to(f'/project/{new_n}/settings')
                    except Exception as e: ui.notify(f"Error: {e}", color='negative')
                ui.button('Rename', on_click=do_rename).props('color=grey-8')

            with ui.dialog() as d_diag, ui.card().classes('q-pa-md'):
                ui.label('Delete Project?').classes('text-h6 text-red')
                ui.label(f'Are you sure you want to delete "{project_name}"? This is permanent.')
                with ui.row().classes('w-full justify-end q-mt-md'):
                    ui.button('Cancel', on_click=d_diag.close).props('flat')
                    ui.button('Delete', color='red', on_click=lambda: [
                        shutil.rmtree(PROJECTS_DIR / project_name),
                        ui.navigate.to('/projects')
                    ])
            ui.button('Delete Project', icon='delete', color='red', on_click=d_diag.open)

        # --- Save ---
        with ui.row().classes('w-full justify-end q-mt-lg'):
            ui.button('Save Project Settings', icon='save', color='positive', 
                      on_click=lambda: save_project_config(project_name, config)).props('size=lg')