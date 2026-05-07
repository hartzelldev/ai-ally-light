import json
import os
from pathlib import Path
from nicegui import ui
from utils.navigation import project_navigation_header, home_button

PROJECTS_DIR = Path("projects")

def get_project_config(project_name: str):
    """Loads existing config or returns standard defaults."""
    path = PROJECTS_DIR / project_name / "project_config.json"
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    # Defaults include the override flags
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
    """Ensures directory existence and saves the configuration."""
    project_path = PROJECTS_DIR / project_name
    config_file = project_path / "project_config.json"
    try:
        if not project_path.exists():
            project_path.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
        ui.notify(f'Settings saved for {project_name.title()}', color='positive')
    except Exception as e:
        ui.notify(f'Error saving settings: {e}', color='negative')

@ui.refreshable
def chunking_fields_container(method: str, config: dict):
    """Dynamically renders fields based on size, delimiter, or full context."""
    if method == 'size':
        with ui.row().classes('w-full gap-4'):
            ui.number(label='Chunk Size', format='%d').classes('col').bind_value(config, 'chunk_size')
            ui.number(label='Chunk Overlap', format='%d').classes('col').bind_value(config, 'chunk_overlap')
    elif method == 'delimiter':
        ui.input(label='Custom Delimiter (e.g. \\n\\n)').classes('w-full').bind_value(config, 'delimiter')
    elif method == 'full':
        ui.label('Full Context: The entire file will be processed as a single block.').classes('text-italic text-grey-7 q-pa-sm')

def project_settings_page(project_name: str):
    home_button()
    project_navigation_header(project_name, current_page='settings')
    config = get_project_config(project_name)

    with ui.column().classes('w-full max-w-4xl mx-auto q-pa-md'):
        ui.label(f'Project Settings: {project_name.title()}').classes('text-h4 q-mb-md')

        # --- Section 1: Provider & Model Override ---
        with ui.card().classes('w-full q-pa-md q-mb-md shadow-2'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Model & Provider').classes('text-h6')
                ui.checkbox('Override Global Provider', value=config.get('override_provider')) \
                    .bind_value(config, 'override_provider') \
                    .on_value_change(lambda: provider_container.refresh())

            @ui.refreshable
            def provider_container():
                if config.get('override_provider'):
                    with ui.column().classes('w-full q-mt-md gap-2'):
                        ui.select(options=['ollama', 'groq', 'openrouter', 'gemini', 'together'], label='Provider') \
                            .classes('w-full').bind_value(config, 'chat_provider')
                        ui.input(label='Model Name (e.g., llama3, mistral-small)') \
                            .classes('w-full').bind_value(config, 'chat_model')
                else:
                    ui.label('Using global provider and model settings.').classes('text-grey-7 q-mt-sm')
            
            provider_container()

        # --- Section 2: Model Tuning Override ---
        with ui.card().classes('w-full q-pa-md q-mb-md shadow-2'):
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Model Tuning (Temp, Prompt, Tokens)').classes('text-h6')
                ui.checkbox('Override Global Tuning', value=config.get('override_tuning')) \
                    .bind_value(config, 'override_tuning') \
                    .on_value_change(lambda: tuning_container.refresh())

            @ui.refreshable
            def tuning_container():
                if config.get('override_tuning'):
                    with ui.column().classes('w-full q-mt-md gap-4'):
                        ui.textarea(label='Project System Prompt') \
                            .classes('w-full').props('autogrow outlined') \
                            .bind_value(config, 'system_prompt')
                        
                        with ui.row().classes('w-full gap-4'):
                            ui.number(label='Temperature', step=0.1, format='%.1f') \
                                .classes('col').props('outlined') \
                                .bind_value(config, 'temperature')
                            ui.number(label='Max Tokens', step=256, format='%d') \
                                .classes('col').props('outlined') \
                                .bind_value(config, 'max_tokens')
                else:
                    ui.label('Using global personality and tuning settings.').classes('text-grey-7 q-mt-sm')
            
            tuning_container()

        # --- Section 3: Data Chunking Strategy ---
        with ui.card().classes('w-full q-pa-md shadow-2'):
            ui.label('Knowledge Base Processing (RAG)').classes('text-h6 q-mb-sm')
            method_select = ui.select(
                options={'size': 'By Character Count', 'delimiter': 'By Custom Delimiter', 'full': 'Full Context'}, 
                label='Chunking Method',
                on_change=lambda e: chunking_fields_container.refresh(e.value, config)
            ).classes('w-full').bind_value(config, 'embed_method')

            with ui.column().classes('w-full q-mt-md'):
                chunking_fields_container(config.get('embed_method'), config)

        # --- Footer Actions ---
        with ui.row().classes('w-full justify-end q-mt-xl'):
            ui.button('Save Project Settings', icon='save', color='positive', 
                      on_click=lambda: save_project_config(project_name, config)) \
                .props('unelevated size=lg aria-label="Save project-specific overrides"')
                