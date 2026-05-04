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
    
    # Defaults to ensure the UI always has data to bind to
    return {
        'chat_provider': 'ollama',
        'chat_model': 'llama3',
        'system_prompt': 'You are a helpful assistant.',
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

@ui.refreshable
def chunking_fields_container(method: str, config: dict):
    """Dynamically renders fields based on size, delimiter, or full context."""
    if method == 'size':
        ui.number(label='Chunk Size', format='%d') \
            .classes('w-full').bind_value(config, 'chunk_size')
        ui.number(label='Chunk Overlap', format='%d') \
            .classes('w-full').bind_value(config, 'chunk_overlap')
    elif method == 'delimiter':
        ui.input(label='Custom Delimiter (e.g. \\n\\n)') \
            .classes('w-full').bind_value(config, 'delimiter')
    elif method == 'full':
        ui.label('Full Context: The entire file will be processed as a single block.') \
            .classes('text-italic text-grey-7 q-pa-sm')

def project_settings_page(project_name: str):
    # Standard header for navigation
    home_button()
    project_navigation_header(project_name, current_page='settings')
    
    # Load state
    config = get_project_config(project_name)

    with ui.column().classes('w-full max-w-4xl mx-auto q-pa-md'):
        ui.label(f'Project Settings: {project_name.title()}').classes('text-h4 q-mb-md')

        # --- Section 1: AI Personality ---
        with ui.card().classes('w-full q-pa-md q-mb-lg shadow-2'):
            ui.label('AI Personality & Provider').classes('text-h6 q-mb-sm')
            
            ui.select(options=['ollama', 'groq', 'openrouter', 'gemini'], 
                      label='Provider') \
                .classes('w-full').bind_value(config, 'chat_provider')
            
            ui.input(label='Model Name (e.g., llama3, deepseek-coder)') \
                .classes('w-full q-mt-sm').bind_value(config, 'chat_model')
            
            ui.textarea(label='System Prompt') \
                .classes('w-full q-mt-sm').props('autogrow') \
                .bind_value(config, 'system_prompt')

# --- Section 2: Data Chunking Strategy ---
        with ui.card().classes('w-full q-pa-md shadow-2'):
            ui.label('Knowledge Base Processing (RAG)').classes('text-h6 q-mb-sm')
            
            # Added 'full' to the options dictionary
            method_select = ui.select(
                options={
                    'size': 'By Character Count', 
                    'delimiter': 'By Custom Delimiter',
                    'full': 'Full Context (No Chunking)'
                }, 
                label='Chunking Method',
                on_change=lambda e: chunking_fields_container.refresh(e.value, config)
            ).classes('w-full').bind_value(config, 'embed_method')

            with ui.column().classes('w-full q-mt-md'):
                # Pass the current value to the container
                chunking_fields_container(config.get('embed_method'), config)

        # --- Footer Actions ---
        with ui.row().classes('w-full justify-end q-mt-xl'):
            ui.button('Save All Settings', icon='save', color='positive', 
                      on_click=lambda: save_project_config(project_name, config)) \
                .props('unelevated size=lg aria-label="Save all changes to disk"')
                