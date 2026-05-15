import os
import re
import sys
from nicegui import ui
from pathlib import Path
from utils.navigation import home_button
from utils.audio import event_beep
from core.config_manager import PROJECTS_DIR

def create_new_project(name: str, refresh_func, dialog):
    """Creates a new project folder structure and refreshes the list."""
    if not name or not name.strip():
        event_beep('info')
        ui.notify("Project name cannot be empty!", color='negative')
        return
    
    # SECURITY: Sanitize name to prevent path traversal or illegal characters
    safe_name = re.sub(r'[^\w\-]', '_', name.strip().replace(' ', '_')).lower()
    project_path = PROJECTS_DIR / safe_name
    
    if project_path.exists():
        event_beep('error')
        ui.notify(f"Project '{safe_name}' already exists.", color='warning')
        return

    try:
        # Create standard project directory structure
        # Using parents=True and absolute pathing to handle Scoop junctions better
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "threads").mkdir(exist_ok=True)
        (project_path / "files").mkdir(exist_ok=True)
        
        ui.notify(f"Created project: {safe_name}", color='positive', icon='check')
        dialog.close()
        refresh_func()  # Refresh the UI list
    except Exception as e:
        event_beep('error')
        ui.notify(f"Error creating project: {e}", color='negative', icon='report_problem')

@ui.refreshable
def projects_list_container():
    """Renders the list of current projects."""
    if not PROJECTS_DIR.exists():
        try:
            PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            ui.label(f'Error accessing projects directory: {e}').classes('text-negative')
            return

    # Use absolute path for listdir to avoid shim ambiguity
    try:
        project_names = sorted([d for d in os.listdir(str(PROJECTS_DIR)) if (PROJECTS_DIR / d).is_dir()])
    except Exception as e:
        ui.label(f'Read Error: {e}').classes('text-negative')
        return

    if not project_names:
        ui.label('No projects found. Create a new one to get started!').classes('text-italic text-grey-7 q-pa-lg')
    else:
        with ui.column().classes('w-full gap-4'):
            for name in project_names:
                with ui.card().classes('w-full q-pa-sm shadow-1 hover:shadow-3'):
                    with ui.row().classes('w-full items-center justify-between'):
                        with ui.column():
                            display_name = name.replace('_', ' ').title()
                            ui.label(display_name).classes('text-h6')
                            ui.label(f'ID: {name}').classes('text-caption text-grey-6')
                        
                        ui.button(f'Open {display_name}', on_click=lambda n=name: ui.navigate.to(f'/project/{n}/chat')) \
                            .props('flat color=primary icon=launch')

def projects_dashboard():
    # --- New Project Dialog ---
    with ui.dialog() as new_project_dialog, ui.card().classes('q-pa-md'):
        ui.label('New Project Details').classes('text-h6 q-mb-md')
        name_input = ui.input('Internal Name (e.g. My New Novel)') \
            .classes('w-full q-mb-md').props('outlined autofocus')
        
        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('Cancel', on_click=new_project_dialog.close).props('flat')
            ui.button('Create Project', on_click=lambda: create_new_project(
                name_input.value, projects_list_container.refresh, new_project_dialog
            )).props('unelevated color=primary')

    # --- Main Dashboard UI ---
    with ui.column().classes('w-full max-w-3xl mx-auto q-pa-md'):
        with ui.row().classes('w-full items-center justify-between'):
            home_button()
            ui.label('Project Library').classes('text-h4')
            ui.button('New Project', icon='add', on_click=new_project_dialog.open) \
                .props('unelevated color=positive aria-label="Create a new project folder"')

        ui.separator().classes('q-my-md')

        # This container refreshes when a new project is created
        projects_list_container()