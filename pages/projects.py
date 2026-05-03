import os
from nicegui import ui
from pathlib import Path
from utils.navigation import home_button

PROJECTS_DIR = Path("projects")

def projects_dashboard():
    if not PROJECTS_DIR.exists():
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    project_names = [d for d in os.listdir(PROJECTS_DIR) if (PROJECTS_DIR / d).is_dir()]

    with ui.column().classes('w-full max-w-3xl mx-auto q-pa-md'):
        with ui.row().classes('w-full items-center justify-between'):
            home_button()
            ui.label('Project Library').classes('text-h4')
            ui.label('') 

        ui.separator().classes('q-my-md')

        if not project_names:
            ui.label('No projects found. Add a folder to the "projects" directory to start.').classes('text-italic text-grey-7 q-pa-lg')
        else:
            with ui.column().classes('w-full gap-4'):
                for name in project_names:
                    with ui.card().classes('w-full q-pa-sm shadow-1 hover:shadow-3'):
                        with ui.row().classes('w-full items-center justify-between'):
                            with ui.column():
                                display_name = name.replace('_', ' ').title()
                                ui.label(display_name).classes('text-h6')
                                ui.label(f'Location: projects/{name}').classes('text-caption')
                            
                            # Entering the project leads to Chat by default
                            ui.button(f'Open {display_name}', on_click=lambda n=name: ui.navigate.to(f'/project/{n}/chat')) \
                                .props('flat color=primary icon=launch')
                                