import os
from nicegui import ui
from pathlib import Path

# This points to the folder where user projects live
PROJECTS_DIR = Path("projects")

def projects_dashboard():
    # Ensure the directory exists
    if not PROJECTS_DIR.exists():
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    # Get list of project folders
    project_names = [d for d in os.listdir(PROJECTS_DIR) if (PROJECTS_DIR / d).is_dir()]

    with ui.column().classes('w-full max-w-3xl mx-auto q-pa-md'):
        with ui.row().classes('w-full items-center justify-between'):
            ui.button(icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat aria-label="Back to Home"')
            ui.label('Project Library').classes('text-h4')
            ui.label('') # Alignment spacer

        ui.separator().classes('q-my-md')

        if not project_names:
            ui.label('No projects found in the workspace. Add a folder to the "projects" directory to get started.') \
                .classes('text-italic text-grey-7 q-pa-lg')
        else:
            with ui.column().classes('w-full gap-4'):
                for name in project_names:
                    with ui.card().classes('w-full q-pa-sm shadow-1 hover:shadow-3'):
                        with ui.row().classes('w-full items-center justify-between'):
                            with ui.column():
                                # Clean up folder names for display
                                display_name = name.replace('_', ' ').title()
                                ui.label(display_name).classes('text-h6')
                                ui.label(f'Location: projects/{name}').classes('text-caption')
                            
                            with ui.row():
                                ui.button('Open', on_click=lambda n=name: ui.navigate.to(f'/projects/{n}/chat')) \
                                    .props('flat color=primary icon=launch')
                                ui.button('Config', on_click=lambda n=name: ui.navigate.to(f'/projects/{n}/settings')) \
                                    .props('flat icon=tune')
                                    