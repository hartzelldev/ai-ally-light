from nicegui import ui
from pathlib import Path

def project_settings_page(project_name: str):
    # This function now knows which project it's working on
    with ui.column().classes('w-full max-w-2xl mx-auto q-pa-md'):
        with ui.row().classes('w-full items-center justify-between'):
            ui.button(icon='arrow_back', on_click=lambda: ui.navigate.to('/projects')).props('flat aria-label="Back to Projects"')
            ui.label(f'Settings: {project_name.title()}').classes('text-h4')
            ui.label('')

        ui.separator().classes('q-my-md')

# Placeholder for project-specific logic
        with ui.card().classes('w-full q-pa-md'):
            ui.label('Project Configuration').classes('text-h6')
            
            # Simplified input: we just set the initial value to the project name
            ui.input(label='Project Display Name', value=project_name.title()) \
                .classes('w-full q-mb-md')
            
            ui.textarea(label='Custom Instructions / Story Bible Summary') \
                .classes('w-full').props('outlined')
            
            ui.button('Save Project Settings', icon='save') \
                .classes('q-mt-md').props('color=primary')
                