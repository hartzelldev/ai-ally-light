from nicegui import ui
from pathlib import Path
from utils.navigation import project_navigation_header, home_button

# Calculate absolute path relative to this file
PROJECTS_DIR = Path(__file__).parent.parent / "projects"

@ui.refreshable
def project_file_grid(project_name: str):
    source_path = PROJECTS_DIR / project_name / "files"
    source_path.mkdir(parents=True, exist_ok=True)
    files = [f for f in source_path.iterdir() if f.is_file()]

    with ui.column().classes('w-full gap-2'):
        if not files:
            ui.label('No files in knowledge base.').classes('text-gray-500 italic q-pa-md')
        for file in files:
            with ui.card().classes('w-full q-pa-sm shadow-1'):
                with ui.row().classes('w-full items-center justify-between'):
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('insert_drive_file', color='primary')
                        ui.label(file.name).classes('font-mono')
                    ui.button(icon='delete', color='negative', 
                              on_click=lambda f=file: [f.unlink(), project_file_grid.refresh()]) \
                        .props('flat dense aria-label="Delete ' + file.name + '"')

def project_files_page(project_name: str):
    home_button()
    project_navigation_header(project_name, current_page='files')
    display_name = project_name.replace("_", " ").title()

    with ui.column().classes('w-full max-w-4xl mx-auto q-pa-md'):
        ui.markdown(f'# Knowledge Base: {display_name}')
        ui.upload(label='Upload Documents', multiple=True, auto_upload=True,
                  on_upload=lambda e: [
                      (PROJECTS_DIR / project_name / "files" / e.name).write_bytes(e.content.read()),
                      project_file_grid.refresh(),
                      ui.notify(f'Uploaded {e.name}')
                  ]).classes('w-full q-mb-lg').props('flat bordered')
        project_file_grid(project_name)
        