from nicegui import ui
from pathlib import Path
from utils.navigation import project_navigation_header, home_button
from core.indexer import reindex_single_file, clear_entire_index, delete_file_from_index

PROJECTS_DIR = Path(__file__).parent.parent / "projects"

def run_reindex_all(project_name: str):
    """Helper to wipe and rebuild the entire project index."""
    source_path = PROJECTS_DIR / project_name / "files"
    files = [f for f in source_path.iterdir() if f.is_file()]
    
    if not files:
        ui.notify("No files found to index.", type='warning')
        return

    # 1. Wipe the current DB collection
    clear_entire_index(project_name)
    
    # 2. Loop through and re-add
    count = 0
    for file in files:
        count += reindex_single_file(project_name, file)
    
    ui.notify(f"Success: Reindexed {len(files)} files into {count} chunks.", color='positive')

@ui.refreshable
def project_file_grid(project_name: str):
    source_path = PROJECTS_DIR / project_name / "files"
    source_path.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in source_path.iterdir() if f.is_file()])

    with ui.column().classes('w-full gap-2'):
        if not files:
            ui.label('No files in knowledge base.').classes('text-gray-500 italic q-pa-md')
        
        for file in files:
            with ui.card().classes('w-full q-pa-sm shadow-1 hover:shadow-2'):
                with ui.row().classes('w-full items-center justify-between'):
                    # File Info
                    with ui.row().classes('items-center gap-2'):
                        ui.icon('insert_drive_file', color='primary')
                        ui.label(file.name).classes('font-mono')
                    
                    # File Actions
                    with ui.row().classes('items-center gap-1'):
                        # Individual Reindex Button
                        ui.button(icon='refresh', color='primary', 
                                  on_click=lambda f=file: [
                                      reindex_single_file(project_name, f),
                                      ui.notify(f"Reindexed {f.name}")
                                  ]) \
                            .props(f'flat dense aria-label="Reindex {file.name}"')
                        
                        # Delete Button
                        ui.button(icon='delete', color='negative', 
                                  on_click=lambda f=file: [
                                      f.unlink(), 
                                      delete_file_from_index(project_name, f.name),
                                      project_file_grid.refresh()
                                  ]) \
                            .props(f'flat dense aria-label="Delete {file.name}"')

def project_files_page(project_name: str):
    home_button()
    project_navigation_header(project_name, current_page='files')
    display_name = project_name.replace("_", " ").title()

    with ui.column().classes('w-full max-w-4xl mx-auto q-pa-md'):
        ui.markdown(f'# Knowledge Base: {display_name}')
        
        # Upload and Bulk Action Row
        with ui.row().classes('w-full items-center justify-between q-mb-md'):
            ui.label('Manage your project documents below:').classes('text-grey-7')
            ui.button('Reindex All Files', icon='layers', color='warning',
                      on_click=lambda: run_reindex_all(project_name)) \
                .props('outline aria-label="Wipe index and process all files again"')

        ui.upload(label='Upload Documents', multiple=True, auto_upload=True,
                  on_upload=lambda e: [
                      (PROJECTS_DIR / project_name / "files" / e.name).write_bytes(e.content.read()),
                      reindex_single_file(project_name, PROJECTS_DIR / project_name / "files" / e.name),
                      project_file_grid.refresh(),
                      ui.notify(f'Uploaded and Indexed {e.name}')
                  ]).classes('w-full q-mb-lg').props('flat bordered')
        
        project_file_grid(project_name)
        