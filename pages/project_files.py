import os
from pathlib import Path
from nicegui import ui, events
from utils.navigation import project_navigation_header, home_button
from core.config_manager import PROJECTS_DIR
from core.indexer import sync_all_project_files, delete_file_from_index

def project_files_page(project_name: str):
    project_path = PROJECTS_DIR / project_name
    files_dir = project_path / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    def get_file_list():
        return [f for f in os.listdir(files_dir) if os.path.isfile(files_dir / f)]

    # --- Logic Functions ---
    # ADDED 'async' here
    async def handle_upload(e: events.UploadEventArguments):
        """Final robust handler that awaits the file stream."""
        try:
            # Check for the file object as confirmed by your debug trace
            if hasattr(e, 'file'):
                filename = e.file.name
                # FIX: We MUST await the read() because it's a coroutine
                content = await e.file.read() 
            else:
                filename = getattr(e, 'name', 'unknown_file')
                # Fallback check for content.read() as a coroutine
                if callable(getattr(e.content, 'read', None)):
                    content = await e.content.read()
                else:
                    content = e.content

            file_path = files_dir / filename
            file_path.write_bytes(content)
            
            ui.notify(f'Uploaded {filename}', color='positive')
            file_list_container.refresh()
            
        except Exception as ex:
            # This will now be caught by your new main.py logger too
            ui.notify(f'Upload failed: {ex}', color='negative', duration=0)

    def delete_file(filename: str):
        try:
            delete_file_from_index(project_name, filename)
            (files_dir / filename).unlink()
            ui.notify(f'Deleted {filename} and removed from AI memory.')
            file_list_container.refresh()
        except Exception as e:
            ui.notify(f'Error: {e}', color='negative')

    async def run_indexing():
        with ui.dialog() as dialog, ui.card().classes('items-center q-pa-lg'):
            ui.label('Updating AI Memory...').classes('text-lg text-weight-bold')
            ui.spinner(size='lg', color='primary')
        
        dialog.open()
        try:
            from nicegui import run
            chunks = await run.io_bound(sync_all_project_files, project_name)
            ui.notify(f'Indexed {chunks} content pieces!', color='positive')
        except Exception as e:
            ui.notify(f'Indexing error: {e}', color='negative')
        finally:
            dialog.close()

    # --- UI Layout ---
    home_button()
    project_navigation_header(project_name, current_page='files')

    with ui.column().classes('w-full q-pa-md max-w-4xl mx-auto'):
        ui.label(f'Files for {project_name.title()}').classes('text-h4 q-mb-md')
        
        ui.html('<h2 style="font-size: 1.5rem;">Upload New Files</h2>')
        
        with ui.card().classes('w-full q-pa-md q-mb-lg'):
            # NiceGUI handles the async call to handle_upload automatically
            ui.upload(on_upload=handle_upload, multiple=True) \
                .props('auto-upload label="Pick files to add" aria-label="File upload picker"') \
                .classes('w-full')

        ui.html('<h2 style="font-size: 1.5rem;">Project Documents</h2>')
        
        with ui.row().classes('w-full justify-between items-center q-mb-md'):
            ui.button('Update AI Memory', icon='psychology', on_click=run_indexing) \
                .props('color=primary aria-label="Scan files and update the AI vector database"')

        @ui.refreshable
        def file_list_container():
            current_files = get_file_list()
            if not current_files:
                ui.label('No files found in this project.').classes('text-grey q-pa-md')
            
            with ui.column().classes('w-full gap-2'):
                for filename in current_files:
                    with ui.card().classes('w-full q-pa-sm shadow-sm'):
                        with ui.row().classes('w-full items-center justify-between'):
                            with ui.row().classes('items-center gap-2'):
                                ui.icon('description', color='grey-7')
                                ui.label(filename).classes('text-weight-medium')
                            
                            ui.button(icon='delete', on_click=lambda f=filename: delete_file(f)) \
                                .props('flat round color=negative dense aria-label="Delete ' + filename + '"')

        file_list_container()
        