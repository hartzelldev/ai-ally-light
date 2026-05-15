from nicegui import ui
import json
from pathlib import Path
from utils.navigation import project_navigation_header, home_button
from pages.project_chat import update_last_active_thread
from core.config_manager import PROJECTS_DIR

def get_threads(project_name: str):
    """Returns a list of thread files for the project."""
    thread_path = PROJECTS_DIR / project_name / "threads"
    if not thread_path.exists():
        return []
    # Sort by modification time so newest is at the top
    return sorted(thread_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

def project_threads_page(project_name: str):
    home_button()
    project_navigation_header(project_name, current_page='threads')
    display_name = project_name.replace("_", " ").title()

    threads = get_threads(project_name)

    # --- Deletion Confirmation Dialog ---
    with ui.dialog() as confirm_delete_dialog, ui.card().classes('q-pa-md'):
        ui.label('Are you sure?').classes('text-h6')
        ui.label('This will permanently delete this conversation thread.').classes('q-my-md')
        with ui.row().classes('w-full justify-end gap-2'):
            ui.button('Cancel', on_click=confirm_delete_dialog.close).props('flat')
            # The on_click for this button is set dynamically by request_delete()
            confirm_delete_btn = ui.button('Delete', color='negative')

    def request_delete(file_path):
        """Sets up and opens the confirmation dialog for a specific file."""
        confirm_delete_btn.on_click(lambda: [
            file_path.unlink(), 
            confirm_delete_dialog.close(), 
            ui.navigate.reload(),
            ui.notify(f"Thread deleted", color='positive')
        ])
        confirm_delete_dialog.open()

    with ui.column().classes('w-full max-w-4xl mx-auto q-pa-md'):
        ui.markdown(f'# Chat History: {display_name}')
        
        if not threads:
            with ui.card().classes('w-full q-pa-md'):
                ui.label('No past conversations found. Start a new chat to begin!').classes('text-italic text-grey-7')
        else:
            for thread_file in threads:
                # Store the stem for the closure
                t_stem = thread_file.stem
                t_path = thread_file
                # Clean up the name for display
                clean_name = t_stem.replace("_", " ")
                
                with ui.card().classes('w-full q-mb-sm hover:bg-blue-50 cursor-pointer'):
                    with ui.row().classes('w-full items-center justify-between'):
                        with ui.row().classes('items-center gap-3'):
                            ui.icon('chat', color='primary')
                            # Heading for screen reader navigation
                            ui.html(f'<h3 style="margin:0; font-size: 1rem; font-weight: bold;">{clean_name}</h3>')

                        with ui.row().classes('gap-2'):
                            # Internal function to handle the "Handshake"
                            def handle_thread_selection(name=t_stem):
                                # 1. Update the config so "Chat" knows to resume this one
                                update_last_active_thread(project_name, name)
                                # 2. Navigate using the clean path format
                                ui.navigate.to(f'/project/{project_name}/chat/{name}')

                            ui.button(icon='open_in_new', color='primary', 
                                      on_click=handle_thread_selection) \
                                .props('flat dense aria-label="Open thread ' + clean_name + '"')

                            # Delete button calls the request_delete function with the file path
                            ui.button(icon='delete', color='negative', 
                                      on_click=lambda f=t_path: request_delete(f)) \
                                .props('flat dense aria-label="Delete thread ' + clean_name + '"')