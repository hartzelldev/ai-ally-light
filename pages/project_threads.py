from nicegui import ui
import json
from pathlib import Path
from utils.navigation import project_navigation_header, home_button

PROJECTS_DIR = Path(__file__).parent.parent / "projects"

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

    with ui.column().classes('w-full max-w-4xl mx-auto q-pa-md'):
        ui.markdown(f'# Chat History: {display_name}')
        
        if not threads:
            with ui.card().classes('w-full q-pa-md'):
                ui.label('No past conversations found. Start a new chat to begin!').classes('text-italic text-grey-7')
        else:
            for thread_file in threads:
                # Clean up the name for display (e.g., "2026-05-06_Plot_Help" -> "2026-05-06 Plot Help")
                clean_name = thread_file.stem.replace("_", " ")
                
                with ui.card().classes('w-full q-mb-sm hover:bg-blue-50 cursor-pointer'):
                    with ui.row().classes('w-full items-center justify-between'):
                        with ui.row().classes('items-center gap-3'):
                            ui.icon('chat', color='primary')
                            # Heading for screen reader navigation
                            ui.html(f'<h3 style="margin:0; font-size: 1rem; font-weight: bold;">{clean_name}</h3>')
                        
                        with ui.row().classes('gap-2'):
                            ui.button(icon='open_in_new', color='primary', 
                                      on_click=lambda n=thread_file.stem: ui.navigate.to(f'/project/{project_name}/chat?thread={n}')) \
                                .props('flat dense aria-label="Open thread ' + clean_name + '"')
                            
                            ui.button(icon='delete', color='negative', 
                                      on_click=lambda f=thread_file: [f.unlink(), ui.navigate.reload()]) \
                                .props('flat dense aria-label="Delete thread ' + clean_name + '"')
                                