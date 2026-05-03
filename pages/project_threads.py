from nicegui import ui
from utils.navigation import project_navigation_header, home_button

def project_threads_page(project_name: str):
    home_button()
    project_navigation_header(project_name, current_page='threads')
    display_name = project_name.replace("_", " ").title()

    with ui.column().classes('w-full max-w-4xl mx-auto q-pa-md'):
        ui.markdown(f'# Chat History: {display_name}')
        
        with ui.card().classes('w-full q-pa-md'):
            ui.label('Past conversations will appear here.').classes('text-italic text-grey-7')
            
            # This is where you will eventually loop through stored JSON/DB threads
            # Example UI structure for a thread:
            # with ui.row().classes('w-full justify-between items-center border-b q-py-sm'):
            #     ui.label('2026-05-02: Plot Brainstorming').classes('cursor-pointer hover:text-blue')
            #     ui.button(icon='delete', color='negative').props('flat dense')
            