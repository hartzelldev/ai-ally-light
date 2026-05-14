from nicegui import ui
import os
import signal

def index_page():
    ui.colors(primary='#387086', secondary='#5e35b1', accent='#111b1e')
    # Standard container to keep everything centered and neat
    with ui.column().classes('w-full max-w-xl mx-auto q-pa-lg items-center'):
        ui.markdown('# AI Ally Light').classes('text-h1 q-mb-md')
        ui.label('AI Workspace for Power Users').classes('text-subtitle1 q-mb-xl')
        
        with ui.card().classes('w-full shadow-2'):
            with ui.list().props('bordered separator'):
                # Using list items for better screen reader grouping
                with ui.item(on_click=lambda: ui.navigate.to('/projects')).props('clickable'):
                    with ui.item_section().props('avatar'):
                        ui.icon('folder')
                    with ui.item_section():
                        ui.item_label('Projects')
                        ui.item_label('Access and manage your organized knowledge bases and workspaces').props('caption')

                with ui.item(on_click=lambda: ui.navigate.to('/settings')).props('clickable'):
                    with ui.item_section().props('avatar'):
                        ui.icon('settings')
                    with ui.item_section():
                        ui.item_label('Global Settings')
                        ui.item_label('API Keys and system-wide defaults').props('caption')

                with ui.item(on_click=lambda: ui.navigate.to('/about')).props('clickable'):
                    with ui.item_section().props('avatar'):
                        ui.icon('help')
                    with ui.item_section():
                        ui.item_label('About')
                        ui.item_label('Application information').props('caption')

        def handle_exit():
            ui.notify('Shutting down server...', type='warning')
            # We use a tiny delay so the notification actually shows up 
            # before the server vanishes
            ui.timer(1.0, lambda: os.kill(os.getpid(), signal.SIGINT), once=True)

        (ui.button('Exit Application', icon='power_settings_new', color='red', 
                  on_click=handle_exit) 
            .classes('q-mt-xl').props('flat'))
            