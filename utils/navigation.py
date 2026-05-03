from nicegui import ui

def home_button():
    """Renders a consistent Home navigation button with guaranteed screen reader text."""
    with ui.row().classes('w-full q-pa-sm border-b items-center'):
        # We put 'Home' inside the button and use 'aria-label' as a backup
        # 'flat' ensures it doesn't look like a bulky block
        ui.button('Home', icon='home', on_click=lambda: ui.navigate.to('/')) \
            .props('flat color=primary aria-label="Home - Return to main dashboard"') \
            .classes('q-px-md')

def project_navigation_header(project_name: str, current_page: str):
    """
    Creates a consistent navigation bar for project-specific pages.
    current_page: 'chat', 'files', or 'settings'
    """
    display_name = project_name.replace("_", " ").title()
    
    with ui.row().classes('w-full justify-center gap-4 q-mb-md border-b q-pb-sm'):
        # Chat Button
        ui.button('Chat', icon='chat', 
                  on_click=lambda: ui.navigate.to(f'/project/{project_name}/chat')) \
            .props(f'flat {"color=primary" if current_page == "chat" else ""}') \
            .classes('font-bold' if current_page == 'chat' else '')

        # Files Button
        ui.button('Files', icon='folder', 
                  on_click=lambda: ui.navigate.to(f'/project/{project_name}/files')) \
            .props(f'flat {"color=primary" if current_page == "files" else ""}') \
            .classes('font-bold' if current_page == 'files' else '')

        # Settings Button
        ui.button('Settings', icon='settings', 
                  on_click=lambda: ui.navigate.to(f'/project/{project_name}/settings')) \
            .props(f'flat {"color=primary" if current_page == "settings" else ""}') \
            .classes('font-bold' if current_page == 'settings' else '')

        # Threads Button (Placeholder for your future feature)
        ui.button('Threads', icon='history') \
            .props('flat disable') \
            .tooltip('Coming soon')
            