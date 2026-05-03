from nicegui import ui
from utils.navigation import project_navigation_header, home_button

messages = []

@ui.refreshable
def update_chat():
    with ui.column().classes('w-full q-pa-md'):
        for role, text in messages:
            with ui.column().classes('w-full q-mb-md'):
                ui.html(f'<h3 style="margin:0; font-size: 1.1rem; font-weight: bold;">{role}</h3>')
                bg_color = 'bg-blue-1' if role == 'You' else 'bg-grey-3'
                ui.label(text).classes(f'q-pa-sm {bg_color} rounded-borders w-full')

def chat_page(project_name: str):
    # Unified Header
    home_button()
    project_navigation_header(project_name, current_page='chat')

    def send_message():
        msg = input_field.value
        if msg:
            messages.append(('You', msg))
            messages.append(('AI Ally', f"Acknowledged. Checking context for {project_name.title()}."))
            input_field.value = ''
            update_chat.refresh()

    with ui.column().classes('w-full h-screen no-wrap'):
        with ui.scroll_area().classes('flex-grow w-full max-w-3xl mx-auto'):
            update_chat()

        with ui.row().classes('w-full q-pa-md bg-white border-t justify-center'):
            with ui.row().classes('w-full max-w-3xl items-center'):
                input_field = ui.input(placeholder='Type your message...') \
                    .classes('flex-grow').on('keydown.enter', send_message) \
                    .props('outlined rounded aria-label="Message input"')
                
                ui.button(icon='send', on_click=send_message) \
                    .props('round flat color=primary aria-label="Send message"')
                    