from nicegui import ui

# We'll keep the session messages here
messages = []

@ui.refreshable
def update_chat():
    with ui.column().classes('w-full q-pa-md'):
        for role, text in messages:
            with ui.column().classes('w-full q-mb-md'):
                # Using ui.html ensures the H3 is rendered as a real heading
                # and not as plain text.
                ui.html(f'<h3 style="margin:0; font-size: 1.1rem; font-weight: bold;">{role}</h3>')
                
                if role == 'You':
                    ui.label(text).classes('q-pa-sm bg-blue-1 rounded-borders w-full')
                else:
                    ui.label(text).classes('q-pa-sm bg-grey-3 rounded-borders w-full')

def chat_page(project_name: str):
    
    def send_message():
        msg = input_field.value
        if msg:
            # 1. Append User message
            messages.append(('You', msg))
            
            # 2. Append AI response
            messages.append(('AI Ally', f"Acknowledged. I'm looking at the context for {project_name.title()}."))
            
            # 3. Clear the input
            input_field.value = ''
            
            # 4. Trigger the visual refresh
            update_chat.refresh()

    # --- MAIN LAYOUT ---
    with ui.column().classes('w-full h-screen no-wrap'):
        # 1. Header
        with ui.row().classes('w-full items-center justify-between q-pa-sm bg-grey-2 shadow-1'):
            ui.button(icon='arrow_back', on_click=lambda: ui.navigate.to('/projects')) \
                .props('flat aria-label="Back to Projects"')
            ui.label(f'Chat: {project_name.title()}').classes('text-h6')
            ui.label('') 

        # 2. Scrollable Chat Area
        with ui.scroll_area().classes('flex-grow w-full max-w-3xl mx-auto'):
            update_chat()

        # 3. Sticky Input
        with ui.row().classes('w-full q-pa-md bg-white border-t justify-center'):
            with ui.row().classes('w-full max-w-3xl items-center'):
                input_field = ui.input(placeholder='Type your message...') \
                    .classes('flex-grow').on('keydown.enter', send_message) \
                    .props('outlined rounded aria-label="Message input"')
                
                ui.button(icon='send', on_click=send_message) \
                    .props('round flat color=primary aria-label="Send message"')
