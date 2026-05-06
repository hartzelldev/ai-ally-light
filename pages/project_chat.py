from nicegui import ui
import datetime
from pathlib import Path
import json
from utils.navigation import project_navigation_header, home_button
from core.config_manager import get_active_config, PROJECTS_DIR, save_thread
from core.indexer import query_vector_db
from core.ai_logic import AIEngine

def chat_page(project_name: str):
    home_button()
    project_navigation_header(project_name, current_page='chat')
    
    config = get_active_config(project_name)
    history = [] 

    # --- Save Logic ---

    def save_current_thread(name: str):
            try:
                saved_as = save_thread(project_name, name, history)
                
                ui.notify(f"Thread saved as {saved_as}", color='positive')
                save_dialog.close()
            except Exception as e:
                ui.notify(f"Error saving thread: {e}", color='negative')

    # --- Save Dialog ---
    with ui.dialog() as save_dialog, ui.card().classes('q-pa-md'):
        ui.label('Save Conversation').classes('text-h6')
        name_input = ui.input('Thread Name', placeholder='e.g., Plot Brainstorming') \
            .classes('w-full').on('keydown.enter', lambda: save_current_thread(name_input.value))
        with ui.row().classes('w-full justify-end'):
            ui.button('Cancel', on_click=save_dialog.close).props('flat')
            ui.button('Save', on_click=lambda: save_current_thread(name_input.value))

    @ui.refreshable
    def update_chat():
        with ui.column().classes('w-full q-pa-md'):
            for role, text in history:
                with ui.column().classes('w-full q-mb-md'):
                    ui.html(f'<h3 style="margin:0; font-size: 1.1rem; font-weight: bold;">{role}</h3>')
                    bg_color = 'bg-blue-1' if role == 'You' else 'bg-grey-2'
                    with ui.column().classes(f'q-pa-md {bg_color} rounded-borders w-full'):
                        ui.markdown(text).classes('w-full')

    def send_message():
        user_text = input_field.value
        if not user_text: return
        history.append(('You', user_text))
        input_field.value = ''
        update_chat.refresh()

        context = query_vector_db(project_name, user_text)
        full_system_prompt = f"{config.get('system_prompt', '')}\n\nContext:\n{context}"
        api_messages = [{"role": "system", "content": full_system_prompt}]
        for role, text in history[-5:]:
            api_messages.append({"role": "user" if role == "You" else "assistant", "content": text})

        engine = AIEngine(config)
        response_text = engine.get_response(api_messages)
        history.append(('AI Ally', response_text))
        update_chat.refresh()
        ui.notify("AI Response received.")

    # --- Layout ---
    with ui.column().classes('w-full h-screen no-wrap'):
        # Top Action Bar for Chat
        with ui.row().classes('w-full q-pa-sm bg-grey-1 border-b justify-end'):
            ui.button('Save Thread', icon='save', on_click=save_dialog.open) \
                .props('flat color=primary aria-label="Save current conversation"')

        with ui.scroll_area().classes('flex-grow w-full max-w-3xl mx-auto'):
            update_chat()

        with ui.row().classes('w-full q-pa-md bg-white border-t justify-center'):
            with ui.row().classes('w-full max-w-3xl items-center'):
                input_field = ui.input(placeholder='Ask about your files...') \
                    .classes('flex-grow').on('keydown.enter', send_message) \
                    .props('outlined rounded aria-label="Message input"')
                ui.button(icon='send', on_click=send_message) \
                    .props('round flat color=primary aria-label="Send message"')
                    