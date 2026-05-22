import datetime
import json
import os
import re
from pathlib import Path
from nicegui import ui, app, run
from utils.navigation import project_navigation_header, home_button
from core.config_manager import get_active_config, PROJECTS_DIR, save_thread
from core.indexer import query_vector_db
from core.ai_logic import AIEngine
from utils.audio import event_beep

# --- Helper Logic for Persistence ---
def update_last_active_thread(project_name, thread_id):
    config_path = PROJECTS_DIR / project_name / "project_config.json"
    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                config = json.load(f)
            except Exception:
                config = {}
    config['last_thread_id'] = thread_id
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def get_last_active_thread(project_name):
    config_path = PROJECTS_DIR / project_name / "project_config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f).get('last_thread_id')
            except Exception:
                return None
    return None

def chat_page(project_name: str, thread: str = None):
    # --- 1. State and Recall Logic ---
    class ChatState:
        def __init__(self):
            self.history = []
            # Check for thread in URL first, then config
            self.thread_id = thread or get_last_active_thread(project_name)
            self.display_name = "New Conversation"

    state = ChatState()

    # If user hits base /chat but we have a last_thread_id, redirect to it
    if not thread and state.thread_id:
        ui.navigate.to(f'/project/{project_name}/chat/{state.thread_id}')
        return

    # If no thread exists at all (brand new project), set an initial timestamp
    if not state.thread_id:
        state.thread_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        update_last_active_thread(project_name, state.thread_id)

    # Update the header display name
    state.display_name = state.thread_id.replace('_', ' ')

    # Load existing thread data from JSON
    thread_file = PROJECTS_DIR / project_name / "threads" / f"{state.thread_id}.json"
    if thread_file.exists():
        try:
            with open(thread_file, 'r', encoding='utf-8') as f:
                state.history = [tuple(msg) for msg in json.load(f)]
        except Exception as e:
            print(f"Load error: {e}")

    config = get_active_config(project_name)
    export_container = {'text': ''}

    # --- 2. Logic Functions ---
    async def auto_save_logic():
        # Save history to file and ensure config knows this is the active thread
        save_thread(project_name, state.thread_id, state.history)
        update_last_active_thread(project_name, state.thread_id)

    def copy_to_clipboard(text: str):
        escaped_text = text.replace('`', '\\`').replace("'", "\\'")
        ui.run_javascript(f'navigator.clipboard.writeText(`{escaped_text}`)')
        ui.notify("Response copied to clipboard")

    def finalize_export():
        folder = folder_input.value or ""
        filename = file_input.value or "export"
        target_dir = PROJECTS_DIR / project_name / "exports" / folder.strip('/')
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename.endswith(('.md', '.txt')):
            filename += '.md'
            
        full_path = target_dir / filename
        full_path.write_text(export_container['text'], encoding='utf-8')
        ui.notify(f"Saved to exports/{folder}/{filename}")
        export_dialog.close()

    def run_save_thread(name: str):
        # Handle manual rename
        saved_as = save_thread(project_name, name, state.history)
        if saved_as:
            state.thread_id = saved_as
            update_last_active_thread(project_name, saved_as)
            title_label.set_content(f'<h2 style="margin:0; font-size: 1.2rem;">Active Thread: {saved_as.replace("_", " ")}</h2>')
            save_thread_dialog.close()
            ui.notify(f"Thread renamed to {saved_as}")

    # --- 3. Dialogs ---
    with ui.dialog() as save_thread_dialog, ui.card().classes('q-pa-md w-80'):
        ui.label('Rename Conversation').classes('text-h6')
        thread_name_input = ui.input('Thread Name', value=state.thread_id).classes('w-full')
        with ui.row().classes('w-full justify-end q-mt-md'):
            ui.button('Cancel', on_click=save_thread_dialog.close).props('flat')
            ui.button('Save', on_click=lambda: run_save_thread(thread_name_input.value))

    with ui.dialog() as export_dialog, ui.card().classes('q-pa-md w-80'):
        ui.label('Export AI Response').classes('text-h6')
        folder_input = ui.input('Sub-folder').classes('w-full')
        file_input = ui.input('Filename').classes('w-full')
        with ui.row().classes('w-full justify-end q-mt-md'):
            ui.button('Cancel', on_click=export_dialog.close).props('flat')
            ui.button('Export', on_click=finalize_export)

    @ui.refreshable
    def update_chat():
        with ui.column().classes('w-full q-pa-md'):
            if not state.history:
                ui.label(f'Start a conversation about {project_name.title()}...').classes('text-grey-5 text-italic mx-auto q-mt-xl')
            
            for role, text in state.history:
                with ui.column().classes('w-full q-mb-md'):
                    ui.html(f'<h3 style="margin:0; font-size: 1.1rem; font-weight: bold;">{role}</h3>')
                    bg_color = 'bg-blue-1' if role == 'You' else 'bg-grey-2'
                    with ui.column().classes(f'q-pa-md {bg_color} rounded-borders w-full shadow-sm'):
                        ui.markdown(text).classes('w-full')
                        if role == 'AI Ally':
                            with ui.row().classes('w-full justify-end gap-2 q-mt-sm'):
                                ui.button(icon='content_copy', on_click=lambda t=text: copy_to_clipboard(t)) \
                                    .props('flat dense color=grey-7 aria-label="Copy to clipboard"')
                                ui.button(icon='download', on_click=lambda t=text: [export_container.update({'text': t}), export_dialog.open()]) \
                                    .props('flat dense color=grey-7 aria-label="Export response"')

    async def send_message():
        user_text = input_field.value
        if not user_text: return
        
        state.history.append(('You', user_text))
        input_field.value = ''
        update_chat.refresh()
        await auto_save_logic()
        
        context = await run.io_bound(query_vector_db, project_name, user_text)
        system_prompt = config.get('system_prompt', 'You are a helpful assistant.')
        
        api_messages = [{"role": "system", "content": f"{system_prompt}\n\nProject Context:\n{context}"}]
        # Load last 10 messages for context
        for r, t in state.history[-10:]:
            api_messages.append({"role": "user" if r == "You" else "assistant", "content": t})

        try:
            engine = AIEngine(config)
            response = await run.io_bound(engine.get_response, api_messages)
            state.history.append(('AI Ally', response))
            update_chat.refresh()
            await auto_save_logic()
            event_beep('complete')
        except Exception as e:
            event_beep('error')
            ui.notify(f"Error: {e}", color='negative')

    # --- 4. Page Layout ---
    home_button()
    project_navigation_header(project_name, current_page='chat')
    
    with ui.column().classes('w-full h-screen no-wrap'):
        with ui.row().classes('w-full q-pa-sm bg-blue-grey-1 border-b items-center justify-between'):
            title_label = ui.html(f'<h2 style="margin:0; font-size: 1.2rem;">Active Thread: {state.display_name}</h2>')
            ui.button('Rename', icon='edit', on_click=save_thread_dialog.open).props('flat color=primary aria-label="Rename thread"')

        with ui.scroll_area().classes('flex-grow w-full max-w-4xl mx-auto'):
            update_chat()

        with ui.row().classes('w-full q-pa-md bg-white border-t justify-center'):
            with ui.row().classes('w-full max-w-4xl items-center gap-2'):
                input_field = ui.input(placeholder='Ask a question...') \
                    .classes('flex-grow').on('keydown.enter', send_message) \
                    .props('outlined rounded aria-label="Chat input field"')
                ui.button(icon='send', on_click=send_message) \
                    .props('round flat color=primary aria-label="Send message"')
