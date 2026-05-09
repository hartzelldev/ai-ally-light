import datetime
import json
from pathlib import Path
from nicegui import ui, app, run  # Added 'run' to the imports here
from utils.navigation import project_navigation_header, home_button
from core.config_manager import get_active_config, PROJECTS_DIR, save_thread
from core.indexer import query_vector_db
from core.ai_logic import AIEngine
from utils.audio import event_beep

def chat_page(project_name: str, thread: str = None):
    # --- 1. Robust State Management ---
    class ChatState:
        def __init__(self):
            self.history = []
            self.display_name = "New Conversation"
            self.thread_id = thread

    state = ChatState()

    # Load existing thread if applicable
    if state.thread_id:
        thread_file = PROJECTS_DIR / project_name / "threads" / f"{state.thread_id}.json"
        if thread_file.exists():
            try:
                with open(thread_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                    state.history = [tuple(msg) for msg in loaded_data]
                state.display_name = state.thread_id.replace('_', ' ')
            except Exception as e:
                event_beep('error')
                ui.notify(f"Error loading thread: {e}", color='negative')

    config = get_active_config(project_name)
    export_container = {'text': '', 'folder': '', 'file': ''}
    initial_save_name = state.thread_id if state.thread_id else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    # --- 2. Logic Functions ---
    def copy_to_clipboard(text: str):
        escaped_text = text.replace('`', '\\`').replace("'", "\\'")
        ui.run_javascript(f'navigator.clipboard.writeText(`{escaped_text}`)')
        ui.notify("Response copied to clipboard")

    def finalize_export():
        text = export_container['text']
        folder = folder_input.value or ""
        filename = file_input.value or ""
        if not filename:
            event_beep('info')
            ui.notify("Filename is required", type='warning')
            return
        
        base_path = PROJECTS_DIR / project_name / "exports"
        target_dir = base_path / folder.strip('/')
        target_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename.endswith(('.md', '.txt')):
            filename += '.md'
            
        full_path = target_dir / filename
        try:
            full_path.write_text(text, encoding='utf-8')
            ui.notify(f"Saved to exports/{folder}/{filename}", color='positive')
            export_dialog.close()
        except Exception as e:
            ui.notify(f"File Error: {e}", color='negative')

    def run_save_thread(name: str):
        try:
            saved_as = save_thread(project_name, name, state.history)
            ui.notify(f"Thread saved as {saved_as}", color='positive')
            state.thread_id = saved_as 
            save_thread_dialog.close()
            title_label.set_content(f'<h2 style="margin:0; font-size: 1.2rem;">Active Thread: {saved_as.replace("_", " ")}</h2>')
        except Exception as e:
            ui.notify(f"Save Error: {e}", color='negative')

    # --- 3. Dialogs ---
    with ui.dialog() as save_thread_dialog, ui.card().classes('q-pa-md w-80'):
        ui.label('Save Conversation Thread').classes('text-h6')
        thread_name_input = ui.input('Thread Name', value=initial_save_name) \
            .classes('w-full').on('keydown.enter', lambda: run_save_thread(thread_name_input.value))
        with ui.row().classes('w-full justify-end q-mt-md'):
            ui.button('Cancel', on_click=save_thread_dialog.close).props('flat')
            ui.button('Save', on_click=lambda: run_save_thread(thread_name_input.value))

    with ui.dialog() as export_dialog, ui.card().classes('q-pa-md w-80'):
        ui.label('Export AI Response').classes('text-h6')
        folder_input = ui.input('Sub-folder (optional)', placeholder='drafts').classes('w-full')
        file_input = ui.input('Filename', placeholder='response-notes').classes('w-full') \
            .on('keydown.enter', finalize_export)
        with ui.row().classes('w-full justify-end q-mt-md'):
            ui.button('Cancel', on_click=export_dialog.close).props('flat')
            ui.button('Export', on_click=finalize_export)

    def open_export(text: str):
        export_container['text'] = text
        export_dialog.open()

    # --- 4. Chat Rendering ---
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
                                    .props('flat dense color=grey-7 aria-label="Copy response to clipboard"')
                                ui.button(icon='download', on_click=lambda t=text: open_export(t)) \
                                    .props('flat dense color=grey-7 aria-label="Export response to file"')

    async def send_message():
        user_text = input_field.value
        if not user_text:
            return

        state.history.append(('You', user_text))
        input_field.value = ''
        update_chat.refresh()

        context = query_vector_db(project_name, user_text)
        system_prompt = config.get('system_prompt', 'You are a helpful assistant.')
        
        api_messages = [{"role": "system", "content": f"{system_prompt}\n\nProject Context:\n{context}"}]
        for r, t in state.history[-10:]:
            api_messages.append({"role": "user" if r == "You" else "assistant", "content": t})

        try:
            engine = AIEngine(config)
            # CHANGED: Use run.io_bound directly from the nicegui import
            response = await run.io_bound(engine.get_response, api_messages)
            
            state.history.append(('AI Ally', response))
            update_chat.refresh()
            event_beep('complete')
            ui.notify("AI Response received")
        except Exception as e:
            event_beep('error')
            ui.notify(f"AI Error: {e}", color='negative')

    # --- 5. Page Layout ---
    home_button()
    project_navigation_header(project_name, current_page='chat')

    with ui.column().classes('w-full h-screen no-wrap'):
        with ui.row().classes('w-full q-pa-sm bg-blue-grey-1 border-b items-center justify-between'):
            title_label = ui.html(f'<h2 style="margin:0; font-size: 1.2rem;">Active Thread: {state.display_name}</h2>')
            
            ui.button('Save Thread', icon='save', on_click=save_thread_dialog.open) \
                .props('flat color=primary aria-label="Save or rename this conversation thread"')

        with ui.scroll_area().classes('flex-grow w-full max-w-4xl mx-auto'):
            update_chat()

        with ui.row().classes('w-full q-pa-md bg-white border-t justify-center'):
            with ui.row().classes('w-full max-w-4xl items-center gap-2'):
                input_field = ui.input(placeholder='Ask a question...') \
                    .classes('flex-grow').on('keydown.enter', send_message) \
                    .props('outlined rounded aria-label="Chat input field"')
                ui.button(icon='send', on_click=send_message) \
                    .props('round flat color=primary aria-label="Send message"')
                    