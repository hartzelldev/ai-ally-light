from nicegui import ui
from utils.navigation import project_navigation_header, home_button
from core.config_manager import get_active_config
from core.indexer import query_vector_db
from core.ai_logic import AIEngine

# Using a list within the page function keeps chat history isolated to the session
def chat_page(project_name: str):
    home_button()
    project_navigation_header(project_name, current_page='chat')
    
    # Load project settings
    config = get_active_config(project_name)
    history = [] # Local chat history

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
        if not user_text:
            return

        # 1. Update UI with user message
        history.append(('You', user_text))
        input_field.value = ''
        update_chat.refresh()

        # 2. Get Context from RAG
        context = query_vector_db(project_name, user_text)
        
        # 3. Construct the prompt for the AI
        # We combine the system prompt from settings + the RAG context
        full_system_prompt = f"{config.get('system_prompt', 'You are a helpful assistant.')}\n\nContext from files:\n{context}"
        
        # 4. Prepare message list for the API
        api_messages = [{"role": "system", "content": full_system_prompt}]
        # Add simple history (last 5 exchanges for brevity)
        for role, text in history[-5:]:
            api_role = "user" if role == "You" else "assistant"
            api_messages.append({"role": api_role, "content": text})

        # 5. Get AI Response
        engine = AIEngine(config)
        response_text = engine.get_response(api_messages)
        
        # 6. Update UI with AI message
        history.append(('AI Ally', response_text))
        update_chat.refresh()
        
        # Notify screen reader that a response has arrived
        ui.notify("AI Response received.")

    # --- Layout ---
    with ui.column().classes('w-full h-screen no-wrap'):
        with ui.scroll_area().classes('flex-grow w-full max-w-3xl mx-auto'):
            update_chat()

        with ui.row().classes('w-full q-pa-md bg-white border-t justify-center'):
            with ui.row().classes('w-full max-w-3xl items-center'):
                input_field = ui.input(placeholder='Ask about your files...') \
                    .classes('flex-grow').on('keydown.enter', send_message) \
                    .props('outlined rounded aria-label="Message input"')
                
                ui.button(icon='send', on_click=send_message) \
                    .props('round flat color=primary aria-label="Send message"')
                    