from nicegui import ui, app
import logging
import os
import secrets


# Import page functions
from pages.index import index_page
from pages.settings import global_settings_page
from pages.about import about_page
from pages.projects import projects_dashboard
from pages.project_chat import chat_page
from pages.project_settings import project_settings_page
from pages.project_files import project_files_page
from pages.project_threads import project_threads_page

# --- Global Routes ---
ui.page('/')(index_page)
ui.page('/settings')(global_settings_page)
ui.page('/about')(about_page)
ui.page('/projects')(projects_dashboard)

# --- Project Specific Routes ---
# We add two decorators for chat: one for the base link and one for specific threads
@ui.page('/project/{project_name}/chat/{thread}')
@ui.page('/project/{project_name}/chat')
def chat_entry(project_name: str, thread: str = None):
    chat_page(project_name, thread)

ui.page('/project/{project_name}/settings')(project_settings_page)
ui.page('/project/{project_name}/files')(project_files_page)
ui.page('/project/{project_name}/threads')(project_threads_page)

# Setup Logging to Terminal
logging.basicConfig(level=logging.INFO)

# This will print the full error to your terminal even if it flashes/vanishes in the UI
@app.on_exception
def handle_exception(exception):
    # Log the full traceback to the console/terminal
    logging.error("--- UI EXCEPTION CAUGHT ---")
    logging.error(exception, exc_info=True)
    ui.notify(f"Ghost Error Caught: {exception}", color='negative', duration=0)

# Try to get the secret from the system environment
# If it doesn't exist, generate a random one for this session only
secret = os.environ.get('AI_ALLY_SECRET', secrets.token_urlsafe(32))

ui.run(native=True, window_size=(1200, 800), storage_secret=secret, title="AI Ally Light", port=8082)
