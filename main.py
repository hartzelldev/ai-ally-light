from nicegui import ui

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

# --- Project Specific Routes
ui.page('/project/{project_name}/chat')(chat_page)
ui.page('/project/{project_name}/settings')(project_settings_page)
ui.page('/project/{project_name}/files')(project_files_page)
ui.page('/project/{project_name}/threads')(project_threads_page)

ui.run(title="AI Ally Light", port=8082, reload=True)
