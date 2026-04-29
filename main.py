from nicegui import ui

# Import page functions
from pages.index import index_page
from pages.settings import settings_page
from pages.about import about_page
from pages.projects import projects_dashboard
from pages.project_settings import project_settings_page 
from pages.chat import chat_page

# Manually register the routes
ui.page('/')(index_page)
ui.page('/settings')(settings_page)
ui.page('/about')(about_page)
ui.page('/projects')(projects_dashboard)
ui.page('/projects/{project_name}/settings')(project_settings_page)
ui.page('/projects/{project_name}/chat')(chat_page)

ui.run(title="AI Ally Light", port=8082, reload=True)
