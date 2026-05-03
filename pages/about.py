from nicegui import ui
from utils.navigation import home_button

def about_page():
    home_button()
    
    with ui.column().classes('w-full max-w-3xl mx-auto q-pa-md'):
        ui.label('About AI Ally Light').classes('text-h3 q-mb-md')
        
        ui.markdown("""
        AI Ally Light is a **privacy-focused, local-first platform** designed to give you complete control over your AI interactions. 
        
        ### What it does:
        *   **Context-Aware Knowledge:** Connect the AI to your specific documents and files for accurate, relevant assistance.
        *   **Local Control:** Manage your data on your own terms, away from cloud-synced storage and external interference.
        *   **Custom Environments:** Configure separate personas and chunking methods for different types of work or research.
        
        Whether you are analyzing technical documentation, managing research projects, or organizing creative archives, AI Ally Light adapts to your workflow.
        """).classes('text-body1')