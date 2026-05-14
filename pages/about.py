from nicegui import ui
from utils.navigation import home_button

def about_page():
    home_button()
    
    with ui.column().classes('w-full max-w-3xl mx-auto q-pa-md'):
        ui.markdown(f"""
        # About AI Ally Light
        Version 0.5.1 Beta (2026-05-13)<br>
        Copyright 2026 Gary Hartzell<br>
        
        AI Ally Light is a **privacy-focused, local-first platform** designed to give you complete control over your AI interactions.         AI Ally Light was born from a professional need for a secure, accessible AI assistant that could handle sensitive technical documentation and long-form creative research without the privacy risks associated with cloud-based platforms.
        
        ### Core Principles:
        * **Local Control:** Manage your data on your own terms. Your files and configurations stay on your hardware, not in a vendor's cloud.
        * **Context-Aware Knowledge:** Connect the AI to your specific documents and files for accurate, relevant assistance via localized RAG.
        * **Custom Environments:** Configure separate personas, model providers, and data-chunking methods tailored to your specific projects.
        * **API Compatibility:** AI Ally Light works with **OpenRouter, Groq, TogetherAI, Google Gemini** and all other **OpenAI** compatible providers, as well as **local models via Ollama** or your favorite local inference engine.
        * **Accessibility First:** Designed with a clean, high-contrast interface and optimized for **screen reader compatibility**. Features include logical heading hierarchies and audio cues for non-visual feedback.

        Whether you are analyzing technical documentation or organizing creative archives, AI Ally Light is built to adapt to your unique workflow.

        ---

        ### About the Author
        **Gary Hartzell** is an IT professional and Cybersecurity Analyst with over twenty years of experience in the field. A dedicated Python developer and AI researcher, he builds tools that bridge the gap between complex security orchestration and user-centric design.
        """).classes('text-body1')

        ui.separator().classes('q-my-lg')

        with ui.row().classes('w-full justify-center gap-4'):
            (ui.button('GitHub Project', icon='code', 
                      on_click=lambda: ui.navigate.to('https://github.com/hartzelldev/ai-ally-light', new_tab=True)) 
                .props('unelevated color=grey-9') 
                .tooltip('View the source code (opens in browser)'))

            (ui.button('Download Page', icon='download', 
                      on_click=lambda: ui.navigate.to('https://hartzelldev.github.io', new_tab=True)) 
                .props('unelevated color=primary') 
                .tooltip('Check for updates (opens in browser)'))

            (ui.button('Contact Author', icon='mail', 
                      on_click=lambda: ui.navigate.to('https://hartzelldev.github.io/contact.html', new_tab=True)) 
                .props('unelevated color=secondary') 
                .tooltip('Get in touch (opens in browser)'))

