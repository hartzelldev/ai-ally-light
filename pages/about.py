from nicegui import ui

@ui.page('/about')
def about_page():
    with ui.column().classes('w-full max-w-xl mx-auto q-pa-lg'):
        ui.button('Back', icon='arrow_back', on_click=lambda: ui.navigate.to('/')) \
            .props('flat')
        
        ui.label('About AI Ally Light').classes('text-h4 q-mt-md')
        ui.markdown('''
        This application is designed to help novelists manage complex story bibles 
        and interact with AI models while maintaining local control of data.
        ''')