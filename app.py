import uvicorn
import webbrowser
import threading
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import os
import logging

# Force offline mode for Hugging Face to prevent the launch hang
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Silence the internal library warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Import the routers
from routers import settings_router, project_router  

app = FastAPI(title="AI Ally Light")

# 1. Setup Directories
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 2. Register Routers
app.include_router(settings_router.router, prefix="/settings")
app.include_router(project_router.router, prefix="/projects")

# 3. Main Landing Page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Renders the main dashboard. 
    Screen readers will land here first.
    """
    return templates.TemplateResponse(
    request=request, 
    name="index.html", 
    context={}
)

# 4. Global Error Handling (Optional but helpful)
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return HTMLResponse("<h2>404: Page Not Found</h2>", status_code=404)

def open_browser():
    """Opens the browser after a short delay to ensure the server is up."""
    webbrowser.open("http://127.0.0.1:5002")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    # Run with auto-reload so it restarts when you save a file
    uvicorn.run("app:app", host="127.0.0.1", port=5002, reload=True)
    