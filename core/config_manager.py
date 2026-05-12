import os
import json
from pathlib import Path
from dotenv import load_dotenv
from schemas.settings import GlobalConfig
import datetime

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"
ENV_PATH = BASE_DIR / ".env"
PROJECTS_DIR = Path("projects")

class ConfigManager:
    def __init__(self):
        self.config = self.load_config()
        # Load the global .env keys into the environment
        load_dotenv(ENV_PATH)

    def load_config(self) -> GlobalConfig:
        """Loads JSON config and validates it with Pydantic."""
        if not CONFIG_PATH.exists():
            # Create a default one if it doesn't exist
            default_config = GlobalConfig()
            self.save_config(default_config)
            return default_config
        
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
            return GlobalConfig(**data)

    def save_config(self, config: GlobalConfig):
        """Saves the Pydantic model back to JSON."""
        with open(CONFIG_PATH, "w") as f:
            json.dump(config.model_dump(), f, indent=4)
        self.config = config

    def save_api_key(self, provider_name: str, key: str):
            """Updates or adds an API key to the .env file."""
            env_key = f"{provider_name.upper()}_API_KEY"
            
            # Read existing lines
            lines = []
            if ENV_PATH.exists():
                with open(ENV_PATH, "r") as f:
                    lines = f.readlines()
            
            # Update existing or add new
            found = False
            new_line = f"{env_key}={key}\n"
            for i, line in enumerate(lines):
                if line.startswith(f"{env_key}="):
                    lines[i] = new_line
                    found = True
                    break
            
            if not found:
                lines.append(new_line)
                
            with open(ENV_PATH, "w") as f:
                f.writelines(lines)
            
            # Reload environment variables so the app sees the change
            load_dotenv(ENV_PATH, override=True)

    def get_api_key(self, provider_name: str) -> str:
        """
        Prioritizes the .env file for keys.
        Example: If provider is 'Groq', it looks for 'GROQ_API_KEY' in .env
        """
        env_key = f"{provider_name.upper()}_API_KEY"
        return os.getenv(env_key, "")

# Create a singleton instance
manager = ConfigManager()

def get_active_config(project_name: str):
    """
    Helper to get the merged global and local project settings.
    This ensures project-specific models override global ones.
    """
    import json
    from pathlib import Path

    # 1. Setup paths
    projects_dir = Path("projects")
    global_config_path = Path("config.json")
    project_config_path = projects_dir / project_name / "project_config.json"

    # 2. Start with a baseline dictionary of defaults
    # This prevents crashes if config.json is missing
    active_config = {
        "chat_provider": "openrouter",
        "chat_model": "google/gemini-2.0-flash-001",
        "temperature": 0.7
    }

    # 3. Load Global Settings (config.json)
    if global_config_path.exists():
        try:
            with open(global_config_path, "r") as f:
                global_data = json.load(f)
                active_config.update(global_data)
        except Exception as e:
            print(f"Error loading global config: {e}")

    # 4. Load Project-Specific Settings (project_config.json)
    # This is where your new model name will overwrite the old global one
    if project_config_path.exists():
        try:
            with open(project_config_path, "r") as f:
                project_data = json.load(f)
                active_config.update(project_data)
                # Success! The project model now takes priority
        except Exception as e:
            print(f"Error loading project config for {project_name}: {e}")

    return active_config


def save_thread(project_name: str, name: str, history: list):
    """Saves a chat thread and returns the standardized safe name used for the file."""
    # Ensure name is filesystem safe
    safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
    
    # Fallback for empty strings
    if not safe_name:
        safe_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

    thread_path = PROJECTS_DIR / project_name / "threads"
    thread_path.mkdir(parents=True, exist_ok=True)
    
    file_path = thread_path / f"{safe_name}.json"
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
        return safe_name
    except Exception as e:
        # We log it to the console for debugging, 
        # but we don't crash the app.
        print(f"FileSystem Error in save_thread: {e}")
        return None
