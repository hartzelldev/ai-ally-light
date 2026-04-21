import os
import json
from pathlib import Path
from dotenv import load_dotenv
from schemas.settings import GlobalConfig

# Set up paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"
ENV_PATH = BASE_DIR / ".env"

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
    """Helper to get the merged global and local project settings."""
    # We define PROJECTS_DIR here locally to be safe
    projects_dir = Path("projects")
    project_path = projects_dir / project_name
    config_path = project_path / "project_config.json"
    project_env = project_path / ".env"
    
    # Use the 'manager' object that's already in this file
    active_config = {
        "chat_provider": manager.config.chat.provider,
        "chat_model": manager.config.chat.model,
        "temperature": manager.config.chat.temperature,
        "system_prompt": manager.config.chat.system_prompt,
        "embed_provider": manager.config.embeddings.provider,
        "embed_model": manager.config.embeddings.model,
        "embed_method": manager.config.embeddings.method,
        "chunk_size": manager.config.embeddings.chunk_size,
        "chunk_overlap": manager.config.embeddings.chunk_overlap,
    }
    
    if project_env.exists():
        load_dotenv(project_env, override=True)
        active_config["chat_api_key"] = os.getenv("PROJECT_CHAT_API_KEY", "")
        active_config["embed_api_key"] = os.getenv("PROJECT_EMBED_API_KEY", "")

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                active_config.update(json.load(f))
        except Exception:
            pass
            
    return active_config
    