from pydantic import BaseModel, Field
from typing import Optional

class ChatSettings(BaseModel):
    provider: str = "ollama"
    api_key: Optional[str] = None
    base_url: Optional[str] = "http://localhost:11434"
    model: str = "llama3"
    system_prompt: str = "You are a helpful assistant."
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0

class EmbeddingSettings(BaseModel):
    provider: str = "ollama"
    api_key: Optional[str] = None
    base_url: Optional[str] = "http://localhost:11434"
    model: str = "mxbai-embed-large"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 5
    method: str = "size"  # size, delimiter, or full
    delimiter: Optional[str] = "\n\n"

class GlobalConfig(BaseModel):
    chat: ChatSettings = ChatSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    event_beep: bool = True
    max_history_turns: int = 10
    max_shown_threads: int = 20