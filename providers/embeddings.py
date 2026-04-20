"""
Embedding providers for AI Ally Light.
Supports: Built-in (Local), Ollama, OpenRouter, HuggingFace, and Custom endpoints.
"""

import requests
import os

EMBED_PROVIDERS = {
    "builtin": {
        "name": "Built-In (Local - MiniLM)",
        "base_url": None,
        "requires_api_key": False,
        "model_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        "group": "Local"
    },
    "ollama": {
        "name": "Ollama (Local)",
        "base_url": "http://127.0.0.1:11434",
        "requires_api_key": False,
        "model_url": "https://ollama.com/library",
        "group": "Local"
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://api.openrouter.ai/v1",
        "requires_api_key": True,
        "model_url": "https://openrouter.ai/models",
        "group": "Cloud"
    },
    "huggingface": {
        "name": "HuggingFace (Remote)",
        "base_url": "https://api-inference.huggingface.co/pipeline/feature-extraction",
        "requires_api_key": True,
        "model_url": "https://huggingface.co/models",
        "group": "Cloud"
    },
    "other": {
        "name": "Other (Custom)",
        "base_url": "",
        "requires_api_key": True,
        "model_url": "",
        "group": "Cloud"
    },
}

# --- ChromaDB Integration Wrapper ---

class AIAllyEmbeddingWrapper:
    """
    Wraps our provider functions so they work directly with ChromaDB.
    ChromaDB requires a class that implements a __call__ method taking List[str].
    """
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, input_texts: list[str]) -> list[list[float]]:
        # Chroma sends a list of strings; we process them one by one or as a batch
        # Note: For efficiency, remote APIs often support batches, but 
        # we'll keep it simple for now to match your existing functions.
        embeddings = []
        provider = self.config.get("embed_provider", "builtin")
        model = self.config.get("embed_model", "all-MiniLM-L6-v2")
        api_key = self.config.get("embed_api_key")
        base_url = self.config.get("embed_base_url")

        for text in input_texts:
            result = get_embedding(provider, text, model, api_key, base_url)
            embeddings.append(result)
        return embeddings

def get_remote_embedding_function(config: dict):
    """Factory function used by indexer.py to get a Chroma-ready object."""
    return AIAllyEmbeddingWrapper(config)

# --- Existing Provider Functions ---

def embed_ollama(text: str, model: str, base_url: str = None) -> list:
    url = (base_url or EMBED_PROVIDERS["ollama"]["base_url"]) + "/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]

def embed_openrouter(text: str, model: str, api_key: str, base_url: str = None) -> list:
    url = (base_url or EMBED_PROVIDERS["openrouter"]["base_url"]) + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": text}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def embed_huggingface(text: str, model: str, api_key: str, base_url: str = None) -> list:
    url = (base_url or EMBED_PROVIDERS["huggingface"]["base_url"]) + f"/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": text}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def embed_custom(text: str, model: str, api_key: str, base_url: str) -> list:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": text}
    r = requests.post(base_url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def get_embedding(provider: str, text: str, model: str, api_key: str = None, base_url: str = None) -> list:
    if provider == "ollama":
        return embed_ollama(text, model, base_url)
    elif provider == "openrouter":
        return embed_openrouter(text, model, api_key, base_url)
    elif provider == "huggingface":
        return embed_huggingface(text, model, api_key, base_url)
    elif provider == "other":
        return embed_custom(text, model, api_key, base_url)
    elif provider == "builtin":
        # Note: This branch is technically handled in indexer.py 
        # by the local SentenceTransformer, but we'll leave it for safety.
        raise ValueError("Built-in model should be handled locally in indexer.py")
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

# --- Utility Functions ---

def check_provider_connection(provider: str, base_url: str = None) -> tuple:
    if provider == "ollama":
        try:
            url = (base_url or EMBED_PROVIDERS["ollama"]["base_url"]) + "/api/tags"
            r = requests.get(url, timeout=5)
            if r.ok:
                return True, f"Ollama ready at {base_url or EMBED_PROVIDERS['ollama']['base_url']}"
            return False, f"Ollama returned status {r.status_code}"
        except Exception as e:
            return False, f"Cannot reach Ollama: {e}"
    return True, f"{provider} connection check not implemented"

def get_provider_info(provider: str) -> dict:
    return EMBED_PROVIDERS.get(provider, EMBED_PROVIDERS["other"])
    