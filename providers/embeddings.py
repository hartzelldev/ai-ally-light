"""
Embedding providers for AI Ally Light.

Supports: Ollama, OpenRouter, HuggingFace, and Custom endpoints.
"""

import requests

EMBED_PROVIDERS = {
    "ollama": {
        "name": "Ollama (Local)",
        "base_url": "http://127.0.0.1:11434",
        "requires_api_key": False,
        "model_url": "https://ollama.com/library",
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://api.openrouter.ai/v1",
        "requires_api_key": True,
        "model_url": "https://openrouter.ai/models",
    },
    "huggingface": {
        "name": "HuggingFace",
        "base_url": "https://api-inference.huggingface.co/pipeline/feature-extraction",
        "requires_api_key": True,
        "model_url": "https://huggingface.co/models",
    },
    "other": {
        "name": "Other (Custom)",
        "base_url": "",
        "requires_api_key": True,
        "model_url": "",
    },
}


def embed_ollama(text: str, model: str, base_url: str = None) -> list:
    """Get embedding using Ollama API."""
    url = (base_url or EMBED_PROVIDERS["ollama"]["base_url"]) + "/api/embeddings"
    payload = {
        "model": model,
        "prompt": text,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]


def embed_openrouter(text: str, model: str, api_key: str, base_url: str = None) -> list:
    """Get embedding using OpenRouter API."""
    url = (base_url or EMBED_PROVIDERS["openrouter"]["base_url"]) + "/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": text,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def embed_huggingface(text: str, model: str, api_key: str, base_url: str = None) -> list:
    """Get embedding using HuggingFace Inference API."""
    url = (base_url or EMBED_PROVIDERS["huggingface"]["base_url"]) + f"/{model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "inputs": text,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def embed_custom(text: str, model: str, api_key: str, base_url: str) -> list:
    """Get embedding using a custom API endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": text,
    }
    r = requests.post(base_url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def get_embedding(provider: str, text: str, model: str, api_key: str = None, base_url: str = None) -> list:
    """
    Route embedding request to the appropriate provider.
    
    Args:
        provider: Provider name (ollama, openrouter, huggingface, other)
        text: Text to embed
        model: Model name/identifier
        api_key: API key (required for providers that need it)
        base_url: Custom base URL (uses provider default if empty)
    
    Returns:
        List of embedding values
    """
    if provider == "ollama":
        return embed_ollama(text, model, base_url)
    elif provider == "openrouter":
        return embed_openrouter(text, model, api_key, base_url)
    elif provider == "huggingface":
        return embed_huggingface(text, model, api_key, base_url)
    elif provider == "other":
        return embed_custom(text, model, api_key, base_url)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def check_provider_connection(provider: str, base_url: str = None) -> tuple:
    """
    Check if a provider is reachable and has the required model.
    
    Returns:
        (success: bool, message: str)
    """
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
    """Get information about a provider."""
    return EMBED_PROVIDERS.get(provider, EMBED_PROVIDERS["other"])
