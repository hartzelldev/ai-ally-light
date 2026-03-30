"""
Chat providers for AI Ally Light.

Supports: OpenRouter, Groq, TogetherAI, Ollama, and Custom endpoints.
"""

import requests

CHAT_PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://api.openrouter.ai/v1",
        "requires_api_key": True,
        "model_url": "https://openrouter.ai/models",
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "requires_api_key": True,
        "model_url": "https://console.groq.com/docs/models",
    },
    "togetherai": {
        "name": "TogetherAI",
        "base_url": "https://api.together.ai/v1",
        "requires_api_key": True,
        "model_url": "https://docs.together.ai/docs/serverless-models",
    },
    "ollama": {
        "name": "Ollama (Local)",
        "base_url": "http://127.0.0.1:11434",
        "requires_api_key": False,
        "model_url": "https://ollama.com/library",
    },
    "other": {
        "name": "Other (Custom)",
        "base_url": "",
        "requires_api_key": True,
        "model_url": "",
    },
}


def chat_openrouter(messages: list, model: str, api_key: str, base_url: str = None) -> str:
    """Chat using OpenRouter API."""
    url = (base_url or CHAT_PROVIDERS["openrouter"]["base_url"]) + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def chat_groq(messages: list, model: str, api_key: str, base_url: str = None) -> str:
    """Chat using Groq API."""
    url = (base_url or CHAT_PROVIDERS["groq"]["base_url"]) + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def chat_togetherai(messages: list, model: str, api_key: str, base_url: str = None) -> str:
    """Chat using TogetherAI API."""
    url = (base_url or CHAT_PROVIDERS["togetherai"]["base_url"]) + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def chat_ollama(messages: list, model: str, base_url: str = None) -> str:
    """Chat using Ollama API."""
    url = (base_url or CHAT_PROVIDERS["ollama"]["base_url"]) + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]


def chat_custom(messages: list, model: str, api_key: str, base_url: str) -> str:
    """Chat using a custom API endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    r = requests.post(base_url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def chat_with_provider(provider: str, messages: list, model: str, api_key: str = None, base_url: str = None) -> str:
    """
    Route chat request to the appropriate provider.
    
    Args:
        provider: Provider name (openrouter, groq, togetherai, ollama, other)
        messages: List of message dicts with 'role' and 'content'
        model: Model name/identifier
        api_key: API key (required for providers that need it)
        base_url: Custom base URL (uses provider default if empty)
    
    Returns:
        Assistant's response text
    """
    if provider == "openrouter":
        return chat_openrouter(messages, model, api_key, base_url)
    elif provider == "groq":
        return chat_groq(messages, model, api_key, base_url)
    elif provider == "togetherai":
        return chat_togetherai(messages, model, api_key, base_url)
    elif provider == "ollama":
        return chat_ollama(messages, model, base_url)
    elif provider == "other":
        return chat_custom(messages, model, api_key, base_url)
    else:
        raise ValueError(f"Unknown chat provider: {provider}")


def get_provider_info(provider: str) -> dict:
    """Get information about a provider."""
    return CHAT_PROVIDERS.get(provider, CHAT_PROVIDERS["other"])
