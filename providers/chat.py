import requests

CHAT_PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "requires_api_key": True,
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "requires_api_key": True,
    },
    "togetherai": {
        "name": "TogetherAI",
        "base_url": "https://api.together.ai/v1",
        "requires_api_key": True,
    },
    "ollama": {
        "name": "Ollama (Local)",
        "base_url": "http://127.0.0.1:11434/v1",
        "requires_api_key": False,
    },
    "other": {
        "name": "Other (Custom)",
        "base_url": "",
        "requires_api_key": True,
    }
}

def get_headers(api_key, provider):
    """Unified header management to prevent 530/403 errors."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI-Ally-Light/1.0"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    if provider == "openrouter":
        headers["HTTP-Referer"] = "http://localhost:5000"
        headers["X-Title"] = "AI Ally Light"
    
    return headers

def chat_openrouter(messages, model, api_key, base_url=None):
    base_url = base_url or CHAT_PROVIDERS["openrouter"]["base_url"]
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = get_headers(api_key, "openrouter")
    
    payload = {"model": model, "messages": messages}
    r = requests.post(api_url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def chat_groq(messages, model, api_key, base_url=None):
    base_url = base_url or CHAT_PROVIDERS["groq"]["base_url"]
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = get_headers(api_key, "groq")
    
    payload = {"model": model, "messages": messages}
    r = requests.post(api_url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def chat_togetherai(messages, model, api_key, base_url=None):
    base_url = base_url or CHAT_PROVIDERS["togetherai"]["base_url"]
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = get_headers(api_key, "togetherai")
    
    payload = {"model": model, "messages": messages}
    r = requests.post(api_url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def chat_ollama(messages, model, base_url=None):
    base_url = base_url or CHAT_PROVIDERS["ollama"]["base_url"]
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = get_headers(None, "ollama")
    
    payload = {"model": model, "messages": messages, "stream": False}
    r = requests.post(api_url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def chat_custom(messages, model, api_key, base_url):
    if not base_url:
        raise ValueError("Custom provider requires a Base URL.")
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = get_headers(api_key, "custom")
    
    payload = {"model": model, "messages": messages}
    r = requests.post(api_url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def chat_with_provider(provider, messages, model, api_key=None, base_url=None):
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
        raise ValueError(f"Unknown provider: {provider}")