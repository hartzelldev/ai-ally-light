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

def get_sampling_params(cfg, provider):
    temperature = float(cfg.get("temperature", 0.8))
    min_p = float(cfg.get("min_p", 0.05))
    top_p = float(cfg.get("top_p", 1.0))
    max_tokens = int(cfg.get("max_tokens", 4096))

    if provider in ("groq", "ollama"):
        return {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
    else:
        return {
            "temperature": temperature,
            "min_p": min_p,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }

def chat_openrouter(messages, model, api_key, base_url=None, sampling_params=None):
    base_url = base_url or CHAT_PROVIDERS["openrouter"]["base_url"]
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = get_headers(api_key, "openrouter")

    payload = {"model": model, "messages": messages}
    if sampling_params:
        payload.update(sampling_params)
    r = requests.post(api_url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def chat_groq(messages, model, api_key, base_url=None, sampling_params=None):
    base_url = base_url or CHAT_PROVIDERS["groq"]["base_url"]
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = get_headers(api_key, "groq")

    payload = {"model": model, "messages": messages}
    if sampling_params:
        payload.update(sampling_params)
    r = requests.post(api_url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def chat_togetherai(messages, model, api_key, base_url=None, sampling_params=None):
    base_url = base_url or CHAT_PROVIDERS["togetherai"]["base_url"]
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = get_headers(api_key, "togetherai")

    payload = {"model": model, "messages": messages}
    if sampling_params:
        payload.update(sampling_params)
    r = requests.post(api_url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def chat_ollama(messages, model, base_url=None, sampling_params=None):
    base_url = base_url or CHAT_PROVIDERS["ollama"]["base_url"]
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = get_headers(None, "ollama")

    payload = {"model": model, "messages": messages, "stream": False}
    if sampling_params:
        payload.update(sampling_params)
    r = requests.post(api_url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def chat_custom(messages, model, api_key, base_url, sampling_params=None):
    if not base_url:
        raise ValueError("Custom provider requires a Base URL.")
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = get_headers(api_key, "custom")

    payload = {"model": model, "messages": messages}
    if sampling_params:
        payload.update(sampling_params)
    r = requests.post(api_url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def chat_with_provider(provider, messages, model, api_key=None, base_url=None, sampling_params=None):
    if provider == "openrouter":
        return chat_openrouter(messages, model, api_key, base_url, sampling_params)
    elif provider == "groq":
        return chat_groq(messages, model, api_key, base_url, sampling_params)
    elif provider == "togetherai":
        return chat_togetherai(messages, model, api_key, base_url, sampling_params)
    elif provider == "ollama":
        return chat_ollama(messages, model, base_url, sampling_params)
    elif provider == "other":
        return chat_custom(messages, model, api_key, base_url, sampling_params)
    else:
        raise ValueError(f"Unknown provider: {provider}")
