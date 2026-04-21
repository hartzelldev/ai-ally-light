import os
from openai import OpenAI

class AIEngine:
    def __init__(self, config: dict):
        self.config = config
        self.provider = config.get("chat_provider", "OpenRouter").lower()
        # Default to a strong, cheap model if none is specified
        self.model = config.get("chat_model", "meta-llama/llama-3-8b-instruct")
        
        self.base_urls = {
            "openrouter": "https://openrouter.ai/api/v1",
            "together": "https://api.together.xyz/v1",
            "groq": "https://api.groq.com/openai/v1"
        }

    def get_response(self, messages: list):
        api_key = os.getenv(f"{self.provider.upper()}_API_KEY")
        base_url = self.base_urls.get(self.provider)

        if not api_key:
            return f"Error: {self.provider.upper()}_API_KEY not found in environment."

        client = OpenAI(api_key=api_key, base_url=base_url)
        
        try:
            # OpenRouter requirements:
            extra_headers = {
                "HTTP-Referer": "http://localhost:8000", 
                "X-Title": "AI-Ally-Light"
            }

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.get("temperature", 0.7),
                extra_headers=extra_headers
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"{self.provider.title()} Error: {str(e)}"