import os
from openai import OpenAI

class AIEngine:
    def __init__(self, config: dict):
        self.config = config
        self.provider = config.get("chat_provider", "openrouter").lower()
        self.model = config.get("chat_model", "meta-llama/llama-3-8b-instruct")
        
        self.base_urls = {
            "openrouter": "https://openrouter.ai/api/v1",
            "together": "https://api.together.xyz/v1",
            "groq": "https://api.groq.com/openai/v1"
        }

    def get_response(self, messages: list):
        api_key = os.getenv(f"{self.provider.upper()}_API_KEY")
        
        # Special case for Gemini
        if self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model)
            # Convert OpenAI format to Gemini format or just pass the last message/prompt
            prompt = messages[-1]['content'] 
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Gemini Error: {str(e)}"

        # OpenAI-compatible providers (Groq, OpenRouter, Together)
        base_url = self.base_urls.get(self.provider)
        if not api_key:
            return f"Error: {self.provider.upper()}_API_KEY not found in environment."

        client = OpenAI(api_key=api_key, base_url=base_url)
        
        try:
            extra_headers = {
                "HTTP-Referer": "http://localhost:8000", 
                "X-Title": "AI-Ally-Light"
            }

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.get("temperature", 0.7),
                extra_headers=extra_headers if self.provider == "openrouter" else None
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"{self.provider.title()} Error: {str(e)}"
            