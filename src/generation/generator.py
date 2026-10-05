import requests


class OllamaGenerator:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(
            self.url,
            json=payload,
        )

        response.raise_for_status()

        data=response.json()
        return data["response"]
