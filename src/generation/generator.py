import requests


class OllamaGenerator:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/generate"
        #self.url = "http://192.168.86.106:11434/api/generate"

    def generate(self, prompt: str) -> str:

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": 250,
                "temperature": 0.1,
            },
        }

        response = requests.post(
            self.url,
            json=payload,
        )

        response.raise_for_status()

        data=response.json()

        print("Load:", data.get("load_duration",0)/1_000_000_000," sec")
        print("Prompt tokens:", data.get("prompt_eval_count"))
        print("Prompt eval:", data.get("prompt_eval_duration",0)/1_000_000_000," sec")
        print("Generated tokens:", data.get("eval_count"))
        print("Generation eval:", data.get("eval_duration",0)/1_000_000_000," sec")
        return data["response"]
