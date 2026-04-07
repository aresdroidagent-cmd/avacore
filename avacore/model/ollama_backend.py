import requests


class OllamaBackend:
    def __init__(self, ollama_url: str, model: str, timeout_ms: int) -> None:
        self.ollama_url = ollama_url
        self.model = model
        self.timeout_s = timeout_ms / 1000.0

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
        }

        response = requests.post(
            self.ollama_url,
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()

        data = response.json()
        message = data.get("message") or {}
        content = (message.get("content") or "").strip()

        if not content:
            raise RuntimeError("Ollama returned empty content")

        return content
