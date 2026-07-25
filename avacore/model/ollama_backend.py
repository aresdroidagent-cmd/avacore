import requests


class OllamaBackend:
    def __init__(
        self,
        ollama_url: str,
        model: str,
        timeout_ms: int,
        session: requests.Session | None = None,
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")

        self.ollama_url = ollama_url.strip()
        self.model = model.strip()
        if not self.ollama_url:
            raise ValueError("ollama_url must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")

        self.timeout_s = timeout_ms / 1000.0
        self.session = session or requests.Session()

    def chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
        }

        response = self.session.post(
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
