import pytest

from avacore.model.ollama_backend import OllamaBackend


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.checked = False

    def raise_for_status(self) -> None:
        self.checked = True

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


def test_chat_sends_expected_ollama_payload() -> None:
    response = FakeResponse({"message": {"content": "  Hallo Roger  "}})
    session = FakeSession(response)
    backend = OllamaBackend(
        "http://localhost:11434/api/chat",
        "test-model",
        2500,
        session=session,
    )

    result = backend.chat([{"role": "user", "content": "Hallo"}])

    assert result == "Hallo Roger"
    assert response.checked is True
    assert session.calls == [
        {
            "url": "http://localhost:11434/api/chat",
            "json": {
                "model": "test-model",
                "stream": False,
                "messages": [{"role": "user", "content": "Hallo"}],
            },
            "timeout": 2.5,
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [{}, {"message": {}}, {"message": {"content": "   "}}],
)
def test_chat_rejects_empty_responses(payload: dict) -> None:
    backend = OllamaBackend(
        "http://ollama/api/chat", "model", 1000, FakeSession(FakeResponse(payload))
    )

    with pytest.raises(RuntimeError, match="empty content"):
        backend.chat([])


@pytest.mark.parametrize(
    ("url", "model", "timeout", "message"),
    [
        ("", "model", 1000, "ollama_url"),
        ("http://ollama", "", 1000, "model"),
        ("http://ollama", "model", 0, "timeout_ms"),
    ],
)
def test_constructor_validates_configuration(
    url: str,
    model: str,
    timeout: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OllamaBackend(url, model, timeout)
