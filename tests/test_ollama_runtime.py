from avacore.system.ollama_runtime import unload_ollama_model


class Response:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {}
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return self.payload


class Session:
    def __init__(self, models, *, post_error=None):
        self.models = models
        self.post_error = post_error
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return Response({"models": self.models})

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return Response(error=self.post_error)


def test_unload_resident_model_uses_keep_alive_zero():
    session = Session([{"name":"gemma4:e2b"}])
    assert unload_ollama_model(
        "gemma4:e2b", "http://localhost:11434/api/chat", session=session
    ) is True
    method, url, kwargs = session.calls[1]
    assert method == "post" and url == "http://localhost:11434/api/generate"
    assert kwargs["json"] == {
        "model":"gemma4:e2b", "prompt":"", "stream":False, "keep_alive":0,
    }


def test_unload_already_absent_model_is_noop():
    session = Session([])
    assert unload_ollama_model(
        "gemma4:e2b", "http://localhost:11434/api/chat", session=session
    ) is False
    assert [call[0] for call in session.calls] == ["get"]


def test_unload_failure_is_nonfatal():
    session = Session([{"model":"gemma4:e2b"}], post_error=RuntimeError("offline"))
    assert unload_ollama_model(
        "gemma4:e2b", "http://localhost:11434/api/chat", session=session
    ) is False
