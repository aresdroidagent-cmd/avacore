from pathlib import Path

import pytest
import torch
from PIL import Image

from avacore.vision import describe


@pytest.fixture(autouse=True)
def reset_vision_client(monkeypatch):
    monkeypatch.setattr(describe, "_client", None)
    monkeypatch.setattr(describe, "_client_failed", False)
    monkeypatch.setattr(describe, "_client_error", "")
    monkeypatch.setattr(describe, "_release_cuda_cache", lambda: None)


def test_transient_initialization_failure_can_retry(monkeypatch):
    client = object()
    attempts = iter([RuntimeError("temporary initialization failure"), client])

    def construct(**_kwargs):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(describe, "SmolVLMClient", construct)
    with pytest.raises(RuntimeError, match="temporary initialization failure"):
        describe.get_vision_client()
    assert describe._client is None
    assert describe._client_failed is False
    assert describe.get_vision_client() is client


def test_cuda_oom_does_not_latch_initialization_failure(monkeypatch):
    attempts = {"count": 0}

    def construct(**_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise torch.cuda.OutOfMemoryError("large multiline CUDA allocator details")
        return object()

    monkeypatch.setattr(describe, "SmolVLMClient", construct)
    with pytest.raises(RuntimeError, match="insufficient GPU memory") as failure:
        describe.get_vision_client()
    assert "allocator details" not in str(failure.value)
    assert describe._client_failed is False
    assert describe.get_vision_client() is not None
    assert attempts["count"] == 2


def test_successful_client_is_cached(monkeypatch):
    calls = {"count": 0}
    client = object()

    def construct(**_kwargs):
        calls["count"] += 1
        return client

    monkeypatch.setattr(describe, "SmolVLMClient", construct)
    assert describe.get_vision_client() is client
    assert describe.get_vision_client() is client
    assert calls["count"] == 1


def test_generate_oom_is_short_and_next_request_reuses_client(monkeypatch, tmp_path):
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (64, 64), "white").save(image_path)

    class Client:
        calls = 0

        def describe_image(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise torch.cuda.OutOfMemoryError("verbose CUDA OOM internals")
            return "recovered"

    client = Client()
    monkeypatch.setattr(describe, "_client", client)
    monkeypatch.setattr(describe.settings, "vision_enabled", True)
    monkeypatch.setattr(describe.settings, "vision_min_image_pixels", 1)

    with pytest.raises(RuntimeError, match="insufficient GPU memory") as failure:
        describe.describe_image_with_smolvlm(image_path, mode="camera")
    assert "OOM internals" not in str(failure.value)
    assert describe.describe_image_with_smolvlm(image_path, mode="camera") == "recovered"
    assert describe._client is client


def test_german_camera_prompt_requests_german_without_identity():
    prompt = describe.camera_scene_prompt("de")
    assert "Antworte auf Deutsch" in prompt
    assert "Identifiziere die Person nicht" in prompt
