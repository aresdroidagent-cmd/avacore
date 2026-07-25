import asyncio

import pytest
import requests

from avacore.channels.telegram import http_client


class FakeResponse:
    ok = True


def test_post_forwards_request_without_changing_protocol(monkeypatch) -> None:
    calls: list[tuple[str, str, dict]] = []
    response = FakeResponse()

    async def fake_to_thread(function, method: str, url: str, **kwargs):
        assert function is http_client.requests.request
        calls.append((method, url, kwargs))
        return response

    monkeypatch.setattr(http_client.asyncio, "to_thread", fake_to_thread)

    result = asyncio.run(
        http_client.post(
            "http://avacore/reply",
            json={"text": "Hallo"},
            timeout=30,
        )
    )

    assert result is response
    assert calls == [
        (
            "POST",
            "http://avacore/reply",
            {"json": {"text": "Hallo"}, "timeout": 30},
        )
    ]


def test_transport_errors_are_preserved(monkeypatch) -> None:
    async def fail(*args, **kwargs):
        raise requests.Timeout("AvaCore timed out")

    monkeypatch.setattr(http_client.asyncio, "to_thread", fail)

    with pytest.raises(requests.Timeout, match="timed out"):
        asyncio.run(http_client.get("http://avacore/health", timeout=1))
