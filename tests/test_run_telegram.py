from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram.error import NetworkError, RetryAfter, TimedOut
from telegram.ext import CommandHandler, MessageHandler


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_telegram.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("run_telegram", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
run_telegram = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(run_telegram)


def test_build_requests_uses_separate_ipv4_transports_with_bounded_retries(
    monkeypatch,
) -> None:
    transport_options: list[dict] = []
    request_options: list[dict] = []

    class FakeTransport:
        def __init__(self, **kwargs) -> None:
            transport_options.append(kwargs)

    class FakeRequest:
        def __init__(self, **kwargs) -> None:
            request_options.append(kwargs)

    monkeypatch.setattr(run_telegram.httpx, "AsyncHTTPTransport", FakeTransport)
    monkeypatch.setattr(run_telegram, "HTTPXRequest", FakeRequest)

    request, polling_request = run_telegram.build_telegram_requests()

    assert request is not polling_request
    assert transport_options == [
        {"local_address": "0.0.0.0", "retries": 2},
        {"local_address": "0.0.0.0", "retries": 2},
    ]
    assert request_options[0]["connect_timeout"] == 15.0
    assert request_options[0]["read_timeout"] == 30.0
    assert request_options[1]["read_timeout"] == 45.0
    assert request_options[0]["httpx_kwargs"]["transport"] is not request_options[1][
        "httpx_kwargs"
    ]["transport"]


@pytest.mark.parametrize(
    "error",
    [NetworkError("temporary DNS failure"), TimedOut(), RetryAfter(2)],
)
def test_temporary_telegram_errors_are_compact_warnings(
    error: Exception,
    caplog,
) -> None:
    context = SimpleNamespace(error=error)

    with caplog.at_level(logging.WARNING, logger=run_telegram.__name__):
        asyncio.run(run_telegram.telegram_error_handler(object(), context))

    record = caplog.records[-1]
    assert record.levelno == logging.WARNING
    assert record.getMessage().startswith("Temporary Telegram network error:")
    assert record.exc_info is None


def test_unexpected_error_keeps_original_traceback(caplog) -> None:
    try:
        raise ValueError("unexpected failure")
    except ValueError as error:
        context = SimpleNamespace(error=error)
        with caplog.at_level(logging.ERROR, logger=run_telegram.__name__):
            asyncio.run(run_telegram.telegram_error_handler(object(), context))

    record = caplog.records[-1]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert record.exc_info[0] is ValueError
    assert "unexpected failure" in caplog.text


def test_main_registers_error_handler_before_polling(monkeypatch, caplog) -> None:
    events: list[object] = []
    regular_request = object()
    polling_request = object()

    class FakeApplication:
        def add_error_handler(self, handler) -> None:
            events.append(("error_handler", handler))

        def run_polling(self) -> None:
            events.append("polling")

    def fake_build_application(**kwargs):
        events.append(("build", kwargs))
        return FakeApplication()

    monkeypatch.setattr(run_telegram.settings, "ollama_autostart", False)
    monkeypatch.setattr(
        run_telegram,
        "build_telegram_requests",
        lambda: (regular_request, polling_request),
    )
    monkeypatch.setattr(run_telegram, "build_application", fake_build_application)

    with caplog.at_level(logging.INFO, logger=run_telegram.__name__):
        run_telegram.main()

    assert events == [
        (
            "build",
            {"request": regular_request, "get_updates_request": polling_request},
        ),
        ("error_handler", run_telegram.telegram_error_handler),
        "polling",
    ]
    assert "Telegram transport configured for IPv4" in caplog.text
    assert "Starting Telegram polling" in caplog.text


def test_application_keeps_existing_handlers_and_uses_both_requests(monkeypatch) -> None:
    from avacore.channels.telegram import bot

    monkeypatch.setattr(bot.settings, "telegram_bot_token", "123456:test-token")
    request, polling_request = run_telegram.build_telegram_requests()

    application = bot.build_application(
        request=request,
        get_updates_request=polling_request,
    )

    handlers = [handler for group in application.handlers.values() for handler in group]
    commands = {
        command
        for handler in handlers
        if isinstance(handler, CommandHandler)
        for command in handler.commands
    }

    assert application.bot.request is request
    assert {"start", "help", "de", "en", "research", "camera", "notes"} <= commands
    assert sum(isinstance(handler, MessageHandler) for handler in handlers) == 2
