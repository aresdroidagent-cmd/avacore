import pytest

from avacore.core.prompts import looks_like_code_request
from avacore.memory.auto_memory import AutoMemoryExtractor


@pytest.mark.parametrize(
    "text",
    [
        "Schreibe Code für einen Sensor",
        "Implementiere eine Funktion in Python",
        "Write code for a parser",
        "Generate code for this API",
    ],
)
def test_code_requests_are_detected(text: str) -> None:
    assert looks_like_code_request(text) is True


def test_ordinary_question_is_not_a_code_request() -> None:
    assert looks_like_code_request("Was ist Python?") is False


@pytest.mark.parametrize(
    ("text", "title"),
    [
        ("Ich nutze Ubuntu 20.04", "Nutzt"),
        ("Ich bevorzuge kurze Antworten", "Präferenz"),
        ("Wir nehmen SQLite für den Speicher", "Entscheidung"),
        ("Mein Standort ist Zürich", "Persönliche Angabe"),
    ],
)
def test_auto_memory_extracts_supported_statements(text: str, title: str) -> None:
    candidates = AutoMemoryExtractor().extract(text)

    assert len(candidates) == 1
    assert candidates[0].title == title
    assert candidates[0].content == text


def test_auto_memory_ignores_general_conversation() -> None:
    assert AutoMemoryExtractor().extract("Wie funktioniert SQLite?") == []
