import pytest

from avacore.core.decision import decide_context


@pytest.mark.parametrize(
    ("text", "expected_flag"),
    [
        ("Wie spät ist es?", "needs_memory"),
        ("Wer bist du?", "needs_memory"),
        ("Was steht heute im Kalender?", "needs_calendar"),
        ("Mach bitte einen Kamera Snapshot", "needs_camera"),
        ("Recherchiere die neueste Python Version", "needs_research"),
        ("Was steht im AvaCore README?", "needs_rag"),
    ],
)
def test_routes_recognized_intents(text: str, expected_flag: str) -> None:
    decision = decide_context(text)

    assert getattr(decision, expected_flag) is True


def test_runtime_question_does_not_trigger_external_context() -> None:
    decision = decide_context("Welches Datum ist heute?")

    assert decision.needs_memory is True
    assert decision.needs_research is False
    assert decision.needs_rag is False
    assert decision.confidence == pytest.approx(0.95)


def test_default_uses_memory_only() -> None:
    decision = decide_context("Erzähl mir etwas Interessantes.")

    assert decision.needs_memory is True
    assert decision.needs_rag is False
    assert decision.needs_research is False
    assert decision.needs_calendar is False
    assert decision.needs_camera is False


def test_decision_can_be_serialized() -> None:
    result = decide_context("news").to_dict()

    assert result["needs_research"] is True
    assert isinstance(result["reason"], str)
    assert 0.0 <= result["confidence"] <= 1.0
