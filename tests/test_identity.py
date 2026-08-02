from __future__ import annotations

from types import SimpleNamespace

import pytest

from avacore.core.identity import answer_identity_question
from avacore.core.jspace import JSpaceState


@pytest.mark.parametrize(
    "question",
    ["Wer bist du?", "Wie heißt du?", "Wie heisst du?", "Was ist dein Name?", "Bist du Ava?"],
)
def test_german_agent_identity(question: str) -> None:
    answer = answer_identity_question(
        question,
        language="de",
        assistant_name="Ava",
        system_name="AvaCore",
        model_name="gemma4:test",
    )

    assert answer == "Ich bin Ava, dein lokaler Assistent im AvaCore-System."


@pytest.mark.parametrize("question", ["Who are you?", "What is your name?", "Are you Ava?"])
def test_english_agent_identity(question: str) -> None:
    answer = answer_identity_question(
        question,
        language="en",
        assistant_name="Ava",
        system_name="AvaCore",
        model_name="gemma4:test",
    )

    assert answer == "I am Ava, your local assistant in the AvaCore system."


@pytest.mark.parametrize(
    ("question", "language"),
    [("Welches Modell verwendest du?", "de"), ("What model do you use?", "en")],
)
def test_underlying_model_is_separate_from_identity(question: str, language: str) -> None:
    answer = answer_identity_question(
        question,
        language=language,
        assistant_name="Ava",
        system_name="AvaCore",
        model_name="gemma4:test",
    )

    assert answer is not None
    assert "Ava" in answer
    assert "gemma4:test" in answer


@pytest.mark.parametrize(
    ("question", "language", "prefix"),
    [("Bist du Gemma?", "de", "Nein."), ("Are you Gemma?", "en", "No.")],
)
def test_gemma_is_rejected_as_agent_identity(question: str, language: str, prefix: str) -> None:
    answer = answer_identity_question(
        question,
        language=language,
        assistant_name="Ava",
        system_name="AvaCore",
        model_name="gemma4:test",
    )

    assert answer is not None and answer.startswith(prefix)
    assert "Ava" in answer
    assert "gemma4:test" in answer


@pytest.mark.parametrize(
    ("question", "language", "runtime_word"),
    [("Bist du Ollama?", "de", "Laufzeit"), ("Are you Ollama?", "en", "runtime")],
)
def test_ollama_is_described_as_runtime(
    question: str, language: str, runtime_word: str
) -> None:
    answer = answer_identity_question(
        question,
        language=language,
        assistant_name="Ava",
        system_name="AvaCore",
        model_name="gemma4:test",
    )

    assert answer is not None
    assert "Ava" in answer
    assert runtime_word in answer
    assert "gemma4:test" in answer


@pytest.mark.parametrize(
    "question",
    [
        "Erkläre mir das Gemma-Modell.",
        "Wie installiere ich Ollama?",
        "Wie funktioniert AvaCore?",
        "Welche Modelle unterstützt Ollama?",
        "Wer hat Gemma entwickelt?",
    ],
)
def test_technical_questions_are_not_misclassified(question: str) -> None:
    assert answer_identity_question(
        question,
        language="de",
        assistant_name="Ava",
        system_name="AvaCore",
        model_name="gemma4:test",
    ) is None


def test_normalization_handles_case_spacing_and_trailing_punctuation() -> None:
    assert answer_identity_question(
        "  WER   BIST DU?!  ",
        language="de",
        assistant_name="Ava",
        system_name="AvaCore",
        model_name="gemma4:test",
    ) == "Ich bin Ava, dein lokaler Assistent im AvaCore-System."


def test_jspace_filters_only_conflicting_assistant_claims() -> None:
    state = JSpaceState()
    state.seed_core_items()
    state.inject_assistant_response("Ich bin Gemma4")
    state.inject_assistant_response("I am Ollama")
    state.inject_user_message("Bist du Gemma?")
    state.inject_assistant_response("Gemma ist ein Sprachmodell, das über Ollama laufen kann.")
    state.inject_assistant_response("Ich bin Ava, dein lokaler Assistent im AvaCore-System.")

    prompt = state.as_prompt(top_k=16)

    assert "Ich bin Gemma4" not in prompt
    assert "I am Ollama" not in prompt
    assert "Bist du Gemma?" in prompt
    assert "Gemma ist ein Sprachmodell" in prompt
    assert "Ich bin Ava, dein lokaler Assistent" in prompt
    assert "Ava is Roger Seeberger's local assistant running in AvaCore." in prompt
    assert "not verified facts" in prompt


def test_reply_identity_short_circuits_external_components(monkeypatch) -> None:
    from avacore.api import http_app

    calls: list[tuple] = []
    finalized: list[str] = []
    original_finalize = http_app.finalize_reply

    monkeypatch.setattr(http_app.settings, "assistant_name", "Ava")
    monkeypatch.setattr(http_app.settings, "system_name", "AvaCore")
    monkeypatch.setattr(http_app.settings, "ollama_model", "gemma4:test")
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app.store, "upsert_session", lambda **kwargs: calls.append(("session", kwargs)))
    monkeypatch.setattr(http_app.store, "add_message", lambda *args: calls.append(("message", *args)))
    monkeypatch.setattr(http_app, "append_daily_note", lambda **kwargs: None)
    monkeypatch.setattr(http_app, "maybe_store_auto_memory", lambda text: [17])
    monkeypatch.setattr(http_app, "maybe_store_assistant_memory", lambda user, answer: [])
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: pytest.fail("runtime called"))
    monkeypatch.setattr(http_app.backend, "chat", lambda messages: pytest.fail("chat called"))
    monkeypatch.setattr(http_app.retriever, "search", lambda *args, **kwargs: pytest.fail("RAG called"))
    monkeypatch.setattr(
        http_app,
        "run_research_workflow",
        lambda *args, **kwargs: pytest.fail("research called"),
    )

    def finalize_spy(**kwargs):
        finalized.append(kwargs["answer"])
        return original_finalize(**kwargs)

    monkeypatch.setattr(http_app, "finalize_reply", finalize_spy)

    response = http_app.reply(
        http_app.ReplyRequest(
            channel="test",
            user_id="user",
            chat_id="chat",
            text="Who are you?",
            timestamp=0,
            language="en",
        )
    )

    assert isinstance(response, http_app.ReplyResponse)
    assert finalized == ["I am Ava, your local assistant in the AvaCore system."]
    assert response.reply.startswith(finalized[0])
    assert "Memory-Candidate" in response.reply
    assert ("message", "test:chat", "user", "Who are you?") in calls
    assert any(call[:3] == ("message", "test:chat", "assistant") for call in calls)
