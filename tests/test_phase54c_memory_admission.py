from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from avacore.core.memory_admission import MemoryAdmissionPolicy, MemoryClass
from avacore.memory.sqlite_store import SQLiteStore
from avacore.tools.web_research import ResearchSource


NOW = datetime(2026, 9, 26, 10, tzinfo=timezone.utc)


def policy_research(**overrides):
    values = dict(research_ok=True, temporal_scope="TIMELESS", response_mode="direct_answer",
                  plan_compliance=True, fallback_used=False, required_fact_coverage=1.0,
                  recommendation_confidence="HIGH", evidence_conflict=False,
                  temporal_claim_conflict=False, search_failed=False,
                  facts=[{"fact_id": "f1", "confidence": .9, "volatility": "LOW",
                          "temporal_basis": "TIMELESS_FACT"}], source_ids=["research_1"])
    values.update(overrides)
    return MemoryAdmissionPolicy().research(**values)


def test_hard_research_guards_and_volatility_classes():
    assert policy_research(search_failed=True).reason == "search_failure"
    assert policy_research(facts=[]).reason == "insufficient_evidence"
    assert policy_research(response_mode="insufficient_current_evidence").reason == "insufficient_evidence"
    assert policy_research(evidence_conflict=True).reason == "unresolved_evidence_conflict"
    assert policy_research(temporal_claim_conflict=True).reason == "temporal_claim_conflict"
    assert policy_research(recommendation_confidence="LOW").reason == "incomplete_recommendation"
    weather = policy_research(temporal_scope="CURRENT", facts=[
        {"fact_id": "weather", "confidence": .9, "volatility": "HIGH",
         "temporal_basis": "CURRENT_PAGE_MARKER"}])
    assert not weather.admit and weather.memory_class == MemoryClass.EPHEMERAL
    timeless = policy_research()
    assert timeless.admit and timeless.memory_class == MemoryClass.CANDIDATE


def configure_research(monkeypatch, tmp_path, sources, answer):
    from avacore.api import http_app

    memory_store = SQLiteStore(tmp_path / "memory.db")
    monkeypatch.setattr(http_app, "store", memory_store)
    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)

    def collect(*, query, diagnostics, requirement_ids=(), **kwargs):
        diagnostics.update(status="SEARCH_OK_WITH_RESULTS", result_count_raw=len(sources),
                           fetch_attempted=len(sources), fetch_succeeded=len(sources), fetch_failed=0)
        for source in sources:
            source.originating_queries = (query,)
            source.originating_providers = ("mock",)
            source.originating_requirement_ids = requirement_ids
        return sources

    monkeypatch.setattr(http_app, "collect_research_sources", collect)
    chat = MagicMock(return_value=answer)
    monkeypatch.setattr(http_app.backend, "chat", chat)
    return http_app, memory_store, chat


def test_successful_current_weather_is_ephemeral_and_not_saved(monkeypatch, tmp_path):
    source = ResearchSource("Wetter Schaffhausen", "https://weather.example/current", "",
                            "Wetter heute in Schaffhausen am 26.09.2026: 17 °C.",
                            time_markers=("2026-09-26",), retrieved_at=NOW.isoformat())
    http_app, store, chat = configure_research(
        monkeypatch, tmp_path, [source], "Heute am 26.09.2026 sind es in Schaffhausen 17 °C.")
    result = http_app.run_research_workflow("Wetter heute in Schaffhausen", save_memory=True)
    assert result["ok"] and result["memory_id"] is None
    assert store.list_memory_items() == []
    decision = http_app.memory_admission_observability.last_decision
    assert decision["memory_class"] == "EPHEMERAL"
    # Stale weather evidence retains its intrinsic class and its specific guard reason.
    expected_reason = ("insufficient_evidence"
                       if http_app._last_research_grounding_debug["response_mode"] == "insufficient_current_evidence"
                       else "high_volatility")
    assert decision["reason"] == expected_reason
    assert chat.call_count == 1


def test_search_failure_never_creates_memory_or_calls_model(monkeypatch, tmp_path):
    from avacore.api import http_app
    store = SQLiteStore(tmp_path / "memory.db")
    monkeypatch.setattr(http_app, "store", store)
    monkeypatch.setattr(http_app.settings, "research_enabled", True)

    def failed(*, diagnostics, **kwargs):
        diagnostics.update(status="SEARCH_PROVIDER_ERROR", result_count_raw=0,
                           fetch_attempted=0, fetch_succeeded=0, fetch_failed=0)
        return []

    monkeypatch.setattr(http_app, "collect_research_sources", failed)
    chat = MagicMock()
    monkeypatch.setattr(http_app.backend, "chat", chat)
    result = http_app.run_research_workflow("Was ist OPC UA?", save_memory=True)
    assert not result["ok"] and result["memory_id"] is None
    assert http_app.memory_admission_observability.last_decision["reason"] == "search_failure"
    assert store.list_memory_items() == []
    chat.assert_not_called()


def test_opc_ua_candidate_uses_validated_fact_and_deduplicates(monkeypatch, tmp_path):
    source = ResearchSource("OPC UA", "https://opcfoundation.org/about/opc-technologies/opc-ua", "",
        "OPC UA is a platform-independent service-oriented architecture for industrial communication.")
    answer = "OPC UA is a platform-independent service-oriented architecture for industrial communication."
    http_app, store, chat = configure_research(monkeypatch, tmp_path, [source], answer)
    first = http_app.run_research_workflow("Was ist OPC UA?", save_memory=True)
    second = http_app.run_research_workflow("Was ist OPC UA?", save_memory=True)
    assert first["memory_id"] and second["memory_id"] is None
    items = store.list_memory_items()
    assert len(items) == 1 and items[0]["status"] == "candidate"
    assert "platform-independent" in items[0]["content"]
    assert "Inhaltsverzeichnis" not in items[0]["content"]
    assert "Zusammenfassung" not in items[0]["content"]
    assert '"created_from_response_plan": true' in items[0]["metadata_json"]
    assert http_app.memory_admission_observability.last_decision["duplicate"] is True
    assert chat.call_count == 2


def test_incomplete_comparison_is_not_saved(monkeypatch, tmp_path):
    sources = [
        ResearchSource("Ubuntu 24.04", "https://ubuntu.com/24", "",
                       "Ubuntu 24.04 released 2024-04 and supported until 2029-04."),
        ResearchSource("Ubuntu 26.04", "https://ubuntu.com/26", "",
                       "Ubuntu 26.04 released 2026-04 and supported until 2031-04."),
    ]
    http_app, store, chat = configure_research(monkeypatch, tmp_path, sources, "Allgemeiner Vergleich.")
    result = http_app.run_research_workflow(
        "Was würdest du im Moment für lokale LLMs empfehlen: Ubuntu 24.04 oder Ubuntu 26.04?",
        save_memory=True)
    assert result["memory_id"] is None and store.list_memory_items() == []
    assert http_app.memory_admission_observability.last_decision["reason"] == "incomplete_recommendation"
    assert chat.call_count == 1


def test_explicit_remember_and_decision_remain_candidates_without_auto_verification(monkeypatch, tmp_path):
    from avacore.api import http_app
    store = SQLiteStore(tmp_path / "memory.db")
    monkeypatch.setattr(http_app, "store", store)
    explicit = http_app.maybe_store_auto_memory("Merk dir bitte, dass mein Testsystem Atlas heißt.")
    decision = http_app.maybe_store_auto_memory("Wir verwenden SearXNG als Primary Search Provider.")
    duplicate = http_app.maybe_store_auto_memory("Wir verwenden SearXNG als Primary Search Provider.")
    assert len(explicit) == 1 and len(decision) == 1 and duplicate == []
    items = store.list_memory_items()
    assert len(items) == 2 and all(item["status"] == "candidate" for item in items)
    assert any('"user_requested": true' in item["metadata_json"] for item in items)


@pytest.mark.anyio
async def test_telegram_memory_marker_only_when_candidate_was_saved(monkeypatch):
    from avacore.channels.telegram import bot

    replies = []

    async def reply(text):
        replies.append(text)

    update = SimpleNamespace(effective_chat=SimpleNamespace(id=42, type="private"),
                             effective_message=SimpleNamespace(reply_text=reply))
    monkeypatch.setattr(bot, "is_allowed_chat", lambda _: True)

    class Response:
        ok = True
        text = ""
        def __init__(self, memory_id): self.memory_id = memory_id
        def json(self):
            return {"answer": "Antwort", "memory_id": self.memory_id, "sources": []}

    responses = iter([Response(None), Response(17)])
    async def post(*args, **kwargs):
        return next(responses)
    monkeypatch.setattr(bot.http_client, "post", post)
    context = SimpleNamespace(args=["Was", "ist", "OPC", "UA?"])
    await bot.research_cmd(update, context)
    await bot.research_cmd(update, context)
    assert "Memory-Kandidat" not in replies[1]
    assert "Als Memory-Kandidat gespeichert: #17" in replies[3]


@pytest.mark.parametrize("guards, reason", [
    ({}, "high_volatility"),
    ({"response_mode":"insufficient_current_evidence"}, "insufficient_evidence"),
    ({"search_failed":True}, "search_failure"),
    ({"temporal_claim_conflict":True}, "temporal_claim_conflict"),
    ({"evidence_conflict":True}, "unresolved_evidence_conflict"),
    ({"recommendation_confidence":"LOW"}, "incomplete_recommendation"),
])
def test_high_volatility_preserves_ephemeral_class_under_suppression(guards, reason):
    decision = policy_research(temporal_scope="CURRENT", facts=[
        {"fact_id":"weather", "confidence":.9, "volatility":"HIGH",
         "temporal_basis":"CURRENT_PAGE_MARKER"}], **guards)
    assert decision.admit is False
    assert decision.memory_class == MemoryClass.EPHEMERAL
    assert decision.reason == reason


def test_medium_volatility_insufficient_evidence_is_unchanged():
    decision = policy_research(response_mode="insufficient_current_evidence", facts=[
        {"fact_id":"medium", "confidence":.9, "volatility":"MEDIUM"}])
    assert decision.admit is False
    assert decision.reason == "insufficient_evidence"
    assert decision.memory_class == MemoryClass.SESSION_RELEVANT


def test_low_volatility_timeless_fact_remains_candidate():
    decision = policy_research()
    assert decision.admit is True
    assert decision.memory_class == MemoryClass.CANDIDATE
    assert decision.reason == "durable_validated_research"


@pytest.mark.parametrize("query", ["Wetter heute", "weather", "current temperature",
                                  "live score", "stock price", "breaking news"])
@pytest.mark.parametrize("search_failed, reason", [(False, "insufficient_evidence"), (True, "search_failure")])
def test_volatile_query_without_facts_remains_ephemeral(query, search_failed, reason):
    decision = policy_research(query=query, facts=[], search_failed=search_failed)
    assert decision.admit is False
    assert decision.memory_class == MemoryClass.EPHEMERAL
    assert decision.volatility == "HIGH"
    assert decision.reason == reason
