from datetime import datetime, timezone
from unittest.mock import MagicMock

from avacore.core.research_grounding import (
    TemporalContext, TemporalScope, TemporalStatus, ResearchMode,
    classify_research_question, normalize_evidence, select_evidence,
    build_research_plan, research_debug,
)
from avacore.tools.web_research import ResearchSource


NOW = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)


def context():
    return TemporalContext.create("Europe/Zurich", now_utc=NOW)


def source(title, text, *, url="https://example.org/source", snippet="", updated=None, retrieved=None):
    return ResearchSource(title, url, snippet, text, retrieved_at=retrieved,
                          updated_at=updated)


def test_temporal_specs_and_runtime_date():
    temporal = context()
    assert temporal.local_date.isoformat() == "2026-09-20"
    weather = classify_research_question("Wetterbericht für heute in Schaffhausen, Schweiz", temporal)
    assert weather.research_mode == ResearchMode.CURRENT_STATE
    assert weather.temporal_scope == TemporalScope.CURRENT
    assert weather.target_date == temporal.local_date
    assert weather.requires_fresh_evidence
    comparison = classify_research_question(
        "Was würdest du im Moment für AI-Applikationen und lokale LLMs empfehlen: Ubuntu 24.04 oder Ubuntu 26.04?", temporal)
    assert comparison.research_mode == ResearchMode.CURRENT_COMPARISON
    assert comparison.comparison_targets == ("Ubuntu 24.04", "Ubuntu 26.04")
    assert classify_research_question("Was ist OPC UA?", temporal).temporal_scope == TemporalScope.TIMELESS


def test_retrieval_time_is_not_evidence_date_and_stale_weather_is_rejected():
    temporal = context()
    spec = classify_research_question("Wetter heute in Schaffhausen", temporal)
    old = source("Wetter Schaffhausen", "Vorhersage für 20.09.2024: 19 °C und Regen.",
                 retrieved="2026-09-20T10:00:00+00:00")
    current = source("Wetter Schaffhausen 20.09.2026", "Schaffhausen heute 20.09.2026: 17 °C.",
                     url="https://weather.example/current")
    evidence = [normalize_evidence(old, 1, spec, temporal), normalize_evidence(current, 2, spec, temporal)]
    selected = select_evidence(spec, evidence)
    assert evidence[0].evidence_date.isoformat() == "2024-09-20"
    assert evidence[0].temporal_status == TemporalStatus.STALE
    assert evidence[0].rejection_reason == "source_explicit_temporal_conflict"
    assert [e.evidence_id for e in selected] == ["research_2"]
    plan = build_research_plan(spec, temporal, selected)
    assert plan.response_mode == "current_state_answer"
    assert all("research_1" not in fact.source_evidence_ids for fact in plan.required_facts)


def test_only_stale_or_unknown_current_evidence_gives_insufficient_plan():
    temporal = context()
    spec = classify_research_question("Wetter heute in Schaffhausen", temporal)
    stale = normalize_evidence(source("Wetter Schaffhausen", "Vorhersage für 20.09.2024: Regen"), 1, spec, temporal)
    unknown = normalize_evidence(source("Wetter Schaffhausen", "Temperatur und Regen in Schaffhausen"), 2, spec, temporal)
    selected = select_evidence(spec, [stale, unknown])
    assert unknown.temporal_status == TemporalStatus.TEMPORAL_UNKNOWN
    assert unknown.rejection_reason == "source_temporal_unknown_but_no_valid_fact"
    plan = build_research_plan(spec, temporal, selected)
    assert plan.response_mode == "insufficient_current_evidence"
    assert "keine ausreichend aktuelle" in plan.render().casefold()
    assert not plan.compliance("Heute sind es 22 °C und sonnig.")["compliant"]
    yesterday = normalize_evidence(source("Wetter Schaffhausen", "Temperatur in Schaffhausen",
                                          updated="2026-09-19"), 3, spec, temporal)
    assert yesterday.temporal_status == TemporalStatus.STALE
    ambiguous = normalize_evidence(source("Wetter Schaffhausen",
                                         "Vorhersage 20.09.2024. Update 20.09.2026."), 4, spec, temporal)
    assert ambiguous.temporal_status == TemporalStatus.TEMPORAL_UNKNOWN


def test_comparison_selects_target_evidence_and_rejects_generic_articles():
    temporal = context()
    spec = classify_research_question("Ubuntu 24.04 oder Ubuntu 26.04 für AI-Applikationen und lokale LLMs im Moment?", temporal)
    sources = [
        source("Ubuntu 24.04 support", "Ubuntu 24.04 supports the listed AI application packages.",
               url="https://ubuntu.com/24", updated="2026-09-19"),
        source("Ubuntu 26.04 support", "Ubuntu 26.04 supports the new AI application packages.",
               url="https://ubuntu.com/26", updated="2026-09-19"),
        source("Local LLM privacy", "Local language models can protect privacy.", updated="2026-09-19"),
        source("Ollama introduction", "Ollama runs models locally.", updated="2026-09-19"),
    ]
    evidence = [normalize_evidence(item, index, spec, temporal) for index, item in enumerate(sources, 1)]
    selected = select_evidence(spec, evidence)
    assert [e.evidence_id for e in selected] == ["research_1", "research_2"]
    assert all(e.rejection_reason == "no_comparison_target" for e in evidence[2:])
    plan = build_research_plan(spec, temporal, selected)
    assert plan.response_mode == "comparison_answer"
    assert {fact.target for fact in plan.required_facts} == set(spec.comparison_targets)
    assert all(fact.source_evidence_ids for fact in plan.required_facts)
    assert plan.compliance("Ubuntu 24.04 supports AI application packages. Ubuntu 26.04 supports new AI application packages.")["compliant"]
    assert not plan.compliance("Lokale LLMs sind wegen Datenschutz nützlich.")["compliant"]
    assert not plan.compliance("Ubuntu 24.04 und Ubuntu 26.04 sind für lokale LLMs interessant.")["compliant"]


def test_missing_comparison_side_requires_caveat():
    temporal = context()
    spec = classify_research_question("Ubuntu 24.04 oder Ubuntu 26.04 im Moment für lokale LLMs?", temporal)
    item = normalize_evidence(source("Ubuntu 24.04 support", "Ubuntu 24.04 supports local LLM packages.",
                                     updated="2026-09-19"), 1, spec, temporal)
    selected = select_evidence(spec, [item])
    plan = build_research_plan(spec, temporal, selected)
    assert plan.required_caveats and "Ubuntu 26.04" in plan.required_caveats[0]
    assert "keine ausreichend belastbare Evidenz" in plan.render()
    debug = research_debug(spec, temporal, [item], plan, plan.compliance("general summary"), True)
    assert debug["comparison_target_coverage"] == .5


def test_timeless_document_is_not_rejected_for_age():
    temporal = context()
    spec = classify_research_question("Was ist OPC UA?", temporal)
    item = normalize_evidence(source("OPC UA specification", "OPC UA is an industrial interoperability standard. Published 2020-01-01."),
                              1, spec, temporal)
    assert item.temporal_status == TemporalStatus.TEMPORALLY_COMPATIBLE
    assert select_evidence(spec, [item]) == [item]


def test_current_claim_with_unsupported_temperature_fails_compliance():
    temporal = context()
    spec = classify_research_question("Wetter heute in Schaffhausen", temporal)
    item = normalize_evidence(source("Wetter Schaffhausen 20.09.2026",
                                     "Schaffhausen heute 20.09.2026: 17 °C und trocken."), 1, spec, temporal)
    plan = build_research_plan(spec, temporal, select_evidence(spec, [item]))
    bad = plan.compliance("Schaffhausen heute 20.09.2026: 19 °C und Regen.")
    assert bad["temporal_claim_conflict"] and not bad["compliant"]
    good = plan.compliance("Schaffhausen heute 20.09.2026: 17 °C und trocken.")
    assert good["compliant"]


def test_conflicting_current_temperatures_are_observable():
    temporal = context()
    spec = classify_research_question("Wetter heute in Schaffhausen", temporal)
    evidence = [normalize_evidence(source("Wetter Schaffhausen 20.09.2026",
                                          "Schaffhausen heute 20.09.2026: 17 °C."), 1, spec, temporal),
                normalize_evidence(source("Wetter Schaffhausen 20.09.2026",
                                          "Schaffhausen heute 20.09.2026: 23 °C.",
                                          url="https://other.example/weather"), 2, spec, temporal)]
    plan = build_research_plan(spec, temporal, select_evidence(spec, evidence))
    assert plan.evidence_conflict
    assert any("widersprüchliche" in caveat for caveat in plan.required_caveats)
    assert "widersprüchliche" in plan.render()


def test_research_workflow_uses_one_synthesis_call_and_cites_only_selected(monkeypatch):
    from avacore.api import http_app

    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app.settings, "daily_briefing_timezone", "Europe/Zurich")
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    monkeypatch.setattr(http_app, "collect_research_sources", lambda **kwargs: [
        source("Wetter Schaffhausen", "Vorhersage für 20.09.2024: 19 °C und Regen.",
               retrieved="2026-09-20T10:00:00+00:00"),
        source("Wetter Schaffhausen 20.09.2026", "Schaffhausen heute 20.09.2026: 17 °C.",
               url="https://weather.example/current")])
    monkeypatch.setattr(http_app, "TemporalContext", type("FixedTemporal", (), {
        "create": staticmethod(lambda timezone_name, target_date=None, now_utc=None:
                               TemporalContext.create(timezone_name, target_date, NOW))}))
    chat = MagicMock(return_value="Heute sind es 19 °C und Regen laut der alten Quelle.")
    monkeypatch.setattr(http_app.backend, "chat", chat)
    result = http_app.run_research_workflow("Wetter heute in Schaffhausen", save_memory=False)
    assert chat.call_count == 1
    assert "19 °C" not in result["answer"]
    assert [s["url"] for s in result["sources"]] == ["https://weather.example/current"]
    assert "20.09.2024" not in chat.call_args.args[0][1]["content"]
    assert http_app._last_research_grounding_debug["fallback_used"]


def test_only_stale_workflow_still_has_one_synthesis_call_and_no_citations(monkeypatch):
    from avacore.api import http_app

    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    monkeypatch.setattr(http_app, "collect_research_sources", lambda **kwargs: [
        source("Wetter Schaffhausen", "Vorhersage 20.09.2024: 19 °C.")])
    monkeypatch.setattr(http_app, "TemporalContext", type("FixedTemporal", (), {
        "create": staticmethod(lambda timezone_name, target_date=None, now_utc=None:
                               TemporalContext.create(timezone_name, target_date, NOW))}))
    chat = MagicMock(return_value="Heute sind es 19 °C.")
    monkeypatch.setattr(http_app.backend, "chat", chat)
    result = http_app.run_research_workflow("Wetter heute in Schaffhausen", save_memory=False)
    assert chat.call_count == 1
    assert result["sources"] == []
    assert "keine ausreichend aktuelle" in result["answer"].casefold()
    assert http_app._last_research_grounding_debug["response_mode"] == "insufficient_current_evidence"


def test_rejected_source_citation_forces_fallback(monkeypatch):
    from avacore.api import http_app

    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    monkeypatch.setattr(http_app, "collect_research_sources", lambda **kwargs: [
        source("Wetter Schaffhausen 20.09.2024", "Schaffhausen 20.09.2024: 19 °C.",
               url="https://old.example/weather"),
        source("Wetter Schaffhausen 20.09.2026", "Schaffhausen heute 20.09.2026: 17 °C.",
               url="https://current.example/weather")])
    monkeypatch.setattr(http_app, "TemporalContext", type("FixedTemporal", (), {
        "create": staticmethod(lambda timezone_name, target_date=None, now_utc=None:
                               TemporalContext.create(timezone_name, target_date, NOW))}))
    chat = MagicMock(return_value="Schaffhausen heute 20.09.2026: 17 °C. Quelle: https://old.example/weather")
    monkeypatch.setattr(http_app.backend, "chat", chat)
    result = http_app.run_research_workflow("Wetter heute in Schaffhausen", save_memory=False)
    assert chat.call_count == 1
    assert "old.example" not in result["answer"]
    assert result["sources"][0]["url"] == "https://current.example/weather"
    assert http_app._last_research_grounding_debug["citation_conflict"]


def test_page_metadata_is_extracted_without_using_fetch_time(monkeypatch):
    from avacore.tools import web_research

    class Response:
        headers = {"content-type": "text/html"}
        text = ('<html><head><title>Example</title>'
                '<meta property="article:published_time" content="2026-09-18T08:00:00Z">'
                '<meta property="article:modified_time" content="2026-09-20T08:00:00Z">'
                '</head><body><main>Current information.</main></body></html>')
        def raise_for_status(self): pass

    monkeypatch.setattr(web_research.requests, "get", lambda *args, **kwargs: Response())
    title, text, published, updated = web_research.fetch_readable_page_text(
        "https://example.org", include_metadata=True)
    assert title == "Example"
    assert published.startswith("2026-09-18") and updated.startswith("2026-09-20")
    assert text == "Current information."
