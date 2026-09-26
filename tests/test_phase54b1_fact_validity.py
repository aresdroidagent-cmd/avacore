from datetime import datetime, timezone
from unittest.mock import MagicMock

from avacore.core.research_grounding import (
    TemporalBasis, TemporalContext, TemporalStatus,
    classify_research_question, normalize_evidence, select_evidence,
    build_research_plan, research_debug,
)
from avacore.tools.web_research import ResearchSource


NOW = datetime(2026, 9, 20, 10, 0, tzinfo=timezone.utc)


def setup(query, sources):
    temporal = TemporalContext.create("Europe/Zurich", now_utc=NOW)
    spec = classify_research_question(query, temporal)
    temporal = TemporalContext.create("Europe/Zurich", spec.target_date, NOW)
    evidence = [normalize_evidence(source, i, spec, temporal) for i, source in enumerate(sources, 1)]
    selected = select_evidence(spec, evidence)
    plan = build_research_plan(spec, temporal, selected, evidence)
    return spec, temporal, evidence, selected, plan


def source(title, text, *, url="https://example.org/page", updated=None, markers=(), retrieved=None):
    return ResearchSource(title, url, "", text, updated_at=updated,
                          time_markers=markers, retrieved_at=retrieved)


def test_weather_fact_date_overrides_today_retrieval_and_current_fact_survives():
    query = "Wetterbericht für heute in Schaffhausen, Schweiz"
    old = source("Wetter Schaffhausen", "Vorhersage Schaffhausen für 20.09.2024: 19 °C.",
                 retrieved="2026-09-20T10:00:00Z", updated="2026-09-20")
    current = source("Wetter Schaffhausen", "Vorhersage Schaffhausen für 20.09.2026: 17 °C.",
                     url="https://current.example/weather")
    spec, temporal, evidence, selected, plan = setup(query, [old, current])
    assert evidence[0].facts[0].temporal_basis == TemporalBasis.EXPLICIT_DATE
    assert evidence[0].facts[0].temporal_status == TemporalStatus.STALE
    assert [item.evidence_id for item in selected] == ["research_2"]
    assert plan.required_facts[0].temporal_basis == TemporalBasis.EXPLICIT_DATE
    assert "17 °C" in plan.render() and "19 °C" not in plan.render()
    debug = research_debug(spec, temporal, evidence, plan, plan.compliance(plan.render()), True)
    assert debug["facts_validated"] > 0


def test_weather_current_page_marker_requires_real_date_binding():
    query = "Wetter heute in Schaffhausen"
    bound = source("Wetter Schaffhausen", "Wetter heute in Schaffhausen: 17 °C und trocken.",
                   markers=("2026-09-20",))
    unbound = source("Wetter Schaffhausen", "Wetter heute in Schaffhausen: 17 °C und trocken.",
                     url="https://unknown.example/weather")
    _, _, evidence, selected, _ = setup(query, [bound, unbound])
    assert evidence[0].facts[0].temporal_basis == TemporalBasis.CURRENT_PAGE_MARKER
    assert evidence[0].facts[0].temporal_status == TemporalStatus.CONFIRMED_CURRENT
    assert evidence[1].facts[0].temporal_basis == TemporalBasis.UNKNOWN
    assert evidence[1].rejection_reason == "source_temporal_unknown_but_no_valid_fact"
    assert [item.evidence_id for item in selected] == ["research_1"]


def test_page_extraction_keeps_structured_dates_and_removes_navigation(monkeypatch):
    from avacore.tools import web_research

    class Response:
        headers = {"content-type": "text/html"}
        text = ('<html><head><title>Forecast</title>'
                '<script type="application/ld+json">'
                '{"@type":"WebPage","dateModified":"2026-09-20"}</script>'
                '</head><body><nav>Navigation Login</nav><main>'
                '<time datetime="2026-09-20">20 September 2026</time>'
                '<p>Wetter heute: 17 °C und trocken.</p>'
                '</main></body></html>')

        def raise_for_status(self):
            pass

    monkeypatch.setattr(web_research.requests, "get", lambda *args, **kwargs: Response())
    _, text, _, updated, markers = web_research.fetch_readable_page_text(
        "https://example.org/weather", include_metadata=True, include_time_markers=True)
    assert "Navigation" not in text
    assert "17 °C" in text
    assert updated == "2026-09-20"
    assert markers == ("2026-09-20",)


def test_ubuntu_lifecycle_intervals_validate_without_publication_date():
    query = "Ubuntu 24.04 oder Ubuntu 26.04 im Moment für lokale LLMs?"
    official = source("Ubuntu release and support table",
        "Ubuntu 24.04 released 2024-04 and supported until 2029-04.\n"
        "Ubuntu 26.04 released 2026-04 and supported until 2031-04.",
        url="https://ubuntu.com/releases")
    generic = source("Local LLM privacy", "Local LLMs can protect privacy.",
                     url="https://blog.example/llm", updated="2026-09-20")
    spec, temporal, evidence, selected, plan = setup(query, [official, generic])
    assert evidence[0].temporal_status == TemporalStatus.TEMPORAL_UNKNOWN
    assert selected == [evidence[0]]
    assert {fact.target for fact in plan.required_facts} == set(spec.comparison_targets)
    assert all(fact.temporal_basis == TemporalBasis.VALIDITY_INTERVAL for fact in plan.required_facts)
    assert all(fact.valid_from <= temporal.local_date <= fact.valid_until for fact in plan.required_facts)
    assert evidence[1].role.value != "PRIMARY"
    debug = research_debug(spec, temporal, evidence, plan, plan.compliance(plan.render()), True)
    assert debug["comparison_fact_coverage"] == 1.0
    assert debug["fact_temporal_basis_counts"]["VALIDITY_INTERVAL"] == 2


def test_lifecycle_exact_release_day_is_not_rounded_to_month_start():
    temporal = TemporalContext.create("Europe/Zurich", now_utc=datetime(
        2026, 4, 10, 10, 0, tzinfo=timezone.utc))
    spec = classify_research_question("Ubuntu 26.04 aktuell?", temporal)
    evidence = normalize_evidence(source("Ubuntu 26.04",
        "Ubuntu 26.04 released 2026-04-25 and supported until 2031-04-30."),
        1, spec, temporal)
    assert evidence.facts[0].valid_from.isoformat() == "2026-04-25"
    assert evidence.facts[0].temporal_status == TemporalStatus.TEMPORAL_CONFLICT


def test_opc_ua_fact_extraction_rejects_boilerplate_and_fallback_uses_definition():
    query = "Was ist OPC UA?"
    page = source("OPC UA", "Inhaltsverzeichnis umschalten OPC Unified Architecture 5 Sprachen Čeština English Polski.\n"
                  "OPC UA is a platform-independent service-oriented architecture for industrial communication.")
    spec, temporal, evidence, selected, plan = setup(query, [page])
    assert selected == [evidence[0]]
    assert any(candidate.rejection_reason == "boilerplate" for candidate in evidence[0].fact_candidates)
    assert len(plan.required_facts) == 1
    assert "platform-independent" in plan.required_facts[0].fact
    assert "Inhaltsverzeichnis" not in plan.render()
    assert "Inhaltsverzeichnis" not in plan.prompt(selected)
    debug = research_debug(spec, temporal, evidence, plan, plan.compliance(plan.render()), True)
    assert debug["fact_candidates_seen"] >= 2 and debug["facts_validated"] == 1
    assert debug["facts_rejected"] >= 1


def test_no_valid_fact_produces_conservative_answer_and_missing_side_reason():
    spec, temporal, evidence, selected, plan = setup("Was ist OPC UA?", [
        source("OPC UA", "Navigation. Inhaltsverzeichnis umschalten. Sprachen. Login.")])
    assert not selected and not plan.required_facts
    assert "keine ausreichend klare belegte Antwort" in plan.render()
    debug = research_debug(spec, temporal, evidence, plan, plan.compliance(plan.render()), True)
    assert debug["facts_validated"] == 0

    spec, temporal, evidence, selected, plan = setup("Ubuntu 24.04 oder Ubuntu 26.04 im Moment?", [
        source("Ubuntu 24.04", "Ubuntu 24.04 released 2024-04 and supported until 2029-04."),
        source("Ubuntu 26.04", "Navigation. Sprachen. Login.", url="https://other.example/26")])
    assert plan.comparison_missing_reasons["Ubuntu 26.04"] == "sources_found_but_no_valid_facts"
    debug = research_debug(spec, temporal, evidence, plan, plan.compliance(plan.render()), True)
    assert debug["comparison_fact_coverage"] == .5
    assert debug["comparison_missing_reasons"]["Ubuntu 26.04"] == "sources_found_but_no_valid_facts"


def test_fact_extraction_does_not_add_model_call(monkeypatch):
    from avacore.api import http_app

    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    monkeypatch.setattr(http_app, "collect_research_sources", lambda **kwargs: [
        source("OPC UA", "Inhaltsverzeichnis umschalten.\n"
               "OPC UA is a platform-independent service-oriented architecture for industrial communication.")])
    chat = MagicMock(return_value="OPC UA is a platform-independent service-oriented architecture for industrial communication.")
    monkeypatch.setattr(http_app.backend, "chat", chat)
    result = http_app.run_research_workflow("Was ist OPC UA?", save_memory=False)
    assert chat.call_count == 1
    assert "Inhaltsverzeichnis" not in chat.call_args.args[0][1]["content"]
    assert "Inhaltsverzeichnis" not in result["answer"]
