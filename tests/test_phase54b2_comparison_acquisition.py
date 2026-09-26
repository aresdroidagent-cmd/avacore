from datetime import datetime, timezone
from unittest.mock import MagicMock

from avacore.core.research_grounding import (
    TemporalContext, TemporalStatus, TemporalBasis, classify_research_question,
    comparison_search_queries, normalize_evidence, select_evidence,
    build_research_plan, research_debug,
)
from avacore.tools.web_research import ResearchSource


NOW = datetime(2026, 9, 20, 10, tzinfo=timezone.utc)
QUERY = ("Was würdest du im Moment für AI-Applikationen und lokale LLMs empfehlen: "
         "Ubuntu 24.04 oder Ubuntu 26.04?")


def source(title, text, url, *, published=None):
    return ResearchSource(title, url, "", text, published_at=published)


def context(query):
    temporal = TemporalContext.create("Europe/Zurich", now_utc=NOW)
    spec = classify_research_question(query, temporal)
    return spec, temporal


def test_comparison_queries_keep_context_and_are_bounded_for_other_versions():
    for query, target_a, target_b in (
        (QUERY, "Ubuntu 24.04", "Ubuntu 26.04"),
        ("Python 3.12 vs Python 3.14 for scientific packages", "Python 3.12", "Python 3.14"),
        ("Python 3.12 vs 3.14 for scientific packages", "Python 3.12", "Python 3.14"),
        ("PostgreSQL 17 vs PostgreSQL 18 for extensions", "PostgreSQL 17", "PostgreSQL 18"),
        ("PostgreSQL 17 vs 18 for extensions", "PostgreSQL 17", "PostgreSQL 18"),
    ):
        spec, _ = context(query)
        searches = comparison_search_queries(spec)
        assert spec.comparison_targets == (target_a, target_b)
        assert len(searches) == 4
        assert target_a in searches[0] and "support lifecycle" in searches[0]
        assert target_b in searches[1] and "support lifecycle" in searches[1]
        assert target_a in searches[2] and target_b in searches[3]
        assert "compatibility" in searches[2] and "compatibility" in searches[3]
    assert "llms" in comparison_search_queries(context(QUERY)[0])[2].casefold()


def test_unknown_or_stale_source_can_supply_valid_interval_fact_but_old_weather_cannot():
    query = "Target X aktuell unterstützt?"
    spec, temporal = context(query)
    unknown = normalize_evidence(source("Target X", "Target X supported until 2029.",
                                        "https://example.org/x"), 1, spec, temporal)
    old = normalize_evidence(source("Target X", "Target X released 2024-04 and supported until 2029-04.",
                                    "https://example.org/old", published="2024-04-01"), 2, spec, temporal)
    selected = select_evidence(spec, [unknown, old])
    assert unknown.temporal_status == TemporalStatus.TEMPORAL_UNKNOWN
    assert unknown.facts[0].temporal_basis == TemporalBasis.VALIDITY_INTERVAL
    assert old.temporal_status == TemporalStatus.STALE
    assert len(selected) == 2
    plan = build_research_plan(spec, temporal, selected)
    debug = research_debug(spec, temporal, [unknown, old], plan, plan.compliance(plan.render()), True)
    assert debug["facts_rescued_from_temporal_unknown"] >= 1

    weather_spec, weather_temporal = context("Wetter heute in Bern")
    weather = normalize_evidence(source("Wetter Bern", "Vorhersage Bern 20.09.2024: 19 °C.",
                                        "https://example.org/old-weather"), 1, weather_spec, weather_temporal)
    assert not select_evidence(weather_spec, [weather])
    assert weather.rejection_reason == "source_explicit_temporal_conflict"


def test_comparison_acquisition_deduplicates_and_keeps_partial_lifecycle_answer(monkeypatch):
    from avacore.api import http_app

    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    searches = []

    def collect(*, query, max_results, max_chars_per_source, diagnostics=None, **kwargs):
        searches.append((query, max_results))
        if len(searches) == 1:
            return [source("Local LLM privacy", "Local LLMs can protect privacy.",
                           "https://example.org/llm?utm_source=search")]
        if len(searches) == 2:
            return [source("Ubuntu 24.04", "Ubuntu 24.04 released 2024-04 and supported until 2029-04.",
                           "https://ubuntu.com/24?utm_source=search")]
        if len(searches) == 3:
            return [source("Ubuntu 26.04", "Ubuntu 26.04 released 2026-04 and supported until 2031-04.",
                           "https://ubuntu.com/26")]
        return [source("Ubuntu 24.04", "Ubuntu 24.04 released 2024-04 and supported until 2029-04.",
                       "https://ubuntu.com/24?utm_medium=duplicate")]

    monkeypatch.setattr(http_app, "collect_research_sources", collect)
    chat = MagicMock(return_value="Lokale LLMs sind privat und Ubuntu ist nützlich.")
    monkeypatch.setattr(http_app.backend, "chat", chat)
    result = http_app.run_research_workflow(QUERY, save_memory=False)
    debug = http_app._last_research_grounding_debug
    assert len(searches) == 4 and all(limit <= 2 for _, limit in searches)
    assert debug["unique_results_seen"] == 3
    assert debug["comparison_fact_coverage"] == 1.0
    assert debug["facts_rescued_from_temporal_unknown"] == 2
    assert debug["comparison_criterion_coverage"] == 0.0
    assert "Ubuntu 24.04" in result["answer"] and "Ubuntu 26.04" in result["answer"]
    assert "Kompatibilität" in result["answer"]
    assert chat.call_count == 1


def test_table_rows_preserve_headers_for_lifecycle_extraction(monkeypatch):
    from avacore.tools import web_research

    class Response:
        headers = {"content-type": "text/html"}
        text = ('<html><body><main><table><tr><th>Version</th><th>Release</th>'
                '<th>Standard support</th></tr><tr><td>Ubuntu 24.04</td>'
                '<td>April 2024</td><td>April 2029</td></tr></table></main></body></html>')

        def raise_for_status(self):
            pass

    monkeypatch.setattr(web_research.requests, "get", lambda *args, **kwargs: Response())
    _, text = web_research.fetch_readable_page_text("https://example.org")
    assert "Version: Ubuntu 24.04" in text
    assert "Release: April 2024" in text
    assert "Standard support: April 2029" in text
    spec, temporal = context(QUERY)
    evidence = normalize_evidence(source("Ubuntu releases", text, "https://ubuntu.com/releases"),
                                  1, spec, temporal)
    assert any(fact.temporal_basis == TemporalBasis.VALIDITY_INTERVAL for fact in evidence.facts)
