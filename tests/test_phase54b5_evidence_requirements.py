from datetime import datetime, timezone
from unittest.mock import MagicMock

from avacore.core.research_grounding import (
    EvidenceVolatility, RecommendationConfidence, TemporalBasis, TemporalContext,
    TemporalStatus, build_research_plan, classify_research_question,
    normalize_evidence, research_debug, research_search_queries, select_evidence,
)
from avacore.tools.web_research import ResearchSource


NOW = datetime(2026, 9, 26, 10, tzinfo=timezone.utc)
QUERY = ("Was würdest du im Moment für AI-Applikationen und lokale LLMs empfehlen: "
         "Ubuntu 24.04 oder Ubuntu 26.04?")


def setup_spec(query=QUERY):
    temporal = TemporalContext.create("Europe/Zurich", now_utc=NOW)
    return classify_research_question(query, temporal), temporal


def source(title, text, url="https://ubuntu.com/docs"):
    return ResearchSource(title, url, "", text, retrieved_at=NOW.isoformat())


def test_requirements_and_queries_are_criterion_driven_and_bounded():
    spec, _ = setup_spec()
    assert [criterion.criterion_id for criterion in spec.comparison_criteria] == [
        "support_lifecycle", "requested_context_compatibility", "current_limitations"]
    assert len(spec.evidence_requirements) == 6
    queries = research_search_queries(spec)
    assert len(queries) == 4
    assert [(query.target, query.criterion) for query in queries] == [
        ("Ubuntu 24.04", "support_lifecycle"),
        ("Ubuntu 26.04", "support_lifecycle"),
        ("Ubuntu 24.04", "requested_context_compatibility+current_limitations"),
        ("Ubuntu 26.04", "requested_context_compatibility+current_limitations"),
    ]
    assert all(query.requirement_ids for query in queries)
    assert "llms" in queries[2].query_text.casefold()


def test_live_authoritative_technical_docs_satisfy_exact_target_requirements():
    spec, temporal = setup_spec()
    evidence = [
        normalize_evidence(source("Ubuntu 24.04 documentation",
            "Ubuntu 24.04 is an LTS release with standard support.", "https://ubuntu.com/24"), 1, spec, temporal),
        normalize_evidence(source("Ubuntu 26.04 documentation",
            "Ubuntu 26.04 is an LTS release with standard support.", "https://ubuntu.com/26"), 2, spec, temporal),
        normalize_evidence(source("Ubuntu 24.04 AI documentation",
            "Ubuntu 24.04 supports local LLM applications.", "https://ubuntu.com/24/ai"), 3, spec, temporal),
        normalize_evidence(source("Ubuntu 26.04 AI documentation",
            "Ubuntu 26.04 supports local LLM applications.", "https://ubuntu.com/26/ai"), 4, spec, temporal),
        normalize_evidence(source("Ubuntu 24.04 limitations",
            "Ubuntu 24.04 has a known issue for one local LLM application.", "https://ubuntu.com/24/issues"), 5, spec, temporal),
        normalize_evidence(source("Ubuntu 26.04 limitations",
            "Ubuntu 26.04 has a known issue for one local LLM application.", "https://ubuntu.com/26/issues"), 6, spec, temporal),
    ]
    selected = select_evidence(spec, evidence)
    assert len(selected) == 6
    facts = [fact for item in selected for fact in item.facts]
    assert all(fact.temporal_basis == TemporalBasis.LIVE_AUTHORITATIVE_DOCUMENTATION for fact in facts)
    assert all(fact.temporal_status == TemporalStatus.TEMPORALLY_COMPATIBLE for fact in facts)
    assert all(fact.volatility == EvidenceVolatility.MEDIUM for fact in facts)
    assert all(fact.requirement_ids for fact in facts)
    plan = build_research_plan(spec, temporal, selected, evidence)
    assert all(requirement.satisfied for requirement in plan.evidence_requirements)
    assert plan.requirement_coverage == {
        "support_lifecycle": 1.0, "requested_context_compatibility": 1.0,
        "current_limitations": 1.0}
    assert plan.recommendation_confidence == RecommendationConfidence.HIGH
    assert len(plan.required_facts) == 6


def test_live_document_policy_needs_authority_exact_version_and_requirement_match():
    spec, temporal = setup_spec()
    generic_product = normalize_evidence(source("Ubuntu AI", "Ubuntu supports local LLM applications.",
                                                  "https://ubuntu.com/ai"), 1, spec, temporal)
    unofficial = normalize_evidence(source("Ubuntu 24.04 AI", "Ubuntu 24.04 supports local LLM applications.",
                                        "https://blog.example/ubuntu-ai"), 2, spec, temporal)
    irrelevant = normalize_evidence(source("Ubuntu 24.04 overview", "Ubuntu 24.04 is a Linux operating system.",
                                         "https://ubuntu.com/24"), 3, spec, temporal)
    assert not generic_product.facts
    assert unofficial.facts[0].temporal_status == TemporalStatus.TEMPORAL_UNKNOWN
    assert irrelevant.facts[0].temporal_status == TemporalStatus.TEMPORAL_UNKNOWN


def test_weather_never_uses_live_authoritative_documentation():
    spec, temporal = setup_spec("Wetter heute in Schaffhausen")
    evidence = normalize_evidence(ResearchSource(
        "Wetter Schaffhausen", "https://weather.example/docs", "",
        "Wetter heute in Schaffhausen: 17 °C.", retrieved_at=NOW.isoformat()), 1, spec, temporal)
    assert evidence.facts[0].volatility == EvidenceVolatility.HIGH
    assert evidence.facts[0].temporal_basis == TemporalBasis.UNKNOWN
    assert evidence.facts[0].temporal_status == TemporalStatus.TEMPORAL_UNKNOWN
    assert not select_evidence(spec, [evidence])


def test_partial_criterion_coverage_produces_structured_partial_comparison():
    spec, temporal = setup_spec()
    evidence = [
        normalize_evidence(source("Ubuntu 24.04 lifecycle",
            "Ubuntu 24.04 released 2024-04 and supported until 2029-04.", "https://ubuntu.com/24"), 1, spec, temporal),
        normalize_evidence(source("Ubuntu 26.04 lifecycle",
            "Ubuntu 26.04 released 2026-04 and supported until 2031-04.", "https://ubuntu.com/26"), 2, spec, temporal),
    ]
    selected = select_evidence(spec, evidence)
    plan = build_research_plan(spec, temporal, selected, evidence)
    assert plan.requirement_coverage["support_lifecycle"] == 1.0
    assert plan.requirement_coverage["requested_context_compatibility"] == 0.0
    assert plan.recommendation_confidence == RecommendationConfidence.INSUFFICIENT
    answer = plan.render()
    assert "Support/Lifecycle" in answer
    assert "Requested-context compatibility" in answer
    assert "Lifecycle-Daten allein" in answer
    assert "Empfehlungssicherheit INSUFFICIENT" in answer
    debug = research_debug(spec, temporal, evidence, plan, plan.compliance(answer), True)
    assert debug["evidence_requirements_satisfied"] == 2
    assert len(debug["missing_requirement_ids"]) == 4
    assert debug["recommendation_confidence"] == "INSUFFICIENT"


def test_requirement_provenance_and_one_synthesis_call(monkeypatch):
    from avacore.api import http_app

    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    seen = []

    def collect(*, query, requirement_ids, diagnostics, **kwargs):
        seen.append((query, requirement_ids))
        diagnostics.update(status="SEARCH_OK_WITH_RESULTS", result_count_raw=1,
                           fetch_attempted=1, fetch_succeeded=1, fetch_failed=0)
        target = "Ubuntu 24.04" if "24.04" in query else "Ubuntu 26.04"
        text = (f"{target} is an LTS release with standard support." if "lifecycle" in query else
                f"{target} supports local LLM applications. "
                f"{target} has a known issue for one local LLM application.")
        return [ResearchSource(target, f"https://ubuntu.com/{target.split()[1]}/{len(seen)}", "", text,
                               originating_queries=(query,), originating_providers=("mock",),
                               originating_requirement_ids=requirement_ids)]

    monkeypatch.setattr(http_app, "collect_research_sources", collect)
    chat = MagicMock(return_value="Ubuntu 24.04 und Ubuntu 26.04 sind dokumentiert.")
    monkeypatch.setattr(http_app.backend, "chat", chat)
    result = http_app.run_research_workflow(QUERY, save_memory=False)
    debug = http_app._last_research_grounding_debug
    assert result["ok"] and len(seen) == 4
    assert all(requirement_ids for _, requirement_ids in seen)
    assert all(value for value in debug["originating_requirement_ids"].values())
    assert debug["evidence_requirements_satisfied"] == 6
    assert debug["recommendation_confidence"] == "HIGH"
    assert chat.call_count == 1
