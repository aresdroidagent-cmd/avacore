from unittest.mock import MagicMock

import requests

from avacore.tools import web_research


class SearchResponse:
    def __init__(self, status=200, urls=(), body=None):
        self.status_code = status
        self.text = body if body is not None else "<html>" + "".join(
            f'<div class="result"><a class="result__a" href="{url}">Result {i}</a></div>'
            for i, url in enumerate(urls)) + "</html>"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


def test_rate_limit_retry_once_then_results(monkeypatch):
    replies = iter([SearchResponse(429), SearchResponse(urls=("https://example.org/a",))])
    post = MagicMock(side_effect=lambda *args, **kwargs: next(replies))
    monkeypatch.setattr(web_research.requests, "post", post)
    monkeypatch.setattr(web_research.time, "sleep", lambda _: None)
    diagnostics = {}
    results = web_research.search_duckduckgo_html("OPC UA", diagnostics=diagnostics)
    assert len(results) == 1
    assert post.call_count == 2
    assert diagnostics["status"] == "SEARCH_OK_WITH_RESULTS"
    assert diagnostics["retry_count"] == 1
    assert diagnostics["http_status"] == 200


def test_persistent_rate_limit_and_permanent_403_are_distinct(monkeypatch):
    monkeypatch.setattr(web_research.time, "sleep", lambda _: None)
    for status, expected, calls in ((429, "SEARCH_RATE_LIMITED", 2),
                                    (403, "SEARCH_HTTP_ERROR", 1)):
        post = MagicMock(return_value=SearchResponse(status))
        monkeypatch.setattr(web_research.requests, "post", post)
        diagnostics = {}
        assert web_research.collect_research_sources("OPC UA", diagnostics=diagnostics) == []
        assert post.call_count == calls
        assert diagnostics["status"] == expected
        assert diagnostics["http_status"] == status


def test_timeout_and_unrecognized_html_are_diagnostic_not_normal_empty(monkeypatch):
    monkeypatch.setattr(web_research.time, "sleep", lambda _: None)
    post = MagicMock(side_effect=requests.Timeout("timeout"))
    monkeypatch.setattr(web_research.requests, "post", post)
    diagnostics = {}
    assert web_research.collect_research_sources("OPC UA", diagnostics=diagnostics) == []
    assert diagnostics["status"] == "SEARCH_TIMEOUT" and post.call_count == 2

    monkeypatch.setattr(web_research.requests, "post", lambda *args, **kwargs:
                        SearchResponse(body="<html><body>Provider challenge</body></html>"))
    diagnostics = {}
    assert web_research.collect_research_sources("OPC UA", diagnostics=diagnostics) == []
    assert diagnostics["status"] == "SEARCH_PARSE_ERROR"


def test_fetch_partial_failure_reports_search_and_fetch_separately(monkeypatch):
    monkeypatch.setattr(web_research.requests, "post", lambda *args, **kwargs:
                        SearchResponse(urls=("https://example.org/a", "https://example.org/b")))

    def fetch(url, **kwargs):
        if url.endswith("/b"):
            raise requests.Timeout("page timeout")
        return "A", "OPC UA is an industrial communication architecture.", None, None, ()

    monkeypatch.setattr(web_research, "fetch_readable_page_text", fetch)
    diagnostics = {}
    sources = web_research.collect_research_sources("OPC UA", diagnostics=diagnostics)
    assert diagnostics["status"] == "SEARCH_OK_WITH_RESULTS"
    assert diagnostics["result_count_raw"] == 2
    assert (diagnostics["fetch_attempted"], diagnostics["fetch_succeeded"],
            diagnostics["fetch_failed"]) == (2, 1, 1)
    assert sources[0].ok and not sources[1].ok


def test_sequential_research_and_duplicate_provenance(monkeypatch):
    from avacore.api import http_app
    from avacore.tools.web_research import ResearchSource

    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    chat = MagicMock(return_value="OPC UA is a platform-independent architecture.")
    monkeypatch.setattr(http_app.backend, "chat", chat)
    calls = []

    def collect(*, query, max_results, max_chars_per_source, diagnostics, **kwargs):
        calls.append(query)
        diagnostics.update(status="SEARCH_OK_WITH_RESULTS", result_count_raw=1,
                           fetch_attempted=1, fetch_succeeded=1, fetch_failed=0)
        if "Ubuntu" in query or "ubuntu" in query:
            return [ResearchSource("Ubuntu 24.04", "https://ubuntu.com/24?utm_source=x", "",
                                   "Ubuntu 24.04 released 2024-04 and supported until 2029-04.")]
        return [ResearchSource("OPC UA", "https://example.org/opc", "",
                               "OPC UA is a platform-independent industrial communication architecture.")]

    monkeypatch.setattr(http_app, "collect_research_sources", collect)
    for query in ("Was ist OPC UA?", "Ubuntu 24.04 oder Ubuntu 26.04 momentan?", "Was ist OPC UA?"):
        result = http_app.run_research_workflow(query, save_memory=False)
        assert result["ok"]
        assert http_app._last_research_grounding_debug["unique_results_seen"] == 1
    assert len(calls) == 6  # 1 + 4 + 1 semantic queries
    debug = http_app._last_research_grounding_debug
    assert debug["search_queries_failed"] == 0
    assert chat.call_count == 3

    http_app.run_research_workflow("Ubuntu 24.04 oder Ubuntu 26.04 momentan?", save_memory=False)
    evidence_origins = http_app._last_research_grounding_debug["originating_queries"]
    assert len(next(iter(evidence_origins.values()))) == 4
    assert http_app._last_research_grounding_debug["raw_results_seen"] == 4
    assert http_app._last_research_grounding_debug["unique_results_seen"] == 1


def test_partial_rate_limit_and_zero_result_fallbacks(monkeypatch):
    from avacore.api import http_app
    from avacore.tools.web_research import ResearchSource

    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    chat = MagicMock(return_value="Keine sichere Empfehlung.")
    monkeypatch.setattr(http_app.backend, "chat", chat)
    calls = []

    def partial(*, query, max_results, max_chars_per_source, diagnostics, **kwargs):
        calls.append(query)
        if len(calls) > 2:
            diagnostics.update(status="SEARCH_RATE_LIMITED", http_status=429,
                               retry_count=1, result_count_raw=0)
            return []
        diagnostics.update(status="SEARCH_OK_WITH_RESULTS", result_count_raw=1)
        return [ResearchSource("Ubuntu 24.04", "https://ubuntu.com/24", "",
                               "Ubuntu 24.04 released 2024-04 and supported until 2029-04.")]

    monkeypatch.setattr(http_app, "collect_research_sources", partial)
    result = http_app.run_research_workflow("Ubuntu 24.04 oder Ubuntu 26.04 momentan?", save_memory=False)
    debug = http_app._last_research_grounding_debug
    assert result["ok"]
    assert (debug["search_queries_attempted"], debug["search_queries_succeeded"],
            debug["search_queries_failed"]) == (4, 2, 2)
    assert debug["search_failure_counts"] == {"SEARCH_RATE_LIMITED": 2}
    assert chat.call_count == 1

    def empty(*, query, max_results, max_chars_per_source, diagnostics, **kwargs):
        diagnostics.update(status="SEARCH_OK_EMPTY", result_count_raw=0)
        return []

    monkeypatch.setattr(http_app, "collect_research_sources", empty)
    assert "keine Suchtreffer" in http_app.run_research_workflow("Was ist OPC UA?", save_memory=False)["answer"]
    assert chat.call_count == 1

    def failed(*, query, max_results, max_chars_per_source, diagnostics, **kwargs):
        diagnostics.update(status="SEARCH_RATE_LIMITED", http_status=429, result_count_raw=0)
        return []

    monkeypatch.setattr(http_app, "collect_research_sources", failed)
    assert "nicht zuverlässig ausgeführt" in http_app.run_research_workflow(
        "Was ist OPC UA?", save_memory=False)["answer"]
    assert chat.call_count == 1

    def unreadable(*, query, max_results, max_chars_per_source, diagnostics, **kwargs):
        diagnostics.update(status="SEARCH_OK_WITH_RESULTS", result_count_raw=1,
                           fetch_attempted=1, fetch_succeeded=0, fetch_failed=1)
        return [ResearchSource("OPC UA", "https://example.org/opc", "", "", ok=False)]

    monkeypatch.setattr(http_app, "collect_research_sources", unreadable)
    assert "Suchtreffer gefunden" in http_app.run_research_workflow(
        "Was ist OPC UA?", save_memory=False)["answer"]
    assert chat.call_count == 1
