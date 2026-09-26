import hashlib
from unittest.mock import MagicMock

import requests

from avacore.tools import web_research


class Response:
    def __init__(self, status=200, body="", payload=None, content_type="text/html"):
        self.status_code = status
        self.text = body
        self.content = body.encode()
        self.headers = {"content-type": content_type}
        self.payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        if self.payload is None:
            raise ValueError("not json")
        return self.payload


RESULT_HTML = ('<html><div class="result"><a class="result__a" '
               'href="https://example.org/opc">OPC UA</a></div></html>')
RESULT_JSON = {"results": [{"title": "OPC UA", "url": "https://example.org/opc",
                            "content": "industrial communication"}]}


def test_ddg_202_unknown_body_is_fingerprinted_safely(monkeypatch):
    body = '<html><body>Captcha challenge token=supersecret1234567890 verify you are human</body></html>'
    monkeypatch.setattr(web_research.requests, "post", lambda *args, **kwargs: Response(202, body))
    result, acquisition = web_research.search_with_fallback("OPC UA", 2)
    assert result.status == "SEARCH_PARSE_ERROR"
    assert result.http_status == 202 and result.results == []
    assert acquisition["provider_fallback_reason"] == "NO_FALLBACK_PROVIDER_CONFIGURED"
    diagnostic = result.diagnostics
    assert diagnostic["content_type"] == "text/html"
    assert diagnostic["body_length"] == len(body.encode())
    assert diagnostic["body_sha256"] == hashlib.sha256(body.encode()).hexdigest()
    assert diagnostic["detected_page_type"] == "challenge"
    assert len(diagnostic["safe_body_prefix"]) <= 200
    assert "supersecret" not in diagnostic["safe_body_prefix"]


def test_primary_success_and_valid_empty_do_not_call_fallback(monkeypatch):
    get = MagicMock()
    monkeypatch.setattr(web_research.requests, "get", get)
    for body, expected in ((RESULT_HTML, "SEARCH_OK_WITH_RESULTS"),
                           ("<html><body>No results found</body></html>", "SEARCH_OK_EMPTY")):
        monkeypatch.setattr(web_research.requests, "post", lambda *args, **kwargs: Response(200, body))
        result, acquisition = web_research.search_with_fallback(
            "OPC UA", 2, fallback_name="searxng", searxng_url="https://search.example")
        assert result.status == expected
        assert not acquisition["provider_fallback_used"]
    get.assert_not_called()


def test_searxng_can_be_selected_as_primary_without_ddg(monkeypatch):
    post = MagicMock()
    monkeypatch.setattr(web_research.requests, "post", post)
    monkeypatch.setattr(web_research.requests, "get", lambda *args, **kwargs:
                        Response(200, payload=RESULT_JSON, content_type="application/json"))
    result, acquisition = web_research.search_with_fallback(
        "OPC UA", 2, primary_name="searxng", searxng_url="https://search.example")
    assert result.status == "SEARCH_OK_WITH_RESULTS"
    assert result.results[0].provider == "searxng"
    assert not acquisition["provider_fallback_used"]
    post.assert_not_called()


def test_202_timeout_and_persistent_429_fall_back_once(monkeypatch):
    monkeypatch.setattr(web_research.time, "sleep", lambda _: None)
    get = MagicMock(return_value=Response(200, payload=RESULT_JSON, content_type="application/json"))
    monkeypatch.setattr(web_research.requests, "get", get)
    cases = (
        (lambda *args, **kwargs: Response(202, "<html>unrecognized</html>"),
         "SEARCH_PARSE_ERROR", 1),
        (MagicMock(side_effect=requests.Timeout("timeout")), "SEARCH_TIMEOUT", 2),
        (lambda *args, **kwargs: Response(429), "SEARCH_RATE_LIMITED", 2),
    )
    for post, first_status, expected_calls in cases:
        spy = MagicMock(side_effect=post)
        monkeypatch.setattr(web_research.requests, "post", spy)
        result, acquisition = web_research.search_with_fallback(
            "OPC UA", 2, fallback_name="searxng", searxng_url="https://search.example")
        assert result.status == "SEARCH_OK_WITH_RESULTS"
        assert result.results[0].provider == "searxng"
        assert result.results[0].originating_query == "OPC UA"
        assert result.results[0].rank == 1
        assert acquisition["provider_fallback_used"]
        assert acquisition["provider_fallback_reason"] == first_status
        assert len(acquisition["provider_attempts"]) == 2
        assert spy.call_count == expected_calls
    assert get.call_count == 3


def test_both_providers_fail_surfaces_search_failure_without_llm(monkeypatch):
    from avacore.api import http_app

    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "search_provider", "ddg_html", raising=False)
    monkeypatch.setattr(http_app.settings, "search_fallback_provider", "searxng", raising=False)
    monkeypatch.setattr(http_app.settings, "searxng_url", "https://search.example", raising=False)
    monkeypatch.setattr(web_research.requests, "post", lambda *args, **kwargs:
                        Response(202, "<html>unrecognized</html>"))
    monkeypatch.setattr(web_research.requests, "get", lambda *args, **kwargs: Response(503))
    chat = MagicMock()
    monkeypatch.setattr(http_app.backend, "chat", chat)
    result = http_app.run_research_workflow("Was ist OPC UA?", save_memory=False)
    debug = http_app._last_research_grounding_debug
    assert "nicht zuverlässig ausgeführt" in result["answer"]
    assert debug["provider_fallback_used"]
    assert debug["search_queries_failed"] == 1
    assert debug["provider_failure_counts"] == {"ddg_html": 1, "searxng": 1}
    chat.assert_not_called()


def test_fallback_result_enters_research_pipeline_once(monkeypatch):
    from avacore.api import http_app

    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app.settings, "search_provider", "ddg_html", raising=False)
    monkeypatch.setattr(http_app.settings, "search_fallback_provider", "searxng", raising=False)
    monkeypatch.setattr(http_app.settings, "searxng_url", "https://search.example", raising=False)
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    monkeypatch.setattr(web_research.requests, "post", lambda *args, **kwargs:
                        Response(202, "<html>unrecognized</html>"))
    monkeypatch.setattr(web_research.requests, "get", lambda *args, **kwargs:
                        Response(200, payload=RESULT_JSON, content_type="application/json"))
    monkeypatch.setattr(web_research, "fetch_readable_page_text", lambda *args, **kwargs:
                        ("OPC UA", "OPC UA is a platform-independent industrial communication architecture.",
                         None, None, ()))
    chat = MagicMock(return_value="OPC UA is a platform-independent industrial communication architecture.")
    monkeypatch.setattr(http_app.backend, "chat", chat)
    result = http_app.run_research_workflow("Was ist OPC UA?", save_memory=False)
    debug = http_app._last_research_grounding_debug
    assert result["ok"] and result["sources"]
    assert debug["provider_fallback_used"]
    assert debug["search_queries_succeeded"] == 1
    assert debug["provider_success_counts"] == {"searxng": 1}
    assert list(debug["originating_providers"].values()) == [["searxng"]]
    assert chat.call_count == 1


def test_url_dedup_keeps_cross_provider_provenance(monkeypatch):
    from avacore.api import http_app
    from avacore.tools.web_research import ResearchSource

    monkeypatch.setattr(http_app.settings, "research_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_enabled", False)
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    monkeypatch.setattr(http_app.backend, "chat", MagicMock(return_value="Keine sichere Empfehlung."))
    calls = []

    def collect(*, query, diagnostics, **kwargs):
        calls.append(query)
        provider = "ddg_html" if len(calls) <= 2 else "searxng"
        diagnostics.update(status="SEARCH_OK_WITH_RESULTS", result_count_raw=1,
                           final_provider=provider,
                           provider_attempts=[{"provider": provider, "status": "SEARCH_OK_WITH_RESULTS"}])
        return [ResearchSource("Ubuntu 24.04", "https://ubuntu.com/24?utm_source=" + provider,
                               "", "Ubuntu 24.04 released 2024-04 and supported until 2029-04.",
                               originating_queries=(query,), originating_providers=(provider,))]

    monkeypatch.setattr(http_app, "collect_research_sources", collect)
    http_app.run_research_workflow("Ubuntu 24.04 oder Ubuntu 26.04 momentan?", save_memory=False)
    debug = http_app._last_research_grounding_debug
    assert debug["unique_results_seen"] == 1
    assert debug["raw_results_seen"] == 4
    assert len(next(iter(debug["originating_queries"].values()))) == 4
    assert next(iter(debug["originating_providers"].values())) == ["ddg_html", "searxng"]
