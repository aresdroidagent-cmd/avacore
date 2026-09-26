from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import time
from urllib.parse import parse_qs, unquote, urlparse
import re
from typing import Protocol

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36 AvaCore/0.8"
)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    provider: str = "ddg_html"
    originating_query: str = ""
    rank: int = 0
    originating_requirement_ids: tuple[str, ...] = ()


@dataclass
class SearchProviderResult:
    provider: str
    status: str
    http_status: int | None
    results: list[SearchResult]
    duration_ms: int
    retry_count: int
    error_type: str | None
    diagnostics: dict


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, limit: int) -> SearchProviderResult: ...


@dataclass
class ResearchSource:
    title: str
    url: str
    snippet: str
    text: str
    ok: bool = True
    error: str = ""
    retrieved_at: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    source_type: str = "web"
    time_markers: tuple[str, ...] = ()
    originating_queries: tuple[str, ...] = ()
    originating_providers: tuple[str, ...] = ()
    originating_requirement_ids: tuple[str, ...] = ()


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _safe_response_fingerprint(response, detected_page_type: str) -> dict:
    body = getattr(response, "content", None)
    if body is None:
        body = str(getattr(response, "text", "")).encode("utf-8", errors="replace")
    visible = BeautifulSoup(str(getattr(response, "text", "")), "html.parser")
    for tag in visible(["script", "style", "form", "input", "noscript"]):
        tag.decompose()
    visible_prefix = _clean_text(visible.get_text(" ", strip=True))[:200].casefold()
    safe_words = {"captcha", "challenge", "verify", "verification", "human", "robot", "consent",
                  "privacy", "redirect", "search", "results", "error", "unusual", "traffic",
                  "access", "denied", "rate", "limited", "waiting", "please", "browser"}
    # A fixed vocabulary prevents arbitrary response text (including unknown secrets) entering debug.
    prefix = " ".join(word for word in re.findall(r"[a-z]+", visible_prefix)
                      if word in safe_words)[:200]
    content_type = str(getattr(response, "headers", {}).get("content-type", "")).split(";", 1)[0].strip().lower()
    if not re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", content_type):
        content_type = "unknown"
    return {"content_type": content_type,
            "body_length": len(body), "body_sha256": hashlib.sha256(body).hexdigest(),
            "safe_body_prefix": prefix[:200], "detected_page_type": detected_page_type}


def _page_type(soup: BeautifulSoup) -> str:
    if soup.select(".result"):
        return "search_results"
    text = soup.get_text(" ", strip=True).casefold()
    if re.search(r"captcha|verify you are human|are you a robot|unusual traffic", text):
        return "challenge"
    if re.search(r"no results|keine ergebnisse|no search results", text):
        return "empty_results"
    if soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)}):
        return "redirect"
    if re.search(r"consent|privacy preferences|cookie preferences", text):
        return "consent"
    return "unknown"


def _extract_duckduckgo_url(href: str) -> str:
    if not href:
        return ""

    # DuckDuckGo HTML often returns redirect URLs containing ?uddg=<real-url>
    parsed = urlparse(href)
    query = parse_qs(parsed.query)

    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])

    return href


def search_duckduckgo_html(query: str, max_results: int = 5, timeout: int = 20,
                           diagnostics: dict | None = None) -> list[SearchResult]:
    query = query.strip()
    if not query:
        raise ValueError("search query is empty")

    started = time.monotonic()
    response = None
    for attempt in range(2):
        try:
            response = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query}, headers={"User-Agent": USER_AGENT}, timeout=timeout,
            )
            status = response.status_code
            if status in {429, 500, 502, 503, 504} and attempt == 0:
                time.sleep(0.1)
                continue
            response.raise_for_status()
            break
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 0:
                time.sleep(0.1)
                continue
            if diagnostics is not None:
                diagnostics.update(status="SEARCH_TIMEOUT" if isinstance(exc, requests.Timeout)
                                   else "SEARCH_PROVIDER_ERROR", error_type=type(exc).__name__,
                                   http_status=None, retry_count=attempt,
                                   request_duration_ms=round((time.monotonic() - started) * 1000))
            raise
        except requests.HTTPError as exc:
            if diagnostics is not None:
                diagnostics.update(status="SEARCH_RATE_LIMITED" if response.status_code == 429
                                   else "SEARCH_HTTP_ERROR", error_type=type(exc).__name__,
                                   http_status=response.status_code, retry_count=attempt,
                                   request_duration_ms=round((time.monotonic() - started) * 1000))
            raise

    assert response is not None
    if diagnostics is not None:
        diagnostics.update(status="SEARCH_OK_EMPTY", error_type=None,
                           http_status=response.status_code, retry_count=attempt,
                           request_duration_ms=round((time.monotonic() - started) * 1000))

    soup = BeautifulSoup(response.text, "html.parser")
    page_type = _page_type(soup)

    results: list[SearchResult] = []
    seen_urls: set[str] = set()

    for result in soup.select(".result"):
        link = result.select_one("a.result__a")
        if not link:
            continue

        title = _clean_text(link.get_text(" ", strip=True))
        url = _extract_duckduckgo_url(link.get("href", ""))

        if not title or not url:
            continue

        if not url.startswith(("http://", "https://")):
            continue

        if url in seen_urls:
            continue

        snippet_node = result.select_one(".result__snippet")
        snippet = _clean_text(snippet_node.get_text(" ", strip=True)) if snippet_node else ""

        seen_urls.add(url)
        results.append(SearchResult(title=title, url=url, snippet=snippet,
                                    originating_query=query, rank=len(results) + 1))

        if len(results) >= max_results:
            break

    recognized = response.status_code == 200 and (bool(results) or page_type == "empty_results")
    if diagnostics is not None:
        if recognized:
            diagnostics["status"] = "SEARCH_OK_WITH_RESULTS" if results else "SEARCH_OK_EMPTY"
        else:
            diagnostics.update(status="SEARCH_PARSE_ERROR", error_type="unrecognized_search_html")
            diagnostics.update(_safe_response_fingerprint(response, page_type))
        diagnostics["result_count_raw"] = len(results) if recognized else 0
    if not recognized:
        if diagnostics is None:
            raise ValueError("unrecognized_search_html")
        return []
    return results


class DdgHtmlProvider:
    name = "ddg_html"

    def search(self, query: str, limit: int) -> SearchProviderResult:
        diagnostics: dict = {}
        try:
            results = search_duckduckgo_html(query, max_results=limit, diagnostics=diagnostics)
        except (requests.RequestException, ValueError) as exc:
            results = []
            diagnostics.setdefault("status", "SEARCH_PROVIDER_ERROR")
            diagnostics.setdefault("error_type", type(exc).__name__)
        return SearchProviderResult(self.name, diagnostics["status"], diagnostics.get("http_status"),
                                    results, diagnostics.get("request_duration_ms", 0),
                                    diagnostics.get("retry_count", 0), diagnostics.get("error_type"),
                                    diagnostics)


class SearxNGProvider:
    name = "searxng"

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, limit: int) -> SearchProviderResult:
        started = time.monotonic()
        diagnostics: dict = {}
        response = None
        try:
            response = requests.get(f"{self.base_url}/search", params={"q": query, "format": "json"},
                                    headers={"User-Agent": USER_AGENT}, timeout=20)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                raise ValueError("unrecognized_search_json")
            results = []
            for item in data["results"]:
                if not isinstance(item, dict):
                    continue
                title, url = item.get("title"), item.get("url")
                if not isinstance(title, str) or not isinstance(url, str) or not url.startswith(("https://", "http://")):
                    continue
                results.append(SearchResult(_clean_text(title)[:180], url,
                                            _clean_text(str(item.get("content") or ""))[:500],
                                            self.name, query, len(results) + 1))
                if len(results) >= limit:
                    break
            status = "SEARCH_OK_WITH_RESULTS" if results else "SEARCH_OK_EMPTY"
            if data["results"] and not results:
                status = "SEARCH_PARSE_ERROR"
                diagnostics["error_type"] = "unrecognized_search_json"
        except requests.Timeout as exc:
            results, status = [], "SEARCH_TIMEOUT"
            diagnostics["error_type"] = type(exc).__name__
        except requests.HTTPError as exc:
            results = []
            status = "SEARCH_RATE_LIMITED" if response is not None and response.status_code == 429 else "SEARCH_HTTP_ERROR"
            diagnostics["error_type"] = type(exc).__name__
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            results, status = [], "SEARCH_PARSE_ERROR" if isinstance(exc, (ValueError, json.JSONDecodeError)) else "SEARCH_PROVIDER_ERROR"
            diagnostics["error_type"] = type(exc).__name__
        duration = round((time.monotonic() - started) * 1000)
        diagnostics.update(status=status, http_status=response.status_code if response is not None else None,
                           request_duration_ms=duration, retry_count=0, result_count_raw=len(results))
        if status == "SEARCH_PARSE_ERROR" and response is not None:
            diagnostics.update(_safe_response_fingerprint(response, "unrecognized_json"))
        return SearchProviderResult(self.name, status, diagnostics["http_status"], results,
                                    duration, 0, diagnostics.get("error_type"), diagnostics)


def _provider(name: str, searxng_url: str | None) -> SearchProvider | None:
    if name == "ddg_html":
        return DdgHtmlProvider()
    if name == "searxng" and searxng_url and urlparse(searxng_url).scheme in {"http", "https"}:
        return SearxNGProvider(searxng_url)
    return None


def search_with_fallback(query: str, limit: int, *, primary_name: str = "ddg_html",
                         fallback_name: str | None = None,
                         searxng_url: str | None = None) -> tuple[SearchProviderResult, dict]:
    primary = _provider(primary_name, searxng_url)
    if primary is None:
        failure = SearchProviderResult(primary_name, "SEARCH_PROVIDER_ERROR", None, [], 0, 0,
                                       "provider_not_configured", {})
        return failure, {"provider_attempts": [failure], "provider_fallback_used": False,
                         "provider_fallback_reason": "NO_FALLBACK_PROVIDER_CONFIGURED"}
    first = primary.search(query, limit)
    attempts = [first]
    fallback_used = False
    fallback_reason = None
    final = first
    if not first.status.startswith("SEARCH_OK"):
        secondary = _provider(fallback_name, searxng_url) if fallback_name and fallback_name != primary_name else None
        if secondary is None:
            fallback_reason = "NO_FALLBACK_PROVIDER_CONFIGURED"
        else:
            fallback_used = True
            fallback_reason = first.status
            final = secondary.search(query, limit)
            attempts.append(final)
    return final, {"provider_attempts": attempts, "provider_fallback_used": fallback_used,
                   "provider_fallback_reason": fallback_reason}


def fetch_readable_page_text(url: str, max_chars: int = 6000, timeout: int = 20,
                             include_metadata: bool = False,
                             include_time_markers: bool = False) -> tuple:
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        raise ValueError(f"unsupported content type: {content_type}")

    soup = BeautifulSoup(response.text, "html.parser")

    schema_dates: dict[str, str] = {}
    for tag in soup.select('script[type="application/ld+json"]')[:4]:
        try:
            data = json.loads(tag.string or tag.get_text())
        except (ValueError, TypeError):
            continue
        nodes = data if isinstance(data, list) else [data]
        for node in nodes:
            if isinstance(node, dict):
                nodes.extend(x for x in node.get("@graph", []) if isinstance(x, dict))
                for key in ("datePublished", "dateModified"):
                    if isinstance(node.get(key), str):
                        schema_dates.setdefault(key, node[key][:80])

    for tag in soup(["script", "style", "noscript", "svg", "canvas", "form",
                     "nav", "header", "footer", "aside", "menu"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = _clean_text(soup.title.string)

    main = soup.find("main") or soup.find("article") or soup.body or soup
    markers = tuple(dict.fromkeys(str(tag.get("datetime"))[:80] for tag in main.find_all("time")
                                  if tag.get("datetime")))[:8]
    blocks = main.find_all(["p", "h1", "h2", "h3", "li", "time", "tr"])
    # Keep table rows intact; nested cells must not become separate facts.
    def block_text(tag) -> str:
        if tag.name != "tr":
            return _clean_text(tag.get_text(" ", strip=True))
        cells = tag.find_all(["td", "th"], recursive=False)
        values = [_clean_text(cell.get_text(" ", strip=True)) for cell in cells]
        table = tag.find_parent("table")
        header = table.find("tr") if table else None
        if header is not None and header is not tag:
            labels = [_clean_text(cell.get_text(" ", strip=True))
                      for cell in header.find_all("th", recursive=False)]
            if len(labels) == len(values):
                return " | ".join(f"{label}: {value}" for label, value in zip(labels, values))
        return " | ".join(values)

    text = "\n".join(block_text(tag) for tag in blocks
                     if tag.get_text(" ", strip=True)) if blocks else _clean_text(main.get_text(" ", strip=True))

    if include_metadata:
        def metadata_value(*names: str) -> str | None:
            for name in names:
                tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name})
                if tag and tag.get("content"):
                    return str(tag["content"])[:80]
            return None
        published = metadata_value("article:published_time", "datePublished", "publish_date") or schema_dates.get("datePublished")
        updated = metadata_value("article:modified_time", "dateModified", "lastmod") or schema_dates.get("dateModified")
        if include_time_markers:
            return title, text[:max_chars], published, updated, markers
        return title, text[:max_chars], published, updated
    return title, text[:max_chars]


def collect_research_sources(
    query: str,
    max_results: int = 4,
    max_chars_per_source: int = 5000,
    diagnostics: dict | None = None,
    provider_name: str = "ddg_html",
    fallback_provider_name: str | None = None,
    searxng_url: str | None = None,
    requirement_ids: tuple[str, ...] = (),
) -> list[ResearchSource]:
    final, acquisition = search_with_fallback(query, max_results, primary_name=provider_name,
                                              fallback_name=fallback_provider_name,
                                              searxng_url=searxng_url)
    if diagnostics is None and not final.status.startswith("SEARCH_OK"):
        raise RuntimeError(f"search acquisition failed: {final.status}")
    search_results = final.results
    if diagnostics is not None:
        diagnostics.update(final.diagnostics)
        diagnostics.update(status=final.status, error_type=final.error_type,
                           http_status=final.http_status, request_duration_ms=final.duration_ms,
                           retry_count=final.retry_count, result_count_raw=len(search_results),
                           final_provider=final.provider,
                           provider_attempts=[{"provider": attempt.provider, "status": attempt.status,
                                               "http_status": attempt.http_status,
                                               "retry_count": attempt.retry_count,
                                               "error_type": attempt.error_type,
                                               **{key: attempt.diagnostics[key] for key in
                                                  ("content_type", "body_length", "body_sha256",
                                                   "safe_body_prefix", "detected_page_type")
                                                  if key in attempt.diagnostics}}
                                              for attempt in acquisition["provider_attempts"]],
                           provider_fallback_used=acquisition["provider_fallback_used"],
                           provider_fallback_reason=acquisition["provider_fallback_reason"])
        diagnostics["fetch_attempted"] = len(search_results)
        diagnostics["fetch_succeeded"] = 0
        diagnostics["fetch_failed"] = 0

    sources: list[ResearchSource] = []

    for result in search_results:
        try:
            page_title, page_text, published, updated, markers = fetch_readable_page_text(
                result.url,
                max_chars=max_chars_per_source,
                include_metadata=True,
                include_time_markers=True,
            )
            if not page_text.strip():
                raise ValueError("empty readable page")

            sources.append(
                ResearchSource(
                    title=page_title or result.title,
                    url=result.url,
                    snippet=result.snippet,
                    text=page_text,
                    ok=True,
                    retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    published_at=published,
                    updated_at=updated,
                    time_markers=markers,
                    originating_queries=(result.originating_query or query,),
                    originating_providers=(result.provider,),
                    originating_requirement_ids=requirement_ids,
                )
            )
            if diagnostics is not None:
                diagnostics["fetch_succeeded"] += 1

        except Exception as exc:
            sources.append(
                ResearchSource(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    text="",
                    ok=False,
                    error=str(exc),
                    retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    originating_queries=(result.originating_query or query,),
                    originating_providers=(result.provider,),
                    originating_requirement_ids=requirement_ids,
                )
            )
            if diagnostics is not None:
                diagnostics["fetch_failed"] += 1

    return sources


def build_research_context(query: str, sources: list[ResearchSource]) -> str:
    parts = [f"Recherchefrage: {query}", ""]

    for index, source in enumerate(sources, start=1):
        parts.append(f"Quelle {index}: {source.title}")
        parts.append(f"URL: {source.url}")

        if source.snippet:
            parts.append(f"Such-Snippet: {source.snippet}")

        if source.ok and source.text:
            parts.append("Seitentext:")
            parts.append(source.text)
        else:
            parts.append(f"Quelle konnte nicht gelesen werden: {source.error}")

        parts.append("")

    return "\n".join(parts)


def serialize_sources(sources: list[ResearchSource]) -> list[dict]:
    return [
        {
            "title": source.title,
            "url": source.url,
            "snippet": source.snippet,
            "ok": source.ok,
            "error": source.error,
            "chars": len(source.text or ""),
        }
        for source in sources
    ]
