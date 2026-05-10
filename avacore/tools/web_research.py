from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse
import re

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


@dataclass
class ResearchSource:
    title: str
    url: str
    snippet: str
    text: str
    ok: bool = True
    error: str = ""


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_duckduckgo_url(href: str) -> str:
    if not href:
        return ""

    # DuckDuckGo HTML often returns redirect URLs containing ?uddg=<real-url>
    parsed = urlparse(href)
    query = parse_qs(parsed.query)

    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])

    return href


def search_duckduckgo_html(query: str, max_results: int = 5, timeout: int = 20) -> list[SearchResult]:
    query = query.strip()
    if not query:
        raise ValueError("search query is empty")

    response = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

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
        results.append(SearchResult(title=title, url=url, snippet=snippet))

        if len(results) >= max_results:
            break

    return results


def fetch_readable_page_text(url: str, max_chars: int = 6000, timeout: int = 20) -> tuple[str, str]:
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

    for tag in soup(["script", "style", "noscript", "svg", "canvas", "form"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = _clean_text(soup.title.string)

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = _clean_text(main.get_text(" ", strip=True))

    return title, text[:max_chars]


def collect_research_sources(
    query: str,
    max_results: int = 4,
    max_chars_per_source: int = 5000,
) -> list[ResearchSource]:
    search_results = search_duckduckgo_html(query=query, max_results=max_results)

    sources: list[ResearchSource] = []

    for result in search_results:
        try:
            page_title, page_text = fetch_readable_page_text(
                result.url,
                max_chars=max_chars_per_source,
            )

            sources.append(
                ResearchSource(
                    title=page_title or result.title,
                    url=result.url,
                    snippet=result.snippet,
                    text=page_text,
                    ok=True,
                )
            )

        except Exception as exc:
            sources.append(
                ResearchSource(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    text="",
                    ok=False,
                    error=str(exc),
                )
            )

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