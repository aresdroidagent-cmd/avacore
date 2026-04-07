from __future__ import annotations

from email.utils import parsedate_to_datetime
from datetime import timezone
import feedparser


def normalize_entry(entry: dict, source: str) -> dict:
    published = ""
    if entry.get("published"):
        try:
            dt = parsedate_to_datetime(entry["published"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            published = dt.isoformat()
        except Exception:
            published = str(entry.get("published", ""))

    return {
        "title": str(entry.get("title", "")).strip(),
        "link": str(entry.get("link", "")).strip(),
        "summary": str(entry.get("summary", "")).strip(),
        "published": published,
        "source": source,
    }


def fetch_feeds(urls: list[str], limit_per_feed: int = 5) -> list[dict]:
    items: list[dict] = []

    for url in urls:
        feed = feedparser.parse(url)
        source = ""
        try:
            source = str(feed.feed.get("title", "")).strip() or url
        except Exception:
            source = url

        for entry in feed.entries[:limit_per_feed]:
            items.append(normalize_entry(entry, source))

    items.sort(key=lambda x: x.get("published", ""), reverse=True)
    return items
