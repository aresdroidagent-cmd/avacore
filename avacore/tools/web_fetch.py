from __future__ import annotations

import re
import requests


def html_to_text(html: str) -> str:
    # Scripts/Styles raus
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)

    # Tags entfernen
    text = re.sub(r"(?s)<[^>]+>", " ", html)

    # HTML entities sehr grob
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )

    # Whitespace normalisieren
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_url_text(url: str, timeout: int = 20) -> str:
    headers = {
        "User-Agent": "AvaCore/0.8 (+local assistant)"
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    ctype = response.headers.get("content-type", "")
    if "text/html" not in ctype and "text/plain" not in ctype:
        raise RuntimeError(f"Unsupported content-type: {ctype}")

    if "text/plain" in ctype:
        return response.text.strip()

    return html_to_text(response.text)
