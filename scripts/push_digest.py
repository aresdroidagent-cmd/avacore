from __future__ import annotations

import argparse
import os
import requests

from avacore.config.settings import settings


def api_base() -> str:
    return f"http://{settings.http_host}:{settings.http_port}"


def get_digest(kind: str) -> str:
    if kind not in {"newsdigest", "mediumdigest"}:
        raise ValueError(f"Unsupported digest kind: {kind}")

    response = requests.get(f"{api_base()}/tools/{kind}", params={"limit": 5}, timeout=180)
    response.raise_for_status()
    data = response.json()
    return str(data.get("digest", "")).strip()


def send_telegram_message(text: str) -> None:
    token = settings.telegram_bot_token
    chat_id = settings.telegram_allowed_chat_id

    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")
    if not chat_id:
        raise RuntimeError("Missing TELEGRAM_ALLOWED_CHAT_ID in .env")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text[:4000],
        },
        timeout=30,
    )
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["newsdigest", "mediumdigest"])
    args = parser.parse_args()

    digest = get_digest(args.kind)
    if not digest:
        digest = f"Kein Inhalt für {args.kind} verfügbar."

    title = "News Digest" if args.kind == "newsdigest" else "Medium Digest"
    message = f"{title}\n\n{digest}"
    send_telegram_message(message)


if __name__ == "__main__":
    main()
