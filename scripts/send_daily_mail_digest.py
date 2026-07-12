from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

import os
import requests


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    api_url = os.environ.get("AVACORE_API_URL", "http://127.0.0.1:8787").rstrip("/")
    admin_password = os.environ.get("AVACORE_WEB_ADMIN_PASSWORD", "").strip()

    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    telegram_chat_id = os.environ.get("TELEGRAM_ALLOWED_CHAT_ID", "").strip()

    digest_limit = int(os.environ.get("AVACORE_MAIL_DIGEST_LIMIT", "8"))

    if not telegram_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    if not telegram_chat_id:
        raise RuntimeError("TELEGRAM_ALLOWED_CHAT_ID is not configured")

    headers = {}
    if admin_password:
        headers["X-Admin-Password"] = admin_password

    response = requests.get(
        f"{api_url}/mail/digest",
        params={"limit": digest_limit},
        headers=headers,
        timeout=180,
    )
    response.raise_for_status()

    data = response.json()
    digest = data.get("digest", "").strip()

    if not digest:
        digest = "Mail-Digest erhalten, aber ohne Inhalt."

    message = "📬 Ava Mail-Digest\n\n" + digest

    telegram_response = requests.post(
        f"https://api.telegram.org/bot{telegram_token}/sendMessage",
        json={
            "chat_id": telegram_chat_id,
            "text": message,
        },
        timeout=30,
    )
    telegram_response.raise_for_status()

    print("[OK] Daily mail digest sent.")


if __name__ == "__main__":
    main()