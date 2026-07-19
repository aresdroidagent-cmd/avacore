from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv


API_RETRY_COUNT = 12
API_RETRY_DELAY_SECONDS = 5
TELEGRAM_CHUNK_SIZE = 3800


def wait_for_api(
    api_url: str,
    headers: dict[str, str],
) -> None:
    """Wait until the local AvaCore API is ready."""

    last_error: Exception | None = None

    for attempt in range(1, API_RETRY_COUNT + 1):
        try:
            response = requests.get(
                f"{api_url}/health",
                headers=headers,
                timeout=5,
            )
            response.raise_for_status()

            print(
                f"[INFO] AvaCore API ready "
                f"after attempt {attempt}/{API_RETRY_COUNT}."
            )
            return

        except requests.RequestException as exc:
            last_error = exc

            if attempt < API_RETRY_COUNT:
                print(
                    f"[INFO] AvaCore API not ready "
                    f"({attempt}/{API_RETRY_COUNT}): {exc}"
                )
                time.sleep(API_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"AvaCore API did not become ready at {api_url}: {last_error}"
    )


def fetch_mail_digest(
    api_url: str,
    headers: dict[str, str],
    limit: int,
) -> str:
    response = requests.get(
        f"{api_url}/mail/digest",
        params={"limit": limit},
        headers=headers,
        timeout=180,
    )
    response.raise_for_status()

    data = response.json()
    digest = str(data.get("digest", "")).strip()

    if not digest:
        return "Mail-Digest erhalten, aber ohne Inhalt."

    return digest


def send_telegram(
    bot_token: str,
    chat_id: str,
    text: str,
) -> None:
    chunks = [
        text[index:index + TELEGRAM_CHUNK_SIZE]
        for index in range(0, len(text), TELEGRAM_CHUNK_SIZE)
    ]

    if not chunks:
        chunks = ["Mail-Digest erhalten, aber ohne Inhalt."]

    for part_number, chunk in enumerate(chunks, start=1):
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": chunk,
            },
            timeout=30,
        )

        if not response.ok:
            raise RuntimeError(
                "Telegram send failed: "
                f"{response.status_code} {response.text}"
            )

        print(
            f"[INFO] Telegram part "
            f"{part_number}/{len(chunks)} sent."
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    api_url = os.environ.get(
        "AVACORE_API_URL",
        "http://127.0.0.1:8787",
    ).rstrip("/")

    admin_password = os.environ.get(
        "AVACORE_WEB_ADMIN_PASSWORD",
        "",
    ).strip()

    telegram_token = os.environ.get(
        "TELEGRAM_BOT_TOKEN",
        "",
    ).strip()

    telegram_chat_id = os.environ.get(
        "TELEGRAM_ALLOWED_CHAT_ID",
        "",
    ).strip()

    try:
        digest_limit = int(
            os.environ.get("AVACORE_MAIL_DIGEST_LIMIT", "8")
        )
    except ValueError as exc:
        raise RuntimeError(
            "AVACORE_MAIL_DIGEST_LIMIT must be an integer"
        ) from exc

    if not telegram_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    if not telegram_chat_id:
        raise RuntimeError("TELEGRAM_ALLOWED_CHAT_ID is not configured")

    headers: dict[str, str] = {}

    if admin_password:
        headers["X-Admin-Password"] = admin_password

    print(
        "[INFO] Starting daily mail digest at "
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}"
    )

    wait_for_api(
        api_url=api_url,
        headers=headers,
    )

    digest = fetch_mail_digest(
        api_url=api_url,
        headers=headers,
        limit=max(1, digest_limit),
    )

    message = f"📬 Ava Mail-Digest\n\n{digest}"

    send_telegram(
        bot_token=telegram_token,
        chat_id=telegram_chat_id,
        text=message,
    )

    print(
        "[OK] Daily mail digest sent at "
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}"
    )


if __name__ == "__main__":
    main()