from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

SUCCESS_STATUSES = {
    "disabled",
    "idle",
    "approval_required",
    "budget_exhausted",
    "completed",
}


def main() -> int:
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
    headers = {"X-Admin-Password": admin_password} if admin_password else {}

    print(
        "[INFO] Autonomous research scheduler started at "
        f"{datetime.now().astimezone().isoformat(timespec='seconds')}"
    )

    try:
        response = requests.post(
            f"{api_url}/research/autonomous/run-next",
            json={},
            headers=headers,
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[ERROR] Autonomous research request failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    status = str(result.get("status") or "failed")
    if status in SUCCESS_STATUSES and result.get("ok", True):
        print(f"[OK] Autonomous research finished with status={status}")
        return 0

    print(
        f"[ERROR] Autonomous research finished with status={status}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
