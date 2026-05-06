from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

CACHE_DIR = Path("./data/cache/camera")
RETENTION_DAYS = 7

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    if not CACHE_DIR.exists():
        print(f"[OK] Cache directory does not exist: {CACHE_DIR}")
        return

    cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)
    deleted = 0

    for path in CACHE_DIR.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        modified = datetime.fromtimestamp(path.stat().st_mtime)

        if modified < cutoff:
            path.unlink()
            deleted += 1
            print(f"[DEL] {path}")

    print(f"[OK] Deleted {deleted} old camera cache file(s).")


if __name__ == "__main__":
    main()