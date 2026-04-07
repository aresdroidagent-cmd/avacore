from __future__ import annotations

import subprocess
from pathlib import Path


def run_tesseract(image_path: Path) -> str:
    try:
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except FileNotFoundError:
        return ""
    except subprocess.CalledProcessError:
        return ""
