from __future__ import annotations

from pathlib import Path
import hashlib

from avacore.rag.ocr import run_tesseract


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ingest_image(path: Path, ocr_enabled: bool = True) -> dict:
    checksum = file_checksum(path)
    title = path.stem
    caption = f"Bilddatei: {path.name}"
    ocr_text = run_tesseract(path) if ocr_enabled else ""

    return {
        "title": title,
        "caption": caption,
        "ocr_text": ocr_text.strip(),
        "checksum": checksum,
    }
