from __future__ import annotations

from pathlib import Path
import re

from PIL import Image

from avacore.config.settings import settings
from avacore.vision.smolvlm_client import SmolVLMClient


_client: SmolVLMClient | None = None
_client_failed: bool = False
_client_error: str = ""


SCREEN_UI_PROMPT = (
    "Analysiere dieses Bild technisch und sachlich. "
    "Wenn es ein Screenshot, UI, Terminal, Dialogfenster, Dashboard oder Konfigurationsfenster ist, beschreibe: "
    "1. sichtbare Anwendung oder Oberfläche, "
    "2. wichtige Texte, Meldungen, Fehlermeldungen oder Statusanzeigen, "
    "3. erkennbare Eingabefelder, Buttons, Menüs oder Bedienelemente, "
    "4. mögliche technische Bedeutung oder Auffälligkeiten. "
    "Antworte kompakt, präzise und ohne Ausschmückung."
)

DIAGRAM_PROMPT = (
    "Analysiere dieses Bild als technisches Diagramm, Schema, Schaltbild, Architekturzeichnung oder Ablaufgrafik. "
    "Beschreibe: "
    "1. die Art der Darstellung, "
    "2. erkennbare Blöcke, Verbindungen, Signalflüsse, Achsen, Labels oder Legenden, "
    "3. technische Kernaussage oder Struktur, "
    "4. auffällige Werte, Bezeichnungen oder Komponenten. "
    "Wenn Text schwer lesbar ist, beschreibe die visuelle Struktur trotzdem. "
    "Antworte kompakt, technisch und sachlich."
)

PHOTO_PROMPT = (
    "Beschreibe dieses Bild technisch und sachlich. "
    "Wenn es ein Foto eines Geräts, Roboters, Aufbaus, Objekts oder einer realen Szene ist, nenne: "
    "1. sichtbare Hauptobjekte oder Komponenten, "
    "2. räumliche Anordnung, "
    "3. mögliche technische Funktion oder Nutzung, "
    "4. auffällige Details oder Zustände. "
    "Antworte kompakt und präzise."
)

ASSEMBLY_PROMPT = (
    "Analysiere dieses Bild als Montageszene aus einer technischen Bauanleitung. "
    "Beschreibe: "
    "1. sichtbare Bauteile oder Baugruppen, "
    "2. sichtbare Handaktionen oder Interaktionen, "
    "3. mögliche Werkzeuge, Schrauben, Halterungen oder Verbindungselemente, "
    "4. den wahrscheinlichen Montageschritt im Aufbauprozess, "
    "5. Unsicherheiten klar benennen. "
    "Antworte kompakt, technisch und sachlich."
)


def get_vision_client() -> SmolVLMClient:
    global _client, _client_failed, _client_error

    if _client is not None:
        return _client

    if _client_failed:
        raise RuntimeError(f"Vision client unavailable: {_client_error}")

    try:
        _client = SmolVLMClient(
            model_name=settings.vision_model,
            max_new_tokens=settings.vision_max_new_tokens,
        )
        return _client
    except Exception as exc:
        _client_failed = True
        _client_error = str(exc)
        raise


def is_image_large_enough(image_path: Path) -> bool:
    try:
        img = Image.open(image_path)
        w, h = img.size
        return (w * h) >= settings.vision_min_image_pixels
    except Exception:
        return False


def choose_prompt(
    image_path: Path,
    ocr_text: str = "",
    explicit_mode: str | None = None,
    page_text: str = "",
) -> tuple[str, str]:
    if explicit_mode == "screen":
        return "screen", SCREEN_UI_PROMPT
    if explicit_mode == "diagram":
        return "diagram", DIAGRAM_PROMPT
    if explicit_mode == "photo":
        return "photo", PHOTO_PROMPT
    if explicit_mode == "assembly":
        return "assembly", ASSEMBLY_PROMPT

    name = image_path.name.lower()
    ocr = (ocr_text or "").lower()
    page = (page_text or "").lower()

    screen_name_hits = [
        "bildschirmfoto", "screenshot", "screen", "terminal", "dialog",
        "window", "config", "settings", "dashboard", "ui"
    ]
    diagram_name_hits = [
        "diagram", "schema", "schematic", "block", "flow", "chart",
        "plot", "graph", "circuit", "wiring", "architektur", "architecture"
    ]
    assembly_hits = [
        "assembly", "montage", "aufbau", "install", "installation",
        "step", "schritt", "attach", "fasten", "screw", "bolt",
        "robot arm", "annin", "joint", "bracket", "mount"
    ]

    if any(token in name for token in screen_name_hits):
        return "screen", SCREEN_UI_PROMPT

    if any(token in name for token in diagram_name_hits):
        return "diagram", DIAGRAM_PROMPT

    if any(token in name for token in assembly_hits):
        return "assembly", ASSEMBLY_PROMPT

    if re.search(r"(error|warning|failed|exception|traceback|stack|terminal|sudo|apt|python|bash)", ocr):
        return "screen", SCREEN_UI_PROMPT

    if re.search(r"(axis|legend|voltage|current|signal|input|output|block|node|chart|flow)", ocr):
        return "diagram", DIAGRAM_PROMPT

    if re.search(r"(screw|bolt|washer|bracket|mount|assembly|install|step|robot arm|joint|fasten)", ocr + " " + page):
        return "assembly", ASSEMBLY_PROMPT

    return "photo", PHOTO_PROMPT


def build_contextual_prompt(
    base_prompt: str,
    page_text: str = "",
) -> str:
    page_text = " ".join((page_text or "").split()).strip()
    if not page_text:
        return base_prompt

    page_text = page_text[:1200]
    return (
        f"{base_prompt}\n\n"
        f"Zusätzlicher PDF-Seitenkontext:\n{page_text}\n\n"
        f"Nutze diesen Kontext nur unterstützend. "
        f"Wenn die Bildaussage unsicher ist, sage das klar."
    )


def describe_image_with_smolvlm(
    image_path: Path,
    prompt: str | None = None,
    ocr_text: str = "",
    mode: str | None = None,
    page_text: str = "",
) -> str:
    if not settings.vision_enabled:
        return ""

    if not is_image_large_enough(image_path):
        return ""

    client = get_vision_client()

    if prompt:
        final_prompt = prompt
    else:
        _, base_prompt = choose_prompt(
            image_path=image_path,
            ocr_text=ocr_text,
            explicit_mode=mode,
            page_text=page_text,
        )
        final_prompt = build_contextual_prompt(base_prompt, page_text=page_text)

    return client.describe_image(
        image_path=image_path,
        prompt=final_prompt,
    )


def detect_image_mode(
    image_path: Path,
    ocr_text: str = "",
    page_text: str = "",
) -> str:
    mode, _ = choose_prompt(
        image_path=image_path,
        ocr_text=ocr_text,
        page_text=page_text,
    )
    return mode