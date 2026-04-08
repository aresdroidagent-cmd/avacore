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
    "Beschreibe dieses Bild als Screenshot, textlastige Oberfläche oder Folie. "
    "Nenne kurz die Art der Oberfläche, sichtbare Titel, wichtige Texte, Meldungen "
    "und auffällige Bedienelemente oder Strukturen. "
    "Wenn das Bild eher eine Präsentationsfolie, ein Poster oder eine textlastige Grafik ist, sage das klar. "
    "Antworte kurz, sachlich und ohne Ausschmückung."
)

DIAGRAM_PROMPT = (
    "Beschreibe dieses Bild als Diagramm oder technische Grafik. "
    "Nenne kurz die Art der Darstellung, sichtbare Blöcke, Verbindungen, Achsen, Labels "
    "oder andere strukturelle Elemente. "
    "Wenn Text schwer lesbar ist, beschreibe trotzdem die visuelle Struktur. "
    "Antworte kurz und sachlich."
)

PHOTO_PROMPT = (
    "Beschreibe dieses Bild kurz und technisch. "
    "Nenne die wichtigsten sichtbaren Objekte, deren Anordnung und eine mögliche Funktion oder Bedeutung. "
    "Wenn das Bild eher eine Illustration, ein Gemälde, ein Poster oder ein Cover als ein echtes Foto zeigt, "
    "sage das ausdrücklich."
)

ASSEMBLY_PROMPT = (
    "Beschreibe dieses Bild als technische Montageszene. "
    "Nenne sichtbare Bauteile, Handlungen, Werkzeuge oder Verbindungselemente "
    "und den wahrscheinlichen Montageschritt. "
    "Wenn unsicher, sage das klar."
)

ARTWORK_PROMPT = (
    "Beschreibe dieses Bild als Illustration, Gemälde, Konzeptgrafik, Cover oder Poster. "
    "Nenne kurz Motiv, Stilcharakter, dominante Bildelemente und sichtbare Texte oder Titel. "
    "Antworte sachlich und ohne Ausschmückung."
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


def _safe_open_image(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return (0, 0)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip().lower()


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    return sum(1 for token in keywords if token in text)


def choose_prompt(
    image_path: Path,
    ocr_text: str = "",
    explicit_mode: str | None = None,
    page_text: str = "",
) -> tuple[str, str]:
    explicit = (explicit_mode or "").strip().lower()
    if explicit == "screen":
        return "screen", SCREEN_UI_PROMPT
    if explicit == "diagram":
        return "diagram", DIAGRAM_PROMPT
    if explicit == "photo":
        return "photo", PHOTO_PROMPT
    if explicit == "assembly":
        return "assembly", ASSEMBLY_PROMPT
    if explicit == "artwork":
        return "artwork", ARTWORK_PROMPT

    name = image_path.name.lower()
    ocr = _normalize_text(ocr_text)
    page = _normalize_text(page_text)
    combined = f"{name} {ocr} {page}".strip()

    width, height = _safe_open_image(image_path)
    aspect_ratio = (width / height) if width and height else 1.0

    screen_name_hits = [
        "screenshot", "screen", "ui", "dialog", "window", "dashboard",
        "settings", "config", "terminal", "console", "slide", "folie",
        "presentation", "ppt", "poster", "cover"
    ]
    diagram_hits = [
        "diagram", "schema", "schematic", "flow", "chart", "graph",
        "plot", "circuit", "wiring", "architecture", "architektur",
        "block", "signal", "input", "output", "legend", "axis"
    ]
    assembly_hits = [
        "assembly", "montage", "aufbau", "installation", "install",
        "step", "schritt", "attach", "fasten", "screw", "bolt",
        "washer", "bracket", "mount", "joint", "robot arm"
    ]
    artwork_hits = [
        "painting", "painted", "art", "artwork", "illustration", "illustrated",
        "poster", "cover", "concept art", "render", "comic", "sketch"
    ]

    screen_score = _count_keyword_hits(combined, screen_name_hits)
    diagram_score = _count_keyword_hits(combined, diagram_hits)
    assembly_score = _count_keyword_hits(combined, assembly_hits)
    artwork_score = _count_keyword_hits(combined, artwork_hits)

    ocr_words = ocr.split()
    ocr_len = len(ocr)
    ocr_word_count = len(ocr_words)

    if re.search(r"(error|warning|failed|exception|traceback|stack|terminal|sudo|apt|python|bash)", ocr):
        screen_score += 4

    if re.search(r"(axis|legend|voltage|current|signal|input|output|block|node|chart|flow|diagram)", ocr):
        diagram_score += 4

    if re.search(r"(screw|bolt|washer|bracket|mount|assembly|install|step|robot arm|joint|fasten)", combined):
        assembly_score += 4

    if re.search(r"(painting|illustration|poster|cover|artwork|concept art|oil on canvas)", combined):
        artwork_score += 4

    if ocr_word_count >= 8 or ocr_len >= 40:
        screen_score += 2

    if ocr_word_count >= 20 or ocr_len >= 120:
        screen_score += 3

    if aspect_ratio > 1.6 and ocr_word_count >= 6:
        screen_score += 1

    if aspect_ratio > 1.3 and diagram_score >= 2:
        diagram_score += 1

    if page and re.search(r"(figure|abbildung|diagramm|schema|flow|chart)", page):
        diagram_score += 2

    if page and re.search(r"(assembly|montage|aufbau|installation|schritt)", page):
        assembly_score += 2

    scores = {
        "screen": screen_score,
        "diagram": diagram_score,
        "assembly": assembly_score,
        "artwork": artwork_score,
        "photo": 0,
    }

    best_mode = max(scores, key=scores.get)
    best_score = scores[best_mode]

    if best_score <= 0:
        return "photo", PHOTO_PROMPT

    if best_mode == "screen":
        return "screen", SCREEN_UI_PROMPT
    if best_mode == "diagram":
        return "diagram", DIAGRAM_PROMPT
    if best_mode == "assembly":
        return "assembly", ASSEMBLY_PROMPT
    if best_mode == "artwork":
        return "artwork", ARTWORK_PROMPT

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
        default_prompt = (settings.vision_prompt or "").strip()
        effective_prompt = default_prompt if default_prompt else base_prompt
        final_prompt = build_contextual_prompt(effective_prompt, page_text=page_text)

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