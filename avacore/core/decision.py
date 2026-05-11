from __future__ import annotations

from dataclasses import asdict, dataclass
import re


@dataclass
class ContextDecision:
    needs_memory: bool = True
    needs_rag: bool = False
    needs_research: bool = False
    needs_calendar: bool = False
    needs_camera: bool = False
    save_memory_candidate: bool = False
    confidence: float = 0.5
    reason: str = "default local answer"

    def to_dict(self) -> dict:
        return asdict(self)


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def decide_context(user_text: str) -> ContextDecision:
    text = (user_text or "").strip().lower()

    # ------------------------------------------------------------
    # Runtime / identity questions
    # These must never trigger web research.
    # Ava knows these from Shared Brain + Runtime Context.
    # ------------------------------------------------------------
    runtime_keywords = [
        "welches datum",
        "was ist heute",
        "welcher tag",
        "heutiges datum",
        "datum heute",
        "wie spät",
        "uhrzeit",
        "welche zeit",
        "current date",
        "what date",
        "what time",
        "today",
        "heute",
    ]

    identity_keywords = [
        "wie heisst du",
        "wie heißt du",
        "wie ist dein name",
        "wer bist du",
        "wer hat dich erschaffen",
        "wer ist dein schöpfer",
        "wer ist dein vater",
        "wer ist roger",
        "who are you",
        "what is your name",
        "who created you",
        "who is your creator",
        "who is your father",
    ]

    location_keywords = [
        "wo bist du",
        "wo befindest du dich",
        "wo läufst du",
        "wo ist dein standort",
        "dein standort",
        "where are you",
        "your location",
    ]

    if any(keyword in text for keyword in runtime_keywords):
        return ContextDecision(
            needs_memory=True,
            needs_rag=False,
            needs_research=False,
            needs_calendar=False,
            needs_camera=False,
            save_memory_candidate=False,
            confidence=0.95,
            reason="question can be answered from runtime context such as current date, time or timezone",
        )

    if any(keyword in text for keyword in identity_keywords):
        return ContextDecision(
            needs_memory=True,
            needs_rag=False,
            needs_research=False,
            needs_calendar=False,
            needs_camera=False,
            save_memory_candidate=False,
            confidence=0.95,
            reason="question can be answered from Ava identity / Shared Brain context",
        )

    if any(keyword in text for keyword in location_keywords):
        return ContextDecision(
            needs_memory=True,
            needs_rag=False,
            needs_research=False,
            needs_calendar=False,
            needs_camera=False,
            save_memory_candidate=False,
            confidence=0.9,
            reason="question can be answered from local runtime/location context",
        )

    # ------------------------------------------------------------
    # Explicit tool intents
    # ------------------------------------------------------------
    calendar_keywords = [
        "kalender",
        "termin",
        "termine",
        "agenda",
        "briefing",
        "tagesaufgaben",
        "heute vor",
    ]

    camera_keywords = [
        "kamera",
        "webcam",
        "bild",
        "snapshot",
        "sehen",
        "siehst du",
    ]

    if any(keyword in text for keyword in calendar_keywords):
        return ContextDecision(
            needs_memory=True,
            needs_rag=False,
            needs_research=False,
            needs_calendar=True,
            needs_camera=False,
            save_memory_candidate=False,
            confidence=0.85,
            reason="question likely requires calendar context",
        )

    if any(keyword in text for keyword in camera_keywords):
        return ContextDecision(
            needs_memory=True,
            needs_rag=False,
            needs_research=False,
            needs_calendar=False,
            needs_camera=True,
            save_memory_candidate=False,
            confidence=0.85,
            reason="question likely requires camera/snapshot context",
        )

    # ------------------------------------------------------------
    # Research intent
    # Only external/current unknown facts should trigger research.
    # Important: 'heute' alone is NOT enough. It was already handled
    # above as runtime context.
    # ------------------------------------------------------------
    research_keywords = [
        "suche im web",
        "recherchiere",
        "websuche",
        "internet",
        "online",
        "aktuelle version",
        "neuste version",
        "neueste version",
        "latest",
        "wo kaufen",
        "preis",
        "preise",
        "lieferbar",
        "bestellen",
        "gesetz",
        "news",
        "nachrichten",
        "release",
        "changelog",
        "hersteller",
    ]

    if any(keyword in text for keyword in research_keywords):
        return ContextDecision(
            needs_memory=True,
            needs_rag=False,
            needs_research=True,
            needs_calendar=False,
            needs_camera=False,
            save_memory_candidate=True,
            confidence=0.85,
            reason="question likely requires current or external web information",
        )

    # ------------------------------------------------------------
    # RAG / local project knowledge
    # ------------------------------------------------------------
    rag_keywords = [
        "dokument",
        "pdf",
        "manual",
        "seite",
        "wissensbasis",
        "rag",
        "avacore",
        "ar4",
        "isaac",
        "opencv",
        "rtsp",
        "kamera",
        "ollama",
        "telegram",
        "repo",
        "readme",
        "setup",
        "installation",
        "welche version verwenden wir",
        "was haben wir",
    ]

    if any(keyword in text for keyword in rag_keywords):
        return ContextDecision(
            needs_memory=True,
            needs_rag=True,
            needs_research=False,
            needs_calendar=False,
            needs_camera=False,
            save_memory_candidate=False,
            confidence=0.75,
            reason="question likely relates to local project knowledge or documents",
        )

    # ------------------------------------------------------------
    # Default: use memory, no research.
    # ------------------------------------------------------------
    return ContextDecision(
        needs_memory=True,
        needs_rag=False,
        needs_research=False,
        needs_calendar=False,
        needs_camera=False,
        save_memory_candidate=False,
        confidence=0.55,
        reason="default local answer path using Shared Brain and verified memory",
    )