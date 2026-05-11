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

    decision = ContextDecision()

    if not text:
        decision.reason = "empty input"
        decision.confidence = 0.2
        return decision

    memory_terms = (
        "erinner", "memory", "gedächtnis", "weisst du", "weißt du",
        "wer bin ich", "wer bist du", "wer ist roger", "vater", "schöpfer",
        "setup", "modell", "kamera", "opencv", "ollama", "avacore",
    )

    rag_terms = (
        "dokument", "pdf", "manual", "handbuch", "seite", "chapter", "kapitel",
        "ar4", "ar3", "isaac", "realsense", "vorgaben", "definitionen",
        "repo", "readme", "projekt", "pipeline", "rtsp", "d-link", "dcs-5222l",
    )

    research_terms = (
        "suche", "recherch", "web", "internet", "google", "aktuell", "neueste",
        "latest", "today", "heute", "preis", "kaufen", "wo bekomme", "version aktuell",
        "release", "news", "hersteller", "datenblatt", "shop", "bestellen",
    )

    calendar_terms = (
        "kalender", "termin", "termine", "briefing", "tagesaufgaben", "agenda", "heute vor",
    )

    camera_terms = (
        "kamera", "bild", "snapshot", "foto", "sehen", "webcam", "rtsp",
    )

    if _contains_any(text, memory_terms):
        decision.needs_memory = True
        decision.reason = "question references identity, memory or known setup"
        decision.confidence = max(decision.confidence, 0.75)

    if _contains_any(text, rag_terms):
        decision.needs_rag = True
        decision.reason = "question likely benefits from local project/RAG knowledge"
        decision.confidence = max(decision.confidence, 0.75)

    if _contains_any(text, research_terms):
        decision.needs_research = True
        decision.save_memory_candidate = True
        decision.reason = "question likely requires current or external web information"
        decision.confidence = max(decision.confidence, 0.85)

    if _contains_any(text, calendar_terms):
        decision.needs_calendar = True
        decision.reason = "question references calendar or daily briefing"
        decision.confidence = max(decision.confidence, 0.8)

    if _contains_any(text, camera_terms):
        decision.needs_camera = True
        decision.reason = "question references camera or visual input"
        decision.confidence = max(decision.confidence, 0.7)

    explicit_research = _matches_any(text, (r"^/research\b", r"\brecherchiere\b", r"\bsuche im web\b"))
    if explicit_research:
        decision.needs_research = True
        decision.save_memory_candidate = True
        decision.reason = "explicit research request"
        decision.confidence = 0.95

    # Local first: if it is clearly about AvaCore/project setup and not explicitly current,
    # prefer memory/RAG over web research.
    if decision.needs_rag and not explicit_research:
        current_markers = ("aktuell", "neueste", "latest", "preis", "kaufen", "bestellen", "news")
        if not _contains_any(text, current_markers):
            decision.needs_research = False
            decision.save_memory_candidate = False

    return decision
