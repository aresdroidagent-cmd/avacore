from __future__ import annotations

from dataclasses import asdict, dataclass

RUNTIME_KEYWORDS = (
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
)

IDENTITY_KEYWORDS = (
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
)

LOCATION_KEYWORDS = (
    "wo bist du",
    "wo befindest du dich",
    "wo läufst du",
    "wo ist dein standort",
    "dein standort",
    "where are you",
    "your location",
)

CALENDAR_KEYWORDS = (
    "kalender",
    "termin",
    "termine",
    "agenda",
    "briefing",
    "tagesaufgaben",
    "heute vor",
)

CAMERA_KEYWORDS = ("kamera", "webcam", "bild", "snapshot", "sehen", "siehst du")

RESEARCH_KEYWORDS = (
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
)

RAG_KEYWORDS = (
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
)


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


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def decide_context(user_text: str) -> ContextDecision:
    text = (user_text or "").strip().lower()

    # ------------------------------------------------------------
    # Runtime / identity questions
    # These must never trigger web research.
    # Ava knows these from Shared Brain + Runtime Context.
    # ------------------------------------------------------------
    # A tool-specific intent wins over a generic time word such as "heute".
    # Otherwise "Was steht heute im Kalender?" is incorrectly treated as a
    # request for the current date.
    if _contains_any(text, RUNTIME_KEYWORDS) and not _contains_any(text, CALENDAR_KEYWORDS):
        return ContextDecision(
            needs_memory=True,
            needs_rag=False,
            needs_research=False,
            needs_calendar=False,
            needs_camera=False,
            save_memory_candidate=False,
            confidence=0.95,
            reason=(
                "question can be answered from runtime context such as "
                "current date, time or timezone"
            ),
        )

    if _contains_any(text, IDENTITY_KEYWORDS):
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

    if _contains_any(text, LOCATION_KEYWORDS):
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
    if _contains_any(text, CALENDAR_KEYWORDS):
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

    if _contains_any(text, CAMERA_KEYWORDS):
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
    if _contains_any(text, RESEARCH_KEYWORDS):
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
    if _contains_any(text, RAG_KEYWORDS):
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
