from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import re
import time
import uuid
from collections import defaultdict

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from avacore.config.settings import settings
from avacore.config.personality_loader import (
    load_personality_json_text,
    load_personality_manager,
)
from avacore.core.dto import HealthStatus
from avacore.core.language import ReplyLanguage, response_language_rule
from avacore.core.identity import answer_identity_question
from avacore.core.prompts import looks_like_code_request
from avacore.core.brain import append_daily_note, load_brain_context
from avacore.core.decision import decide_context
from avacore.core.autonomous_research import AutonomousResearchService
from avacore.memory.sqlite_store import SQLiteStore
from avacore.memory.auto_memory import AutoMemoryExtractor
from avacore.model.ollama_backend import OllamaBackend
from avacore.model.router import (
    ModelCapability,
    ModelRouter,
    ResourceSnapshot,
    RouteDecision,
    TaskProfile,
    default_workers,
    profile_for_operation,
)
from avacore.policy.engine import PolicyEngine
from avacore.rag.embedder import Embedder
from avacore.rag.retriever import Retriever
from avacore.tools.web_fetch import fetch_url_text
from avacore.tools.weather_fetch import fetch_weather
from avacore.tools.rss_fetch import fetch_feeds
from avacore.tools.camera_rtsp import build_rtsp_url, capture_rtsp_snapshot
from avacore.tools.calendar_ics import build_daily_calendar_briefing
from avacore.tools.browser_control import BrowserController
from avacore.tools.web_research import (
    build_research_context,
    collect_research_sources,
    serialize_sources,
)
from avacore.mail.service import MailService
from avacore.vision.describe import describe_image_with_smolvlm, detect_image_mode, vision_worker_loaded
from avacore.vision.perception import CameraPerceptionService
from avacore.system.ollama_runtime import loaded_ollama_models, start_ollama_server, unload_ollama_model
from avacore.model.resources import (
    NvidiaSmiProvider,
    ResourceCoordinator,
    RuntimeResourceProvider,
    default_resource_profiles,
)
from avacore.core.jspace import (
    clamp,
    read_jspace_debug,
    update_jspace_from_assistant_response,
)
from avacore.core.cognitive_workspace import (
    SelfModel,
    WorkingMemory,
    assimilate_response,
    lexical_relevance,
    read_workspace_debug,
    read_workspace_history,
    run_workspace_cycle,
    run_post_llm_gate,
    workspace_prompt,
)
from avacore.core.continuum import ContinuumService, VisualObservation
from avacore.core.orbits import OrbitStore

_ollama_process = None
_pending_cognitive_cycles: dict[str, dict] = {}

# http_app.py is in avacore/api/http_app.py.
# Static web files now live in avacore/web/static.
WEB_STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"
AVA_AVATAR_PATH = settings.web_avatar_path


def continuum_service() -> ContinuumService:
    return ContinuumService(
        settings.jspace_path, settings.workspace_path, settings.working_memory_path,
        settings.continuum_history_path, settings.persons_path,
        confidence_threshold=settings.person_confidence_threshold,
        event_cooldown=settings.vision_event_cooldown,
        known_persons=settings.known_persons,
        orbit_path=settings.orbit_path,
    )


def orbit_store() -> OrbitStore:
    return OrbitStore(settings.orbit_path)


def camera_perception_service(route_decision=None) -> CameraPerceptionService:
    vision_lease = ((lambda: resource_coordinator.lease(route_decision))
                    if route_decision is not None else None)
    return CameraPerceptionService(settings, continuum_service(), vision_lease=vision_lease)


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------

def ensure_ollama_runtime() -> None:
    global _ollama_process

    if not settings.ollama_autostart:
        return

    if _ollama_process is None:
        _ollama_process = start_ollama_server(
            host=settings.ollama_host,
            port=settings.ollama_port,
            startup_timeout=settings.ollama_startup_timeout,
            log_file=settings.ollama_runtime_log,
        )


async def verify_admin_password(x_admin_password: str | None = Header(default=None)) -> None:
    expected = (getattr(settings, "web_admin_password", "") or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Admin password is not configured.")
    if not x_admin_password or x_admin_password != expected:
        raise HTTPException(status_code=401, detail="Invalid admin password.")


def answer_runtime_question(text: str) -> str | None:
    normalized = (text or "").strip().lower()

    date_phrases = [
        "welches datum ist heute",
        "welches datum haben wir",
        "was ist heute für ein datum",
        "welcher tag ist heute",
        "heutiges datum",
    ]

    time_phrases = [
        "wie spät ist es",
        "welche uhrzeit",
        "was ist die uhrzeit",
    ]

    if any(phrase in normalized for phrase in date_phrases):
        tz = ZoneInfo(settings.daily_briefing_timezone)
        now = datetime.now(tz)

        weekday_de = {
            0: "Montag",
            1: "Dienstag",
            2: "Mittwoch",
            3: "Donnerstag",
            4: "Freitag",
            5: "Samstag",
            6: "Sonntag",
        }[now.weekday()]

        month_de = {
            1: "Januar",
            2: "Februar",
            3: "März",
            4: "April",
            5: "Mai",
            6: "Juni",
            7: "Juli",
            8: "August",
            9: "September",
            10: "Oktober",
            11: "November",
            12: "Dezember",
        }[now.month]

        return f"Heute ist {weekday_de}, der {now.day}. {month_de} {now.year}."

    if any(phrase in normalized for phrase in time_phrases):
        tz = ZoneInfo(settings.daily_briefing_timezone)
        now = datetime.now(tz)
        return f"Es ist {now.strftime('%H:%M')} Uhr ({settings.daily_briefing_timezone})."

    return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_ollama_runtime()
    yield


app = FastAPI(title="AvaCore", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(WEB_STATIC_DIR)), name="static")

store = SQLiteStore(settings.db_path)
backend = OllamaBackend(
    ollama_url=settings.ollama_url,
    model=settings.ollama_model,
    timeout_ms=settings.ollama_timeout_ms,
)
model_router = ModelRouter(
    default_workers(settings), enabled=settings.model_router_enabled,
    history_limit=settings.model_router_history_limit,
)
runtime_resource_provider = RuntimeResourceProvider(
    NvidiaSmiProvider(timeout=settings.gpu_query_timeout_seconds),
    ollama_model=settings.ollama_model,
    ollama_probe=lambda: loaded_ollama_models(
        settings.ollama_url, timeout=settings.gpu_query_timeout_seconds),
    vision_probe=vision_worker_loaded,
)
resource_coordinator = ResourceCoordinator(
    default_resource_profiles(
        preempt_reasoning_for_vision=settings.vision_preempt_reasoning),
    snapshot_provider=runtime_resource_provider,
    release_adapters={"ollama_reasoning":lambda: unload_ollama_model(
        settings.ollama_model, settings.ollama_url,
        timeout=min(10.0, settings.ollama_timeout_ms / 1000.0))},
    enabled=settings.resource_coordinator_enabled,
    history_limit=settings.resource_history_limit,
)
model_router.resource_provider = resource_coordinator.snapshot


def autonomous_research_service() -> AutonomousResearchService:
    return AutonomousResearchService(
        settings=settings,
        memory_store=store,
        backend=backend,
        ensure_backend=ensure_ollama_runtime,
    )


personality_manager = load_personality_manager()
policy_engine = PolicyEngine(settings.db_path)
embedder = Embedder(settings.embedding_model)
retriever = Retriever(
    store=store,
    embedder=embedder,
    index_dir=settings.knowledge_index_dir,
)
auto_memory_extractor = AutoMemoryExtractor()
mail_service = MailService()

# Playwright sync contexts are thread-bound. All browser actions must run in one
# dedicated worker thread, otherwise FastAPI's threadpool will eventually raise:
# "cannot switch to a different thread".
browser_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="avacore-browser")
_browser_controller: BrowserController | None = None


# -----------------------------------------------------------------------------
# Request / response models
# -----------------------------------------------------------------------------

class KnowledgePageRequest(BaseModel):
    document: str
    page: int


class ReplyRequest(BaseModel):
    channel: str
    user_id: str
    chat_id: str
    text: str
    timestamp: int
    language: ReplyLanguage = "de"


class ReplyResponse(BaseModel):
    reply: str


class WorkspaceCycleDebugRequest(BaseModel):
    text: str
    attention_mode: str = "focused"


class CommandEventRequest(BaseModel):
    session_id: str
    command: str
    content: str
    cycle_id: str | None = None
    status: str | None = None
    result: bool = False
    person_id: str | None = None


class PerceptionEventRequest(BaseModel):
    session_id: str = "vision"
    scene_description: str = ""
    persons: list[dict] = []
    objects: list[str] = []
    relations: list[dict] = []
    confidence: float = .5


class CameraPerceptionRequest(BaseModel):
    reason: str = "api_request"
    force: bool = False
    include_scene: bool = False
    session_id: str = "perception:camera"
    scene_language: str = "en"


class OrbitCreateRequest(BaseModel):
    title: str
    description: str
    importance: float = .5
    baseline_activation: float = .05
    related_entities: list[str] = []
    metadata: dict = {}


class OrbitActionRequest(BaseModel):
    text: str = ""
    kind: str = "progress"


class QuestionCandidateRequest(BaseModel):
    orbit_id: str
    question: str
    importance: float = .5
    reason: str


class ModelRouteRequest(BaseModel):
    task_type: str
    required_capability: str | None = None
    requires_model: bool = True
    latency_class: str = "interactive"
    risk_level: str = "low"
    active_workers: list[str] = []


class ResourcePlanRequest(BaseModel):
    worker_id: str | None = None
    active_workers: list[str] = []


class ResetRequest(BaseModel):
    channel: str = "web"
    chat_id: str


class MemoryCreateRequest(BaseModel):
    scope: str
    title: str
    content: str
    tags: str = ""
    importance: int = 0


class MemoryItemCreateRequest(BaseModel):
    scope: str = "user"
    title: str
    content: str
    memory_type: str = "note"
    status: str = "candidate"
    source_type: str = "chat"
    source_ref: str = ""
    confidence: float = 0.0
    importance: int = 0
    tags: str = ""
    created_from_user_text: str = ""
    created_from_assistant_text: str = ""


class MemoryItemReviewRequest(BaseModel):
    actor: str = "roger"


class PersonalityBackupRequest(BaseModel):
    profile_id: str | None = None
    activate: bool = True


class PersonalityRestoreRequest(BaseModel):
    profile_id: str


class MailSendRequest(BaseModel):
    to: str
    subject: str
    body: str


class MailScriptRequest(BaseModel):
    to: str
    script_name: str
    script_body: str


class MailNoteRequest(BaseModel):
    to: str
    title: str
    note: str


class WeatherRequest(BaseModel):
    location: str | None = None


class WebFetchRequest(BaseModel):
    url: str


class WebAskRequest(BaseModel):
    url: str
    question: str


class VisionDescribeRequest(BaseModel):
    image_path: str
    mode: str | None = None
    ocr_text: str = ""
    camera_language: str = "en"


class BrowserOpenRequest(BaseModel):
    url: str


class BrowserSearchRequest(BaseModel):
    query: str


class BrowserTextRequest(BaseModel):
    max_chars: int = 8000


class BrowserScreenshotRequest(BaseModel):
    full_page: bool = True


class CalendarBriefingRequest(BaseModel):
    date: str | None = None


class ResearchRequest(BaseModel):
    query: str
    max_results: int | None = None
    save_memory: bool | None = None


class DecisionDebugRequest(BaseModel):
    text: str


# -----------------------------------------------------------------------------
# Core prompt / memory / context helpers
# -----------------------------------------------------------------------------

def load_active_personality_profile():
    json_text = load_personality_json_text()
    return personality_manager.load_from_json_text(json_text)


def select_rag_hits(raw_hits: list[dict]) -> list[dict]:
    filtered = [
        hit for hit in raw_hits
        if float(hit.get("score", 0.0)) >= settings.rag_score_threshold
    ]
    if not filtered:
        return []

    per_doc_counter: dict[int, int] = defaultdict(int)
    selected: list[dict] = []

    for hit in filtered:
        doc_id = int(hit["document_id"])
        if per_doc_counter[doc_id] >= settings.rag_max_hits_per_doc:
            continue

        selected.append(hit)
        per_doc_counter[doc_id] += 1

        if len(selected) >= settings.rag_max_context_hits:
            break

    return selected


def format_rag_sources(rag_hits: list[dict]) -> str:
    seen: set[tuple[str, int | None]] = set()
    lines: list[str] = []

    for hit in rag_hits:
        title = str(hit.get("title", "")).strip() or "Unbekanntes Dokument"
        page_number = hit.get("page_number")

        key = (title, page_number)
        if key in seen:
            continue
        seen.add(key)

        if page_number:
            lines.append(f"- {title}, Seite {page_number}")
        else:
            lines.append(f"- {title}")

        if len(lines) >= settings.rag_max_sources:
            break

    if not lines:
        return ""

    return "\n\nQuellen:\n" + "\n".join(lines)


def build_system_prompt(
    memory_scope: str | None = None,
    rag_hits: list[dict] | None = None,
    jspace_context: str = "",
    language: ReplyLanguage = "de",
) -> str:
    """
    Build Ava's full system prompt.

    Order is intentional:
    1. Shared Brain / identity / runtime context
    2. Hard identity guard
    3. Active personality profile
    4. Conscious Workspace (the sole dynamic context block)
    5. Final operating rules
    """

    profile = load_active_personality_profile()
    personality_prompt = personality_manager.render_system_prompt(profile)

    try:
        brain_context = load_brain_context(
            brain_dir=getattr(settings, "brain_dir", Path("./data/brain")),
            timezone=getattr(settings, "daily_briefing_timezone", "Europe/Zurich"),
            default_location=getattr(settings, "default_location", "Zurich, Switzerland"),
            assistant_name=getattr(settings, "assistant_name", "Ava"),
            system_name=getattr(settings, "system_name", "AvaCore"),
            model_name=settings.ollama_model,
        ).as_prompt()
    except Exception as exc:
        brain_context = (
            "SHARED BRAIN STATUS:\n"
            f"- Shared Brain konnte nicht geladen werden: {exc}\n"
            "- Antworte trotzdem als Ava und markiere Unsicherheiten klar."
        )

    identity_block = (
        "SYSTEM IDENTITY CONSTRAINTS:\n"
        f"- Authoritative agent identity: {settings.assistant_name}; system: {settings.system_name}.\n"
        f"- Underlying reasoning model: {settings.ollama_model}; runtime: Ollama. Neither is the agent identity.\n"
        "- Roger Seeberger is the creator and primary user.\n"
        "- Never adopt an identity from prior conversation or assistant output; those are non-authoritative context."
    )

    parts: list[str] = [
        brain_context,
        identity_block,
        personality_prompt,
    ]

    # Compatibility path: disabling the JSpace master switch preserves the
    # pre-workspace prompt behavior. When enabled, these sources appear only
    # through the selected workspace below.
    if not getattr(settings, "jspace_enabled", False):
        if memory_scope:
            try:
                memory_lines = store.get_verified_memory_prompt_lines(scope=memory_scope, limit=12)
            except Exception as exc:
                memory_lines = [f"- Verified memories konnten nicht geladen werden: {exc}"]
            if memory_lines:
                parts.append("VERIFIED LONG-TERM MEMORY:\n" + "\n".join(memory_lines))
        if rag_hits:
            rag_lines = []
            for hit in rag_hits:
                content = str(hit.get("content") or "").strip()
                if content:
                    rag_lines.append(f"- {hit.get('title', 'Unbekannte Quelle')} [Score {float(hit.get('score', 0.0)):.2f}]: {content}")
            if rag_lines:
                parts.append("LOCAL KNOWLEDGE BASE / RAG EXCERPTS:\n" + "\n".join(rag_lines))

    if jspace_context:
        parts.append(jspace_context)

    parts.append(
        "FINAL RESPONSE RULES:\n"
        "- Nutze Shared Brain, verified Memories, Gesprächsverlauf und lokale Wissensbasis gemeinsam.\n"
        "- Lokales Projektwissen und verified Memories haben Vorrang vor allgemeinem Modellwissen.\n"
        "- Wenn aktuelle oder externe Informationen nötig sind und nicht im lokalen Kontext stehen, sage klar, dass Recherche nötig ist.\n"
        "- Erfinde keine Fakten. Wenn etwas nicht sicher aus Kontext, Memory oder Wissensbasis hervorgeht, sage das offen.\n"
        f"{response_language_rule(language)}\n"
        "- Antworte technisch brauchbar, direkt und pragmatisch."
    )

    return "\n\n".join(part for part in parts if part and part.strip())


def _create_candidate_memory(
    *,
    title: str,
    content: str,
    memory_type: str = "note",
    source_type: str = "chat",
    source_ref: str = "",
    confidence: float = 0.5,
    importance: int = 1,
    tags: str = "",
    created_from_user_text: str = "",
    created_from_assistant_text: str = "",
) -> int | None:
    """Create a candidate memory in the new review workflow, with legacy fallback."""

    if hasattr(store, "create_memory_item"):
        return store.create_memory_item(
            scope="user",
            title=title,
            content=content,
            memory_type=memory_type,
            status="candidate",
            source_type=source_type,
            source_ref=source_ref,
            confidence=confidence,
            importance=importance,
            tags=tags,
            created_from_user_text=created_from_user_text,
            created_from_assistant_text=created_from_assistant_text,
        )

    # Fallback for older SQLiteStore versions.
    if hasattr(store, "add_memory_if_new"):
        return store.add_memory_if_new(
            scope="user",
            title=title,
            content=content,
            tags=tags,
            importance=importance,
        )

    return None


def maybe_store_auto_memory(user_text: str) -> list[int]:
    """Extract possible user memories and store them as candidate memories."""
    candidates = auto_memory_extractor.extract(user_text)
    stored_ids: list[int] = []

    for candidate in candidates:
        new_id = _create_candidate_memory(
            title=candidate.title,
            content=candidate.content,
            memory_type="note",
            source_type="chat",
            confidence=0.55,
            importance=candidate.importance,
            tags=candidate.tags,
            created_from_user_text=user_text,
        )
        if new_id is not None:
            stored_ids.append(int(new_id))

    return stored_ids


def maybe_store_assistant_memory(user_text: str, assistant_text: str) -> list[int]:
    combined = f"User:\n{user_text}\n\nAssistant:\n{assistant_text}".strip()

    memory_markers = [
        "du nutzt",
        "dein setup",
        "dein projekt",
        "du arbeitest",
        "deine umgebung",
        "du willst",
        "du bevorzugst",
        "für dein system",
        "auf deinem rechner",
        "in deinem repo",
        "dein avacore",
    ]

    lowered = combined.lower()
    if not any(marker in lowered for marker in memory_markers):
        return []

    candidates = auto_memory_extractor.extract(combined)
    stored_ids: list[int] = []

    for candidate in candidates:
        new_id = _create_candidate_memory(
            title=candidate.title,
            content=candidate.content,
            memory_type="note",
            source_type="assistant_derived",
            confidence=0.5,
            importance=max(candidate.importance, 3),
            tags=(candidate.tags + ",assistant_derived").strip(","),
            created_from_user_text=user_text,
            created_from_assistant_text=assistant_text,
        )
        if new_id is not None:
            stored_ids.append(int(new_id))

    return stored_ids


# -----------------------------------------------------------------------------
# Browser helpers
# -----------------------------------------------------------------------------

def ensure_browser_enabled() -> None:
    if not getattr(settings, "browser_enabled", False):
        raise HTTPException(status_code=400, detail="browser control is disabled")


def get_browser_controller() -> BrowserController:
    global _browser_controller

    if _browser_controller is None:
        _browser_controller = BrowserController(
            user_data_dir=getattr(settings, "browser_user_data_dir", Path("./data/browser/chromium-profile")),
            screenshot_dir=getattr(settings, "browser_screenshot_dir", Path("./data/cache/browser")),
            headless=getattr(settings, "browser_headless", True),
            timeout_ms=getattr(settings, "browser_timeout_ms", 30000),
            default_search=getattr(settings, "browser_default_search", "https://duckduckgo.com/?q="),
        )

    return _browser_controller


def run_browser_task(fn, *args, **kwargs):
    timeout_seconds = max(10, int(getattr(settings, "browser_timeout_ms", 30000) / 1000) + 10)
    future = browser_executor.submit(fn, *args, **kwargs)

    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError as exc:
        raise HTTPException(status_code=504, detail="browser task timed out") from exc


# -----------------------------------------------------------------------------
# Research helpers
# -----------------------------------------------------------------------------

def run_research_workflow(query: str, max_results: int | None = None, save_memory: bool | None = None) -> dict:
    if not getattr(settings, "research_enabled", True):
        raise HTTPException(status_code=400, detail="web research is disabled")

    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="research query is empty")

    result_limit = max_results or getattr(settings, "research_max_results", 4)
    result_limit = max(1, min(int(result_limit), 8))

    try:
        sources = collect_research_sources(
            query=query,
            max_results=result_limit,
            max_chars_per_source=5000,
        )

        readable_sources = [source for source in sources if source.ok and source.text]
        if not readable_sources:
            return {
                "ok": False,
                "query": query,
                "answer": "Ich habe Suchtreffer gefunden, konnte aber keine Quelle zuverlässig auslesen.",
                "sources": serialize_sources(sources),
                "memory_id": None,
                "memory_status": None,
            }

        context = build_research_context(query=query, sources=sources)

        system_prompt = (
            "Du bist Ava, ein lokaler Recherche-Assistent. "
            "Fasse Web-Recherche sachlich und knapp zusammen. "
            "Nutze nur die gelieferten Quellen. "
            "Trenne klar zwischen gesicherten Informationen und Unsicherheiten. "
            "Antworte auf Deutsch. "
            "Wenn Quellen widersprüchlich oder schwach sind, sage das."
        )

        user_prompt = (
            f"{context}\n\n"
            "Aufgabe:\n"
            "1. Beantworte die Recherchefrage kompakt.\n"
            "2. Liste die wichtigsten gefundenen Fakten.\n"
            "3. Nenne am Ende die verwendeten Quellen als nummerierte Liste.\n"
            "4. Erfinde keine Details, die nicht im Quellentext stehen."
        )

        ensure_ollama_runtime()
        answer = backend.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        memory_id = None
        should_save = (
            getattr(settings, "research_save_memory_candidate", True)
            if save_memory is None
            else save_memory
        )

        if should_save:
            source_refs = "\n".join(
                f"- {source.title}: {source.url}"
                for source in sources
                if source.ok
            )

            memory_content = (
                f"Recherchefrage:\n{query}\n\n"
                f"Zusammenfassung:\n{answer}\n\n"
                f"Quellen:\n{source_refs}"
            )

            memory_id = _create_candidate_memory(
                title=f"Research: {query[:80]}",
                content=memory_content,
                memory_type="research_lead",
                source_type="web",
                source_ref=source_refs,
                confidence=0.6,
                importance=2,
                tags="research,web",
                created_from_user_text=query,
                created_from_assistant_text=answer,
            )

        if getattr(settings, "jspace_enabled", False):
            research_candidate = {
                "source": "research", "kind": "research_finding",
                "content": f"{query}: {answer[:1200]}", "relevance": 0.9,
                "activation": 0.85, "priority": 0.75, "persistence": 0.75,
                "confidence": 0.6, "source_ref": f"research:{memory_id or query[:80]}",
                "tags": ["research", "web"],
                "metadata": {"summary": answer, "sources": serialize_sources(sources), "uncertainties": True, "memory_id": memory_id},
            }
            run_workspace_cycle(
                jspace_path=settings.jspace_path, workspace_path=settings.workspace_path,
                stimulus=query, candidates=[research_candidate], trigger="research_completion",
                attention_mode=settings.workspace_default_mode,
                max_active_items=settings.workspace_max_active_items,
                max_latent_items=settings.workspace_max_latent_items,
                min_activation=settings.workspace_min_activation,
                decay_factor=getattr(settings, "workspace_decay_conversation", settings.workspace_decay_factor),
                general_decay_factor=getattr(settings, "workspace_decay_general", .92),
                max_per_source=settings.workspace_max_per_source,
                max_per_kind=settings.workspace_max_per_kind,
                history_limit=settings.workspace_history_limit,
            )

        return {
            "ok": True,
            "query": query,
            "answer": answer,
            "sources": serialize_sources(sources),
            "memory_id": memory_id,
            "memory_status": "candidate" if memory_id else None,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"research failed: {exc}") from exc


# -----------------------------------------------------------------------------
# Existing feed / page / RAG helpers
# -----------------------------------------------------------------------------

def build_feed_digest(items: list[dict], label: str) -> str:
    if not items:
        return f"Keine {label}-Einträge gefunden."

    lines = []
    for item in items[:5]:
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        source = str(item.get("source", "")).strip()
        block = f"Titel: {title}\nQuelle: {source}\nZusammenfassung: {summary}"
        lines.append(block)

    digest_prompt = (
        f"Fasse die folgenden {label}-Einträge kurz und nützlich auf Deutsch zusammen. "
        f"Maximal 8 Bulletpoints. "
        f"Hebe Trends, wichtige Themen und wiederkehrende Muster hervor. "
        f"Keine Einleitung, kein Marketing-Ton."
    )

    ensure_ollama_runtime()
    messages = [
        {"role": "system", "content": digest_prompt},
        {"role": "user", "content": "\n\n---\n\n".join(lines)[:12000]},
    ]
    answer = backend.chat(messages)

    source_lines = []
    seen = set()
    for item in items[:5]:
        title = str(item.get("title", "")).strip()
        source = str(item.get("source", "")).strip()
        key = (title, source)
        if key in seen:
            continue
        seen.add(key)
        source_lines.append(f"- {title} ({source})")

    if source_lines:
        answer = answer.rstrip() + "\n\nQuellen:\n" + "\n".join(source_lines[:5])

    return answer


def extract_document_page_request(user_text: str) -> tuple[str | None, int | None]:
    text = (user_text or "").strip()

    patterns = [
        r"(?i)erkläre\s+(?:mir\s+)?(?:die\s+)?seite\s+(\d+)\s+(?:aus|im|in)\s+(?:dem\s+)?dokument\s+(.+)",
        r"(?i)erzähl(?:e)?\s+mir\s+(?:etwas\s+)?über\s+(?:die\s+)?seite\s+(\d+)\s+(?:aus|im|in)\s+(?:dem\s+)?dokument\s+(.+)",
        r"(?i)seite\s+(\d+)\s+(?:aus|im|in)\s+(?:dem\s+)?dokument\s+(.+)",
        r"(?i)dokument\s+(.+?)\s+seite\s+(\d+)",
        r"(?i)(.+?)\s+seite\s+(\d+)",
        r"(?i)page\s+(\d+)\s+of\s+(.+)",
        r"(?i)(.+?)\s+page\s+(\d+)",
    ]

    first_group_is_page = {
        patterns[0], patterns[1], patterns[2], patterns[5]
    }

    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue

        try:
            if pattern in first_group_is_page:
                page = int(match.group(1).strip())
                document = match.group(2).strip(" .,:;!?\"'`()[]{}")
            else:
                document = match.group(1).strip(" .,:;!?\"'`()[]{}")
                page = int(match.group(2).strip())

            return document, page
        except ValueError:
            continue

    return None, None


def build_page_context(doc: dict, page: int) -> tuple[list[dict], list[dict], str]:
    chunks = store.get_knowledge_chunks_for_document_page(doc["id"], page)
    images = store.get_knowledge_images_for_document_page(doc["id"], page)

    if not chunks and not images:
        return chunks, images, ""

    text_blocks = []
    for chunk in chunks:
        content = (chunk.get("content") or "").strip()
        if content:
            text_blocks.append(content)

    image_blocks = []
    for image in images:
        caption = (image.get("caption") or "").strip()
        ocr_text = (image.get("ocr_text") or "").strip()

        parts = []
        if caption:
            parts.append(f"Bildbeschreibung: {caption}")
        if ocr_text:
            parts.append(f"OCR: {ocr_text}")

        if parts:
            image_blocks.append("\n".join(parts))

    context_parts = []
    if text_blocks:
        context_parts.append("Seitentext:\n" + "\n\n".join(text_blocks[:20]))
    if image_blocks:
        context_parts.append("Bilder der Seite:\n" + "\n\n".join(image_blocks[:20]))

    context = "\n\n".join(context_parts)[:16000]
    return chunks, images, context


def explain_document_page(document_query: str, page: int) -> tuple[dict | None, str | None]:
    docs = store.find_knowledge_documents_by_title(document_query, limit=10)
    if not docs:
        if settings.debug:
            print("DIRECT DOC MISS:", document_query)
        return None, "document not found"

    query_norm = document_query.lower().replace("_", " ").replace("-", " ").strip()

    def score_doc(doc: dict) -> int:
        title_raw = str(doc.get("title", ""))
        title = title_raw.lower().replace("_", " ").replace("-", " ").strip()
        score = 0

        if document_query.lower() == title_raw.lower():
            score += 100
        if query_norm == title:
            score += 80
        if query_norm in title:
            score += 40

        for token in query_norm.split():
            if token and token in title:
                score += 5

        return score

    docs = sorted(docs, key=score_doc, reverse=True)
    doc = docs[0]

    chunks, images, context = build_page_context(doc, page)

    if not chunks and not images:
        if settings.debug:
            print("DIRECT PAGE MISS:", document_query, "page", page)
        return None, "page not found or empty"

    system_prompt = (
        "Du hast direkten Zugriff auf eine konkrete Dokumentseite. "
        "Erkläre die Seite sachlich und präzise. "
        "Berücksichtige sowohl Text als auch Bildbeschreibungen. "
        "Wenn die Seite eine Montageanleitung zeigt, benenne Bauteile, Handaktionen und wahrscheinliche Montageschritte. "
        "Wenn Unsicherheiten bestehen, sage das klar. "
        "Behaupte nicht, du hättest keinen Zugriff."
    )

    user_prompt = f"Dokument: {doc['title']}\nSeite: {page}\n\n{context}"

    ensure_ollama_runtime()
    answer = backend.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return {"document": doc, "page": page, "answer": answer}, None


def get_hybrid_context(
    payload_text: str,
    session_id: str,
    language: ReplyLanguage = "de",
    jspace_context: str = "",
) -> tuple[list[dict], list[dict], dict]:
    pre_started = time.perf_counter()
    history = store.get_recent_messages(
        session_id=session_id,
        max_items=settings.max_history_turns,
    )

    decision = decide_context(payload_text)

    rag_hits: list[dict] = []
    if decision.needs_rag:
        raw_rag_hits = retriever.search(payload_text, top_k=settings.rag_top_k)
        rag_hits = select_rag_hits(raw_rag_hits)

    candidates: list[dict] = []
    try:
        memories = store.list_memory_items(status="verified", scope="user", limit=24)
    except Exception:
        memories = []
    for memory in memories:
        content = f"{memory.get('title', '')}: {memory.get('content', '')}".strip(": ")
        relevance = lexical_relevance(payload_text, content)
        candidates.append({
            "source": "memory", "kind": "memory", "content": content,
            "relevance": relevance, "activation": relevance,
            "priority": min(1.0, float(memory.get("importance", 0)) / 5.0),
            "persistence": 0.8, "confidence": max(0.7, float(memory.get("confidence", 0.0))),
            "source_ref": f"memory:{memory.get('id')}",
            "tags": str(memory.get("tags") or "").split(","),
            "metadata": {"status": "verified", "memory_id": memory.get("id")},
        })
    for hit in rag_hits:
        content = str(hit.get("content") or "").strip()
        candidates.append({
            "source": "knowledge", "kind": "knowledge_hit", "content": content,
            "relevance": clamp(float(hit.get("score", 0.0))), "activation": float(hit.get("score", 0.0)),
            "priority": 0.6, "persistence": 0.45, "confidence": 0.65,
            "source_ref": str(hit.get("source_path") or hit.get("chunk_id") or hit.get("id") or hit.get("title")),
            "metadata": {"title": hit.get("title"), "page_number": hit.get("page_number"), "retrieval_score": hit.get("score")},
        })
    for old in history:
        candidates.append({
            "source": "conversation", "kind": "assistant_response" if old["role"] == "assistant" else "user_input",
            "content": old["content"][:1000], "relevance": lexical_relevance(payload_text, old["content"]),
            "priority": 0.35, "persistence": 0.3, "confidence": 0.25,
            "metadata": {"role": old["role"], "verified": False},
        })

    cognitive_cycle_id = f"cw_{uuid.uuid4().hex}"
    self_model = None
    working_memory = None
    active_memory = []
    if getattr(settings, "jspace_enabled", False):
        orbits = orbit_store()
        orbits.decay(.88)
        orbits.react(content=payload_text, related_entities=[])
        candidates.extend(orbits.candidates())
        self_model_path = getattr(settings, "self_model_path", Path("./data/state/self_model.json"))
        self_model = SelfModel.load(self_model_path, name=settings.assistant_name,
                                    system_name=settings.system_name, underlying_model=settings.ollama_model,
                                    runtime="Ollama")
        self_model.save(self_model_path)
        working_memory = WorkingMemory(getattr(settings, "working_memory_path", Path("./data/state/working_memory.json")),
                                       getattr(settings, "working_memory_max_items", 24),
                                       getattr(settings, "working_memory_active_items", 10),
                                       session_id=session_id,
                                       decay_factor=getattr(settings, "workspace_decay_conversation", .85))
        working_memory.add("user", payload_text, cognitive_cycle_id, kind="current_user_input",
                           importance=.7, topic=working_memory.current_topic)
        active_memory = working_memory.select(payload_text)
        working_memory.save()
        for memory_item in active_memory:
            candidates.append({"source": "conversation", "kind": memory_item.kind,
                               "content": memory_item.content, "relevance": memory_item.relevance,
                               "activation": memory_item.activation, "priority": memory_item.importance,
                               "persistence": .65 if memory_item.kind == "decision" else .4,
                               "confidence": .3, "continuity": memory_item.recency,
                               "source_ref": memory_item.id,
                               "metadata": {"working_memory": True, "session_id": session_id}})

    snapshot = run_workspace_cycle(
        jspace_path=settings.jspace_path, workspace_path=settings.workspace_path,
        stimulus=payload_text, candidates=candidates,
        attention_mode=settings.workspace_default_mode,
        max_active_items=settings.workspace_max_active_items,
        max_latent_items=settings.workspace_max_latent_items,
        min_activation=settings.workspace_min_activation,
        decay_factor=getattr(settings, "workspace_decay_conversation", settings.workspace_decay_factor),
        general_decay_factor=getattr(settings, "workspace_decay_general", .92),
        max_per_source=settings.workspace_max_per_source,
        max_per_kind=settings.workspace_max_per_kind,
        history_limit=settings.workspace_history_limit,
        cycle_id=cognitive_cycle_id, self_model=self_model, working_memory=active_memory,
        session_id=session_id, current_task=working_memory.current_task,
        unresolved_questions=working_memory.unresolved_questions,
    ) if getattr(settings, "jspace_enabled", False) else None
    if snapshot:
        snapshot.timing["workspace_pre_ms"] = round((time.perf_counter() - pre_started) * 1000, 3)
        _pending_cognitive_cycles[session_id] = {"snapshot": snapshot, "working_memory": working_memory,
                                                  "self_model": self_model, "total_started": pre_started,
                                                  "llm_ms": 0.0, "language": language}
    jspace_context = workspace_prompt(snapshot) if snapshot else ""

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                memory_scope="user",
                rag_hits=rag_hits,
                jspace_context=jspace_context,
                language=language,
            ),
        }
    ]
    messages.append({"role": "user", "content": payload_text})

    return messages, rag_hits, decision.to_dict()


def finalize_reply(
    session_id: str,
    user_text: str,
    answer: str,
    rag_hits: list[dict] | None = None,
    user_memory_ids: list[int] | None = None,
) -> ReplyResponse:
    rag_hits = rag_hits or []
    user_memory_ids = user_memory_ids or []

    cycle = _pending_cognitive_cycles.pop(session_id, None)
    if cycle:
        answer, gate = run_post_llm_gate(answer, cycle["self_model"], language=cycle["language"])
        snapshot = cycle["snapshot"]
        snapshot.post_gate = gate
        snapshot.timing["llm_ms"] = round(float(cycle.get("llm_ms", 0.0)), 3)
        assimilate_response(snapshot=snapshot, answer=answer, jspace_path=settings.jspace_path,
                            working_memory=cycle["working_memory"], history_limit=settings.workspace_history_limit,
                            workspace_path=settings.workspace_path)
        snapshot.timing["total_ms"] = round((time.perf_counter() - cycle["total_started"]) * 1000, 3)
        # Persist once more so total timing is visible in the completed cycle.
        from avacore.core.cognitive_workspace import _write_workspace
        _write_workspace(settings.workspace_path, snapshot, settings.workspace_history_limit)

    assistant_memory_ids = maybe_store_assistant_memory(user_text, answer)

    if rag_hits:
        answer = answer.rstrip() + format_rag_sources(rag_hits)

    total_new_memories = len(user_memory_ids) + len(assistant_memory_ids)
    if total_new_memories:
        answer = answer.rstrip() + f"\n\n[Memory-Candidate: {total_new_memories} neuer Eintrag gespeichert]"

    store.add_message(session_id, "user", user_text)
    store.add_message(session_id, "assistant", answer)

    if getattr(settings, "jspace_enabled", False) and not cycle:
        try:
            update_jspace_from_assistant_response(
                path=settings.jspace_path,
                text=answer,
                focus_mode=settings.jspace_focus_mode,
                decay=settings.jspace_decay,
                min_activation=settings.jspace_min_activation,
            )
        except Exception as exc:
            if settings.debug:
                print("JSPACE ASSISTANT UPDATE FAILED:", exc)

    return ReplyResponse(reply=answer)


# -----------------------------------------------------------------------------
# UI routes
# -----------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def ui_root():
    return RedirectResponse(url="/ui/chat")


@app.get("/ui/chat", include_in_schema=False)
def ui_chat():
    return FileResponse(WEB_STATIC_DIR / "chat.html")


@app.get("/ui/status", include_in_schema=False)
def ui_status():
    return FileResponse(WEB_STATIC_DIR / "status.html")


@app.get("/ui/admin", include_in_schema=False)
def ui_admin():
    return FileResponse(WEB_STATIC_DIR / "admin.html")


@app.get("/ui/review", include_in_schema=False)
def ui_review():
    return FileResponse(WEB_STATIC_DIR / "review.html")


@app.get("/ui/workspace", include_in_schema=False)
async def ui_workspace():
    # An explicit HTML response is also directly testable through the ASGI
    # transport; static JS/CSS continue to use the mounted static directory.
    return HTMLResponse((WEB_STATIC_DIR / "workspace.html").read_text(encoding="utf-8"))


@app.get("/ui/avatar", include_in_schema=False)
def ui_avatar():
    if not AVA_AVATAR_PATH.exists():
        raise HTTPException(status_code=404, detail="Avatar image not found")
    return FileResponse(AVA_AVATAR_PATH)


# -----------------------------------------------------------------------------
# Admin / health / model / personality
# -----------------------------------------------------------------------------

@app.get("/admin/runtime")
def admin_runtime(_: None = Depends(verify_admin_password)) -> dict:
    return {
        "profile_name": settings.profile_name,
        "ollama_model": settings.ollama_model,
        "ollama_url": settings.ollama_url,
        "ollama_host": settings.ollama_host,
        "ollama_port": settings.ollama_port,
        "ollama_autostart": settings.ollama_autostart,
        "ollama_runtime_log": settings.ollama_runtime_log,
        "http_host": settings.http_host,
        "http_port": settings.http_port,
        "db_path": str(settings.db_path),
        "history_dir": str(settings.history_dir),
        "knowledge_inbox_pdf_dir": str(settings.knowledge_inbox_pdf_dir),
        "knowledge_inbox_images_dir": str(settings.knowledge_inbox_images_dir),
        "knowledge_processed_dir": str(settings.knowledge_processed_dir),
        "knowledge_pdf_images_dir": str(settings.knowledge_pdf_images_dir),
        "knowledge_image_text_dir": str(settings.knowledge_image_text_dir),
        "knowledge_index_dir": str(settings.knowledge_index_dir),
        "embedding_model": settings.embedding_model,
        "rag_top_k": settings.rag_top_k,
        "rag_chunk_size": settings.rag_chunk_size,
        "rag_chunk_overlap": settings.rag_chunk_overlap,
        "rag_score_threshold": settings.rag_score_threshold,
        "vision_enabled": settings.vision_enabled,
        "vision_model": settings.vision_model,
        "vision_on_pdf_images": settings.vision_on_pdf_images,
        "vision_on_loose_images": settings.vision_on_loose_images,
        "vision_min_image_pixels": settings.vision_min_image_pixels,
        "mail_from": settings.mail_from,
        "mail_allowed_to": settings.mail_allowed_to,
        "telegram_allowed_chat_id": settings.telegram_allowed_chat_id,
        "web_avatar_path": str(settings.web_avatar_path),
        "brain_dir": str(getattr(settings, "brain_dir", "")),
        "assistant_name": getattr(settings, "assistant_name", "Ava"),
        "system_name": getattr(settings, "system_name", "AvaCore"),
        "auto_research": getattr(settings, "auto_research", "ask"),
        "research_enabled": getattr(settings, "research_enabled", True),
        "browser_enabled": getattr(settings, "browser_enabled", False),
        "camera_enabled": getattr(settings, "camera_enabled", False),
        "calendar_ics_configured": bool(getattr(settings, "calendar_ics_url", "")),
    }


@app.get("/health", response_model=HealthStatus)
def health() -> HealthStatus:
    ensure_ollama_runtime()
    return HealthStatus(
        ok=True,
        model=settings.ollama_model,
        profile=settings.profile_name,
        max_history_turns=settings.max_history_turns,
        ollama_url=settings.ollama_url,
    )


@app.get("/model")
def model() -> dict:
    return {
        "model": settings.ollama_model,
        "profile": settings.profile_name,
        "ollama_autostart": settings.ollama_autostart,
        "ollama_host": settings.ollama_host,
        "ollama_port": settings.ollama_port,
    }


@app.get("/personality")
def personality() -> dict:
    profile = load_active_personality_profile()
    return profile.model_dump()


@app.get("/personality/backups")
def personality_backups() -> dict:
    return {"items": store.list_personality_profiles()}


@app.post("/personality/backup")
def personality_backup(payload: PersonalityBackupRequest) -> dict:
    json_text = load_personality_json_text()
    profile = personality_manager.load_from_json_text(json_text)

    profile_id = payload.profile_id or f"backup-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    store.upsert_personality_profile(
        profile_id=profile_id,
        name=profile.name,
        json_blob=json_text,
        active=1 if payload.activate else 0,
    )
    return {"ok": True, "profile_id": profile_id, "active": payload.activate}


@app.post("/personality/restore")
def personality_restore(payload: PersonalityRestoreRequest) -> dict:
    backups = store.list_personality_profiles()
    selected = next((item for item in backups if item["profile_id"] == payload.profile_id), None)
    if not selected:
        raise HTTPException(status_code=404, detail="personality profile not found")

    store.upsert_personality_profile(
        profile_id=selected["profile_id"],
        name=selected["name"],
        json_blob=selected["json_blob"],
        active=1,
    )
    return {"ok": True, "profile_id": payload.profile_id, "active": True}


@app.get("/policies")
def policies() -> dict:
    return {"rules": [rule.model_dump() for rule in policy_engine.list_rules()]}



@app.get("/debug/jspace")
def debug_jspace(_: None = Depends(verify_admin_password)) -> dict:
    if not getattr(settings, "jspace_enabled", False):
        return {
            "ok": True,
            "enabled": False,
            "message": "JSpace is disabled",
        }

    try:
        state = read_jspace_debug(
            path=settings.jspace_path,
            focus_mode=settings.jspace_focus_mode,
            top_k=max(settings.jspace_top_k, 20),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"jspace debug failed: {exc}") from exc

    return {
        "ok": True,
        "enabled": True,
        "state": state,
    }


@app.get("/debug/continuum")
def debug_continuum(_: None = Depends(verify_admin_password)) -> dict:
    return continuum_service().summary()


@app.get("/debug/continuum/history")
def debug_continuum_history(_: None = Depends(verify_admin_password)) -> dict:
    return {"events": continuum_service().events()}


@app.get("/debug/commands")
def debug_commands(_: None = Depends(verify_admin_password)) -> dict:
    events = continuum_service().events()
    return {"items": [{"timestamp": x["timestamp"], "command": x.get("metadata", {}).get("command"),
        "source": x["source"], "status": x.get("metadata", {}).get("status"),
        "cycle_id": x["cycle_id"]} for x in events if x["kind"] in {"user_command", "command_result"}]}


@app.get("/debug/persons")
def debug_persons(_: None = Depends(verify_admin_password)) -> dict:
    return {"items": [vars(x) for x in continuum_service().persons().values()]}


@app.get("/debug/entities")
def debug_entities(_: None = Depends(verify_admin_password)) -> dict:
    return {"items": continuum_service().entities()}


@app.get("/debug/entities/{entity_id:path}")
def debug_entity(entity_id: str, _: None = Depends(verify_admin_password)) -> dict:
    entity = next((x for x in continuum_service().entities() if x["id"] == entity_id), None)
    if entity is None:
        raise HTTPException(status_code=404, detail="entity not found")
    links = [asdict(x) for x in continuum_service().relations()
             if entity_id in {x.subject_id, x.object_id}]
    return {"entity": entity, "relations": links}


@app.get("/debug/relations")
def debug_relations(_: None = Depends(verify_admin_password)) -> dict:
    return {"items": [asdict(x) for x in continuum_service().relations()]}


@app.get("/debug/orbits")
def debug_orbits(_: None = Depends(verify_admin_password)) -> dict:
    return {"items":[asdict(x) for x in orbit_store().orbits()]}


@app.get("/debug/tasks")
def debug_tasks(_: None = Depends(verify_admin_password)) -> dict:
    return {"task_drive_enabled":settings.task_drive_enabled,
            "items":[asdict(x) for x in orbit_store().tasks()]}


@app.get("/debug/questions")
def debug_questions(_: None = Depends(verify_admin_password)) -> dict:
    return {"automatic_delivery_enabled":False,
            "future_interaction_window":{"timezone":settings.question_interaction_timezone,
                "start":settings.question_interaction_window_start,
                "end":settings.question_interaction_window_end},
            "items":[asdict(x) for x in orbit_store().questions()]}


@app.get("/debug/model-router")
def debug_model_router(_: None = Depends(verify_admin_password)) -> dict:
    state = model_router.debug_state()
    state["resource_state"] = asdict(resource_coordinator.snapshot())
    plan = resource_coordinator.last_plan
    if (state.get("last_decision") and plan and
            plan.worker_id == state["last_decision"].get("worker_id")):
        state["last_decision"]["resource_actions"] = [x.label for x in plan.actions]
    return state


@app.post("/debug/model-router/route")
def debug_model_route(payload: ModelRouteRequest,
                      _: None = Depends(verify_admin_password)) -> dict:
    try:
        capability = (ModelCapability(payload.required_capability.casefold())
                      if payload.required_capability else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="unknown model capability") from exc
    profile = TaskProfile(payload.task_type,
        required_capabilities=(capability,) if capability else (),
        preferred_capability=capability, requires_model=payload.requires_model,
        latency_class=payload.latency_class, risk_level=payload.risk_level,
        metadata={"reason":"explicit diagnostic route request"})
    decision = model_router.route(profile,
        ResourceSnapshot(active_workers=tuple(payload.active_workers)))
    plan = resource_coordinator.plan(decision,
        ResourceSnapshot(active_workers=tuple(payload.active_workers)))
    return {**asdict(decision),
            "resource_actions":[action.label for action in plan.actions],
            "resource_plan":asdict(plan)}


@app.get("/debug/resources")
def debug_resources(_: None = Depends(verify_admin_password)) -> dict:
    return resource_coordinator.debug_state()


@app.post("/debug/resources/plan")
def debug_resource_plan(payload: ResourcePlanRequest,
                        _: None = Depends(verify_admin_password)) -> dict:
    worker = next((item for item in model_router.workers
                   if item.worker_id == payload.worker_id), None)
    if payload.worker_id and worker is None:
        raise HTTPException(status_code=400, detail="unknown worker")
    decision = RouteDecision("diagnostic.resource-plan", bool(worker), None,
        worker.worker_id if worker else None, worker.model_name if worker else None,
        worker.runtime if worker else None, "diagnostic resource plan")
    plan = resource_coordinator.plan(decision,
        ResourceSnapshot(active_workers=tuple(payload.active_workers)))
    return asdict(plan)


@app.post("/orbits")
def create_orbit(payload: OrbitCreateRequest, _: None = Depends(verify_admin_password)) -> dict:
    orbit = orbit_store().create_orbit(payload.title, payload.description,
        importance=payload.importance, baseline_activation=payload.baseline_activation,
        related_entities=payload.related_entities, metadata=payload.metadata)
    continuum_service().relate(f"orbit:{orbit.orbit_id}", "participates_in", "continuum:ava",
        confidence=1.0, source="orbit", metadata={"orbit_id":orbit.orbit_id})
    for entity_id in orbit.related_entities:
        continuum_service().relate(f"orbit:{orbit.orbit_id}", "related_to", entity_id,
            confidence=1.0, source="orbit", metadata={"orbit_id":orbit.orbit_id})
    return asdict(orbit)


@app.post("/orbits/{orbit_id}/{action}")
def update_orbit(orbit_id: str, action: str, payload: OrbitActionRequest,
                 _: None = Depends(verify_admin_password)) -> dict:
    store = orbit_store()
    try:
        if action == "progress": orbit = store.record_progress(orbit_id, payload.text, kind=payload.kind)
        elif action == "hypothesis": orbit = store.add_hypothesis(orbit_id, payload.text)
        elif action == "question": orbit = store.add_question(orbit_id, payload.text)
        elif action == "resolve": orbit = store.resolve(orbit_id, payload.text)
        elif action == "reopen": orbit = store.reopen(orbit_id)
        else: raise HTTPException(status_code=400, detail="unsupported orbit action")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="orbit not found") from exc
    return asdict(orbit)


@app.post("/questions/candidates")
def create_orbit_question(payload: QuestionCandidateRequest,
                          _: None = Depends(verify_admin_password)) -> dict:
    try:
        candidate = orbit_store().create_question_candidate(payload.orbit_id, payload.question,
            importance=payload.importance, reason=payload.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="orbit not found") from exc
    return {"created":candidate is not None, "item":asdict(candidate) if candidate else None,
            "automatic_delivery_enabled":False}


@app.post("/debug/task-drive/run")
def run_task_drive(_: None = Depends(verify_admin_password)) -> dict:
    tasks = orbit_store().run_task_drive(enabled=settings.task_drive_enabled,
        minimum_interval_seconds=settings.task_drive_minimum_interval_seconds,
        max_tasks=settings.task_drive_max_tasks_per_cycle,
        priority_threshold=settings.task_drive_priority_threshold)
    return {"enabled":settings.task_drive_enabled, "created":[asdict(x) for x in tasks]}


@app.get("/debug/perception")
def debug_perception(_: None = Depends(verify_admin_password)) -> dict:
    data = continuum_service()._read(settings.persons_path, {})
    return {"enabled": settings.camera_enabled, "camera_enabled": settings.camera_enabled,
            **camera_perception_service().state(), "last_observation": data.get("last_observation"),
            "persons": [vars(x) for x in continuum_service().persons().values()]}


@app.post("/cognitive/command")
def cognitive_command(payload: CommandEventRequest, _: None = Depends(verify_admin_password)) -> dict:
    service = continuum_service()
    event = (service.command_result(session_id=payload.session_id, command=payload.command,
             content=payload.content, cycle_id=payload.cycle_id or "missing-cycle",
             status=payload.status or "success") if payload.result else
             service.command(session_id=payload.session_id, command=payload.command,
             content=payload.content, cycle_id=payload.cycle_id, person_id=payload.person_id))
    return vars(event)


@app.post("/cognitive/perception")
def cognitive_perception(payload: PerceptionEventRequest, _: None = Depends(verify_admin_password)) -> dict:
    observation = VisualObservation(payload.scene_description, payload.persons, payload.objects,
                                    payload.relations, payload.confidence)
    return {"events": [vars(x) for x in continuum_service().observe(observation, session_id=payload.session_id)]}


@app.post("/perception/camera")
def request_camera_perception(payload: CameraPerceptionRequest,
                              _: None = Depends(verify_admin_password)) -> dict:
    try:
        operation = "vision.camera" if payload.include_scene else (
            "telegram:/idcheck" if payload.reason == "idcheck" else "telegram:/who"
        )
        route_decision = model_router.route(profile_for_operation(operation))
        return asdict(camera_perception_service(route_decision).request(reason=payload.reason, force=payload.force,
            include_scene=payload.include_scene, session_id=payload.session_id,
            scene_language=payload.scene_language))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"camera perception failed: {exc}") from exc


@app.get("/debug/workspace")
async def debug_workspace(_: None = Depends(verify_admin_password)) -> dict:
    return read_workspace_debug(
        settings.workspace_path,
        enabled=getattr(settings, "jspace_enabled", False),
    )


@app.get("/debug/workspace/history")
async def debug_workspace_history(_: None = Depends(verify_admin_password)) -> dict:
    return {"enabled": getattr(settings, "jspace_enabled", False), "history": read_workspace_history(settings.workspace_path)}


@app.post("/debug/workspace/cycle")
def debug_workspace_cycle(
    payload: WorkspaceCycleDebugRequest,
    _: None = Depends(verify_admin_password),
) -> dict:
    if payload.attention_mode not in {"focused", "associative", "urgent"}:
        raise HTTPException(status_code=400, detail="invalid attention mode")
    if not getattr(settings, "jspace_enabled", False):
        return {"enabled": False, "message": "JSpace is disabled"}
    snapshot = run_workspace_cycle(
        jspace_path=settings.jspace_path, workspace_path=settings.workspace_path,
        stimulus=payload.text, attention_mode=payload.attention_mode,
        max_active_items=settings.workspace_max_active_items,
        max_latent_items=settings.workspace_max_latent_items,
        min_activation=settings.workspace_min_activation,
        decay_factor=getattr(settings, "workspace_decay_conversation", settings.workspace_decay_factor),
        general_decay_factor=getattr(settings, "workspace_decay_general", .92),
        max_per_source=settings.workspace_max_per_source,
        max_per_kind=settings.workspace_max_per_kind,
        history_limit=settings.workspace_history_limit,
        trigger="debug_override",
    )
    return read_workspace_debug(settings.workspace_path, enabled=True)


# -----------------------------------------------------------------------------
# Memory routes
# -----------------------------------------------------------------------------

@app.get("/memories")
def memories(scope: str | None = None, limit: int = 20) -> dict:
    return {"items": store.list_memories(scope=scope, limit=limit)}


@app.post("/memories")
def create_memory(payload: MemoryCreateRequest) -> dict:
    memory_id = store.add_memory(
        scope=payload.scope,
        title=payload.title,
        content=payload.content,
        tags=payload.tags,
        importance=payload.importance,
    )
    return {"ok": True, "id": memory_id}


@app.get("/memories/items")
def memory_items(
    status: str | None = None,
    memory_type: str | None = None,
    scope: str | None = None,
    limit: int = 50,
    _: None = Depends(verify_admin_password),
) -> dict:
    items = store.list_memory_items(
        status=status,
        memory_type=memory_type,
        scope=scope,
        limit=limit,
    )
    return {"items": items}


@app.get("/memories/candidates")
def memory_candidates(
    limit: int = 50,
    scope: str | None = None,
    _: None = Depends(verify_admin_password),
) -> dict:
    return {
        "items": store.list_memory_items(status="candidate", scope=scope, limit=limit)
    }


@app.get("/memories/verified")
def memory_verified(
    limit: int = 50,
    scope: str | None = None,
    _: None = Depends(verify_admin_password),
) -> dict:
    return {
        "items": store.list_memory_items(status="verified", scope=scope, limit=limit)
    }


@app.get("/memories/rejected")
def memory_rejected(
    limit: int = 50,
    scope: str | None = None,
    _: None = Depends(verify_admin_password),
) -> dict:
    return {
        "items": store.list_memory_items(status="rejected", scope=scope, limit=limit)
    }


@app.get("/memories/items/{memory_id}")
def memory_item(memory_id: int, _: None = Depends(verify_admin_password)) -> dict:
    item = store.get_memory_item(memory_id)
    if not item:
        raise HTTPException(status_code=404, detail="memory item not found")
    return {"item": item}


@app.post("/memories/items")
def create_memory_item(
    payload: MemoryItemCreateRequest,
    _: None = Depends(verify_admin_password),
) -> dict:
    memory_id = store.create_memory_item(
        scope=payload.scope,
        title=payload.title,
        content=payload.content,
        memory_type=payload.memory_type,
        status=payload.status,
        source_type=payload.source_type,
        source_ref=payload.source_ref,
        confidence=payload.confidence,
        importance=payload.importance,
        tags=payload.tags,
        created_from_user_text=payload.created_from_user_text,
        created_from_assistant_text=payload.created_from_assistant_text,
    )
    return {"ok": True, "id": memory_id}


@app.post("/memories/items/{memory_id}/verify")
def verify_memory_item(
    memory_id: int,
    payload: MemoryItemReviewRequest,
    _: None = Depends(verify_admin_password),
) -> dict:
    ok = store.verify_memory_item(memory_id, verified_by=payload.actor)
    if not ok:
        raise HTTPException(status_code=404, detail="memory item not found")
    return {"ok": True, "id": memory_id, "status": "verified"}


@app.post("/memories/items/{memory_id}/reject")
def reject_memory_item(
    memory_id: int,
    payload: MemoryItemReviewRequest,
    _: None = Depends(verify_admin_password),
) -> dict:
    ok = store.reject_memory_item(memory_id, rejected_by=payload.actor)
    if not ok:
        raise HTTPException(status_code=404, detail="memory item not found")
    return {"ok": True, "id": memory_id, "status": "rejected"}


@app.delete("/memories/items/{memory_id}")
def delete_memory_item(memory_id: int, _: None = Depends(verify_admin_password)) -> dict:
    ok = store.delete_memory_item(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="memory item not found")
    return {"ok": True, "id": memory_id}


# -----------------------------------------------------------------------------
# Knowledge / RAG routes
# -----------------------------------------------------------------------------

@app.get("/knowledge/search")
def knowledge_search(q: str, top_k: int | None = None) -> dict:
    raw_results = retriever.search(q, top_k=top_k or settings.rag_top_k)
    selected = select_rag_hits(raw_results)
    return {"items": selected}


@app.get("/knowledge/documents")
def knowledge_documents(q: str = "", limit: int = 20) -> dict:
    items = store.find_knowledge_documents_by_title(q, limit=limit)
    return {"items": items}


@app.post("/knowledge/page")
def knowledge_page(payload: KnowledgePageRequest) -> dict:
    docs = store.find_knowledge_documents_by_title(payload.document, limit=5)
    if not docs:
        raise HTTPException(status_code=404, detail="document not found")

    doc = docs[0]
    chunks = store.get_knowledge_chunks_for_document_page(doc["id"], payload.page)
    images = store.get_knowledge_images_for_document_page(doc["id"], payload.page)

    if not chunks and not images:
        raise HTTPException(status_code=404, detail="page not found or empty")

    return {
        "ok": True,
        "document": doc,
        "page": payload.page,
        "chunks": chunks,
        "images": images,
    }


@app.post("/knowledge/explain_page")
def knowledge_explain_page(payload: KnowledgePageRequest) -> dict:
    try:
        explained, error = explain_document_page(payload.document, payload.page)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if explained is None:
        if error == "document not found":
            raise HTTPException(status_code=404, detail="document not found")
        raise HTTPException(status_code=404, detail="page not found or empty")

    return {
        "ok": True,
        "document": explained["document"],
        "page": explained["page"],
        "answer": explained["answer"],
    }


# -----------------------------------------------------------------------------
# Mail routes
# -----------------------------------------------------------------------------

@app.post("/mail/send")
def mail_send(payload: MailSendRequest) -> dict:
    rule = policy_engine.resolve("external", "send_mail", channel="api", user_id=None)
    if rule and rule.mode == "deny":
        raise HTTPException(status_code=403, detail="mail sending denied by policy")

    try:
        mail_service.send_allowed_mail(to=payload.to, subject=payload.subject, body=payload.body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"mail send failed: {exc}") from exc

    return {"ok": True, "to": payload.to, "subject": payload.subject}


@app.post("/mail/send_python_script")
def mail_send_python_script(payload: MailScriptRequest) -> dict:
    try:
        mail_service.send_python_script_mail(
            script_name=payload.script_name,
            script_body=payload.script_body,
            to=payload.to,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"mail script send failed: {exc}") from exc

    return {"ok": True, "to": payload.to, "script_name": payload.script_name}


@app.post("/mail/send_important_note")
def mail_send_important_note(payload: MailNoteRequest) -> dict:
    try:
        mail_service.send_important_note_mail(title=payload.title, note=payload.note, to=payload.to)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"mail note send failed: {exc}") from exc

    return {"ok": True, "to": payload.to, "title": payload.title}


@app.get("/mail/inbox")
def mail_inbox(limit: int = 10) -> dict:
    if not settings.mail_imap_host or not settings.mail_username or not settings.mail_password:
        raise HTTPException(status_code=400, detail="mail configuration incomplete")
    try:
        items = mail_service.list_recent(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"mail inbox failed: {exc}") from exc
    return {"ok": True, "items": items}


@app.get("/mail/digest")
def mail_digest(limit: int = 8) -> dict:
    if not settings.mail_imap_host or not settings.mail_username or not settings.mail_password:
        raise HTTPException(status_code=400, detail="mail configuration incomplete")
    try:
        digest = mail_service.build_digest(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"mail digest failed: {exc}") from exc
    return {"ok": True, "digest": digest}


# -----------------------------------------------------------------------------
# Tools: weather, feeds, web fetch / web ask, vision
# -----------------------------------------------------------------------------

@app.post("/tools/weather")
def tools_weather(payload: WeatherRequest) -> dict:
    rule = policy_engine.resolve("web", "web_fetch", channel="api", user_id=None)
    if rule and rule.mode == "deny":
        raise HTTPException(status_code=403, detail="weather fetch denied by policy")

    raw_location = payload.location if payload.location is not None else settings.default_location
    location = str(raw_location).strip()

    if not location:
        raise HTTPException(status_code=400, detail="missing location")

    try:
        data = fetch_weather(location)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"weather fetch failed: {exc}") from exc

    return {"ok": True, "weather": data}


@app.post("/vision/detect_mode")
def vision_detect_mode(payload: VisionDescribeRequest) -> dict:
    mode = detect_image_mode(Path(payload.image_path), ocr_text=payload.ocr_text)
    return {"ok": True, "mode": mode}


@app.post("/vision/describe_image")
def vision_describe_image(payload: VisionDescribeRequest) -> dict:
    try:
        caption = describe_image_with_smolvlm(
            Path(payload.image_path),
            ocr_text=payload.ocr_text,
            mode=payload.mode,
            camera_language=payload.camera_language,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"vision describe failed: {exc}") from exc

    return {"ok": True, "caption": caption}


@app.get("/tools/mediumdigest")
def tools_mediumdigest(limit: int = 5) -> dict:
    if not settings.medium_feeds:
        return {"ok": True, "digest": "Keine Medium-Feeds konfiguriert."}

    items = fetch_feeds(settings.medium_feeds, limit_per_feed=max(limit, 1))
    digest = build_feed_digest(items[:limit], "Medium")
    return {"ok": True, "digest": digest, "count": len(items[:limit])}


@app.get("/tools/newsdigest")
def tools_newsdigest(limit: int = 5) -> dict:
    if not settings.news_feeds:
        return {"ok": True, "digest": "Keine News-Feeds konfiguriert."}

    items = fetch_feeds(settings.news_feeds, limit_per_feed=max(limit, 1))
    digest = build_feed_digest(items[:limit], "News")
    return {"ok": True, "digest": digest, "count": len(items[:limit])}


@app.get("/tools/medium")
def tools_medium(limit: int = 5) -> dict:
    if not settings.medium_feeds:
        return {"ok": True, "items": []}

    items = fetch_feeds(settings.medium_feeds, limit_per_feed=max(limit, 1))
    return {"ok": True, "items": items[:limit]}


@app.get("/tools/news")
def tools_news(limit: int = 5) -> dict:
    if not settings.news_feeds:
        return {"ok": True, "items": []}

    items = fetch_feeds(settings.news_feeds, limit_per_feed=max(limit, 1))
    return {"ok": True, "items": items[:limit]}


@app.post("/tools/web_fetch")
def tools_web_fetch(payload: WebFetchRequest) -> dict:
    rule = policy_engine.resolve("web", "web_fetch", channel="api", user_id=None)
    if rule and rule.mode == "deny":
        raise HTTPException(status_code=403, detail="web_fetch denied by policy")

    try:
        text = fetch_url_text(payload.url, timeout=20)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    shortened = text[:4000]
    return {
        "ok": True,
        "url": payload.url,
        "text": shortened,
        "truncated": len(text) > len(shortened),
    }


@app.post("/tools/web_ask")
def tools_web_ask(payload: WebAskRequest) -> dict:
    rule = policy_engine.resolve("web", "web_fetch", channel="api", user_id=None)
    if rule and rule.mode == "deny":
        raise HTTPException(status_code=403, detail="web_fetch denied by policy")

    try:
        page_text = fetch_url_text(payload.url, timeout=20)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    page_text = page_text[:6000]

    system_prompt = (
        "Du beantwortest Fragen zu einer geladenen Webseite. "
        "Antworte kurz, präzise und nur auf Basis des bereitgestellten Seitentexts. "
        "Wenn die Information nicht klar im Seitentext steht, sage das offen."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"URL: {payload.url}\n\n"
                f"Frage: {payload.question}\n\n"
                f"Seitentext:\n{page_text}"
            ),
        },
    ]

    try:
        ensure_ollama_runtime()
        answer = backend.chat(messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = answer.rstrip() + f"\n\nQuelle:\n{payload.url}"

    return {"ok": True, "url": payload.url, "question": payload.question, "answer": answer}


# -----------------------------------------------------------------------------
# Camera / calendar / browser / research / decision routes
# -----------------------------------------------------------------------------

@app.post("/camera/snapshot")
def camera_snapshot() -> dict:
    if not getattr(settings, "camera_enabled", False):
        raise HTTPException(status_code=400, detail="camera is disabled")

    if not getattr(settings, "camera_ip", ""):
        raise HTTPException(status_code=400, detail="camera IP is not configured")

    try:
        url = build_rtsp_url(
            user=getattr(settings, "camera_user", "admin"),
            password=getattr(settings, "camera_password", ""),
            ip=settings.camera_ip,
            rtsp_path=getattr(settings, "camera_rtsp_path", "/play1.sdp"),
        )

        image_path = capture_rtsp_snapshot(
            url=url,
            output_dir=getattr(settings, "camera_cache_dir", Path("./data/cache/camera")),
            camera_name="dlink-dcs-5222l",
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"camera snapshot failed: {exc}") from exc

    return {"ok": True, "image_path": str(image_path)}


@app.post("/briefing/calendar")
def briefing_calendar(
    payload: CalendarBriefingRequest,
    _: None = Depends(verify_admin_password),
) -> dict:
    if not getattr(settings, "calendar_ics_url", ""):
        raise HTTPException(status_code=400, detail="calendar ICS URL is not configured")

    try:
        target_day = None
        if payload.date:
            target_day = datetime.fromisoformat(payload.date).date()

        return build_daily_calendar_briefing(
            ics_url=settings.calendar_ics_url,
            target_day=target_day,
            timezone=getattr(settings, "daily_briefing_timezone", "Europe/Zurich"),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"calendar briefing failed: {exc}") from exc


@app.get("/browser/status")
def browser_status(_: None = Depends(verify_admin_password)) -> dict:
    ensure_browser_enabled()
    try:
        return run_browser_task(lambda: get_browser_controller().status())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"browser status failed: {exc}") from exc


@app.post("/browser/open")
def browser_open(
    payload: BrowserOpenRequest,
    _: None = Depends(verify_admin_password),
) -> dict:
    ensure_browser_enabled()
    try:
        return run_browser_task(lambda: get_browser_controller().open_url(payload.url))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"browser open failed: {exc}") from exc


@app.post("/browser/search")
def browser_search(
    payload: BrowserSearchRequest,
    _: None = Depends(verify_admin_password),
) -> dict:
    ensure_browser_enabled()
    try:
        return run_browser_task(lambda: get_browser_controller().search(payload.query))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"browser search failed: {exc}") from exc


@app.post("/browser/text")
def browser_text(
    payload: BrowserTextRequest,
    _: None = Depends(verify_admin_password),
) -> dict:
    ensure_browser_enabled()
    try:
        return run_browser_task(lambda: get_browser_controller().get_text(max_chars=payload.max_chars))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"browser text failed: {exc}") from exc


@app.post("/browser/screenshot")
def browser_screenshot(
    payload: BrowserScreenshotRequest,
    _: None = Depends(verify_admin_password),
) -> dict:
    ensure_browser_enabled()
    try:
        return run_browser_task(lambda: get_browser_controller().screenshot(full_page=payload.full_page))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"browser screenshot failed: {exc}") from exc


@app.post("/browser/close")
def browser_close(_: None = Depends(verify_admin_password)) -> dict:
    ensure_browser_enabled()
    try:
        return run_browser_task(lambda: get_browser_controller().close())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"browser close failed: {exc}") from exc


@app.post("/calendar/browser_day")
def calendar_browser_day(_: None = Depends(verify_admin_password)) -> dict:
    ensure_browser_enabled()
    calendar_url = "https://calendar.google.com/calendar/u/0/r/day"

    try:
        def task():
            controller = get_browser_controller()
            controller.open_url(calendar_url)
            return controller.get_text(max_chars=12000)

        page_text = run_browser_task(task)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"calendar browser read failed: {exc}") from exc

    return {
        "ok": True,
        "source": "browser",
        "calendar_url": calendar_url,
        "title": page_text.get("title", ""),
        "url": page_text.get("url", ""),
        "text": page_text.get("text", ""),
        "truncated": page_text.get("truncated", False),
    }


@app.post("/research")
def research(
    payload: ResearchRequest,
    _: None = Depends(verify_admin_password),
) -> dict:
    return run_research_workflow(
        query=payload.query,
        max_results=payload.max_results,
        save_memory=payload.save_memory,
    )


@app.get("/debug/research_queue")
def debug_research_queue(_: None = Depends(verify_admin_password)) -> dict:
    try:
        queue = autonomous_research_service().load_queue()
        return {"ok": True, "status": "idle", "queue": queue.to_dict()}
    except Exception as exc:
        return {"ok": False, "status": "failed", "error": str(exc)}


@app.post("/research/autonomous/derive")
def autonomous_research_derive(
    _: None = Depends(verify_admin_password),
) -> dict:
    try:
        return autonomous_research_service().derive()
    except Exception as exc:
        return {"ok": False, "status": "failed", "error": str(exc)}


@app.post("/research/autonomous/run-next")
def autonomous_research_run_next(
    _: None = Depends(verify_admin_password),
) -> dict:
    try:
        return autonomous_research_service().run_next()
    except Exception as exc:
        return {"ok": False, "status": "failed", "error": str(exc)}


@app.post("/research/autonomous/topics/{topic_id}/dismiss")
def autonomous_research_dismiss(
    topic_id: str,
    _: None = Depends(verify_admin_password),
) -> dict:
    try:
        return autonomous_research_service().dismiss(topic_id)
    except Exception as exc:
        return {"ok": False, "status": "failed", "error": str(exc)}


@app.post("/debug/decision")
def debug_decision(
    payload: DecisionDebugRequest,
    _: None = Depends(verify_admin_password),
) -> dict:
    return decide_context(payload.text).to_dict()


# -----------------------------------------------------------------------------
# Main reply route
# -----------------------------------------------------------------------------

@app.post("/reply", response_model=ReplyResponse)
def reply(payload: ReplyRequest) -> ReplyResponse:
    session_id = f"{payload.channel}:{payload.chat_id}"

    store.upsert_session(
        session_id=session_id,
        channel=payload.channel,
        user_id=payload.user_id,
        chat_id=payload.chat_id,
    )

    decision = decide_context(payload.text)

    try:
        append_daily_note(
            brain_dir=getattr(settings, "brain_dir", Path("./data/brain")),
            text=f"User asked: {payload.text[:500]} | Decision: {decision.to_dict()}",
            section="Interactions",
            timezone=getattr(settings, "daily_briefing_timezone", "Europe/Zurich"),
        )
    except Exception:
        # Daily notes should never break the reply path.
        pass

    user_memory_ids = maybe_store_auto_memory(payload.text)

    messages, rag_hits, _decision_dict = get_hybrid_context(
        payload.text,
        session_id=session_id,
        language=payload.language,
    )

    identity_answer = answer_identity_question(
        payload.text,
        language=payload.language,
        assistant_name=settings.assistant_name,
        system_name=settings.system_name,
        model_name=settings.ollama_model,
    )
    if identity_answer is not None:
        return finalize_reply(
            session_id=session_id,
            user_text=payload.text,
            answer=identity_answer,
            rag_hits=[],
            user_memory_ids=user_memory_ids,
        )

    ensure_ollama_runtime()

    name_question = payload.text.strip().lower()
    if name_question in {
        "wer ist dein vater",
        "wer ist dein schöpfer",
        "wer hat dich erschaffen",
        "who created you",
        "who is your creator",
    }:
        answer = "Mein Schöpfer, Vater und primärer Nutzer ist Roger Seeberger."
        return finalize_reply(
            session_id=session_id,
            user_text=payload.text,
            answer=answer,
            rag_hits=[],
            user_memory_ids=user_memory_ids,
        )

    document_query, requested_page = extract_document_page_request(payload.text)
    if settings.debug:
        print("DOC PAGE DETECT:", repr(payload.text), "->", repr(document_query), requested_page)

    if document_query and requested_page:
        try:
            explained, error = explain_document_page(document_query, requested_page)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if explained is not None:
            if settings.debug:
                print("DIRECT PAGE HIT:", explained["document"]["title"], "page", explained["page"])

            answer = explained["answer"].rstrip()
            answer += f"\n\nQuelle:\n{explained['document']['title']}, Seite {explained['page']}"

            return finalize_reply(
                session_id=session_id,
                user_text=payload.text,
                answer=answer,
                rag_hits=[],
                user_memory_ids=user_memory_ids,
            )

        if error == "document not found":
            answer = f'Ich konnte kein passendes Dokument zu "{document_query}" finden.'
            return finalize_reply(
                session_id=session_id,
                user_text=payload.text,
                answer=answer,
                rag_hits=[],
                user_memory_ids=user_memory_ids,
            )

        if error == "page not found or empty":
            answer = (
                f"Ich habe das Dokument gefunden, aber Seite {requested_page} "
                f"ist nicht vorhanden oder enthält keine verarbeitbaren Inhalte."
            )
            return finalize_reply(
                session_id=session_id,
                user_text=payload.text,
                answer=answer,
                rag_hits=[],
                user_memory_ids=user_memory_ids,
            )

    if looks_like_code_request(payload.text):
        rule = policy_engine.resolve(
            "coding",
            "generate_code",
            channel=payload.channel,
            user_id=payload.user_id,
        )

        if rule and rule.mode == "ask":
            answer = (
                "Bevor ich Code erstelle: "
                "möchtest du zuerst nur ein Konzept/eine Lösungsskizze "
                "oder direkt konkreten Code?"
            )
            return finalize_reply(
                session_id=session_id,
                user_text=payload.text,
                answer=answer,
                rag_hits=[],
                user_memory_ids=user_memory_ids,
            )

        if rule and rule.mode == "deny":
            answer = "Code-Erzeugung ist aktuell durch Policy gesperrt."
            return finalize_reply(
                session_id=session_id,
                user_text=payload.text,
                answer=answer,
                rag_hits=[],
                user_memory_ids=user_memory_ids,
            )

    auto_research_mode = getattr(settings, "auto_research", "ask").strip().lower()
    if decision.needs_research and auto_research_mode == "ask":
        answer = (
            "Dazu brauche ich wahrscheinlich aktuelle oder externe Webinformationen. "
            "Starte bitte `/research <deine Frage>` oder stelle AVACORE_AUTO_RESEARCH=auto, "
            "wenn Ava bei harmlosen Sachfragen selbst recherchieren darf."
        )
        return finalize_reply(
            session_id=session_id,
            user_text=payload.text,
            answer=answer,
            rag_hits=[],
            user_memory_ids=user_memory_ids,
        )

    if decision.needs_research and auto_research_mode == "auto":
        research_result = run_research_workflow(
            query=payload.text,
            max_results=getattr(settings, "research_max_results", 4),
            save_memory=True,
        )
        answer = research_result.get("answer", "Recherche abgeschlossen, aber ohne Antworttext.")
        memory_id = research_result.get("memory_id")
        if memory_id:
            answer = answer.rstrip() + f"\n\n[Research-Memory-Candidate: #{memory_id}]"
        return finalize_reply(
            session_id=session_id,
            user_text=payload.text,
            answer=answer,
            rag_hits=[],
            user_memory_ids=user_memory_ids,
        )

    try:
        llm_started = time.perf_counter()
        route_decision = model_router.route(profile_for_operation("dialogue.reply"))
        with resource_coordinator.lease(route_decision):
            answer = backend.chat(messages)
        if session_id in _pending_cognitive_cycles:
            _pending_cognitive_cycles[session_id]["llm_ms"] = (time.perf_counter() - llm_started) * 1000
    except Exception as exc:
        _pending_cognitive_cycles.pop(session_id, None)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return finalize_reply(
        session_id=session_id,
        user_text=payload.text,
        answer=answer,
        rag_hits=rag_hits,
        user_memory_ids=user_memory_ids,
    )


@app.delete("/reply")
def reset_reply(payload: ResetRequest) -> dict:
    session_id = f"{payload.channel}:{payload.chat_id}"
    store.reset_session_messages(session_id)
    return {"ok": True, "session_id": session_id}
