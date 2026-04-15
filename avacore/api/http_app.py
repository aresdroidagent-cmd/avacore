from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from avacore.config.settings import settings
from avacore.config.personality_loader import (
    load_personality_json_text,
    load_personality_manager,
)
from avacore.core.dto import HealthStatus
from avacore.core.prompts import looks_like_code_request
from avacore.memory.sqlite_store import SQLiteStore
from avacore.memory.auto_memory import AutoMemoryExtractor
from avacore.model.ollama_backend import OllamaBackend
from avacore.policy.engine import PolicyEngine
from avacore.rag.embedder import Embedder
from avacore.rag.retriever import Retriever
from avacore.tools.web_fetch import fetch_url_text
from avacore.tools.weather_fetch import fetch_weather
from avacore.tools.rss_fetch import fetch_feeds
from avacore.mail.service import MailService
from avacore.vision.describe import describe_image_with_smolvlm, detect_image_mode
from avacore.system.ollama_runtime import start_ollama_server


_ollama_process = None

WEB_STATIC_DIR = Path(__file__).resolve().parents[2] / "web" / "static"
AVA_AVATAR_PATH = settings.web_avatar_path

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


def verify_admin_password(x_admin_password: str | None = Header(default=None)) -> None:
    expected = (getattr(settings, "web_admin_password", "") or "").strip()
    if not expected:
        raise HTTPException(status_code=503, detail="Admin password is not configured.")
    if not x_admin_password or x_admin_password != expected:
        raise HTTPException(status_code=401, detail="Invalid admin password.")


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


class KnowledgePageRequest(BaseModel):
    document: str
    page: int


class ReplyRequest(BaseModel):
    channel: str
    user_id: str
    chat_id: str
    text: str
    timestamp: int


class ReplyResponse(BaseModel):
    reply: str


class ResetRequest(BaseModel):
    chat_id: str


class MemoryCreateRequest(BaseModel):
    scope: str
    title: str
    content: str
    tags: str = ""
    importance: int = 0


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
) -> str:
    profile = load_active_personality_profile()
    base = personality_manager.render_system_prompt(profile)

    parts = [base]

    if memory_scope:
        memory_lines = store.get_memory_prompt_lines(scope=memory_scope, limit=5)
        if memory_lines:
            parts.append("Relevante bekannte Erinnerungen:\n" + "\n".join(memory_lines))

    if rag_hits:
        rag_lines = []
        for hit in rag_hits:
            label = hit["title"]
            if hit.get("page_number"):
                label += f" (Seite {hit['page_number']})"
            score = float(hit.get("score", 0.0))
            rag_lines.append(f"- {label} [Score {score:.2f}]: {hit['content']}")
        if rag_lines:
            parts.append(
                "Relevante Dokumentauszüge aus der Wissensbasis:\n" + "\n".join(rag_lines)
            )

    return "\n\n".join(parts)


def maybe_store_auto_memory(user_text: str) -> list[int]:
    candidates = auto_memory_extractor.extract(user_text)
    stored_ids: list[int] = []

    for candidate in candidates:
        new_id = store.add_memory_if_new(
            scope="user",
            title=candidate.title,
            content=candidate.content,
            tags=candidate.tags,
            importance=candidate.importance,
        )
        if new_id is not None:
            stored_ids.append(new_id)

    return stored_ids


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
        r"(?i)erkläre\s+(?:das\s+)?dokument\s+(.+?)\s+seite\s+(\d+)",
        r"(?i)was\s+siehst\s+du\s+im\s+dokument\s+(.+?)\s+auf\s+seite\s+(\d+)",
        r"(?i)fasse\s+(?:das\s+)?dokument\s+(.+?)\s+seite\s+(\d+)\s+zusammen",
        r"(?i)dokument\s+(.+?)\s+seite\s+(\d+)",
        r"(?i)in\s+(.+?)\s+auf\s+seite\s+(\d+)",
        r"(?i)(.+?)\s+auf\s+seite\s+(\d+)",
        r"(?i)page\s+(\d+)\s+of\s+(.+)",
        r"(?i)(.+?)\s+page\s+(\d+)",
        r"(?i)(.+?)\s+seite\s+(\d+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if not m:
            continue

        if pattern == r"(?i)page\s+(\d+)\s+of\s+(.+)":
            try:
                document = m.group(2).strip(" .,:;!?\"'`()[]{}")
                page = int(m.group(1).strip())
                return document, page
            except ValueError:
                return None, None

        try:
            document = m.group(1).strip(" .,:;!?\"'`()[]{}")
            page = int(m.group(2).strip())
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
    docs = store.find_knowledge_documents_by_title(document_query, limit=5)
    if not docs:
        if settings.debug:
            print("DIRECT DOC MISS:", document_query)
        return None, "document not found"

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

    user_prompt = (
        f"Dokument: {doc['title']}\n"
        f"Seite: {page}\n\n"
        f"{context}"
    )

    ensure_ollama_runtime()
    answer = backend.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return {"document": doc, "page": page, "answer": answer}, None


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


@app.get("/knowledge/search")
def knowledge_search(q: str, top_k: int | None = None) -> dict:
    raw_results = retriever.search(q, top_k=top_k or settings.rag_top_k)
    selected = select_rag_hits(raw_results)
    return {"items": selected}


@app.get("/knowledge/documents")
def knowledge_documents(q: str = "", limit: int = 20) -> dict:
    items = store.find_knowledge_documents_by_title(q, limit=limit)
    return {"items": items}

@app.get("/ui/avatar", include_in_schema=False)
def ui_avatar():
    if not AVA_AVATAR_PATH.exists():
        raise HTTPException(status_code=404, detail="Avatar image not found")
    return FileResponse(AVA_AVATAR_PATH)

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


@app.post("/mail/send")
def mail_send(payload: MailSendRequest) -> dict:
    rule = policy_engine.resolve("external", "send_mail", channel="api", user_id=None)
    if rule and rule.mode == "deny":
        raise HTTPException(status_code=403, detail="mail sending denied by policy")

    try:
        mail_service.send_allowed_mail(
            to=payload.to,
            subject=payload.subject,
            body=payload.body,
        )
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
        mail_service.send_important_note_mail(
            title=payload.title,
            note=payload.note,
            to=payload.to,
        )
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


@app.post("/reply", response_model=ReplyResponse)
def reply(payload: ReplyRequest) -> ReplyResponse:
    ensure_ollama_runtime()

    session_id = f"{payload.channel}:{payload.chat_id}"

    store.upsert_session(
        session_id=session_id,
        channel=payload.channel,
        user_id=payload.user_id,
        chat_id=payload.chat_id,
    )

    auto_memory_ids = maybe_store_auto_memory(payload.text)

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
            if auto_memory_ids:
                answer += f"\n\n[Auto-Memory: {len(auto_memory_ids)} neuer Eintrag gespeichert]"

            store.add_message(session_id, "user", payload.text)
            store.add_message(session_id, "assistant", answer)
            return ReplyResponse(reply=answer)

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
            store.add_message(session_id, "user", payload.text)
            store.add_message(session_id, "assistant", answer)
            return ReplyResponse(reply=answer)

        if rule and rule.mode == "deny":
            answer = "Code-Erzeugung ist aktuell durch Policy gesperrt."
            store.add_message(session_id, "user", payload.text)
            store.add_message(session_id, "assistant", answer)
            return ReplyResponse(reply=answer)

    history = store.get_recent_messages(
        session_id=session_id,
        max_items=settings.max_history_turns,
    )
    raw_rag_hits = retriever.search(payload.text, top_k=settings.rag_top_k)
    rag_hits = select_rag_hits(raw_rag_hits)

    messages = [{"role": "system", "content": build_system_prompt(memory_scope="user", rag_hits=rag_hits)}]
    messages.extend(history)
    messages.append({"role": "user", "content": payload.text})

    try:
        answer = backend.chat(messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if rag_hits:
        answer = answer.rstrip() + format_rag_sources(rag_hits)

    if auto_memory_ids:
        answer = answer.rstrip() + f"\n\n[Auto-Memory: {len(auto_memory_ids)} neuer Eintrag gespeichert]"

    store.add_message(session_id, "user", payload.text)
    store.add_message(session_id, "assistant", answer)

    return ReplyResponse(reply=answer)


@app.delete("/reply")
def reset_reply(payload: ResetRequest) -> dict:
    session_id = f"telegram:{payload.chat_id}"
    store.reset_session_messages(session_id)
    return {"ok": True, "session_id": session_id}