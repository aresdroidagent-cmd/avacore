from __future__ import annotations
from avacore.tools.mystrom import light_on, light_off, light_status
from avacore.tools.speech_to_text import transcribe_audio_file
from avacore.tools.notes_export import export_and_sync_notes
from avacore.tools.camera_rtsp import (
    build_rtsp_url,
    capture_rtsp_snapshot,
    crop_camera_overlay,
)
from avacore.tools.identity_rag import (
    build_identity_index,
    copy_capture_to_identity_dataset,
    format_identity_decision,
    recognize_face_image,
)
import time
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import BaseRequest

from avacore.config.settings import settings
from avacore.channels.telegram import http_client
from avacore.tools.notes import (
    append_to_note,
    create_note,
    format_note,
    format_note_list,
    list_notes,
    search_notes,
    update_note_status,
)


@dataclass(frozen=True)
class CommandSpec:
    name: str
    handler: object
    description: str
    aliases: tuple[str, ...] = ()
    requires_llm: bool = False
    cognitive_visibility: bool = True


COMMAND_REGISTRY: dict[str, CommandSpec] = {}


def register_commands(specs: list[CommandSpec]) -> dict[str, CommandSpec]:
    registry: dict[str, CommandSpec] = {}
    for spec in specs:
        for name in (spec.name, *spec.aliases):
            key = name.casefold()
            if key in registry:
                raise ValueError(f"duplicate Telegram command or alias: {key}")
            registry[key] = spec
    return registry


async def _record_command(update: Update, command: str, content: str, cycle_id: str,
                          *, result: bool = False, status: str = "success") -> None:
    if not settings.command_events_enabled or not update.effective_chat:
        return
    try:
        await http_client.post(f"{api_base()}/cognitive/command", json={
            "session_id": f"telegram:{update.effective_chat.id}", "command": command,
            "content": content, "cycle_id": cycle_id, "result": result, "status": status,
            "person_id": settings.telegram_person_id,
        }, headers=admin_headers(), timeout=15)
    except Exception:
        # Cognitive observability must not make an otherwise working command
        # unavailable when the local API is temporarily restarting.
        return


def cognitive_handler(spec: CommandSpec, invoked_name: str):
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        cycle_id = f"cy_{uuid.uuid4().hex}"
        text = (update.effective_message.text if update.effective_message else None) or f"/{invoked_name}"
        await _record_command(update, spec.name, text, cycle_id)
        try:
            await spec.handler(update, context)
        except Exception as exc:
            await _record_command(update, spec.name, f"/{spec.name} failed: {type(exc).__name__}", cycle_id, result=True, status="failed")
            raise
        await _record_command(update, spec.name, f"/{spec.name} completed", cycle_id, result=True)
    return wrapped


def api_base() -> str:
    return f"http://{settings.http_host}:{settings.http_port}"


def admin_headers() -> dict:
    password = os.environ.get("AVACORE_WEB_ADMIN_PASSWORD", "").strip()
    if not password:
        return {}
    return {"X-Admin-Password": password}


def is_allowed_chat(chat_id: str) -> bool:
    allowed = (settings.telegram_allowed_chat_id or "").strip()
    return bool(allowed) and chat_id == allowed


def telegram_reply_language(context: ContextTypes.DEFAULT_TYPE) -> str:
    language = context.chat_data.get("reply_language", "de")
    return "en" if language == "en" else "de"


def default_mail_recipient() -> str | None:
    if not settings.mail_allowed_to:
        return None
    recipient = (settings.mail_allowed_to[0] or "").strip()
    return recipient or None

def detect_switch_intent(text: str) -> str | None:
    normalized = (text or "").strip().lower()

    switch_words = [
        "licht",
        "lampe",
        "switch",
        "mystrom",
        "steckdose",
    ]

    if not any(word in normalized for word in switch_words):
        return None

    on_words = [
        "einschalten",
        "ein schalten",
        "mach an",
        "mache an",
        "mach das licht an",
        "mache das licht an",
        "schalte das licht an",
        "schalt das licht an",
        "schalt licht an",
        "licht einschalten",
        "licht ein",
        "lampe ein",
        "anschalten",
        "an machen",
        "licht an",
        "lampe an",
        "switch on",
        "turn on",
    ]

    off_words = [
        "ausschalten",
        "aus schalten",
        "mach aus",
        "mache aus",
        "mach das licht aus",
        "mache das licht aus",
        "schalte das licht aus",
        "schalt das licht aus",
        "schalt licht aus",
        "licht ausschalten",
        "licht aus",
        "lampe aus",
        "switch off",
        "turn off",
    ]

    state_words = [
        "status",
        "zustand",
        "ist das licht an",
        "ist die lampe an",
        "ist eingeschaltet",
        "ist ausgeschaltet",
        "leistung",
        "verbrauch",
        "power",
        "state",
    ]

    if any(word in normalized for word in on_words):
        return "on"

    if any(word in normalized for word in off_words):
        return "off"

    if any(word in normalized for word in state_words):
        return "state"

    return None

def command_help_text() -> str:
    return (
        "Ava Befehle:\n\n"
        "Allgemein:\n"
        "/start - Ava starten\n"
        "/help - diese Übersicht\n"
        "/health - AvaCore Status\n"
        "/status - kompakter Betriebsstatus\n"
        "/model - aktives Modell\n"
        "/de - Antworten auf Deutsch\n"
        "/en - replies in English\n"
        "/personality - aktive Persönlichkeit\n"
        "/personalitybackup - Personality in SQLite sichern\n"
        "/personalityrestore <profile_id> - Personality wiederherstellen\n\n"
        "Memory / Policies:\n"
        "/memories - gespeicherte Memories anzeigen\n"
        "/remember <text> - etwas explizit merken\n"
        "/policies - aktive Policies anzeigen\n"
        "/reset - Chatverlauf zurücksetzen\n\n"
        "Dokumente / Wissen:\n"
        "/docs [suchwort] - Dokumente auflisten\n"
        "/page <dokumentname> | <seite> - konkrete Dokumentseite erklären\n\n"
        "/weather [ort] - Wetter kurz anzeigen\n"
        "/medium - aktuelle Medium-Einträge\n"
        "/news - aktuelle News-Einträge\n"
        "/mediumdigest - Medium kurz zusammenfassen\n"
        "/newsdigest - News kurz zusammenfassen\n\n"
        "/webfetch <url> - Rohtext einer Seite holen\n"
        "/webask <url> <frage> - Frage zu einer Webseite beantworten\n"
        "/browsersearch <Suchbegriff> - Websuche über kontrollierten Chromium-Browser\n"
        "/research <Frage> - Web-Recherche mit Quellen und Memory-Kandidat\n\n"
        "Mail:\n"
        "/mail - letzte Mails anzeigen\n"
        "/maildigest - Mails kurz zusammenfassen\n"
        "/sendmail <subject> | <text> - Mail an Standardempfänger senden\n"
        "/mailscript <dateiname.py> | <scriptinhalt> - Python-Script mailen\n"
        "/mailnote <titel> | <inhalt> - wichtigen Inhalt mailen\n\n"
        "/camera - aktuelles Kamerabild holen\n"        
        "/see - aktuelle visuelle Wahrnehmung anfordern\n"
        "/snapshot - Alias für /camera\n"
        "/idcapture roger - aktuelles Kamerabild als Roger-Beispiel speichern\n"
        "/idcapture unknown - aktuelles Kamerabild als Nicht-Roger-Beispiel speichern\n"
        "/idcapture empty - aktuelles Kamerabild als leere Szene speichern\n"
        "/idtrain - visuellen Identity-Index bauen\n"
        "/idcheck - aktuelle Kameraaufnahme gegen Identity-Index prüfen\n\n"
        "/switchon - Switch einschalten\n"
        "/switchoff - Switch ausschalten\n"
        "/switchstate - Status abfragen\n\n"
        "/briefing - heutiges Kalender-Briefing abrufen\n\n"
        "/note <Text> - neue lokale Notiz erfassen\n"
        "/notes [open|done|archived|all] - Notizen anzeigen\n"
        "/notesearch <Suchbegriff> - Notizen durchsuchen\n"
        "/noteadd <id> <Text> - Notiz ergänzen\n"
        "/notedone <id> - Notiz als erledigt markieren\n"
        "/notearchive <id> - Notiz archivieren\n"
        "/notesync - lokale Ava Notes als Markdown exportieren und optional zu Google Drive syncen\n\n"
        "Ava Continuum:\n"
        "/focus - aktueller Spotlight\n/continuum - Continuum-Zusammenfassung\n"
        "/workspace - Conscious Workspace\n/memory - aktuelle Working Memory\n"
        "/persons - registrierte Personen\n/who - aktuell anwesende Person\n"
        "/why - strukturierte Aktivierungsgründe\n/bsp - installationsspezifische BSP-Aktion"

    )

def detect_note_intent(text: str) -> str | None:
    original = (text or "").strip()
    if not original:
        return None

    normalized = " ".join(original.split())
    lowered = normalized.lower()

    # Remove common wake word at the beginning.
    lowered_no_wake = lowered
    original_no_wake = normalized

    for wake in ["ava,", "ava"]:
        if lowered_no_wake.startswith(wake + " "):
            cut = len(wake)
            original_no_wake = original_no_wake[cut:].strip(" ,:")
            lowered_no_wake = lowered_no_wake[cut:].strip(" ,:")
            break

    # Strong explicit patterns.
    patterns = [
        r"^(?:bitte\s+)?notiere(?:\s+bitte)?[:\s]+(.+)$",
        r"^(?:bitte\s+)?notier(?:\s+bitte)?[:\s]+(.+)$",
        r"^(?:bitte\s+)?mach(?:e)?\s+eine\s+notiz(?:\s+bitte)?[:\s]+(.+)$",
        r"^(?:bitte\s+)?erstelle\s+eine\s+notiz(?:\s+bitte)?[:\s]+(.+)$",
        r"^(?:bitte\s+)?merk(?:e)?\s+dir\s+als\s+notiz[:\s]+(.+)$",
        r"^(?:bitte\s+)?speichere\s+als\s+notiz[:\s]+(.+)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, lowered_no_wake, flags=re.IGNORECASE)
        if match:
            # Use the same span on original_no_wake where possible.
            start = match.start(1)
            note = original_no_wake[start:].strip(" .,:;")
            return note or None

    # Fallback for common Whisper variants.
    fallback_phrases = [
        "notiere",
        "notier",
        "notiert",
        "mach eine notiz",
        "mache eine notiz",
        "erstelle eine notiz",
        "speichere als notiz",
    ]

    for phrase in fallback_phrases:
        if lowered_no_wake.startswith(phrase):
            note = original_no_wake[len(phrase):].strip(" .,:;")
            return note or None

    return None


def camera_rtsp_url() -> str:
    return build_rtsp_url(
        user=settings.camera_user,
        password=settings.camera_password,
        ip=settings.camera_ip,
        rtsp_path=settings.camera_rtsp_path,
    )


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    await update.effective_message.reply_text("Ava ist bereit.\n\n" + command_help_text())


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    await update.effective_message.reply_text(command_help_text())


async def de_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    context.chat_data["reply_language"] = "de"
    await update.effective_message.reply_text("Antwortsprache: Deutsch.")


async def en_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("This chat is not authorized.")
        return

    context.chat_data["reply_language"] = "en"
    await update.effective_message.reply_text("Reply language: English.")


async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = await http_client.get(f"{api_base()}/health", timeout=15)
    if not response.ok:
        await update.effective_message.reply_text(f"Health fehlgeschlagen: {response.text}")
        return

    data = response.json()
    msg = (
        f"ok: {data.get('ok')}\n"
        f"model: {data.get('model')}\n"
        f"profile: {data.get('profile')}\n"
        f"max_history_turns: {data.get('max_history_turns')}\n"
        f"ollama_url: {data.get('ollama_url')}"
    )
    await update.effective_message.reply_text(msg)


async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = await http_client.get(f"{api_base()}/model", timeout=15)
    if not response.ok:
        await update.effective_message.reply_text(f"Model-Abfrage fehlgeschlagen: {response.text}")
        return

    data = response.json()
    await update.effective_message.reply_text(
        f"Model: {data.get('model')}\nProfile: {data.get('profile')}"
    )


async def personality_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = await http_client.get(f"{api_base()}/personality", timeout=30)
    if not response.ok:
        await update.effective_message.reply_text(f"Personality-Abfrage fehlgeschlagen: {response.text}")
        return

    data = response.json()
    text = str(data)
    if len(text) > 3800:
        text = text[:3800] + " ..."
    await update.effective_message.reply_text(text)


async def personalitybackup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = await http_client.post(
        f"{api_base()}/personality/backup",
        json={},
        timeout=30,
    )
    if not response.ok:
        await update.effective_message.reply_text(f"Backup fehlgeschlagen: {response.text}")
        return

    data = response.json()
    await update.effective_message.reply_text(
        f"Backup erstellt.\nprofile_id: {data.get('profile_id')}\nactive: {data.get('active')}"
    )


async def personalityrestore_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    profile_id = " ".join(context.args).strip()
    if not profile_id:
        await update.effective_message.reply_text("Format: /personalityrestore <profile_id>")
        return

    response = await http_client.post(
        f"{api_base()}/personality/restore",
        json={"profile_id": profile_id},
        timeout=30,
    )
    if not response.ok:
        await update.effective_message.reply_text(f"Restore fehlgeschlagen: {response.text}")
        return

    data = response.json()
    await update.effective_message.reply_text(
        f"Restore ok.\nprofile_id: {data.get('profile_id')}\nactive: {data.get('active')}"
    )


async def policies_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = await http_client.get(f"{api_base()}/policies", timeout=30)
    if not response.ok:
        await update.effective_message.reply_text(f"Policies fehlgeschlagen: {response.text}")
        return

    data = response.json()
    text = str(data.get("rules", []))
    if len(text) > 3800:
        text = text[:3800] + " ..."
    await update.effective_message.reply_text(text)


async def memories_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = await http_client.get(
        f"{api_base()}/memories", params={"limit": 20}, timeout=30
    )
    if not response.ok:
        await update.effective_message.reply_text(f"Memories fehlgeschlagen: {response.text}")
        return

    items = response.json().get("items", [])
    if not items:
        await update.effective_message.reply_text("Keine Memories gefunden.")
        return

    lines = ["Memories:"]
    for item in items[:20]:
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()
        lines.append(f"- {title}: {content}")

    out = "\n".join(lines)
    if len(out) > 3800:
        out = out[:3800] + "\n..."
    await update.effective_message.reply_text(out)


async def remember_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    content = " ".join(context.args).strip()
    if not content:
        await update.effective_message.reply_text("Format: /remember <text>")
        return

    response = await http_client.post(
        f"{api_base()}/memories",
        json={
            "scope": "user",
            "title": "Merker",
            "content": content,
            "tags": "manual",
            "importance": 5,
        },
        timeout=30,
    )
    if not response.ok:
        await update.effective_message.reply_text(f"Remember fehlgeschlagen: {response.text}")
        return

    data = response.json()
    await update.effective_message.reply_text(f"Gemerkt. id={data.get('id')}")


async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    payload = {"chat_id": str(update.effective_chat.id)}
    response = await http_client.delete(f"{api_base()}/reply", json=payload, timeout=30)

    if not response.ok:
        await update.effective_message.reply_text(f"Reset fehlgeschlagen: {response.text}")
        return

    await update.effective_message.reply_text("Chatverlauf zurückgesetzt.")


async def briefing_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    await update.effective_message.reply_text("Ich hole dein Kalender-Briefing...")

    try:
        response = await http_client.post(
            f"{api_base()}/briefing/calendar",
            json={},
            headers=admin_headers(),
            timeout=30,
        )

        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text

            await update.effective_message.reply_text(
                f"Kalender-Briefing fehlgeschlagen: {detail}"
            )
            return

        data = response.json()
        briefing = data.get("briefing", "").strip()

        if not briefing:
            briefing = "Kalender-Briefing erhalten, aber ohne Inhalt."

        await update.effective_message.reply_text(briefing)

    except Exception as exc:
        await update.effective_message.reply_text(
            f"Briefing-Befehl fehlgeschlagen: {exc}"
        )


async def notesync_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    if not settings.notes_export_enabled:
        await update.effective_message.reply_text(
            "Notes Export ist deaktiviert. Setze AVACORE_NOTES_EXPORT_ENABLED=1."
        )
        return

    try:
        result = export_and_sync_notes(
            db_path=settings.db_path,
            export_path=settings.notes_export_path,
            timezone_name=settings.daily_briefing_timezone,
            rclone_enabled=settings.notes_rclone_enabled,
            rclone_remote=settings.notes_rclone_remote,
        )

        message = (
            "Notes Export abgeschlossen.\n\n"
            f"Lokale Datei:\n{result['exported_path']}"
        )

        if result["rclone_enabled"]:
            message += (
                "\n\n"
                f"Google Drive Ziel:\n{result['rclone_remote']}\n\n"
                "Sync: abgeschlossen."
            )
        else:
            message += "\n\nRclone Sync ist deaktiviert."

        await update.effective_message.reply_text(message)

    except Exception as exc:
        await update.effective_message.reply_text(
            f"Notes Sync fehlgeschlagen: {exc}"
        )


def weather_code_label(code: int | None) -> str:
    mapping = {
        0: "klar",
        1: "überwiegend klar",
        2: "teilweise bewölkt",
        3: "bedeckt",
        45: "Nebel",
        48: "Raureifnebel",
        51: "leichter Nieselregen",
        53: "Nieselregen",
        55: "starker Nieselregen",
        61: "leichter Regen",
        63: "Regen",
        65: "starker Regen",
        71: "leichter Schneefall",
        73: "Schneefall",
        75: "starker Schneefall",
        80: "Regenschauer",
        81: "kräftige Regenschauer",
        82: "sehr kräftige Regenschauer",
        95: "Gewitter",
    }
    return mapping.get(code, f"Wettercode {code}")


def clean_camera_description(description: str) -> str:
    text = (description or "").strip()

    if not text:
        return "Die Szene ist nicht zuverlässig erkennbar."

    bad_fragments = [
        "du möchtest",
        "schriftfreiheit",
        "leben-konto",
        "lebenkonto",
        "spiel mit",
        "karte von",
        "zeitbewertung",
        "schritte verwendet",
        "there is a text",
        "the image is a screenshot",
    ]

    lowered = text.lower()

    if any(fragment in lowered for fragment in bad_fragments):
        return "Die Szene ist nicht zuverlässig erkennbar. Das Kamerabild wurde aufgenommen, aber die automatische Bildbeschreibung ist unsicher."

    # Very short OCR-only response, e.g. only timestamp/camera model.
    if "dcs-5222l" in lowered and len(text) < 80:
        return "Die reale Szene ist nicht zuverlässig erkennbar; das Modell hat hauptsächlich das Kamera-Overlay erkannt."

    return text


async def translate_camera_description_to_german(description: str) -> str:
    text = (description or "").strip()

    if not text:
        return ""

    # Wenn es schon deutsch wirkt, nicht unnötig übersetzen.
    german_markers = [
        "ich sehe",
        "eine person",
        "ein sofa",
        "eine tür",
        "wohnzimmer",
        "raum",
        "sichtbar",
        "nicht zuverlässig erkennbar",
    ]

    if any(marker in text.lower() for marker in german_markers):
        return text

    try:
        response = await http_client.post(
            f"{api_base()}/reply",
            json={
                "channel": "internal",
                "user_id": "system",
                "chat_id": "camera-translation",
                "text": (
                    "Übersetze die folgende Kamerabeschreibung ins Deutsche. "
                    "Formuliere sie kurz, sachlich und natürlich. "
                    "Erfinde keine zusätzlichen Details. "
                    "Wenn die Beschreibung unsicher klingt, behalte diese Unsicherheit bei.\n\n"
                    f"Beschreibung:\n{text}"
                ),
                "timestamp": int(time.time()),
            },
            timeout=120,
        )

        if not response.ok:
            return text

        data = response.json()
        translated = (data.get("reply") or data.get("answer") or "").strip()

        return translated or text

    except Exception:
        return text


async def weather_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    location = " ".join(context.args).strip()

    response = await http_client.post(
        f"{api_base()}/tools/weather",
        json={"location": location or None},
        timeout=30,
    )

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        await update.effective_message.reply_text(f"Wetterabfrage fehlgeschlagen: {detail}")
        return

    data = response.json().get("weather", {})
    current_temp = data.get("current_temperature")
    current_label = weather_code_label(data.get("current_weather_code"))

    dates = data.get("dates", [])
    temp_max = data.get("temp_max", [])
    temp_min = data.get("temp_min", [])
    codes = data.get("weather_codes", [])

    lines = [
        f"Wetter für {data.get('location')}:",
        f"Aktuell: {current_label}, {current_temp}°C",
    ]

    for i in range(min(2, len(dates))):
        lines.append(
            f"{dates[i]}: {weather_code_label(codes[i] if i < len(codes) else None)}, "
            f"max {temp_max[i] if i < len(temp_max) else '?'}°C, "
            f"min {temp_min[i] if i < len(temp_min) else '?'}°C"
        )

    await update.effective_message.reply_text("\n".join(lines))


async def medium_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = await http_client.get(
        f"{api_base()}/tools/medium", params={"limit": 5}, timeout=30
    )
    response.raise_for_status()
    items = response.json().get("items", [])

    if not items:
        await update.effective_message.reply_text("Keine Medium-Einträge gefunden.")
        return

    lines = ["Medium:"]
    for item in items[:5]:
        title = item.get("title", "").strip()
        source = item.get("source", "").strip()
        lines.append(f"- {title} ({source})")

    await update.effective_message.reply_text("\n".join(lines))


async def news_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = await http_client.get(
        f"{api_base()}/tools/news", params={"limit": 5}, timeout=30
    )
    response.raise_for_status()
    items = response.json().get("items", [])

    if not items:
        await update.effective_message.reply_text("Keine News-Einträge gefunden.")
        return

    lines = ["News:"]
    for item in items[:5]:
        title = item.get("title", "").strip()
        source = item.get("source", "").strip()
        lines.append(f"- {title} ({source})")

    await update.effective_message.reply_text("\n".join(lines))


async def mediumdigest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = await http_client.get(
        f"{api_base()}/tools/mediumdigest", params={"limit": 5}, timeout=180
    )
    response.raise_for_status()
    data = response.json()

    digest = data.get("digest", "")
    if len(digest) > 3800:
        digest = digest[:3800] + " ..."
    await update.effective_message.reply_text(digest or "Kein Medium-Digest verfügbar.")


async def newsdigest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = await http_client.get(
        f"{api_base()}/tools/newsdigest", params={"limit": 5}, timeout=180
    )
    response.raise_for_status()
    data = response.json()

    digest = data.get("digest", "")
    if len(digest) > 3800:
        digest = digest[:3800] + " ..."
    await update.effective_message.reply_text(digest or "Kein News-Digest verfügbar.")


async def webfetch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    url = " ".join(context.args).strip()
    if not url:
        await update.effective_message.reply_text("Bitte gib eine URL an.")
        return

    response = await http_client.post(
        f"{api_base()}/tools/web_fetch",
        json={"url": url},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    text = data.get("text", "")
    if len(text) > 3500:
        text = text[:3500] + " ..."

    await update.effective_message.reply_text(text or "Keine lesbaren Inhalte gefunden.")


async def webask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    raw = " ".join(context.args).strip()
    if not raw:
        await update.effective_message.reply_text("Bitte gib URL und Frage an.")
        return

    parts = raw.split(maxsplit=1)
    if len(parts) < 2:
        await update.effective_message.reply_text("Format: /webask <url> <frage>")
        return

    url, question = parts[0], parts[1].strip()

    response = await http_client.post(
        f"{api_base()}/tools/web_ask",
        json={"url": url, "question": question},
        headers=admin_headers(),
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    answer = data.get("answer", "")
    if len(answer) > 3500:
        answer = answer[:3500] + " ..."

    await update.effective_message.reply_text(answer or "Keine Antwort erhalten.")

async def browser_search_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    query = " ".join(context.args).strip()

    if not query:
        await update.effective_message.reply_text(
            "Bitte gib eine Suchanfrage an, z.B.:\n"
            "/browsersearch D-Link DCS-5222L RTSP play1.sdp"
        )
        return

    await update.effective_message.reply_text(f"Ich suche im Browser nach:\n{query}")

    try:
        search_response = await http_client.post(
            f"{api_base()}/browser/search",
            json={"query": query},
            headers=admin_headers(),
            timeout=60,
        )

        if not search_response.ok:
            try:
                detail = search_response.json().get("detail", search_response.text)
            except Exception:
                detail = search_response.text

            await update.effective_message.reply_text(
                f"Browser-Suche fehlgeschlagen: {detail}"
            )
            return

        text_response = await http_client.post(
            f"{api_base()}/browser/text",
            json={"max_chars": 3000},
            headers=admin_headers(),
            timeout=60,
        )

        if not text_response.ok:
            try:
                detail = text_response.json().get("detail", text_response.text)
            except Exception:
                detail = text_response.text

            await update.effective_message.reply_text(
                f"Browser-Text konnte nicht gelesen werden: {detail}"
            )
            return

        data = text_response.json()

        title = data.get("title", "(ohne Titel)")
        url = data.get("url", "")
        text = data.get("text", "").strip()

        if not text:
            text = "Kein lesbarer Text gefunden."

        message = (
            f"Browser-Suche geöffnet.\n\n"
            f"Titel: {title}\n"
            f"URL: {url}\n\n"
            f"Textauszug:\n{text[:3000]}"
        )

        await update.effective_message.reply_text(message)

    except Exception as exc:
        await update.effective_message.reply_text(
            f"Browser-Suche fehlgeschlagen: {exc}"
        )


async def mail_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    response = await http_client.get(
        f"{api_base()}/mail/inbox", params={"limit": 5}, timeout=60
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        await update.effective_message.reply_text(f"Mailabruf fehlgeschlagen: {detail}")
        return

    items = response.json().get("items", [])
    if not items:
        await update.effective_message.reply_text("Keine Mails gefunden.")
        return

    lines = ["Inbox:"]
    for item in items[:5]:
        subject = item.get("subject", "").strip()
        sender = item.get("from", "").strip()
        lines.append(f"- {subject} ({sender})")

    await update.effective_message.reply_text("\n".join(lines))


async def maildigest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    response = await http_client.get(
        f"{api_base()}/mail/digest", params={"limit": 8}, timeout=180
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        await update.effective_message.reply_text(f"Mail-Digest fehlgeschlagen: {detail}")
        return

    digest = response.json().get("digest", "")
    if len(digest) > 3800:
        digest = digest[:3800] + " ..."

    await update.effective_message.reply_text(digest or "Kein Mail-Digest verfügbar.")


async def sendmail_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    recipient = default_mail_recipient()
    if not recipient:
        await update.effective_message.reply_text(
            "Kein Standardempfänger konfiguriert. Setze AVACORE_MAIL_ALLOWED_TO in .env."
        )
        return

    raw = " ".join(context.args).strip()
    if not raw or "|" not in raw:
        await update.effective_message.reply_text("Format: /sendmail <subject> | <text>")
        return

    subject, body = [part.strip() for part in raw.split("|", 1)]

    response = await http_client.post(
        f"{api_base()}/mail/send",
        json={
            "to": recipient,
            "subject": subject,
            "body": body,
        },
        timeout=60,
    )

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        await update.effective_message.reply_text(f"Mailversand fehlgeschlagen: {detail}")
        return

    await update.effective_message.reply_text(f"Mail gesendet an {recipient}.")


async def mailscript_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    recipient = default_mail_recipient()
    if not recipient:
        await update.effective_message.reply_text(
            "Kein Standardempfänger konfiguriert. Setze AVACORE_MAIL_ALLOWED_TO in .env."
        )
        return

    raw = " ".join(context.args).strip()
    if not raw or "|" not in raw:
        await update.effective_message.reply_text(
            "Format: /mailscript <dateiname.py> | <scriptinhalt>"
        )
        return

    script_name, script_body = [part.strip() for part in raw.split("|", 1)]

    response = await http_client.post(
        f"{api_base()}/mail/send_python_script",
        json={
            "to": recipient,
            "script_name": script_name,
            "script_body": script_body,
        },
        timeout=60,
    )

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        await update.effective_message.reply_text(f"Script-Mail fehlgeschlagen: {detail}")
        return

    await update.effective_message.reply_text(f"Python-Script per Mail gesendet an {recipient}.")


async def mailnote_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    recipient = default_mail_recipient()
    if not recipient:
        await update.effective_message.reply_text(
            "Kein Standardempfänger konfiguriert. Setze AVACORE_MAIL_ALLOWED_TO in .env."
        )
        return

    raw = " ".join(context.args).strip()
    if not raw or "|" not in raw:
        await update.effective_message.reply_text("Format: /mailnote <titel> | <inhalt>")
        return

    title, note = [part.strip() for part in raw.split("|", 1)]

    response = await http_client.post(
        f"{api_base()}/mail/send_important_note",
        json={
            "to": recipient,
            "title": title,
            "note": note,
        },
        timeout=60,
    )

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        await update.effective_message.reply_text(f"Wichtige-Mail fehlgeschlagen: {detail}")
        return

    await update.effective_message.reply_text(f"Wichtiger Inhalt per Mail gesendet an {recipient}.")


async def docs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    query = " ".join(context.args).strip()

    response = await http_client.get(
        f"{api_base()}/knowledge/documents",
        params={"q": query, "limit": 20},
        timeout=30,
    )

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        await update.effective_message.reply_text(f"Dokumentliste fehlgeschlagen: {detail}")
        return

    items = response.json().get("items", [])
    if not items:
        await update.effective_message.reply_text("Keine Dokumente gefunden.")
        return

    lines = ["Dokumente:"]
    for item in items:
        title = (item.get("title") or "").strip()
        doc_type = (item.get("doc_type") or "").strip()
        lines.append(f"- {title} [{doc_type}]")

    out = "\n".join(lines)
    if len(out) > 3800:
        out = out[:3800] + "\n..."

    await update.effective_message.reply_text(out)


async def page_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    raw = " ".join(context.args).strip()
    if not raw or "|" not in raw:
        await update.effective_message.reply_text("Format: /page <dokumentname> | <seite>")
        return

    document, page_str = [part.strip() for part in raw.split("|", 1)]

    try:
        page = int(page_str)
    except ValueError:
        await update.effective_message.reply_text("Seite muss eine Zahl sein.")
        return

    response = await http_client.post(
        f"{api_base()}/knowledge/explain_page",
        json={"document": document, "page": page},
        timeout=180,
    )

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        await update.effective_message.reply_text(f"Seitenabfrage fehlgeschlagen: {detail}")
        return

    data = response.json()
    answer = data.get("answer", "").strip()

    if len(answer) > 3800:
        answer = answer[:3800] + " ..."

    await update.effective_message.reply_text(answer or "Keine Erklärung erhalten.")

async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    text = (update.effective_message.text or "").strip()
    if not text:
        return
    if text.startswith("/"):
        await unknown_command(update, context)
        return

    if settings.debug:
        print("TELEGRAM RAW TEXT:", repr(update.effective_message.text))
        print("TELEGRAM CHAT ID:", repr(chat_id))
        print("TELEGRAM TO /reply:", repr(text))

    # ------------------------------------------------------------
    # Local myStrom natural language control
    # ------------------------------------------------------------
    # This is handled before the normal /reply call so simple device
    # commands do not need to go through the LLM.
    # ------------------------------------------------------------
    switch_intent = detect_switch_intent(text)
    if switch_intent:
        handled = await handle_switch_intent(update, switch_intent)
        if handled:
            return

    # ------------------------------------------------------------
    # Local Ava Notes natural language capture
    # ------------------------------------------------------------
    # Example:
    # "Notiere: D405 Halterung prüfen"
    # "Ava, notiere: myStrom Switch funktioniert"
    # ------------------------------------------------------------
    note_text = detect_note_intent(text)
    if note_text:
        try:
            note = create_note(
                db_path=settings.db_path,
                content=note_text,
                source="telegram-natural-language",
            )
            await update.effective_message.reply_text(
                "Notiz gespeichert:\n\n" + format_note(note)
            )
            return
        except Exception as exc:
            await update.effective_message.reply_text(
                f"Notiz konnte nicht gespeichert werden: {exc}"
            )
            return

    response = await http_client.post(
        f"{api_base()}/reply",
        json={
            "channel": "telegram",
            "user_id": str(update.effective_user.id) if update.effective_user else "telegram-user",
            "chat_id": chat_id,
            "text": text,
            "timestamp": int(time.time()),
            "language": telegram_reply_language(context),
        },
        timeout=300,
    )

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        await update.effective_message.reply_text(f"Reply fehlgeschlagen: {detail}")
        return

    data = response.json()
    reply_text = (data.get("reply") or data.get("answer") or "").strip()

    if not reply_text:
        reply_text = "Keine Antwort erhalten."

    if len(reply_text) <= 4000:
        await update.effective_message.reply_text(reply_text)
        return

    chunk_size = 3800
    for i in range(0, len(reply_text), chunk_size):
        await update.effective_message.reply_text(reply_text[i:i + chunk_size])

async def camera_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    language = telegram_reply_language(context)
    if language == "en":
        await update.effective_message.reply_text(
            "I am fetching the current camera image and taking a look..."
        )
    else:
        await update.effective_message.reply_text(
            "Ich hole ein aktuelles Kamerabild und schaue es mir an..."
        )

    try:
        snapshot_response = await http_client.post(
            f"{api_base()}/camera/snapshot",
            json={},
            timeout=90,
        )

        if not snapshot_response.ok:
            try:
                detail = snapshot_response.json().get("detail", snapshot_response.text)
            except Exception:
                detail = snapshot_response.text

            await update.effective_message.reply_text(
                f"Kamera-Snapshot fehlgeschlagen: {detail}"
            )
            return

        snapshot_data = snapshot_response.json()
        image_path = Path(snapshot_data.get("image_path", ""))

        if not image_path.exists():
            await update.effective_message.reply_text(
                f"Snapshot wurde erzeugt, aber Datei nicht gefunden: {image_path}"
            )
            return

        # Originalbild wird an Telegram gesendet.
        # Für das VLM verwenden wir ein oben beschnittenes Bild,
        # damit D-Link-Zeitstempel/Kameraname nicht die Beschreibung dominieren.
        try:
            scene_image_path = crop_camera_overlay(image_path)
        except Exception:
            scene_image_path = image_path

        description = ""
        perceived_persons = []

        if settings.identity_enabled and settings.person_recognition_enabled:
            try:
                identity = recognize_face_image(
                    image_path=scene_image_path, identity_dir=settings.identity_dir,
                    model_name=settings.identity_model, device=settings.identity_device,
                    threshold=settings.person_confidence_threshold,
                    margin_threshold=settings.identity_margin, top_k=settings.identity_top_k,
                    min_roger_votes=settings.identity_min_roger_votes,
                )
                perceived_persons = [{"track_id": "camera_primary",
                    "person_id": identity.identity if identity.identity != "unknown" else None,
                    "confidence": identity.confidence, "location": "camera_view"}]
            except Exception:
                perceived_persons = []

        try:
            vision_response = await http_client.post(
                f"{api_base()}/vision/describe_image",
                json={
                    "image_path": str(scene_image_path),
                    "mode": "camera",
                    "ocr_text": "",
                },
                timeout=180,
            )

            if vision_response.ok:
                vision_data = vision_response.json()
                description = (
                    vision_data.get("caption")
                    or vision_data.get("description")
                    or vision_data.get("answer")
                    or vision_data.get("text")
                    or ""
                ).strip()
                description = clean_camera_description(description)
                if language == "de":
                    description = await translate_camera_description_to_german(description)
            else:
                try:
                    detail = vision_response.json().get("detail", vision_response.text)
                except Exception:
                    detail = vision_response.text
                description = f"VLM-Beschreibung fehlgeschlagen: {detail}"

        except Exception as exc:
            description = f"VLM-Beschreibung fehlgeschlagen: {exc}"

        caption = (
            "Current Ava camera image"
            if language == "en"
            else "Aktuelles Ava-Kamerabild"
        )

        if description:
            label = "Ava sees" if language == "en" else "Ava sieht"
            caption += f"\n\n{label}:\n{description}"

        # The adapter reports structured perception back into the same
        # Continuum.  No frame or face embedding is persisted here.
        await http_client.post(
            f"{api_base()}/cognitive/perception",
            json={"session_id": f"telegram:{chat_id}",
                  "scene_description": description or "Camera snapshot captured",
                  "persons": perceived_persons, "objects": [], "relations": [],
                  "confidence": .6 if description else .3},
            headers=admin_headers(), timeout=15,
        )

        if len(caption) > 1000:
            caption = caption[:1000] + "\n\n[Beschreibung gekürzt]"

        with image_path.open("rb") as photo:
            await update.effective_message.reply_photo(
                photo=photo,
                caption=caption,
            )

    except Exception as exc:
        await update.effective_message.reply_text(
            f"Kamera-Befehl fehlgeschlagen: {exc}"
        )

async def research_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    query = " ".join(context.args).strip()

    if not query:
        await update.effective_message.reply_text(
            "Bitte gib eine Recherchefrage an, z.B.:\n"
            "/research D-Link DCS-5222L RTSP play1.sdp"
        )
        return

    await update.effective_message.reply_text(f"Ich recherchiere:\n{query}")

    try:
        response = await http_client.post(
            f"{api_base()}/research",
            json={
                "query": query,
                "max_results": 4,
                "save_memory": True,
            },
            headers=admin_headers(),
            timeout=180,
        )

        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text

            await update.effective_message.reply_text(
                f"Recherche fehlgeschlagen: {detail}"
            )
            return

        data = response.json()

        answer = data.get("answer", "").strip()
        memory_id = data.get("memory_id")
        sources = data.get("sources", [])

        if not answer:
            answer = "Recherche abgeschlossen, aber ohne Antworttext."

        source_lines = []
        for i, source in enumerate(sources[:4], start=1):
            title = source.get("title", "(ohne Titel)")
            url = source.get("url", "")
            ok = source.get("ok", False)

            marker = "OK" if ok else "nicht gelesen"
            source_lines.append(f"{i}. {title} [{marker}]\n{url}")

        suffix = ""

        if source_lines:
            suffix += "\n\nQuellen:\n" + "\n".join(source_lines)

        if memory_id:
            suffix += f"\n\nAls Memory-Kandidat gespeichert: #{memory_id}"

        message = answer + suffix

        # Telegram limit is 4096 characters.
        if len(message) <= 3900:
            await update.effective_message.reply_text(message)
        else:
            await update.effective_message.reply_text(message[:3900] + "\n\n[gekürzt]")

    except Exception as exc:
        await update.effective_message.reply_text(
            f"Recherche-Befehl fehlgeschlagen: {exc}"
        )

async def switch_on_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    try:
        result = light_on()
        await update.effective_message.reply_text(result)
    except Exception as exc:
        await update.effective_message.reply_text(f"Switch konnte nicht eingeschaltet werden: {exc}")


async def switch_off_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    try:
        result = light_off()
        await update.effective_message.reply_text(result)
    except Exception as exc:
        await update.effective_message.reply_text(f"Switch konnte nicht ausgeschaltet werden: {exc}")


async def switch_state_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    try:
        status = light_status()

        relay = status.get("relay")
        power = status.get("power")
        temperature = status.get("temperature")

        relay_text = "ein" if relay else "aus"

        lines = [
            "myStrom Switch Status:",
            f"- Relais: {relay_text}",
        ]

        if power is not None:
            lines.append(f"- Leistung: {power} W")

        if temperature is not None:
            lines.append(f"- Temperatur: {temperature} °C")

        await update.effective_message.reply_text("\n".join(lines))

    except Exception as exc:
        await update.effective_message.reply_text(f"Switch-Status konnte nicht gelesen werden: {exc}")

async def handle_switch_intent(
    update: Update,
    intent: str,
) -> bool:
    if not update.effective_message:
        return False

    try:
        if intent == "on":
            result = light_on()
            await update.effective_message.reply_text(result)
            return True

        if intent == "off":
            result = light_off()
            await update.effective_message.reply_text(result)
            return True

        if intent == "state":
            status = light_status()

            relay = status.get("relay")
            power = status.get("power")
            temperature = status.get("temperature")

            relay_text = "ein" if relay else "aus"

            lines = [
                "myStrom Switch Status:",
                f"- Relais: {relay_text}",
            ]

            if power is not None:
                lines.append(f"- Leistung: {power} W")

            if temperature is not None:
                lines.append(f"- Temperatur: {temperature} °C")

            await update.effective_message.reply_text("\n".join(lines))
            return True

    except Exception as exc:
        await update.effective_message.reply_text(
            f"myStrom-Aktion fehlgeschlagen: {exc}"
        )
        return True

    return False

async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    if not settings.voice_enabled:
        await update.effective_message.reply_text("Spracherkennung ist deaktiviert.")
        return

    voice = update.effective_message.voice
    audio = update.effective_message.audio

    tg_file_id = None
    suffix = ".ogg"

    if voice:
        tg_file_id = voice.file_id
        suffix = ".ogg"
    elif audio:
        tg_file_id = audio.file_id
        suffix = ".oga"
    else:
        return

    try:
        settings.voice_cache_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        audio_path = settings.voice_cache_dir / f"telegram-voice-{chat_id}-{timestamp}{suffix}"

        tg_file = await context.bot.get_file(tg_file_id)
        await tg_file.download_to_drive(custom_path=str(audio_path))

        await update.effective_message.reply_text("Ich höre kurz zu...")

        result = transcribe_audio_file(
            audio_path=audio_path,
            model_name=settings.voice_model,
            device=settings.voice_device,
            compute_type=settings.voice_compute_type,
            language=settings.voice_language,
        )

        text = (result.get("text") or "").strip()

        if not text:
            await update.effective_message.reply_text(
                "Ich konnte die Sprachnachricht nicht verständlich transkribieren."
            )
            return

        await update.effective_message.reply_text(f"Verstanden:\n{text}")
        note_text_debug = detect_note_intent(text)
        if settings.debug:
            print("VOICE NOTE INTENT:", repr(note_text_debug))

        # ------------------------------------------------------------
        # Reuse local natural-language switch control for voice input
        # ------------------------------------------------------------
        switch_intent = detect_switch_intent(text)
        if switch_intent:
            handled = await handle_switch_intent(update, switch_intent)
            if handled:
                return

        # ------------------------------------------------------------
        # Reuse local Ava Notes natural-language capture for voice input
        # ------------------------------------------------------------
        # Example spoken:
        # "Ava, notiere: D405 Halterung nochmals prüfen"
        # ------------------------------------------------------------
        note_text = detect_note_intent(text)
        if note_text:
            try:
                note = create_note(
                    db_path=settings.db_path,
                    content=note_text,
                    source="telegram-voice",
                )
                await update.effective_message.reply_text(
                    "Notiz gespeichert:\n\n" + format_note(note)
                )
                return
            except Exception as exc:
                await update.effective_message.reply_text(
                    f"Notiz konnte nicht gespeichert werden: {exc}"
                )
                return

        response = await http_client.post(
            f"{api_base()}/reply",
            json={
                "channel": "telegram",
                "user_id": str(update.effective_user.id) if update.effective_user else "telegram-user",
            "chat_id": chat_id,
            "text": text,
            "timestamp": int(time.time()),
            "language": telegram_reply_language(context),
        },
            timeout=300,
        )

        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            await update.effective_message.reply_text(f"Reply fehlgeschlagen: {detail}")
            return

        data = response.json()
        reply_text = (data.get("reply") or data.get("answer") or "").strip()

        if not reply_text:
            reply_text = "Keine Antwort erhalten."

        if len(reply_text) <= 4000:
            await update.effective_message.reply_text(reply_text)
            return

        chunk_size = 3800
        for i in range(0, len(reply_text), chunk_size):
            await update.effective_message.reply_text(reply_text[i:i + chunk_size])

    except Exception as exc:
        await update.effective_message.reply_text(
            f"Sprachverarbeitung fehlgeschlagen: {exc}"
        )

async def note_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    content = " ".join(context.args).strip()

    if not content:
        await update.effective_message.reply_text(
            "Bitte gib eine Notiz an, z.B.:\n"
            "/note D405 Halterung nochmals prüfen"
        )
        return

    try:
        note = create_note(
            db_path=settings.db_path,
            content=content,
            source="telegram",
        )
        await update.effective_message.reply_text(
            "Notiz gespeichert:\n\n" + format_note(note)
        )
    except Exception as exc:
        await update.effective_message.reply_text(f"Notiz konnte nicht gespeichert werden: {exc}")


async def notes_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    try:
        status = "open"
        limit = 10

        if context.args:
            first = context.args[0].strip().lower()
            if first in {"open", "done", "archived", "all"}:
                status = first

        notes = list_notes(
            db_path=settings.db_path,
            status=status,
            limit=limit,
        )

        await update.effective_message.reply_text(
            f"Notizen ({status}):\n\n" + format_note_list(notes)
        )

    except Exception as exc:
        await update.effective_message.reply_text(f"Notizen konnten nicht geladen werden: {exc}")


async def notesearch_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    query = " ".join(context.args).strip()

    if not query:
        await update.effective_message.reply_text(
            "Bitte gib einen Suchbegriff an, z.B.:\n"
            "/notesearch D405"
        )
        return

    try:
        notes = search_notes(
            db_path=settings.db_path,
            query=query,
            limit=10,
        )

        await update.effective_message.reply_text(
            f"Suchergebnis für '{query}':\n\n" + format_note_list(notes)
        )

    except Exception as exc:
        await update.effective_message.reply_text(f"Notizensuche fehlgeschlagen: {exc}")


async def noteadd_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    if len(context.args) < 2:
        await update.effective_message.reply_text(
            "Bitte nutze:\n"
            "/noteadd <id> <Text>\n\n"
            "Beispiel:\n"
            "/noteadd 3 Zusätzlich Schrauben M2.5 prüfen"
        )
        return

    try:
        note_id = int(context.args[0])
        extra = " ".join(context.args[1:]).strip()

        note = append_to_note(
            db_path=settings.db_path,
            note_id=note_id,
            extra_content=extra,
        )

        await update.effective_message.reply_text(
            "Notiz ergänzt:\n\n" + format_note(note)
        )

    except Exception as exc:
        await update.effective_message.reply_text(f"Notiz konnte nicht ergänzt werden: {exc}")


async def notedone_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    if not context.args:
        await update.effective_message.reply_text(
            "Bitte nutze:\n"
            "/notedone <id>"
        )
        return

    try:
        note_id = int(context.args[0])

        note = update_note_status(
            db_path=settings.db_path,
            note_id=note_id,
            status="done",
        )

        await update.effective_message.reply_text(
            "Notiz als erledigt markiert:\n\n" + format_note(note, include_content=False)
        )

    except Exception as exc:
        await update.effective_message.reply_text(f"Notiz konnte nicht erledigt werden: {exc}")


async def notearchive_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    if not context.args:
        await update.effective_message.reply_text(
            "Bitte nutze:\n"
            "/notearchive <id>"
        )
        return

    try:
        note_id = int(context.args[0])

        note = update_note_status(
            db_path=settings.db_path,
            note_id=note_id,
            status="archived",
        )

        await update.effective_message.reply_text(
            "Notiz archiviert:\n\n" + format_note(note, include_content=False)
        )

    except Exception as exc:
        await update.effective_message.reply_text(f"Notiz konnte nicht archiviert werden: {exc}")


async def idcapture_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    if not settings.identity_enabled:
        await update.effective_message.reply_text(
            "Identity RAG ist deaktiviert. Setze AVACORE_IDENTITY_ENABLED=1."
        )
        return

    if not context.args:
        await update.effective_message.reply_text(
            "Bitte nutze:\n"
            "/idcapture roger\n"
            "/idcapture unknown\n"
            "/idcapture empty"
        )
        return

    label = context.args[0].strip().lower()

    try:
        await update.effective_message.reply_text(f"Erfasse Identity-Beispiel: {label} ...")

        url = camera_rtsp_url()

        raw_snapshot = capture_rtsp_snapshot(
            url=url,
            output_dir=settings.camera_cache_dir,
            camera_name="identity",
        )

        try:
            scene_image = crop_camera_overlay(raw_snapshot)
        except Exception:
            scene_image = raw_snapshot

        dataset_path = copy_capture_to_identity_dataset(
            image_path=scene_image,
            identity_dir=settings.identity_dir,
            label=label,
        )

        await update.effective_message.reply_photo(
            photo=open(scene_image, "rb"),
            caption=(
                "Identity-Beispiel gespeichert.\n\n"
                f"Label: {label}\n"
                f"Datei: {dataset_path}\n\n"
                "Danach später /idtrain ausführen."
            ),
        )

    except Exception as exc:
        await update.effective_message.reply_text(
            f"Identity Capture fehlgeschlagen: {exc}"
        )


async def idtrain_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    if not settings.identity_enabled:
        await update.effective_message.reply_text(
            "Identity RAG ist deaktiviert. Setze AVACORE_IDENTITY_ENABLED=1."
        )
        return

    try:
        await update.effective_message.reply_text(
            "Baue Identity-Index. Beim ersten Lauf kann das Modell geladen werden..."
        )

        result = build_identity_index(
            identity_dir=settings.identity_dir,
            model_name=settings.identity_model,
            device=settings.identity_device,
        )

        counts = result["counts"]

        await update.effective_message.reply_text(
            "Identity-Index erstellt.\n\n"
            f"Roger Face Embeddings: {counts['roger']}\n"
            f"Unknown Face Embeddings: {counts['unknown']}\n"
            f"Übersprungen: {counts['skipped']}\n"
            f"Total im Index: {counts['total']}\n\n"
            f"Index:\n{result['index_path']}"
        )

    except Exception as exc:
        await update.effective_message.reply_text(
            f"Identity Training fehlgeschlagen: {exc}"
        )


async def idcheck_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)

    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    if not settings.identity_enabled:
        await update.effective_message.reply_text(
            "Identity RAG ist deaktiviert. Setze AVACORE_IDENTITY_ENABLED=1."
        )
        return

    try:
        await update.effective_message.reply_text("Prüfe aktuelle Kameraaufnahme...")

        url = camera_rtsp_url()

        raw_snapshot = capture_rtsp_snapshot(
            url=url,
            output_dir=settings.camera_cache_dir,
            camera_name="identity-check",
        )

        try:
            scene_image = crop_camera_overlay(raw_snapshot)
        except Exception:
            scene_image = raw_snapshot

        decision = recognize_face_image(
            image_path=scene_image,
            identity_dir=settings.identity_dir,
            model_name=settings.identity_model,
            device=settings.identity_device,
            threshold=settings.identity_threshold,
            margin_threshold=settings.identity_margin,
            top_k=settings.identity_top_k,
            min_roger_votes=settings.identity_min_roger_votes,
        )

        await http_client.post(
            f"{api_base()}/cognitive/perception",
            json={"session_id": f"telegram:{chat_id}", "scene_description": "Identity check",
                  "persons": [{"track_id": "camera_primary",
                    "person_id": decision.identity if decision.identity != "unknown" else None,
                    "confidence": decision.confidence, "location": "camera_view"}],
                  "objects": [], "relations": [], "confidence": decision.confidence},
            headers=admin_headers(), timeout=15,
        )

        caption = "Identity Check:\n\n" + format_identity_decision(decision)

        if len(caption) > 1000:
            caption = caption[:997] + "..."

        await update.effective_message.reply_photo(
            photo=open(scene_image, "rb"),
            caption=caption,
        )

    except Exception as exc:
        await update.effective_message.reply_text(
            f"Identity Check fehlgeschlagen: {exc}"
        )       


async def _debug_json(path: str):
    return await http_client.get(f"{api_base()}{path}", headers=admin_headers(), timeout=15)


async def _request_camera_perception(update: Update, *, reason: str, force: bool,
                                     include_scene: bool) -> tuple[dict, str | None]:
    if not update.effective_chat:
        return {}, "No Telegram chat context."
    try:
        response = await http_client.post(f"{api_base()}/perception/camera", json={
            "reason": reason, "force": force, "include_scene": include_scene,
            "session_id": f"telegram:{update.effective_chat.id}"},
            headers=admin_headers(), timeout=240 if include_scene else 120)
    except Exception as exc:
        return {}, f"Camera perception failed: {exc}"
    if not response.ok:
        try: detail = response.json().get("detail", response.text)
        except Exception: detail = response.text
        return {}, str(detail)
    return response.json(), None


async def active_camera_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message: return
    data, error = await _request_camera_perception(update, reason="see_command", force=True, include_scene=True)
    if error:
        await update.effective_message.reply_text(error); return
    description = clean_camera_description(data.get("scene_description") or "")
    if description and telegram_reply_language(context) == "de":
        description = await translate_camera_description_to_german(description)
    known = data.get("identities_resolved") or []
    unknown_count = sum(1 for x in data.get("persons", []) if not x.get("person_id"))
    structured = []
    if known: structured.append("Recognized: " + ", ".join(known))
    if unknown_count: structured.append(f"Unknown persons: {unknown_count}")
    caption = "Aktuelle Ava-Wahrnehmung\n\n" + (description or f"Persons visible: {len(data.get('persons') or [])}")
    if structured: caption += "\n" + "\n".join(structured)
    image_path = Path(data.get("image_path") or "")
    if image_path.exists():
        with image_path.open("rb") as photo:
            await update.effective_message.reply_photo(photo=photo, caption=caption[:1000])
    else:
        await update.effective_message.reply_text(caption[:4000])


async def active_idcheck_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message: return
    data, error = await _request_camera_perception(update, reason="idcheck", force=True, include_scene=False)
    if error:
        await update.effective_message.reply_text(error); return
    lines = ["Identity Check:", f"persons detected: {len(data.get('persons') or [])}"]
    for person in data.get("persons") or []:
        lines.append(f"{person.get('track_id')}: {person.get('person_id') or 'unknown'} ({person.get('confidence',0):.3f})")
    image_path = Path(data.get("image_path") or "")
    if image_path.exists():
        with image_path.open("rb") as photo:
            await update.effective_message.reply_photo(photo=photo, caption="\n".join(lines)[:1000])
    else:
        await update.effective_message.reply_text("\n".join(lines))


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return
    health, continuum, perception = await _debug_json("/health"), await _debug_json("/debug/continuum"), await _debug_json("/debug/perception")
    h = health.json() if health.ok else {}; c = continuum.json() if continuum.ok else {}; p = perception.json() if perception.ok else {}
    present = [x.get("display_name") for x in p.get("persons", []) if x.get("current_presence")]
    await update.effective_message.reply_text(
        f"Ava: {'online' if h.get('ok') else 'unavailable'}\nLLM: {'available' if h.get('ok') else 'unavailable'}\n"
        f"Continuum: {'active' if c.get('enabled') else 'inactive'} ({c.get('entities', 0)} entities)\n"
        f"Workspace: {c.get('workspace', 0)} items\nVision: {'on' if p.get('enabled') else 'off'}\n"
        f"Present: {', '.join(present) if present else 'none known'}")


async def continuum_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message: return
    response = await _debug_json("/debug/continuum"); data = response.json() if response.ok else {}
    await update.effective_message.reply_text(f"Continuum:\nentities: {data.get('entities', 0)}\nactive: {data.get('active', 0)}\nworkspace: {data.get('workspace', 0)}\nvision entities: {data.get('vision_entities', 0)}\npersons active: {data.get('persons_active', 0)}")


async def workspace_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message: return
    response = await _debug_json("/debug/workspace"); data = response.json() if response.ok else {}
    items = data.get("active_items") or []
    text = "Conscious Workspace:\n" + ("\n".join(f"{i+1}. {x.get('content','')[:100]}" for i, x in enumerate(items[:8])) or "empty")
    await update.effective_message.reply_text(text)


async def focus_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message: return
    response = await _debug_json("/debug/workspace"); data = response.json() if response.ok else {}
    items = data.get("active_items") or []
    await update.effective_message.reply_text("Current focus:\n" + ("\n".join(f"{i+1}. {x.get('content','')[:90]} – {x.get('activation_score', x.get('activation', 0)):.2f}" for i, x in enumerate(items[:5])) or "No active focus."))


async def memory_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message: return
    response = await _debug_json("/debug/workspace"); data = response.json() if response.ok else {}
    items = data.get("working_memory") or []
    await update.effective_message.reply_text("Current Working Memory:\n" + ("\n".join(f"- {x.get('content','')[:120]}" for x in items[-8:]) or "empty"))


async def persons_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message: return
    _, error = await _request_camera_perception(update, reason="persons_command", force=False, include_scene=False)
    if error and settings.camera_enabled:
        await update.effective_message.reply_text(error); return
    response = await _debug_json("/debug/persons"); data = response.json() if response.ok else {}
    items = data.get("items") or []
    await update.effective_message.reply_text("Persons:\n" + ("\n".join(f"{x.get('display_name')} – {'present' if x.get('current_presence') else 'not present'} – confidence {x.get('confidence',0):.2f}" for x in items) or "No deliberately registered persons."))


async def who_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message: return
    _, error = await _request_camera_perception(update, reason="who_command", force=False, include_scene=False)
    if error:
        await update.effective_message.reply_text(error); return
    response = await _debug_json("/debug/persons"); items = response.json().get("items", []) if response.ok else []
    present = [x["display_name"] for x in items if x.get("current_presence") and x.get("confidence", 0) >= settings.person_confidence_threshold]
    uncertain = [x for x in items if x.get("current_presence") and not x.get("known", True)]
    if present:
        answer = f"{', '.join(present)} is currently present."
    elif len(uncertain) == 1:
        answer = "One person is currently present, but I do not know who it is."
    elif uncertain:
        answer = f"{len(uncertain)} people are present; identities uncertain."
    else:
        answer = "No person is currently visible."
    await update.effective_message.reply_text(answer)


async def why_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message: return
    response = await _debug_json("/debug/workspace"); items = response.json().get("active_items", []) if response.ok else []
    lines = []
    labels = {"recency":"recency", "continuity":"conversation/visual continuity", "self_affinity":"self relevance", "relevance":"current relevance", "urgency":"urgency", "confidence":"confidence"}
    for item in items[:4]:
        factors = item.get("score_components") or {}
        reasons = [label for key, label in labels.items() if factors.get(key, 0) >= .5]
        lines.append(f"{item.get('content','')[:70]}:\n- " + ("\n- ".join(reasons) or item.get("selection_reason", "activation competition")))
    await update.effective_message.reply_text("Why current focus:\n" + ("\n".join(lines) or "No active focus."))


async def bsp_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message:
        await update.effective_message.reply_text("BSP ist in dieser Installation nicht konfiguriert.")


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_chat: return
    text = update.effective_message.text or "/unknown"
    name = text.split()[0].lstrip("/").split("@", 1)[0].casefold()
    cycle_id = f"cy_{uuid.uuid4().hex}"
    await _record_command(update, name, text, cycle_id)
    await update.effective_message.reply_text(f"Unbekannter Befehl /{name}. Verwende /help.")
    await _record_command(update, name, f"Unknown command: /{name}", cycle_id, result=True, status="failed")


def build_app(
    *,
    request: BaseRequest | None = None,
    get_updates_request: BaseRequest | None = None,
) -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")

    builder = Application.builder().token(settings.telegram_bot_token)
    if request is not None:
        builder = builder.request(request)
    if get_updates_request is not None:
        builder = builder.get_updates_request(get_updates_request)
    app = builder.build()

    specs = [
        CommandSpec("start", start_cmd, "Ava starten"), CommandSpec("help", help_cmd, "Befehle anzeigen"),
        CommandSpec("de", de_cmd, "Deutsch"), CommandSpec("en", en_cmd, "English"),
        CommandSpec("health", health_cmd, "AvaCore health"), CommandSpec("status", status_cmd, "Operational status"),
        CommandSpec("model", model_cmd, "Active model"), CommandSpec("personality", personality_cmd, "Active personality"),
        CommandSpec("personalitybackup", personalitybackup_cmd, "Backup personality"), CommandSpec("personalityrestore", personalityrestore_cmd, "Restore personality"),
        CommandSpec("policies", policies_cmd, "Policies"), CommandSpec("memories", memories_cmd, "Long-term memories"),
        CommandSpec("remember", remember_cmd, "Remember text"), CommandSpec("reset", reset_cmd, "Reset chat"),
        CommandSpec("docs", docs_cmd, "Documents"), CommandSpec("page", page_cmd, "Explain page", requires_llm=True),
        CommandSpec("weather", weather_cmd, "Weather"), CommandSpec("medium", medium_cmd, "Medium feed"), CommandSpec("news", news_cmd, "News feed"),
        CommandSpec("mediumdigest", mediumdigest_cmd, "Medium digest", requires_llm=True), CommandSpec("newsdigest", newsdigest_cmd, "News digest", requires_llm=True),
        CommandSpec("webfetch", webfetch_cmd, "Fetch URL"), CommandSpec("webask", webask_cmd, "Ask about URL", requires_llm=True),
        CommandSpec("browsersearch", browser_search_cmd, "Browser search"), CommandSpec("research", research_cmd, "Web research", requires_llm=True),
        CommandSpec("camera", active_camera_cmd, "Capture and describe camera", aliases=("snapshot", "see"), requires_llm=False),
        CommandSpec("idcapture", idcapture_cmd, "Register identity sample"), CommandSpec("idtrain", idtrain_cmd, "Build local identity index"), CommandSpec("idcheck", active_idcheck_cmd, "Check local identity"),
        CommandSpec("mail", mail_cmd, "Recent mail"), CommandSpec("maildigest", maildigest_cmd, "Mail digest", requires_llm=True),
        CommandSpec("sendmail", sendmail_cmd, "Send mail"), CommandSpec("mailscript", mailscript_cmd, "Mail script"), CommandSpec("mailnote", mailnote_cmd, "Mail note"),
        CommandSpec("briefing", briefing_cmd, "Calendar briefing"), CommandSpec("switchon", switch_on_cmd, "Switch on"),
        CommandSpec("switchoff", switch_off_cmd, "Switch off"), CommandSpec("switchstate", switch_state_cmd, "Switch state"),
        CommandSpec("note", note_cmd, "Create note"), CommandSpec("notes", notes_cmd, "List notes"), CommandSpec("notesearch", notesearch_cmd, "Search notes"),
        CommandSpec("noteadd", noteadd_cmd, "Append note"), CommandSpec("notedone", notedone_cmd, "Complete note"), CommandSpec("notearchive", notearchive_cmd, "Archive note"), CommandSpec("notesync", notesync_cmd, "Sync notes"),
        CommandSpec("focus", focus_cmd, "Current Spotlight"), CommandSpec("continuum", continuum_cmd, "Continuum summary"),
        CommandSpec("workspace", workspace_cmd, "Conscious Workspace"), CommandSpec("memory", memory_cmd, "Working Memory"),
        CommandSpec("persons", persons_cmd, "Known persons"), CommandSpec("who", who_cmd, "Who is present"), CommandSpec("why", why_cmd, "Activation reasons"),
        CommandSpec("bsp", bsp_cmd, "Installation-specific BSP action"),
    ]
    global COMMAND_REGISTRY
    COMMAND_REGISTRY = register_commands(specs)
    for invoked_name, spec in COMMAND_REGISTRY.items():
        app.add_handler(CommandHandler(invoked_name, cognitive_handler(spec, invoked_name)))

    # Message handlers should stay last.
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_message))
    app.add_handler(MessageHandler(filters.TEXT, text_message))

    return app


def build_application(
    *,
    request: BaseRequest | None = None,
    get_updates_request: BaseRequest | None = None,
) -> Application:
    return build_app(request=request, get_updates_request=get_updates_request)


def main() -> None:
    app = build_application()
    app.run_polling()


if __name__ == "__main__":
    main()
