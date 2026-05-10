from __future__ import annotations

import time
import requests
import os
from pathlib import Path
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from avacore.config.settings import settings


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


def default_mail_recipient() -> str | None:
    if not settings.mail_allowed_to:
        return None
    recipient = (settings.mail_allowed_to[0] or "").strip()
    return recipient or None


def command_help_text() -> str:
    return (
        "Ava Befehle:\n\n"
        "Allgemein:\n"
        "/start - Ava starten\n"
        "/help - diese Übersicht\n"
        "/health - AvaCore Status\n"
        "/model - aktives Modell\n"
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
        "/page <dokumentname> | <seite> - konkrete Dokumentseite erklären\n"
        "/weather [ort] - Wetter kurz anzeigen\n"
        "/medium - aktuelle Medium-Einträge\n"
        "/news - aktuelle News-Einträge\n"
        "/mediumdigest - Medium kurz zusammenfassen\n"
        "/newsdigest - News kurz zusammenfassen\n"
        "/webfetch <url> - Rohtext einer Seite holen\n"
        "/webask <url> <frage> - Frage zu einer Webseite beantworten\n"
        "/browsersearch <Suchbegriff> - Websuche über kontrollierten Chromium-Browser\n"
        "/research <Frage> - Web-Recherche mit Quellen und Memory-Kandidat\n\n"
        "Mail:\n"
        "/mail - letzte Mails anzeigen\n"
        "/maildigest - Mails kurz zusammenfassen\n"
        "/sendmail <subject> | <text> - Mail an Standardempfänger senden\n"
        "/mailscript <dateiname.py> | <scriptinhalt> - Python-Script mailen\n"
        "/mailnote <titel> | <inhalt> - wichtigen Inhalt mailen\n"
        "/camera - aktuelles Kamerabild holen\n"
        "/snapshot - Alias für /camera\n"
        "/briefing - heutiges Kalender-Briefing abrufen\n"
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


async def health_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = requests.get(f"{api_base()}/health", timeout=15)
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

    response = requests.get(f"{api_base()}/model", timeout=15)
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

    response = requests.get(f"{api_base()}/personality", timeout=30)
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

    response = requests.post(
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

    response = requests.post(
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

    response = requests.get(f"{api_base()}/policies", timeout=30)
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

    response = requests.get(f"{api_base()}/memories", params={"limit": 20}, timeout=30)
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

    response = requests.post(
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
    response = requests.delete(f"{api_base()}/reply", json=payload, timeout=30)

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
        response = requests.post(
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


async def weather_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or not update.effective_message:
        return

    chat_id = str(update.effective_chat.id)
    if update.effective_chat.type != "private" or not is_allowed_chat(chat_id):
        await update.effective_message.reply_text("Dieser Chat ist nicht freigegeben.")
        return

    location = " ".join(context.args).strip()

    response = requests.post(
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

    response = requests.get(f"{api_base()}/tools/medium", params={"limit": 5}, timeout=30)
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

    response = requests.get(f"{api_base()}/tools/news", params={"limit": 5}, timeout=30)
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

    response = requests.get(f"{api_base()}/tools/mediumdigest", params={"limit": 5}, timeout=180)
    response.raise_for_status()
    data = response.json()

    digest = data.get("digest", "")
    if len(digest) > 3800:
        digest = digest[:3800] + " ..."
    await update.effective_message.reply_text(digest or "Kein Medium-Digest verfügbar.")


async def newsdigest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message:
        return

    response = requests.get(f"{api_base()}/tools/newsdigest", params={"limit": 5}, timeout=180)
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

    response = requests.post(
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

    response = requests.post(
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
        search_response = requests.post(
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

        text_response = requests.post(
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

    response = requests.get(f"{api_base()}/mail/inbox", params={"limit": 5}, timeout=60)
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

    response = requests.get(f"{api_base()}/mail/digest", params={"limit": 8}, timeout=180)
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

    response = requests.post(
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

    response = requests.post(
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

    response = requests.post(
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

    response = requests.get(
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

    response = requests.post(
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

    if settings.debug:
        print("TELEGRAM RAW TEXT:", repr(update.effective_message.text))
        print("TELEGRAM CHAT ID:", repr(chat_id))
        print("TELEGRAM TO /reply:", repr(text))

    response = requests.post(
        f"{api_base()}/reply",
        json={
            "channel": "telegram",
            "user_id": str(update.effective_user.id) if update.effective_user else "telegram-user",
            "chat_id": chat_id,
            "text": text,
            "timestamp": int(time.time()),
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
    reply_text = (data.get("reply") or "").strip()
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

    await update.effective_message.reply_text("Ich hole ein aktuelles Kamerabild...")

    try:
        response = requests.post(
            f"{api_base()}/camera/snapshot",
            json={},
            timeout=30,
        )

        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            await update.effective_message.reply_text(f"Kamera-Snapshot fehlgeschlagen: {detail}")
            return

        data = response.json()
        image_path = Path(data.get("image_path", ""))

        if not image_path.exists():
            await update.effective_message.reply_text(
                f"Snapshot wurde erzeugt, aber Datei nicht gefunden: {image_path}"
            )
            return

        with image_path.open("rb") as photo:
            await update.effective_message.reply_photo(
                photo=photo,
                caption="Aktuelles Ava-Kamerabild"
            )

    except Exception as exc:
        await update.effective_message.reply_text(f"Kamera-Befehl fehlgeschlagen: {exc}")

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
        response = requests.post(
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


def build_app() -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")

    app = Application.builder().token(settings.telegram_bot_token).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("health", health_cmd))
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("personality", personality_cmd))
    app.add_handler(CommandHandler("personalitybackup", personalitybackup_cmd))
    app.add_handler(CommandHandler("personalityrestore", personalityrestore_cmd))

    app.add_handler(CommandHandler("policies", policies_cmd))
    app.add_handler(CommandHandler("memories", memories_cmd))
    app.add_handler(CommandHandler("remember", remember_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))

    app.add_handler(CommandHandler("docs", docs_cmd))
    app.add_handler(CommandHandler("page", page_cmd))

    app.add_handler(CommandHandler("weather", weather_cmd))
    app.add_handler(CommandHandler("medium", medium_cmd))
    app.add_handler(CommandHandler("news", news_cmd))
    app.add_handler(CommandHandler("mediumdigest", mediumdigest_cmd))
    app.add_handler(CommandHandler("newsdigest", newsdigest_cmd))
    app.add_handler(CommandHandler("webfetch", webfetch_cmd))
    app.add_handler(CommandHandler("webask", webask_cmd))
    app.add_handler(CommandHandler("browsersearch", browser_search_cmd))
    app.add_handler(CommandHandler("research", research_cmd))

    app.add_handler(CommandHandler("camera", camera_cmd))
    app.add_handler(CommandHandler("snapshot", camera_cmd))

    app.add_handler(CommandHandler("mail", mail_cmd))
    app.add_handler(CommandHandler("maildigest", maildigest_cmd))
    app.add_handler(CommandHandler("sendmail", sendmail_cmd))
    app.add_handler(CommandHandler("mailscript", mailscript_cmd))
    app.add_handler(CommandHandler("mailnote", mailnote_cmd))

    app.add_handler(CommandHandler("briefing", briefing_cmd))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))

    return app


def build_application() -> Application:
    return build_app()


def main() -> None:
    app = build_app()
    app.run_polling()


if __name__ == "__main__":
    main()