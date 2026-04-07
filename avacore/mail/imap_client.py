from __future__ import annotations

import imaplib
import email
from email.header import decode_header
from email.message import Message
from typing import Any


def decode_mime_header(value: str | None) -> str:
    if not value:
        return ""
    parts = decode_header(value)
    decoded = []
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded).strip()


def extract_text_from_message(msg: Message) -> str:
    texts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))

            if "attachment" in disposition.lower():
                continue

            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                if payload:
                    texts.append(payload.decode(charset, errors="replace"))
    else:
        if msg.get_content_type() == "text/plain":
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or "utf-8"
            if payload:
                texts.append(payload.decode(charset, errors="replace"))

    text = "\n".join(t.strip() for t in texts if t.strip())
    return " ".join(text.split())


class ImapMailClient:
    def __init__(self, host: str, port: int, username: str, password: str) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def list_recent_messages(self, limit: int = 10) -> list[dict[str, Any]]:
        conn = imaplib.IMAP4_SSL(self.host, self.port)
        try:
            conn.login(self.username, self.password)
            conn.select("INBOX")

            status, data = conn.search(None, "ALL")
            if status != "OK":
                raise RuntimeError("IMAP search failed")

            ids = data[0].split()
            ids = ids[-limit:]

            results: list[dict[str, Any]] = []

            for msg_id in reversed(ids):
                status, msg_data = conn.fetch(msg_id, "(RFC822)")
                if status != "OK" or not msg_data or not msg_data[0]:
                    continue

                raw = msg_data[0][1]
                msg = email.message_from_bytes(raw)

                results.append(
                    {
                        "id": msg_id.decode("utf-8", errors="replace"),
                        "from": decode_mime_header(msg.get("From")),
                        "to": decode_mime_header(msg.get("To")),
                        "subject": decode_mime_header(msg.get("Subject")),
                        "date": decode_mime_header(msg.get("Date")),
                        "body": extract_text_from_message(msg)[:2000],
                    }
                )

            return results
        finally:
            try:
                conn.close()
            except Exception:
                pass
            conn.logout()
