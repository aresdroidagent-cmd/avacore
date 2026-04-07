from __future__ import annotations

from avacore.config.settings import settings
from avacore.mail.imap_client import ImapMailClient
from avacore.mail.smtp_client import SmtpMailClient
from avacore.model.ollama_backend import OllamaBackend


class MailService:
    def __init__(self) -> None:
        self.client = ImapMailClient(
            host=settings.mail_imap_host,
            port=settings.mail_imap_port,
            username=settings.mail_username,
            password=settings.mail_password,
        )
        self.smtp = SmtpMailClient(
            host=settings.mail_smtp_host,
            port=settings.mail_smtp_port,
            username=settings.mail_username,
            password=settings.mail_password,
            mail_from=settings.mail_from or settings.mail_username,
        )
        self.backend = OllamaBackend(
            ollama_url=settings.ollama_url,
            model=settings.ollama_model,
            timeout_ms=settings.ollama_timeout_ms,
        )

    def list_recent(self, limit: int = 10) -> list[dict]:
        return self.client.list_recent_messages(limit=limit)

    def build_digest(self, limit: int = 8) -> str:
        items = self.list_recent(limit=limit)
        if not items:
            return "Keine Mails gefunden."

        blocks = []
        for item in items:
            blocks.append(
                "\n".join(
                    [
                        f"Von: {item.get('from', '')}",
                        f"Betreff: {item.get('subject', '')}",
                        f"Datum: {item.get('date', '')}",
                        f"Inhalt: {item.get('body', '')}",
                    ]
                )
            )

        prompt = (
            "Fasse die folgenden E-Mails kurz und nützlich auf Deutsch zusammen. "
            "Maximal 8 Bulletpoints. "
            "Hebe wichtige, dringende oder handlungsrelevante Nachrichten hervor. "
            "Keine Einleitung."
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "\n\n---\n\n".join(blocks)[:12000]},
        ]

        answer = self.backend.chat(messages)

        sources = []
        for item in items[:5]:
            sources.append(f"- {item.get('subject', '')} ({item.get('from', '')})")

        if sources:
            answer = answer.rstrip() + "\n\nMails:\n" + "\n".join(sources)

        return answer

    def send_allowed_mail(self, to: str, subject: str, body: str) -> None:
        allowed = set(settings.mail_allowed_to)
        if to not in allowed:
            raise RuntimeError(f"Recipient not allowed: {to}")

        self.smtp.send_mail(to=to, subject=subject, body=body)

    def send_python_script_mail(self, script_name: str, script_body: str, to: str) -> None:
        subject = f"Ava: Python-Script erstellt - {script_name}"
        body = (
            f"Hallo Roger,\n\n"
            f"Ava hat ein Python-Script erstellt.\n\n"
            f"Datei: {script_name}\n\n"
            f"Inhalt:\n"
            f"{script_body}\n"
        )
        self.send_allowed_mail(to=to, subject=subject, body=body)

    def send_important_note_mail(self, title: str, note: str, to: str) -> None:
        subject = f"Ava: Wichtiger Inhalt - {title}"
        body = (
            f"Hallo Roger,\n\n"
            f"Ava hat folgenden wichtigen Inhalt markiert:\n\n"
            f"{note}\n"
        )
        self.send_allowed_mail(to=to, subject=subject, body=body)
