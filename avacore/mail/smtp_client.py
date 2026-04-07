from __future__ import annotations

import smtplib
from email.message import EmailMessage


class SmtpMailClient:
    def __init__(self, host: str, port: int, username: str, password: str, mail_from: str) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.mail_from = mail_from

    def send_mail(self, to: str, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["From"] = self.mail_from
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(self.host, self.port, timeout=30) as smtp:
            smtp.starttls()
            smtp.login(self.username, self.password)
            smtp.send_message(msg)
