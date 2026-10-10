"""Help documents existing commands without evaluating or changing governance."""
from types import SimpleNamespace

import pytest

from avacore.channels.telegram import bot
from avacore.governance.service import GovernanceService


@pytest.mark.anyio
async def test_authorized_help_documents_governance_without_side_effects(tmp_path, monkeypatch):
    from avacore.api import http_app
    governance = GovernanceService(tmp_path / "governance.json")
    before = governance.snapshot
    persisted = governance.path.read_text()
    monkeypatch.setattr(bot.settings, "telegram_allowed_chat_id", "42")
    monkeypatch.setattr(http_app, "ava_governance", lambda: governance)
    def forbidden(*args, **kwargs):
        pytest.fail("help invoked governance, permission, model or API logic")
    for name in ("evaluate", "permission_decision", "request_permission", "review_permission", "execute_authorized"):
        monkeypatch.setattr(governance, name, forbidden)
    monkeypatch.setattr(http_app.backend, "chat", forbidden)
    monkeypatch.setattr(bot.http_client, "get", forbidden)
    monkeypatch.setattr(bot.http_client, "post", forbidden)
    replies = []
    async def reply_text(text):
        replies.append(text)
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=42, type="private"),
        effective_user=SimpleNamespace(id=42), effective_message=SimpleNamespace(text="/help", reply_text=reply_text))
    await bot.help_cmd(update, SimpleNamespace(args=[]))
    text = replies[0]
    assert "Governance & Autonomy (nur autorisierter privater Nutzer)" in text
    assert "/autonomy - Entwicklungsstufe und Capability-Autonomie anzeigen" in text
    assert "/permissions - offene Permission Requests anzeigen" in text
    assert "/permission <id> <once|scope|deny|cancel>" in text
    for explanation in ("once = einmalig erlauben", "scope = vorgesehenen Scope erlauben", "deny = verweigern", "cancel = abbrechen"):
        assert explanation in text
    assert "/permissiontest <capability> <scope|-> <action> - PermissionGate-Diagnose" in text
    assert "keine reale Aktion ausführen; - = ohne Scope" in text
    assert "/govtest <agent|llm|roger> <content> - Governance-/Integrity-Diagnose" in text
    assert "simulierter Provenance; keine Core-Mutation" in text
    for command in ("/start", "/help", "/health", "/research", "/see", "/idcheck", "/sendmail", "/notes", "/questions"):
        assert command in text
    assert governance.snapshot == before
    assert governance.path.read_text() == persisted


@pytest.mark.anyio
@pytest.mark.parametrize("chat_id, chat_type", [(43, "private"), (42, "group"), (42, "supergroup")])
async def test_unauthorized_help_exposes_no_governance_commands(monkeypatch, chat_id, chat_type):
    monkeypatch.setattr(bot.settings, "telegram_allowed_chat_id", "42")
    replies = []
    async def reply_text(text):
        replies.append(text)
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
        effective_message=SimpleNamespace(text="/help", reply_text=reply_text))
    await bot.help_cmd(update, SimpleNamespace(args=[]))
    assert replies == ["Dieser Chat ist nicht freigegeben."]
    assert not any(command in replies[0] for command in ("/autonomy", "/permissions", "/permission", "/permissiontest", "/govtest"))
