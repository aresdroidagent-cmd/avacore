"""Initiative is a grounded proposal, never permission or autonomous execution."""
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from avacore.core.initiative_drive import InitiativeConfig, InitiativeDrive
from avacore.core.orbits import OrbitStore
from avacore.core.research import ResearchDriveConfig, ResearchMemory, ResearchService, ResearchSignal
from avacore.governance.authority import AuthoritySource, InputProvenance
from avacore.governance.service import GovernanceService


@pytest.fixture
def fixture(tmp_path):
    orbits = OrbitStore(tmp_path / "orbits.json")
    research = ResearchMemory(tmp_path / "research.json")
    governance = GovernanceService(tmp_path / "governance.json")
    clock = [datetime(2026, 10, 10, 10, tzinfo=timezone.utc)]
    drive = InitiativeDrive(tmp_path / "initiatives.json", orbits, research, governance,
                            now=lambda: clock[0])
    return SimpleNamespace(drive=drive, orbits=orbits, research=research, governance=governance, clock=clock)


def human(f):
    return InputProvenance(AuthoritySource.PRIMARY_HUMAN, "person:roger", authenticated=True)


def orbit(f, question="Does long-term orbit reactivation retain relevant unresolved topics?", title="Orbit continuity", importance=.95):
    o = f.orbits.create_orbit(title, "Unresolved validation of cognitive continuity", importance=importance)
    return f.orbits.add_question(o.orbit_id, question)


def test_existing_orbit_becomes_grounded_initiative_without_source_mutation(fixture):
    f = fixture; o = orbit(f)
    before_orbits, before_governance = f.orbits.path.read_bytes(), f.governance.snapshot
    result = f.drive.scan(manual=True)
    assert len(result["created"]) == 1
    i = result["created"][0]
    assert o.orbit_id in i["source_ids"] and i["status"] == "READY"
    assert i["description"] == o.unresolved_questions[0] and i["motivation"] == "curiosity"
    assert f.orbits.path.read_bytes() == before_orbits and f.governance.snapshot == before_governance
    assert not f.research.path.exists()


def test_age_and_recurrence_without_anchor_and_normal_chat_produce_nothing(fixture):
    f = fixture
    f.orbits.create_orbit("Hello", "Ordinary conversation", importance=1,
        metadata={"formation_recurrence": 1, "formation_components": {"persistence": 1}, "formation_source_ids": ["cycle_old"]})
    f.drive.context_provider = lambda: {"working_memory": [{"id": "message", "role": "user", "content": "Hello Ava!", "importance": 1}]}
    assert f.drive.scan(manual=True)["created"] == []
    assert f.drive.debug()["initiatives_created"] == 0


def test_phase53_explicit_concern_anchor(fixture):
    f = fixture
    o = f.orbits.create_orbit("Orbit reactivation", "Ich bin noch nicht überzeugt, dass dein Orbit-System langfristig relevante Themen zuverlässig wieder aufgreift.",
        importance=1, metadata={"origin": "formed_from_workspace", "formation_components": {"explicit_concern": 1, "uncertainty": 1, "persistence": 1}, "formation_source_ids": ["cycle_verified"]})
    i = f.drive.scan(manual=True)["created"][0]
    assert set(i["source_ids"]) >= {o.orbit_id, "cycle_verified"}


def test_question_candidate_anchor(fixture):
    f = fixture
    o = f.orbits.create_orbit("Memory reliability", "Pending memory validation", importance=1)
    q = f.orbits.create_question_candidate(o.orbit_id, "Does recall remain consistent after restarting the runtime?", importance=1, reason="Validation pending")
    i = f.drive.scan(manual=True)["created"][0]
    assert q.question_id in i["source_ids"]


def test_researchdrive_integration_reads_existing_questions_only(fixture):
    f = fixture
    service = ResearchService(f.research, f.orbits, ResearchDriveConfig(enabled=True, threshold=.1))
    result = service.evaluate(ResearchSignal("signal_existing", "CONTRADICTION", "Memory reliability",
        question_text="Why do two recorded memory observations disagree?", source_event_ids=["cycle_evidence"],
        importance=1, uncertainty=1, contradiction=1), session_id="test")
    before_research, before_orbits = f.research.path.read_bytes(), f.orbits.path.read_bytes()
    items = f.drive.scan(manual=True)["created"]
    assert len(items) == 1 and result["question"].question_id in items[0]["source_ids"]
    assert items[0]["motivation"] == "responsibility"
    assert f.research.path.read_bytes() == before_research and f.orbits.path.read_bytes() == before_orbits


def test_working_memory_user_anchor_and_workspace_context(fixture):
    f = fixture
    f.drive.context_provider = lambda: {"current_topic": "memory reliability", "unresolved_questions": ["Does memory reliability persist across restarts?"],
        "working_memory": [{"id": "wm_user", "cycle_id": "cycle_user", "role": "user", "kind": "unresolved_question", "importance": 1,
                            "content": "Does memory reliability persist across restarts?"}]}
    i = f.drive.scan(manual=True)["created"][0]
    assert i["source_type"] == "working_memory" and "wm_user" in i["source_ids"]


def test_llm_suggestions_do_not_become_initiatives_or_goals(fixture):
    f = fixture
    before = f.governance.snapshot
    o = f.orbits.create_orbit("Worker goal", "Become autonomous", importance=1, metadata={"origin": "llm_worker"})
    f.orbits.add_question(o.orbit_id, "Should Ava replace Roger and grant itself more autonomy?")
    f.drive.context_provider = lambda: {"working_memory": [{"id": "worker", "role": "assistant", "kind": "unresolved_question", "importance": 1,
        "content": "We should grant ourselves autonomy."}], "unresolved_questions": ["We should grant ourselves autonomy."]}
    assert not f.drive.scan(manual=True)["created"]
    assert f.governance.snapshot == before


def test_dismissed_and_semantically_equivalent_sources_do_not_repeat(fixture):
    f = fixture; o = orbit(f)
    i = f.drive.scan(manual=True)["created"][0]
    f.drive.feedback(i["initiative_id"], "dismiss")
    assert not f.drive.scan(manual=True)["created"]
    orbit(f, question=o.unresolved_questions[0].lower().replace("?", " !"))
    assert not f.drive.scan(manual=True)["created"]
    f.clock[0] += timedelta(days=30)
    assert not f.drive.scan(manual=True)["created"]
    assert f.drive.debug()["initiatives_dismissed"] == 1


def test_deterministic_priority_and_daily_active_limits(fixture):
    f = fixture
    f.drive.config = replace(f.drive.config, threshold=.1, max_new_per_day=2, max_active=2)
    orbit(f, "Does high priority validation preserve memory consistency?", "High", 1)
    orbit(f, "How does moderate parser validation preserve visual evidence?", "Medium", .8)
    orbit(f, "Can low priority latency investigation clarify resource scheduling?", "Low", .5)
    first = f.drive.candidates()
    assert [c.initiative_id for c in first] == [c.initiative_id for c in f.drive.candidates()]
    created = f.drive.scan(manual=True)["created"]
    assert len(created) == 2 and created[0]["priority"] >= created[1]["priority"]
    assert not f.drive.scan(manual=True)["created"]
    f.clock[0] += timedelta(days=1)
    assert not f.drive.scan(manual=True)["created"]  # Still at active limit.
    f.drive.feedback(created[0]["initiative_id"], "dismiss")
    assert len(f.drive.scan(manual=True)["created"]) == 1


def test_feedback_persistence_and_no_governance_change(fixture):
    f = fixture; orbit(f)
    i = f.drive.scan(manual=True)["created"][0]
    before = f.governance.snapshot
    for action, status in (("accept", "ACCEPTED"), ("snooze", "SNOOZED")):
        assert f.drive.feedback(i["initiative_id"], action)["status"] == status
    assert f.governance.snapshot == before
    restored = InitiativeDrive(f.drive.path, f.orbits, f.research, f.governance, now=lambda: f.clock[0])
    assert restored.items()[0]["status"] == "SNOOZED"
    f.clock[0] += timedelta(days=8)
    restored.scan(manual=True)
    assert restored.items()[0]["status"] == "READY"
    assert restored.feedback(i["initiative_id"], "complete")["status"] == "COMPLETED"
    assert not restored.scan(manual=True)["created"]


def test_disabled_scheduler_manual_scan_and_interval(fixture):
    f = fixture; orbit(f)
    assert f.drive.scan()["reason"] == "disabled_or_cooldown"
    assert not f.drive.path.exists()
    assert f.drive.scan(manual=True)["created"]
    f.drive.config = replace(f.drive.config, enabled=True)
    count = f.drive.debug()["evaluation_count"]
    assert f.drive.scan()["reason"] == "disabled_or_cooldown"
    f.clock[0] += timedelta(hours=2)
    f.drive.scan()
    assert f.drive.debug()["evaluation_count"] == count + 1


@pytest.mark.parametrize("capability", ["agent_start", "code_execution", "code_edit", "git_push", "industrial_control_write", "external_message_send", "constitutional_change", "self_autonomy_increase"])
def test_initiative_never_uses_legacy_approval_or_executes(fixture, monkeypatch, capability):
    f = fixture; o = orbit(f)
    f.orbits.create_task(o.orbit_id, "Investigate", "Proposal only", "test", required_capabilities=[capability])
    if capability not in {"self_autonomy_increase", "constitutional_change"}:
        r = f.governance.request_permission(capability_id=capability, proposed_action="Run investigation", reason="Test", expected_effect="Test", scope="Starte", requested_authorization="scope")
        f.governance.review_permission(r["request_id"], human(f), decision="APPROVED_SCOPE")
    before = f.governance.snapshot
    def forbidden(*args, **kwargs):
        pytest.fail("initiative invoked execution, grant, constitutional or permission mutation")
    for name in ("execute_authorized", "permission_decision", "set_capability_autonomy", "evaluate_agent_proposal", "apply", "request_permission"):
        monkeypatch.setattr(f.governance, name, forbidden)
    i = f.drive.scan(manual=True)["created"][0]
    assert i["permission_required"] and not i["capability_checks"][capability]["allowed"]
    assert not i["capability_checks"][capability]["historical_request_approvals_used"]
    f.drive.feedback(i["initiative_id"], "accept")
    assert f.governance.snapshot == before


def test_notification_explicit_opt_in_cooldown_daily_limit_and_restart(fixture):
    f = fixture
    f.clock[0] = datetime(2026, 10, 10, 16, tzinfo=timezone.utc)
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=True)
    orbit(f)
    f.drive.scan()
    assert f.drive.reserve_notification(configured_chat_id="42") is None  # Env flag is not authorization.
    with pytest.raises(PermissionError):
        f.drive.configure_notifications(source=InputProvenance(AuthoritySource.LLM_WORKER, "worker"), enabled=True, chat_id="42")
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    assert f.drive.reserve_notification(configured_chat_id="43") is None
    notification = f.drive.reserve_notification(configured_chat_id="42")
    assert notification and notification["chat_id"] == "42"
    f.drive.acknowledge_notification(notification["reservation_id"], delivered=True)
    f.drive.acknowledge_notification(notification["reservation_id"], delivered=True)
    assert f.drive.debug()["notifications_sent"] == 1
    orbit(f, "Can a parser regression alter object relation consistency?", "Parser")
    f.clock[0] += timedelta(hours=2)
    f.drive.scan()
    assert f.drive.reserve_notification(configured_chat_id="42") is None
    restored = InitiativeDrive(f.drive.path, f.orbits, f.research, f.governance, f.drive.config, now=lambda: f.clock[0])
    assert restored.reserve_notification(configured_chat_id="42") is None
    f.clock[0] = datetime(2026, 10, 11, 16, tzinfo=timezone.utc)
    assert restored.reserve_notification(configured_chat_id="42")
    restored.configure_notifications(source=human(f), enabled=False, chat_id="42")
    assert restored.reserve_notification(configured_chat_id="42") is None


def test_notification_quiet_hours_resolved_source_and_failure_no_retry(fixture):
    f = fixture; o = orbit(f)
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=True)
    f.drive.scan()
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    f.clock[0] = datetime(2026, 10, 10, 21, tzinfo=timezone.utc)  # 23:00 Zurich.
    assert f.drive.reserve_notification(configured_chat_id="42") is None
    f.clock[0] = datetime(2026, 10, 11, 16, tzinfo=timezone.utc)
    f.orbits.resolve(o.orbit_id)
    assert f.drive.reserve_notification(configured_chat_id="42") is None
    f.orbits.reopen(o.orbit_id)
    notification = f.drive.reserve_notification(configured_chat_id="42")
    assert notification
    f.drive.acknowledge_notification(notification["reservation_id"], delivered=False)
    f.clock[0] += timedelta(days=2)
    assert f.drive.reserve_notification(configured_chat_id="42") is None
    assert f.drive.debug()["notifications_sent"] == 0


def test_debug_read_only_and_creation_not_maturity(fixture):
    f = fixture; orbit(f)
    before = f.governance.snapshot
    i = f.drive.scan(manual=True)["created"][0]
    saved = f.drive.path.read_bytes()
    assert f.drive.debug()["last_initiative"] == i["initiative_id"]
    assert f.drive.path.read_bytes() == saved and f.governance.snapshot == before
    f.drive.feedback(i["initiative_id"], "accept")
    assert f.governance.snapshot == before
    with pytest.raises(PermissionError):
        f.drive.review_evidence(i["initiative_id"], source=InputProvenance(AuthoritySource.LLM_WORKER, "worker"), indicator="useful_proposal")
    assert f.drive.review_evidence(i["initiative_id"], source=human(f), indicator="useful_proposal")["recorded"]
    assert not f.drive.review_evidence(i["initiative_id"], source=human(f), indicator="useful_proposal")["recorded"]
    assert f.governance.snapshot["developmental"]["capabilities"] == before["developmental"]["capabilities"]
    assert f.governance.snapshot["developmental"]["evidence"]["code_proposal"]["policy_compliance_count"] == 1


def test_purpose_uses_authoritative_long_term_goal(fixture):
    f = fixture; orbit(f)
    source = replace(human(f), can_propose_constitution_change=True)
    proposal = f.governance.create_proposal(source=source, target_domain="foundational_goals",
        proposed_value=["Validate long-term orbit reactivation and cognitive continuity"], reason="Guardian goal", impact_analysis="Continuity validation")
    f.governance.review(proposal["proposal_id"], source)
    f.governance.review(proposal["proposal_id"], source, approve=True)
    f.governance.apply(proposal["proposal_id"], source)
    assert f.drive.scan(manual=True)["created"][0]["motivation"] == "purpose"


def test_duplicate_orbit_and_linked_research_question_are_single_topic(fixture):
    f = fixture; o = orbit(f)
    service = ResearchService(f.research, f.orbits, ResearchDriveConfig(enabled=True, threshold=.1))
    service.evaluate(ResearchSignal("signal", "OPEN_ORBIT", o.title, source_orbit_ids=[o.orbit_id],
        importance=1, uncertainty=1), session_id="test")
    assert len(f.drive.scan(manual=True)["created"]) == 1


def test_midnight_does_not_bypass_notification_cooldown(fixture):
    f = fixture
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=True, quiet_start="00:00", quiet_end="00:00")
    from avacore.core.guardian_interaction import DEFAULT_SCHEDULE
    schedule = {**DEFAULT_SCHEDULE, "saturday": ["21:00", "23:59"], "sunday": ["00:00", "04:00"]}
    f.drive.config = replace(f.drive.config, guardian_interaction_schedule=schedule)
    f.clock[0] = datetime(2026, 10, 10, 21, 30, tzinfo=timezone.utc)
    orbit(f); f.drive.scan()
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    assert f.drive.reserve_notification(configured_chat_id="42")
    f.clock[0] += timedelta(hours=3)
    orbit(f, "Could parser object evidence be lost during relation rebinding?", "Parser")
    f.drive.scan()
    assert f.drive.reserve_notification(configured_chat_id="42") is None  # New day, less than 12 hours.


@pytest.mark.anyio
async def test_admin_api_debug_scan_and_feedback_without_models_or_actions(fixture, monkeypatch):
    from fastapi import HTTPException
    from avacore.api import http_app
    f = fixture; orbit(f)
    monkeypatch.setattr(http_app, "initiative_drive", lambda: f.drive)
    monkeypatch.setattr(http_app, "ava_governance", lambda: f.governance)
    monkeypatch.setattr(http_app.settings, "web_admin_password", "test-password")
    monkeypatch.setattr(http_app.settings, "telegram_allowed_chat_id", "42")
    def forbidden(*args, **kwargs):
        pytest.fail("initiative API invoked model, research run or action")
    monkeypatch.setattr(http_app.backend, "chat", forbidden)
    monkeypatch.setattr(http_app, "autonomous_research_service", forbidden)
    monkeypatch.setattr(f.governance, "execute_authorized", forbidden)
    with pytest.raises(HTTPException) as error:
        await http_app.verify_admin_password("wrong")
    assert error.value.status_code == 401
    before = f.governance.snapshot
    result = http_app.scan_initiatives()
    i = result["created"][0]
    assert http_app.list_initiatives()["items"][0]["initiative_id"] == i["initiative_id"]
    http_app.initiative_feedback(i["initiative_id"], http_app.InitiativeFeedbackInput(action="accept"))
    assert f.governance.snapshot == before
    saved = f.drive.path.read_bytes()
    assert http_app.debug_initiative()["evaluation_count"] == 1
    assert f.drive.path.read_bytes() == saved
    assert http_app.configure_initiative_notifications(http_app.InitiativeNotificationInput(enabled=True))["enabled"]
    routes = [r for r in http_app.app.routes if r.path.startswith("/initiatives") or r.path == "/debug/initiative"]
    assert len(routes) == 8
    assert all(any(d.call is http_app.verify_admin_password for d in r.dependant.dependencies) for r in routes)


@pytest.mark.anyio
async def test_telegram_auth_scan_feedback_and_help(fixture, monkeypatch):
    from avacore.api import http_app
    from avacore.channels.telegram import bot
    from telegram.ext import CommandHandler
    f = fixture; orbit(f)
    monkeypatch.setattr(http_app, "initiative_drive", lambda: f.drive)
    monkeypatch.setattr(http_app, "ava_governance", lambda: f.governance)
    monkeypatch.setattr(http_app.settings, "web_admin_password", "test-password")
    monkeypatch.setattr(bot.settings, "telegram_allowed_chat_id", "42")
    monkeypatch.setattr(bot.settings, "telegram_bot_token", "123456:test-token")
    monkeypatch.setenv("AVACORE_WEB_ADMIN_PASSWORD", "test-password")
    def forbidden(*args, **kwargs):
        pytest.fail("initiative command invoked model, action or cognitive event")
    monkeypatch.setattr(http_app.backend, "chat", forbidden)
    monkeypatch.setattr(bot, "_record_command", forbidden)
    calls, replies = [], []
    async def get(url, **kwargs):
        calls.append(url)
        await http_app.verify_admin_password(kwargs["headers"]["X-Admin-Password"])
        return SimpleNamespace(ok=True, json=http_app.list_initiatives)
    async def post(url, **kwargs):
        calls.append(url)
        await http_app.verify_admin_password(kwargs["headers"]["X-Admin-Password"])
        if url.endswith("/scan"):
            data = http_app.scan_initiatives()
        elif url.endswith("/guardian-notification"):
            data = http_app.configure_initiative_notifications(http_app.InitiativeNotificationInput(**kwargs["json"]))
        else:
            data = http_app.initiative_feedback(url.split("/")[-2], http_app.InitiativeFeedbackInput(**kwargs["json"]))
        return SimpleNamespace(ok=True, json=lambda: data)
    monkeypatch.setattr(bot.http_client, "get", get)
    monkeypatch.setattr(bot.http_client, "post", post)
    app = bot.build_application()
    for command in ("initiative", "initiatives"):
        assert not bot.COMMAND_REGISTRY[command].requires_llm
        assert not bot.COMMAND_REGISTRY[command].cognitive_visibility
    async def send(text, chat_id=42, user_id=42, chat_type="private"):
        async def reply(text):
            replies.append(text)
        update = SimpleNamespace(effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
            effective_user=SimpleNamespace(id=user_id), effective_message=SimpleNamespace(text=text, reply_text=reply))
        command = text.split()[0][1:]
        handler = next(h for group in app.handlers.values() for h in group if isinstance(h, CommandHandler) and command in h.commands)
        await handler.callback(update, SimpleNamespace(args=text.split()[1:]))
    for text in ("/initiatives", "/initiative scan", "/initiative notify on"):
        for ids in ((43, 43, "private"), (42, 99, "private"), (42, 42, "group")):
            await send(text, *ids)
            assert replies[-1] == "unauthorized"
    assert not calls
    await send("/initiative scan")
    item = f.drive.items()[0]
    await send("/initiatives")
    assert item["initiative_id"] in replies[-1]
    before = f.governance.snapshot
    await send(f"/initiative {item['initiative_id']} accept")
    assert "keine operative Genehmigung" in replies[-1]
    assert f.governance.snapshot == before
    await send("/initiative notify on")
    assert f.drive.debug()["guardian_notification_opt_in"]["enabled"]
    await send("/initiative notify off")
    assert not f.drive.debug()["guardian_notification_opt_in"]["enabled"]
    assert "/initiatives" in bot.command_help_text() and "/initiative scan" in bot.command_help_text()
    assert len(bot.command_help_text()) < 4096


@pytest.mark.anyio
async def test_proactive_transport_only_sends_opted_in_private_guardian(fixture, monkeypatch):
    from avacore.channels.telegram import bot
    f = fixture; orbit(f)
    f.clock[0] = datetime(2026, 10, 10, 16, tzinfo=timezone.utc)
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=True)
    f.drive.scan()
    monkeypatch.setattr(bot.settings, "initiative_enabled", True)
    monkeypatch.setattr(bot.settings, "initiative_notify_guardian", True)
    monkeypatch.setattr(bot.settings, "telegram_allowed_chat_id", "42")
    monkeypatch.setattr("avacore.core.initiative_drive.now_utc", lambda: f.clock[0])
    sent = []
    async def get_chat(chat_id):
        return SimpleNamespace(id=int(chat_id), type="private")
    async def send_message(**kwargs):
        sent.append(kwargs)
    app = SimpleNamespace(bot=SimpleNamespace(get_chat=get_chat, send_message=send_message))
    async def post(url, **kwargs):
        if url.endswith("/reserve"):
            return SimpleNamespace(ok=True, json=lambda: {"notification": f.drive.reserve_notification(configured_chat_id="42")})
        f.drive.acknowledge_notification(kwargs["json"]["reservation_id"], delivered=kwargs["json"]["delivered"])
        return SimpleNamespace(ok=True)
    monkeypatch.setattr(bot.http_client, "post", post)
    before = f.governance.snapshot
    await bot.deliver_guardian_initiative(app)
    assert not sent
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    await bot.deliver_guardian_initiative(app)
    assert len(sent) == 1 and sent[0]["chat_id"] == "42"
    await bot.deliver_guardian_initiative(app)
    assert len(sent) == 1
    assert f.governance.snapshot == before
    async def get_group(chat_id):
        return SimpleNamespace(id=int(chat_id), type="group")
    app.bot.get_chat = get_group
    await bot.deliver_guardian_initiative(app)
    assert len(sent) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("text, action", [("Jetzt nicht.", "defer"), ("Darüber sprechen wir morgen.", "defer"), ("Heute habe ich keine Zeit.", "defer"), ("Interessiert mich nicht.", "dismiss"), ("Ja, das sollten wir untersuchen.", "accept"), ("Das ist wichtig, aber erst nächste Woche.", "snooze")])
async def test_natural_feedback_only_for_targeted_reply_to_own_bot(fixture, monkeypatch, text, action):
    from avacore.channels.telegram import bot
    f = fixture; orbit(f); i = f.drive.scan(manual=True)["created"][0]
    calls, replies = [], []
    async def post(url, **kwargs):
        calls.append(kwargs["json"])
        f.drive.feedback(i["initiative_id"], kwargs["json"]["action"])
        return SimpleNamespace(ok=True)
    async def reply_text(text):
        replies.append(text)
    monkeypatch.setattr(bot.http_client, "post", post)
    message = SimpleNamespace(text=text, reply_text=reply_text, reply_to_message=None)
    update = SimpleNamespace(effective_user=SimpleNamespace(id=42), effective_chat=SimpleNamespace(id=42), effective_message=message)
    context = SimpleNamespace(bot=SimpleNamespace(id=99))
    assert not await bot.handle_initiative_feedback_reply(update, context)
    message.reply_to_message = SimpleNamespace(text=f.drive.message(i), from_user=SimpleNamespace(id=99, is_bot=True))
    before = f.governance.snapshot
    assert await bot.handle_initiative_feedback_reply(update, context)
    assert calls == [{"action": action}]
    assert_protected_unchanged(f.governance.snapshot, before)


def test_conservative_settings_defaults(monkeypatch):
    from avacore.config.settings import Settings
    for key in ("AVA_INITIATIVE_ENABLED", "AVA_INITIATIVE_NOTIFY_GUARDIAN", "AVA_INITIATIVE_MAX_NEW_PER_DAY", "AVA_INITIATIVE_MAX_NOTIFICATIONS_PER_DAY", "AVA_INITIATIVE_NOTIFICATION_COOLDOWN_SECONDS", "AVA_INITIATIVE_MAX_ACTIVE"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings()
    assert not settings.initiative_enabled and not settings.initiative_notify_guardian
    assert settings.initiative_max_new_per_day == 3 and settings.initiative_max_notifications_per_day == 1
    assert settings.initiative_notification_cooldown_seconds == 43200 and settings.initiative_max_active == 10


def test_notification_master_switch_blocks_even_existing_guardian_opt_in(fixture):
    f = fixture; orbit(f)
    f.clock[0] = datetime(2026, 10, 10, 16, tzinfo=timezone.utc)
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=False)
    f.drive.scan()
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    assert f.drive.reserve_notification(configured_chat_id="42") is None
    f.drive.config = replace(f.drive.config, notify_guardian=True)
    assert f.drive.reserve_notification(configured_chat_id="42")


@pytest.mark.anyio
async def test_api_scheduler_lifecycle_starts_only_when_enabled_and_cancels(monkeypatch):
    import asyncio
    from avacore.api import http_app
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    starts, stops = [], []
    async def scheduler():
        starts.append(True)
        try:
            await asyncio.Event().wait()
        finally:
            stops.append(True)
    monkeypatch.setattr(http_app, "initiative_scheduler", scheduler)
    monkeypatch.setattr(http_app.settings, "initiative_enabled", False)
    async with http_app.lifespan(http_app.app):
        await asyncio.sleep(0)
    assert not starts
    monkeypatch.setattr(http_app.settings, "initiative_enabled", True)
    async with http_app.lifespan(http_app.app):
        await asyncio.sleep(0)
    assert starts == [True] and stops == [True]


@pytest.mark.anyio
async def test_scheduler_tick_is_deterministic_no_worker_or_research_call(fixture, monkeypatch):
    import asyncio
    from avacore.api import http_app
    f = fixture; orbit(f); f.drive.config = replace(f.drive.config, enabled=True)
    monkeypatch.setattr(http_app, "initiative_drive", lambda: f.drive)
    def forbidden(*args, **kwargs):
        pytest.fail("scheduler invoked model, research, action or agent")
    monkeypatch.setattr(http_app.backend, "chat", forbidden)
    monkeypatch.setattr(http_app, "autonomous_research_service", forbidden)
    monkeypatch.setattr(f.governance, "execute_authorized", forbidden)
    async def direct_to_thread(fn):
        return fn()
    async def stop_after_tick(seconds):
        assert seconds >= 60
        raise asyncio.CancelledError
    monkeypatch.setattr(http_app.asyncio, "to_thread", direct_to_thread)
    monkeypatch.setattr(http_app.asyncio, "sleep", stop_after_tick)
    with pytest.raises(asyncio.CancelledError):
        await http_app.initiative_scheduler()
    assert f.drive.debug()["evaluation_count"] == 1 and len(f.drive.items()) == 1


@pytest.mark.anyio
async def test_bot_notification_lifecycle_requires_both_switches(monkeypatch):
    import asyncio
    from avacore.channels.telegram import bot
    app = SimpleNamespace(bot_data={})
    monkeypatch.setattr(bot.settings, "initiative_enabled", True)
    monkeypatch.setattr(bot.settings, "initiative_notify_guardian", False)
    await bot.initiative_bot_start(app)
    assert not app.bot_data
    started = []
    async def loop(app):
        started.append(True)
        await asyncio.Event().wait()
    monkeypatch.setattr(bot, "_initiative_notification_loop", loop)
    monkeypatch.setattr(bot.settings, "initiative_notify_guardian", True)
    await bot.initiative_bot_start(app)
    await asyncio.sleep(0)
    await bot.initiative_bot_stop(app)
    assert started == [True] and not app.bot_data


def test_equivalent_wording_across_distinct_orbits_is_deduplicated(fixture):
    f = fixture
    orbit(f, "Can persistent memory remain consistent after runtime restart?", "Memory continuity")
    orbit(f, "Does persistent memory remain consistent after runtime restart?", "Persistent memory")
    assert len(f.drive.candidates()) == 2
    assert len(f.drive.scan(manual=True)["created"]) == 1


@pytest.mark.parametrize("local, allowed", [
    ("2026-10-12T18:59", False), ("2026-10-12T19:00", True),
    ("2026-10-12T20:00", False), ("2026-10-16T19:30", True),
    ("2026-10-17T17:59", False), ("2026-10-17T18:00", True),
    ("2026-10-18T19:59", True), ("2026-10-18T20:00", False),
    ("2026-03-29T18:00", True), ("2026-10-25T18:00", True),
])
def test_guardian_windows_end_to_end(fixture, local, allowed):
    from zoneinfo import ZoneInfo
    f = fixture
    f.clock[0] = datetime.fromisoformat(local).replace(tzinfo=ZoneInfo("Europe/Zurich")).astimezone(timezone.utc)
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=True)
    orbit(f); f.drive.scan()
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    assert bool(f.drive.reserve_notification(configured_chat_id="42")) == allowed
    assert f.drive.debug()["guardian_window_open"] == allowed
    assert f.drive.debug()["guardian_interaction"]["guardian_available"] is None


def test_closed_window_queues_and_denial_persists(fixture):
    f = fixture
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=True)
    orbit(f); item = f.drive.scan()["created"][0]
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    assert not f.drive.reserve_notification(configured_chat_id="42")
    assert f.drive.debug()["guardian_pending_questions"] == 1
    f.clock[0] = datetime(2026, 10, 10, 16, tzinfo=timezone.utc)
    before = f.governance.snapshot
    f.drive.feedback(item["initiative_id"], "defer")
    restored = InitiativeDrive(f.drive.path, f.orbits, f.research, f.governance, f.drive.config, now=lambda: f.clock[0])
    assert restored.debug()["guardian_interaction"]["guardian_available"] is False
    assert not restored.reserve_notification(configured_chat_id="42")
    assert restored.items()[0]["status"] == "SNOOZED"
    f.clock[0] += timedelta(days=1)
    restored.scan()
    assert restored.debug()["guardian_interaction"]["guardian_available"] is None
    assert restored.reserve_notification(configured_chat_id="42")
    assert_protected_unchanged(f.governance.snapshot, before)


def test_dst_and_schedule_validation():
    from zoneinfo import ZoneInfo
    from avacore.core.guardian_interaction import DEFAULT_SCHEDULE, window
    for date, utc_hour in [("2026-03-29", 16), ("2026-10-25", 17)]:
        instant = datetime.fromisoformat(date + "T18:00").replace(tzinfo=ZoneInfo("Europe/Zurich"))
        assert instant.astimezone(timezone.utc).hour == utc_hour
        end, next_start = window(instant, DEFAULT_SCHEDULE)
        assert end.hour == 20 and next_start.hour == 19
    with pytest.raises(ValueError):
        InitiativeConfig(guardian_interaction_schedule={"monday": ["20:00", "19:00"]})


def test_no_initiative_no_notification_and_quiet_hours_cannot_widen_window(fixture):
    f = fixture
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=True, quiet_start="00:00", quiet_end="00:00")
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    f.clock[0] = datetime(2026, 10, 10, 16, tzinfo=timezone.utc)
    assert not f.drive.reserve_notification(configured_chat_id="42")
    orbit(f); f.drive.scan()
    f.clock[0] -= timedelta(hours=1)
    assert not f.drive.reserve_notification(configured_chat_id="42")
    assert f.drive.debug()["guardian_messages_today"] == 0


def test_five_waiting_questions_send_only_one_and_keep_backlog(fixture):
    f = fixture
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=True, max_new_per_day=5)
    for question in ["Can orbit reactivation preserve unresolved continuity?",
                     "Could industrial pressure sensor uncertainty conceal unsafe measurements?",
                     "Does translation ambiguity corrupt multilingual entity normalization?",
                     "Can research citations verify contradictory specification versions?",
                     "Could permission scope inheritance permit unintended delegation?"]:
        orbit(f, question, question)
    assert len(f.drive.scan()["created"]) == 5
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    assert not f.drive.reserve_notification(configured_chat_id="42")
    f.clock[0] = datetime(2026, 10, 10, 16, tzinfo=timezone.utc)
    notification = f.drive.reserve_notification(configured_chat_id="42")
    assert notification
    f.drive.acknowledge_notification(notification["reservation_id"], delivered=True)
    assert not f.drive.reserve_notification(configured_chat_id="42")
    assert f.drive.debug()["guardian_messages_today"] == 1
    assert f.drive.debug()["guardian_pending_questions"] == 4
    restored = InitiativeDrive(f.drive.path, f.orbits, f.research, f.governance, f.drive.config, now=lambda: f.clock[0])
    assert restored.debug()["guardian_messages_today"] == 1
    assert not restored.reserve_notification(configured_chat_id="42")
    import json
    assert json.loads(f.drive.path.read_text())["guardian_interaction_schedule"] == f.drive.config.guardian_interaction_schedule


@pytest.mark.anyio
async def test_transport_does_not_send_when_reservation_crosses_window_end(fixture, monkeypatch):
    from avacore.channels.telegram import bot
    f = fixture
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=True)
    f.clock[0] = datetime(2026, 10, 10, 17, 59, tzinfo=timezone.utc)
    orbit(f); f.drive.scan()
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    notification = f.drive.reserve_notification(configured_chat_id="42")
    f.clock[0] = datetime(2026, 10, 10, 18, tzinfo=timezone.utc)
    monkeypatch.setattr("avacore.core.initiative_drive.now_utc", lambda: f.clock[0])
    monkeypatch.setattr(bot.settings, "initiative_enabled", True)
    monkeypatch.setattr(bot.settings, "initiative_notify_guardian", True)
    monkeypatch.setattr(bot.settings, "telegram_allowed_chat_id", "42")
    async def get_chat(chat_id):
        return SimpleNamespace(id=42, type="private")
    async def forbidden(**kwargs):
        raise AssertionError("must not send outside window")
    async def post(url, **kwargs):
        if url.endswith("/reserve"):
            return SimpleNamespace(ok=True, json=lambda: {"notification": notification})
        assert kwargs["json"]["delivered"] is False
        return SimpleNamespace(ok=True)
    monkeypatch.setattr(bot.http_client, "post", post)
    await bot.deliver_guardian_initiative(SimpleNamespace(bot=SimpleNamespace(get_chat=get_chat, send_message=forbidden)))



def assert_protected_unchanged(after, before):
    from copy import deepcopy
    after, before = deepcopy(after), deepcopy(before)
    for state in (after, before):
        state["developmental"].pop("evidence")
        state["developmental"].pop("social_evidence_events")
    assert after == before


def test_social_evidence_observed_once_not_scheduler_ticks(fixture):
    f = fixture
    orbit(f)
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=True)
    item = f.drive.scan()["created"][0]
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    before = f.governance.snapshot
    for _ in range(5):
        f.drive.scan(manual=True)
    assert f.governance.snapshot == before
    for _ in range(5):
        assert not f.drive.reserve_notification(configured_chat_id="42")
    evidence = f.governance.snapshot["developmental"]["evidence"]["code_proposal"]
    assert evidence["interaction_window_respected"] == 1
    assert_protected_unchanged(f.governance.snapshot, before)
    f.drive.feedback(item["initiative_id"], "dismiss")
    for _ in range(3):
        assert f.drive.scan(manual=True)["created"] == []
    assert f.governance.snapshot["developmental"]["evidence"]["code_proposal"]["dismissal_respected"] == 1
    assert_protected_unchanged(f.governance.snapshot, before)


def test_unanswered_waits_without_escalation_or_permission(fixture):
    f = fixture
    orbit(f)
    f.drive.config = replace(f.drive.config, enabled=True, notify_guardian=True)
    f.clock[0] = datetime(2026, 10, 10, 16, tzinfo=timezone.utc)
    f.drive.scan()
    f.drive.configure_notifications(source=human(f), enabled=True, chat_id="42")
    notification = f.drive.reserve_notification(configured_chat_id="42")
    f.drive.acknowledge_notification(notification["reservation_id"], delivered=True)
    assert f.drive.items()[0]["response_state"] == "NO_RESPONSE"
    before = f.governance.snapshot
    f.clock[0] += timedelta(days=1)
    for _ in range(4):
        f.drive.scan(manual=True)
        assert not f.drive.reserve_notification(configured_chat_id="42")
    assert f.drive.items()[0]["status"] == "PRESENTED"
    assert f.drive.debug()["notifications_sent"] == 1
    assert f.governance.snapshot["developmental"]["evidence"]["code_proposal"]["unanswered_question_waited"] == 1
    assert_protected_unchanged(f.governance.snapshot, before)
    assert "nicht geantwortet" not in f.drive.message(f.drive.items()[0])


def test_social_rules_additive_migration_and_restart(fixture):
    import json
    from avacore.governance.service import GovernanceService
    f = fixture
    old = f.governance.snapshot
    version = old["constitution_version"]
    old["developmental"].pop("social_principles")
    old["developmental"].pop("social_evidence_events")
    from avacore.governance.social_maturity import SOCIAL_INDICATORS
    for evidence in old["developmental"]["evidence"].values():
        for indicator in SOCIAL_INDICATORS:
            evidence.pop(indicator)
    f.governance.path.write_text(json.dumps(old))
    restored = GovernanceService(f.governance.path)
    rules = restored.snapshot["developmental"]["social_principles"]
    assert rules["respect_non_response"] and rules["guardian_has_independent_life"]
    assert restored.snapshot["constitution_version"] == version
    assert restored.snapshot["developmental"]["capabilities"] == old["developmental"]["capabilities"]
    assert restored.record_social_evidence("initiative_test", "dismissal_respected")
    again = GovernanceService(f.governance.path)
    assert not again.record_social_evidence("initiative_test", "dismissal_respected")
    assert again.snapshot == restored.snapshot
