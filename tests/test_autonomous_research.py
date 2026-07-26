from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from avacore.core.autonomous_research import AutonomousResearchService
from avacore.core.jspace import JSpaceState
from avacore.core.research_curiosity import derive_topics, is_researchable_item, score_item
from avacore.core.research_queue import (
    ResearchQueue,
    ResearchRun,
    ResearchTopic,
    stable_research_id,
)
from avacore.tools.web_research import ResearchSource


class FakeStore:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def list_memory_items(self, **_: object) -> list[dict]:
        return list(self.items)

    def create_memory_item(self, **values: object) -> int:
        memory_id = len(self.items) + 1
        self.items.append({"id": memory_id, **values})
        return memory_id


class FakeBackend:
    def __init__(self, answer: str = "Gesicherte Erkenntnis. Unsicherheit bleibt.") -> None:
        self.answer = answer
        self.calls = 0

    def chat(self, messages: list[dict]) -> str:
        assert messages
        self.calls += 1
        return self.answer


def readable_sources(**_: object) -> list[ResearchSource]:
    return [
        ResearchSource(
            title="Primary source",
            url="https://example.test/source",
            snippet="Relevant",
            text="Reliable readable content",
        )
    ]


def make_settings(tmp_path: Path, mode: str = "bounded", notify_score: float = 0.8):
    jspace_path = tmp_path / "jspace.json"
    state = JSpaceState()
    state.inject(
        source="conversation",
        kind="user_message",
        content="Welche aktuelle Python-Version ist für AvaCore stabil und sicher?",
        tags=["avacore", "python", "security"],
        activation_boost=0.95,
        priority=0.95,
        persistence=0.7,
    )
    state.save(jspace_path)
    return SimpleNamespace(
        research_enabled=True,
        auto_research=mode,
        research_queue_path=tmp_path / "research_queue.json",
        research_max_runs_per_day=3,
        research_max_topics_per_run=5,
        research_max_sources_per_topic=5,
        research_min_score=0.1,
        research_cooldown_hours=24,
        research_notify_score=notify_score,
        research_curiosity_weight=0.15,
        jspace_enabled=True,
        jspace_path=jspace_path,
        jspace_focus_mode="balanced",
        jspace_top_k=8,
        telegram_bot_token="",
        telegram_allowed_chat_id="",
    )


def make_service(
    tmp_path: Path,
    *,
    mode: str = "bounded",
    notify_score: float = 0.8,
    telegram_sender=lambda *_: None,
) -> tuple[AutonomousResearchService, FakeStore, FakeBackend]:
    settings = make_settings(tmp_path, mode=mode, notify_score=notify_score)
    store = FakeStore()
    backend = FakeBackend()
    service = AutonomousResearchService(
        settings=settings,
        memory_store=store,
        backend=backend,
        source_collector=readable_sources,
        telegram_sender=telegram_sender,
    )
    return service, store, backend


def test_queue_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state" / "queue.json"
    topic = ResearchTopic(
        id=stable_research_id("What changed?"),
        title="Changes",
        question="What changed?",
        score=0.8,
    )
    queue = ResearchQueue()
    queue.add_or_update(topic)
    queue.save(path)

    restored = ResearchQueue.load(path)

    assert restored.get(topic.id) is not None
    assert restored.get(topic.id).score == pytest.approx(0.8)


def test_queue_save_uses_atomic_temporary_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "queue.json"
    observed: list[Path] = []
    original_replace = Path.replace

    def checked_replace(source: Path, target: Path):
        assert source.name.endswith(".json.tmp")
        assert source.exists()
        observed.append(source)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", checked_replace)
    ResearchQueue().save(path)

    assert observed
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


def test_stable_ids_and_normalized_deduplication() -> None:
    assert stable_research_id(" What  changed? ") == stable_research_id("what changed")
    queue = ResearchQueue()
    first = ResearchTopic(stable_research_id("Question?"), "One", "Question?", score=0.7)
    second = ResearchTopic(stable_research_id("question"), "Two", " question ", score=0.9)

    queue.add_or_update(first)
    stored, created = queue.add_or_update(second)

    assert created is False
    assert len(queue.topics) == 1
    assert stored.score == pytest.approx(0.9)


def test_cooldown_excludes_recently_researched_topic() -> None:
    now = datetime(2026, 7, 26, 10, tzinfo=timezone.utc)
    topic = ResearchTopic(
        stable_research_id("Question?"),
        "Question",
        "Question?",
        score=0.9,
        last_researched_at=(now - timedelta(hours=2)).isoformat(),
    )
    queue = ResearchQueue(topics={topic.id: topic})

    assert queue.candidates(0.5, cooldown_hours=24, now=now) == []
    assert queue.candidates(0.5, cooldown_hours=1, now=now) == [topic]


def test_derivation_ignores_question_inside_cooldown(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path)
    completed = service.run_next()

    derived = service.derive()

    assert completed["status"] == "completed"
    assert derived["topics"] == []


def test_score_is_bounded_and_components_are_visible() -> None:
    state = JSpaceState()
    item = state.inject(
        "conversation",
        "user_message",
        "What is the latest critical AvaCore security update?",
        activation_boost=5,
        priority=5,
        tags=["avacore", "security"],
    )

    score, components = score_item(item, curiosity_weight=1.0)

    assert 0.0 <= score <= 1.0
    assert set(components) == {
        "activation",
        "priority",
        "knowledge_gap",
        "freshness_need",
        "urgency",
        "curiosity",
    }
    assert all(0.0 <= value <= 1.0 for value in components.values())


def test_identity_rules_are_not_derived_as_topics() -> None:
    state = JSpaceState()
    state.seed_core_items()

    assert derive_topics(state.top_items()) == []


def test_internal_camera_translation_prompt_is_not_researchable() -> None:
    state = JSpaceState()
    item = state.inject(
        "conversation",
        "user_message",
        "Übersetze die folgende Kamerabeschreibung ins Deutsche. "
        "Erfinde keine zusätzlichen Details. Beschreibung: a couch.",
        activation_boost=0.9,
        priority=0.8,
    )

    assert is_researchable_item(item) is False
    assert derive_topics([item]) == []


def test_unsicherheit_does_not_trigger_security_urgency() -> None:
    state = JSpaceState()
    item = state.inject(
        "conversation",
        "user_message",
        "Welche Unsicherheit bleibt bei dieser neutralen technischen Beschreibung?",
        activation_boost=0.8,
        priority=0.5,
    )

    _, components = score_item(item)

    assert components["urgency"] != 0.9


def test_existing_internal_prompt_candidate_is_auto_dismissed(tmp_path: Path) -> None:
    service, _, _ = make_service(tmp_path, mode="ask")
    state = JSpaceState()
    internal_item = state.inject(
        "conversation",
        "user_message",
        "Übersetze die folgende Kamerabeschreibung ins Deutsche. "
        "Beschreibung: a black couch.",
        activation_boost=0.9,
        priority=0.8,
    )
    state.save(service.settings.jspace_path)
    topic = ResearchTopic(
        stable_research_id("Internal camera prompt?"),
        "Internal camera prompt",
        "Internal camera prompt?",
        origin_item_ids=[internal_item.id],
        score=0.9,
    )
    queue = ResearchQueue(topics={topic.id: topic})
    queue.save(service.queue_path)

    result = service.derive()
    stored = service.load_queue().get(topic.id)

    assert result["status"] == "idle"
    assert stored.status == "dismissed"


def test_highest_scored_candidate_is_selected() -> None:
    low = ResearchTopic(stable_research_id("Low?"), "Low", "Low?", score=0.6)
    high = ResearchTopic(stable_research_id("High?"), "High", "High?", score=0.9)
    queue = ResearchQueue(topics={low.id: low, high.id: high})

    assert queue.candidates(0.5, 0)[0].id == high.id


def test_mode_off_does_not_derive_or_run(tmp_path: Path) -> None:
    service, store, backend = make_service(tmp_path, mode="off")

    result = service.run_next()

    assert result["status"] == "disabled"
    assert store.items == []
    assert backend.calls == 0
    assert not service.queue_path.exists()


def test_mode_ask_derives_but_requires_approval(tmp_path: Path) -> None:
    service, store, backend = make_service(tmp_path, mode="ask")

    result = service.run_next()

    assert result["status"] == "approval_required"
    assert service.load_queue().topics
    assert store.items == []
    assert backend.calls == 0


def test_bounded_run_completes_exactly_one_topic_and_candidate_memory(
    tmp_path: Path,
) -> None:
    service, store, backend = make_service(tmp_path, mode="bounded")
    service.derive()
    queue = service.load_queue()
    extra = ResearchTopic(
        stable_research_id("Second research question?"),
        "Second",
        "Second research question?",
        score=0.99,
    )
    queue.add_or_update(extra)
    queue.save(service.queue_path)

    result = service.run_next()

    assert result["status"] == "completed"
    assert backend.calls == 1
    assert len(store.items) == 1
    assert store.items[0]["status"] == "candidate"
    assert store.items[0]["source_type"] == "autonomous_research"
    completed = [
        topic for topic in service.load_queue().topics.values()
        if topic.status == "completed"
    ]
    assert len(completed) == 1


def test_daily_budget_stops_execution(tmp_path: Path) -> None:
    service, store, backend = make_service(tmp_path)
    service.derive()
    queue = service.load_queue()
    now = datetime.now(timezone.utc).isoformat()
    for index in range(service.settings.research_max_runs_per_day):
        queue.append_run(ResearchRun(f"topic-{index}", "completed", now, now))
    queue.save(service.queue_path)

    result = service.run_next()

    assert result["status"] == "budget_exhausted"
    assert store.items == []
    assert backend.calls == 0


def test_completed_topic_does_not_create_duplicate_memory(tmp_path: Path) -> None:
    service, store, _ = make_service(tmp_path)
    first = service.run_next()
    topic = service.load_queue().get(first["topic"]["id"])

    memory_id = service._get_or_create_memory(
        topic,
        "Another summary",
        readable_sources(),
    )

    assert memory_id == first["memory_id"]
    assert len(store.items) == 1


def test_telegram_only_above_threshold(tmp_path: Path) -> None:
    messages: list[str] = []
    service, _, _ = make_service(
        tmp_path,
        notify_score=1.0,
        telegram_sender=lambda _token, _chat, text: messages.append(text),
    )
    service.settings.telegram_bot_token = "token"
    service.settings.telegram_allowed_chat_id = "chat"

    result = service.run_next()

    assert result["status"] == "completed"
    assert messages == []


def test_telegram_is_sent_above_threshold(tmp_path: Path) -> None:
    messages: list[str] = []
    service, _, _ = make_service(
        tmp_path,
        notify_score=0.0,
        telegram_sender=lambda _token, _chat, text: messages.append(text),
    )
    service.settings.telegram_bot_token = "token"
    service.settings.telegram_allowed_chat_id = "chat"

    result = service.run_next()

    assert result["status"] == "completed"
    assert len(messages) == 1
    assert "Memory-Kandidat" in messages[0]


def test_telegram_failure_does_not_fail_research(tmp_path: Path) -> None:
    def broken_sender(*_: object) -> None:
        raise RuntimeError("telegram unavailable")

    service, store, _ = make_service(
        tmp_path,
        notify_score=0.0,
        telegram_sender=broken_sender,
    )
    service.settings.telegram_bot_token = "token"
    service.settings.telegram_allowed_chat_id = "chat"

    result = service.run_next()

    assert result["status"] == "completed"
    assert result["notification_error"] == "telegram unavailable"
    assert len(store.items) == 1


def test_manual_research_route_remains_registered() -> None:
    source = (
        Path(__file__).parents[1] / "avacore" / "api" / "http_app.py"
    ).read_text(encoding="utf-8")

    assert '@app.post("/research")' in source
    assert "return run_research_workflow(" in source
    assert '@app.post("/research/autonomous/run-next")' in source
