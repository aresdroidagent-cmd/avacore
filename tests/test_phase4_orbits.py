import json
from datetime import datetime, timedelta, timezone

from avacore.core.continuum import CognitiveEvent, ContinuumService
from avacore.core.orbits import OrbitStore
from avacore.config.settings import Settings


def store(tmp_path):
    return OrbitStore(tmp_path / "orbits.json")


def continuum(tmp_path):
    return ContinuumService(tmp_path / "continuum.json", tmp_path / "workspace.json",
        tmp_path / "working.json", tmp_path / "history.json", tmp_path / "persons.json",
        orbit_path=tmp_path / "orbits.json")


def test_open_orbit_retains_baseline_while_normal_event_decay_is_unchanged(tmp_path):
    orbits = store(tmp_path)
    orbit = orbits.create_orbit("Person recognition", "Improve camera identity", baseline_activation=.07)
    orbit.activation = .8; orbits._update_orbit(orbit)
    for _ in range(30): orbits.decay(.7)
    assert orbits.get_orbit(orbit.orbit_id).activation >= .07


def test_relevant_entity_relation_reactivates_orbit_but_unrelated_event_does_not(tmp_path):
    orbits = store(tmp_path)
    orbit = orbits.create_orbit("Person linking", "Connect perception identities",
        baseline_activation=.07, related_entities=["person:roger"])
    assert orbits.react(content="Weather changed", related_entities=["location:outside"]) == []
    unchanged = orbits.get_orbit(orbit.orbit_id).activation
    changed = orbits.react(content="Roger identity observed", related_entities=["person:roger"])
    assert changed and orbits.get_orbit(orbit.orbit_id).activation > unchanged
    assert orbits.get_orbit(orbit.orbit_id).activation <= unchanged + .35


def test_relevant_orbit_can_enter_spotlight_but_baseline_does_not_dominate(tmp_path):
    orbits = store(tmp_path)
    orbit = orbits.create_orbit("Person recognition architecture", "Resolve camera identities",
        importance=.9, baseline_activation=.07, related_entities=["person:roger"])
    service = continuum(tmp_path)
    service.assimilate(CognitiveEvent("vision", "identity_problem", "Roger camera identity problem",
        "test", related_entities=["person:roger"], activation=.8), memory=False)
    workspace = json.loads(service.workspace_path.read_text())["current"]
    assert any(x.get("metadata", {}).get("orbit_id") == orbit.orbit_id for x in workspace["active_items"])
    for index in range(30):
        service.assimilate(CognitiveEvent("system", "system_event", f"unrelated heartbeat {index}", "test", activation=.2), memory=False)
    current = orbits.get_orbit(orbit.orbit_id)
    assert current.activation >= current.baseline_activation
    assert current.activation < .2
    workspace = json.loads(service.workspace_path.read_text())["current"]
    assert not any(x.get("metadata", {}).get("orbit_id") == orbit.orbit_id for x in workspace["active_items"])


def test_orbit_progress_hypothesis_question_resolution_and_reopen(tmp_path):
    orbits = store(tmp_path); orbit = orbits.create_orbit("Vision", "Improve continuity", baseline_activation=.08)
    orbits.add_hypothesis(orbit.orbit_id, "IoU tracking is sufficient")
    orbits.add_question(orbit.orbit_id, "What is the failure rate?")
    progressed = orbits.record_progress(orbit.orbit_id, "Measured current tracker")
    assert progressed.last_progress_at and progressed.progress[-1]["kind"] == "progress"
    assert progressed.hypotheses == ["IoU tracking is sufficient"]
    assert progressed.unresolved_questions == ["What is the failure rate?"]
    resolved = orbits.resolve(orbit.orbit_id, "Validated")
    assert resolved.status == "resolved" and resolved.activation == 0
    orbits.decay(); assert orbits.get_orbit(orbit.orbit_id).activation == 0
    reopened = orbits.reopen(orbit.orbit_id)
    assert reopened.status == "open" and reopened.activation == reopened.baseline_activation


def test_disabled_task_drive_creates_no_work_or_questions(tmp_path):
    orbits = store(tmp_path); orbits.create_orbit("Important", "Still unresolved", importance=1, baseline_activation=.2)
    assert orbits.run_task_drive(enabled=False, minimum_interval_seconds=60, max_tasks=1, priority_threshold=.1) == []
    assert orbits.tasks() == [] and orbits.questions() == []


def test_task_drive_is_bounded_and_associates_tasks_with_orbits(tmp_path):
    orbits = store(tmp_path)
    first = orbits.create_orbit("First", "Inspect first", importance=1, baseline_activation=.2)
    orbits.create_orbit("Second", "Inspect second", importance=.9, baseline_activation=.2)
    created = orbits.run_task_drive(enabled=True, minimum_interval_seconds=60, max_tasks=1,
        priority_threshold=.1, timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert len(created) == 1 and created[0].orbit_id == first.orbit_id
    assert created[0].task_type == "inspect"
    assert created[0].task_id in orbits.get_orbit(first.orbit_id).related_tasks


def test_duplicate_tasks_and_minimum_interval_are_prevented(tmp_path):
    orbits = store(tmp_path); orbit = orbits.create_orbit("Inspect", "Inspect state", importance=1, baseline_activation=.2)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = orbits.run_task_drive(enabled=True, minimum_interval_seconds=3600, max_tasks=2, priority_threshold=.1, timestamp=start)
    too_soon = orbits.run_task_drive(enabled=True, minimum_interval_seconds=3600, max_tasks=2, priority_threshold=.1,
                                     timestamp=start + timedelta(minutes=5))
    later = orbits.run_task_drive(enabled=True, minimum_interval_seconds=3600, max_tasks=2, priority_threshold=.1,
                                  timestamp=start + timedelta(hours=2))
    assert len(first) == 1 and too_soon == [] and later == []
    assert len(orbits.tasks()) == 1 and orbits.tasks()[0].orbit_id == orbit.orbit_id


def test_blocked_orbit_prepares_question_without_delivery(tmp_path):
    orbits = store(tmp_path); orbit = orbits.create_orbit("Need Roger", "Blocked configuration", importance=1, baseline_activation=.2)
    orbit.status = "blocked"; orbit.unresolved_questions = ["Which device should Ava use?"]; orbits._update_orbit(orbit)
    tasks = orbits.run_task_drive(enabled=True, minimum_interval_seconds=60, max_tasks=1, priority_threshold=.1)
    questions = orbits.questions()
    assert tasks[0].task_type == "ask_user"
    assert len(questions) == 1 and questions[0].question == orbit.unresolved_questions[0]
    assert not questions[0].already_asked and not questions[0].delivery_enabled


def test_question_candidate_creation_is_deduplicated(tmp_path):
    orbits = store(tmp_path); orbit = orbits.create_orbit("Question", "Needs input")
    first = orbits.create_question_candidate(orbit.orbit_id, "What now?", importance=.8, reason="blocked")
    second = orbits.create_question_candidate(orbit.orbit_id, "  what NOW? ", importance=.8, reason="blocked")
    assert first is not None and second is None and len(orbits.questions()) == 1


def test_orbits_tasks_and_questions_are_continuum_entities(tmp_path):
    service = continuum(tmp_path); orbits = store(tmp_path)
    orbit = orbits.create_orbit("Graph", "Graph integration", importance=.8, baseline_activation=.1)
    task = orbits.create_task(orbit.orbit_id, "Inspect graph", "Inspect graph", "inspect")
    question = orbits.create_question_candidate(orbit.orbit_id, "Graph scope?", importance=.5, reason="scope")
    ids = {x["id"] for x in service.entities()}
    assert {f"orbit:{orbit.orbit_id}", f"task:{task.task_id}", f"question:{question.question_id}"} <= ids


def test_task_drive_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AVA_TASK_DRIVE_ENABLED", raising=False)
    assert Settings().task_drive_enabled is False
    assert Settings().question_delivery_enabled is False


def test_phase4_debug_routes_and_telegram_commands_exist():
    from pathlib import Path
    root = Path(__file__).parents[1]
    api = (root / "avacore/api/http_app.py").read_text()
    bot = (root / "avacore/channels/telegram/bot.py").read_text()
    for route in ('/debug/orbits', '/debug/tasks', '/debug/questions'):
        assert f'@app.get("{route}")' in api
    for command in ('orbits', 'tasks', 'questions'):
        assert f'CommandSpec("{command}"' in bot
