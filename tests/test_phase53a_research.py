import json
from dataclasses import asdict

import pytest

from avacore.config.settings import Settings
from avacore.core.continuum import CognitiveEvent, ContinuumService
from avacore.core.jspace import ContinuumState
from avacore.core.orbits import OrbitStore
from avacore.core.research import (
    ResearchDrive,
    ResearchDriveConfig,
    ResearchMemory,
    ResearchQuestion,
    ResearchService,
    ResearchSignal,
    deterministic_question,
)


def signal(**changes):
    values = dict(signal_id="signal_1", signal_type="REPEATED_UNCERTAINTY",
        topic="person recognition robustness", source_event_ids=["event_1"],
        importance=.9, uncertainty=.9, recurrence=.9, contradiction=.5,
        novelty=.8, estimated_cost=.1)
    values.update(changes)
    return ResearchSignal(**values)


def service(tmp_path, **config):
    memory = ResearchMemory(tmp_path / "research.json")
    orbits = OrbitStore(tmp_path / "orbits.json")
    defaults = dict(enabled=True, threshold=.5, max_open_questions=20,
                    max_new_per_session=3, dedupe_window_seconds=86400)
    defaults.update(config)
    return ResearchService(memory, orbits, ResearchDriveConfig(**defaults)), memory, orbits


def test_research_drive_disabled_observes_but_creates_nothing(tmp_path):
    research, memory, _ = service(tmp_path, enabled=False)
    result = research.evaluate(signal(), session_id="session")
    assert result["status"] == "suppressed_disabled"
    assert memory.questions() == []
    assert memory.debug()["signals_seen"] == 1
    assert memory.debug()["suppressed_disabled"] == 1


def test_threshold_cost_clamping_and_determinism():
    drive = ResearchDrive(ResearchDriveConfig(enabled=True, threshold=.8))
    extreme = signal(importance=9, uncertainty=-2, recurrence=3,
                     contradiction=-1, novelty=8, estimated_cost=1)
    first = drive.score(extreme)
    assert first == drive.score(extreme)
    assert 0 <= first[0] <= 1
    assert all(0 <= value <= 1 for value in first[1].values())
    costly = drive.score(signal(estimated_cost=1))[0]
    cheap = drive.score(signal(estimated_cost=0))[0]
    assert costly < cheap
    assert drive.accepts(signal(importance=.1, uncertainty=.1, recurrence=0,
                                contradiction=0, novelty=0))[0] is False


def test_question_candidate_becomes_research_question_without_model(tmp_path):
    research, memory, orbits = service(tmp_path, threshold=.3)
    orbit = orbits.create_orbit("Camera uncertainty", "Identity is unresolved", importance=.9)
    candidate = orbits.create_question_candidate(
        orbit.orbit_id, "Why is the camera identity uncertain?", importance=.9,
        reason="blocked on identity evidence")
    result = research.from_question_candidate(candidate, orbit, session_id="session")
    question = result["question"]
    assert question.text == candidate.question
    assert question.origin == "QUESTION_CANDIDATE"
    assert question.source_question_candidate_ids == [candidate.question_id]
    assert question.source_orbit_ids == [orbit.orbit_id]
    assert question.hypothesis_count == question.evidence_count == 0
    assert memory.orbits()[0].cognitive_orbit_id == orbit.orbit_id


@pytest.mark.parametrize("kind,expected", [
    ("OPEN_ORBIT", "Why has the cognitive orbit concerning"),
    ("REPEATED_UNCERTAINTY", "Why does uncertainty about"),
    ("CONTRADICTION", "What explains the unresolved contradiction"),
])
def test_supported_signals_use_fact_neutral_templates(kind, expected):
    question = deterministic_question(signal(signal_type=kind, question_text=None))
    assert question.startswith(expected)
    assert "because" not in question.casefold()


def test_duplicate_signal_reactivates_one_question_and_merges_sources(tmp_path):
    research, memory, orbits = service(tmp_path)
    first = research.evaluate(signal(), session_id="session")
    before = orbits.get_orbit(first["orbit"].cognitive_orbit_id).activation
    second = research.evaluate(signal(signal_id="signal_2", source_event_ids=["event_2"]),
                               session_id="session")
    after = orbits.get_orbit(second["orbit"].cognitive_orbit_id).activation
    questions = memory.questions()
    assert second["status"] == "reactivated" and second["deduplicated"]
    assert len(questions) == 1
    assert questions[0].recurrence > first["question"].recurrence
    assert questions[0].source_event_ids == ["event_1", "event_2"]
    assert questions[0].reactivation_count == 1
    assert after > before


def test_cross_origin_signals_share_deficit_fingerprint_and_preserve_provenance(tmp_path):
    research, memory, orbits = service(tmp_path, threshold=.3)
    orbit = orbits.create_orbit("Identity uncertainty", "Camera identity remains unresolved",
        importance=.9, metadata={"uncertainty":.9})
    candidate = orbits.create_question_candidate(
        orbit.orbit_id, "Why is the camera identity uncertain?", importance=.9,
        reason="identity evidence is incomplete")

    first = research.from_question_candidate(candidate, orbit, session_id="session")
    second = research.from_open_orbit(orbit, event_id="event_2", session_id="session")

    questions = memory.questions()
    assert first["question"].question_id == second["question"].question_id
    assert len(questions) == 1
    assert questions[0].recurrence > first["question"].recurrence
    assert questions[0].origin == "QUESTION_CANDIDATE"
    assert set(questions[0].observed_origins) == {"QUESTION_CANDIDATE", "OPEN_ORBIT"}

    other_orbit = orbits.create_orbit("Identity uncertainty", "A separate sensor uncertainty",
        importance=.9, metadata={"uncertainty":.9})
    research.from_open_orbit(other_orbit, event_id="event_3", session_id="session")
    assert len(memory.questions()) == 2


def test_repeated_below_threshold_signals_accumulate_recurrence(tmp_path):
    research, memory, _ = service(tmp_path, threshold=.45)
    weak = signal(importance=.5, uncertainty=.5, recurrence=0,
                  contradiction=0, novelty=.3, estimated_cost=.1)
    statuses = [research.evaluate(ResearchSignal(**{**asdict(weak), "signal_id":f"s{i}"}),
                                  session_id="session")["status"] for i in range(6)]
    assert statuses[0] == "suppressed_threshold"
    assert statuses[-1] == "created"
    assert len(memory.questions()) == 1


def test_open_and_session_bounds_suppress_without_losing_signal(tmp_path):
    research, memory, _ = service(tmp_path, max_open_questions=1, max_new_per_session=1)
    assert research.evaluate(signal(), session_id="session")["status"] == "created"
    other = signal(signal_id="other", topic="another unknown", source_event_ids=["event_2"])
    assert research.evaluate(other, session_id="session")["status"] == "suppressed_limits"
    assert memory.debug()["signals_seen"] == 2
    assert memory.debug()["suppressed_limits"] == 1


def test_closed_question_respects_dedupe_cooldown(tmp_path):
    research, memory, _ = service(tmp_path)
    created = research.evaluate(signal(), session_id="session")
    question, link = created["question"], created["orbit"]
    question.status = "RESOLVED"
    memory.save_question(question, link, new=False, session_id="session")
    result = research.evaluate(signal(signal_id="later"), session_id="other")
    assert result["status"] == "suppressed_cooldown"


def test_research_memory_survives_reload_and_bounds_reads(tmp_path):
    research, memory, _ = service(tmp_path)
    created = research.evaluate(signal(), session_id="session")
    reloaded = ResearchMemory(memory.path)
    question = reloaded.questions(limit=1)[0]
    assert question.question_id == created["question"].question_id
    assert question.status == "NEW"
    assert question.source_event_ids == ["event_1"]
    assert reloaded.questions(limit=0) == []
    raw = memory.path.read_text()
    assert "user_prompt" not in raw and "password" not in raw and "api_key" not in raw


def test_continuum_event_reuses_orbit_competition_and_persists_across_restart(tmp_path):
    research, memory, orbits = service(tmp_path, threshold=.25)
    orbit = orbits.create_orbit("Identity uncertainty", "Unresolved camera identity",
        importance=.9, related_entities=["person:roger"], metadata={"uncertainty":.9})
    continuum = ContinuumService(tmp_path / "continuum.json", tmp_path / "workspace.json",
        tmp_path / "working.json", tmp_path / "history.json", tmp_path / "persons.json",
        orbit_path=orbits.path, research_service=research)
    continuum.assimilate(CognitiveEvent("vision", "identity_uncertain", "Identity uncertainty",
        "session", related_entities=["person:roger"]), memory=False)
    first = memory.questions()[0]
    assert first.source_orbit_ids == [orbit.orbit_id]
    assert any(item.kind == "research_question" for item in
               ContinuumState.load(tmp_path / "continuum.json").items.values())
    restarted = ResearchService(ResearchMemory(memory.path), OrbitStore(orbits.path),
        research.config)
    before = first.recurrence
    restarted.from_open_orbit(orbits.get_orbit(orbit.orbit_id), event_id="event_3",
                              session_id="session")
    assert len(ResearchMemory(memory.path).questions()) == 1
    assert ResearchMemory(memory.path).questions()[0].recurrence > before


def test_research_event_cannot_self_reactivate_but_independent_event_can(tmp_path):
    research, memory, orbits = service(tmp_path, threshold=.25)
    orbit = orbits.create_orbit("Identity uncertainty", "Unresolved camera identity",
        importance=.9, related_entities=["person:roger"], metadata={"uncertainty":.9})
    created = research.from_open_orbit(orbit, event_id="external_1", session_id="session")
    continuum = ContinuumService(tmp_path / "continuum.json", tmp_path / "workspace.json",
        tmp_path / "working.json", tmp_path / "history.json", tmp_path / "persons.json",
        orbit_path=orbits.path, research_service=research)
    before_orbit = orbits.get_orbit(orbit.orbit_id)
    before_question = memory.questions()[0]
    before_signals = memory.debug()["signals_seen"]

    continuum.assimilate(CognitiveEvent("research", "research_question", created["question"].text,
        "session", related_entities=["person:roger"]), memory=False)

    after_research_orbit = orbits.get_orbit(orbit.orbit_id)
    after_research_question = memory.questions()[0]
    assert after_research_orbit.activation == before_orbit.activation
    assert after_research_orbit.last_activated_at == before_orbit.last_activated_at
    assert memory.debug()["signals_seen"] == before_signals
    assert after_research_question.recurrence == before_question.recurrence
    assert after_research_question.reactivation_count == before_question.reactivation_count

    continuum.assimilate(CognitiveEvent("conversation", "user_message", "Identity uncertainty",
        "session", related_entities=["person:roger"]), memory=False)
    assert orbits.get_orbit(orbit.orbit_id).activation > after_research_orbit.activation
    assert memory.questions()[0].reactivation_count == before_question.reactivation_count + 1


def test_research_service_has_no_action_or_model_dependencies():
    import inspect
    import avacore.core.research as module
    source = inspect.getsource(module)
    for forbidden in ("requests.", "httpx", "ModelRouter", "Ollama", "subprocess",
                      "git ", "systemctl", "telegram"):
        assert forbidden not in source


def test_research_defaults_are_safe(monkeypatch):
    monkeypatch.delenv("AVA_RESEARCH_DRIVE_ENABLED", raising=False)
    settings = Settings()
    assert settings.research_drive_enabled is False
    assert settings.research_threshold == .65
    assert settings.research_max_open == 20
    assert settings.research_max_new_per_session == 3


def test_research_debug_routes_are_admin_protected_by_declaration():
    from pathlib import Path
    source = (Path(__file__).parents[1] / "avacore/api/http_app.py").read_text()
    for route in ("/debug/research", "/debug/research/questions", "/debug/research/orbits"):
        assert f'@app.get("{route}")' in source
    assert source.count("def debug_research") >= 3
    assert "Depends(verify_admin_password)" in source
