from dataclasses import replace

from avacore.config.settings import Settings
from avacore.core.cognitive_workspace import (
    WorkingMemory,
    extract_unresolved_questions,
    run_workspace_cycle,
)
from avacore.core.orbit_formation import (
    EpistemicSalienceEvaluator,
    OrbitFormationCandidate,
    OrbitFormationConfig,
    has_explicit_concern,
)
from avacore.core.orbits import OrbitStore
from avacore.core.research import ResearchDriveConfig, ResearchMemory, ResearchService


def evaluator(tmp_path, **changes):
    values = dict(enabled=True, threshold=.7, max_new_per_session=2,
                  max_open=20, cooldown_seconds=3600)
    values.update(changes)
    store = OrbitStore(tmp_path / "orbits.json")
    return EpistemicSalienceEvaluator(store, OrbitFormationConfig(**values)), store


def candidate(**changes):
    values = dict(topic="long-term coherence", title="long-term coherence",
        description="This remains unresolved", source_event_ids=["cycle-1"],
        importance=1, uncertainty=1, persistence=1, explicit_unresolved=1,
        formation_reason="EXPLICIT_UNRESOLVED")
    values.update(changes)
    return OrbitFormationCandidate(**values)


def snapshot_and_memory(tmp_path, topic="orbit coherence", cycle="cycle-1"):
    memory = WorkingMemory(tmp_path / "working.json", session_id="session")
    memory.current_topic = topic
    snapshot = run_workspace_cycle(jspace_path=tmp_path / "continuum.json",
        workspace_path=tmp_path / "workspace.json", stimulus=topic,
        cycle_id=cycle, session_id="session")
    snapshot.active_topic = topic
    return snapshot, memory


def test_generic_assistant_questions_are_not_unresolved():
    for text in ("Wie kann ich Ihnen helfen?", "Könnten Sie mir mehr Kontext geben?",
                 "Was möchten Sie als Nächstes tun?"):
        assert extract_unresolved_questions(text) == []
    assert extract_unresolved_questions(
        "Offen bleibt, welcher Resolver antwortet?") == ["Offen bleibt, welcher Resolver antwortet"]


def test_formation_disabled_and_threshold_are_enforced(tmp_path):
    disabled, store = evaluator(tmp_path, enabled=False)
    assert disabled.evaluate(candidate(), session_id="session")["status"] == "suppressed_disabled"
    assert store.orbits() == []

    enabled, store = evaluator(tmp_path, threshold=.7)
    weak = candidate(importance=.2, uncertainty=.2, persistence=.2,
                     explicit_unresolved=0, recurrence=0)
    assert enabled.evaluate(weak, session_id="session")["status"] == "suppressed_salience"
    assert enabled.evaluate(candidate(), session_id="session")["status"] == "formed"
    assert len(store.orbits()) == 1
    debug = enabled.debug()
    assert debug["formation_candidates_seen"] == 3
    assert debug["orbits_formed"] == 1


def test_score_is_deterministic_clamped_and_explainable(tmp_path):
    formation, _ = evaluator(tmp_path)
    item = candidate(importance=5, uncertainty=-1, recurrence=4, persistence=2,
                     explicit_unresolved=3, explicit_concern=-2, blocked_state=2)
    first = formation.score(item)
    assert first == formation.score(item)
    assert 0 <= first[0] <= 1
    assert all(0 <= value <= 1 for value in first[1].values())
    assert set(first[1]) == {"importance", "uncertainty", "recurrence", "persistence",
                            "novelty", "explicit_unresolved", "explicit_concern", "blocked_state"}


def test_explicit_concern_is_conservative():
    assert has_explicit_concern("Ich bin noch nicht überzeugt, dass das Orbit-System zuverlässig ist.")
    assert has_explicit_concern("This remains unclear and needs attention.")
    assert not has_explicit_concern("Heute sprechen wir über das Orbit-System.")
    assert not has_explicit_concern("Das klingt interessant.")


def test_recurrent_topic_requires_multiple_cycles(tmp_path):
    formation, store = evaluator(tmp_path)
    statuses = []
    for index in range(3):
        snapshot, memory = snapshot_and_memory(tmp_path, cycle=f"cycle-{index}")
        statuses.extend(result["status"] for result in formation.observe_cycle(
            source="conversation", user_text="A normal statement.", snapshot=snapshot,
            working_memory=memory, session_id="session"))
    assert statuses == ["suppressed_missing_epistemic_anchor"]
    assert store.orbits() == []
    assert formation.debug()["suppressed_missing_epistemic_anchor"] == 1


def test_generic_recurrent_topics_do_not_form_without_epistemic_anchor(tmp_path):
    for topic in ("avacore", "camera"):
        folder = tmp_path / topic
        folder.mkdir()
        formation, store = evaluator(folder)
        results = []
        for index in range(4):
            snapshot, memory = snapshot_and_memory(folder, topic=topic, cycle=f"cycle-{index}")
            results.extend(formation.observe_cycle(source="conversation", user_text="Normal statement.",
                snapshot=snapshot, working_memory=memory, session_id="session"))
        assert store.orbits() == []
        assert results
        assert all(item["status"] == "suppressed_missing_epistemic_anchor" for item in results)


def test_recurrent_topic_with_structured_epistemic_anchors_is_eligible(tmp_path):
    cases = (
        candidate(formation_reason="RECURRENT_TOPIC", explicit_unresolved=0,
                  explicit_concern=1, recurrence=1),
        candidate(formation_reason="RECURRENT_TOPIC", explicit_unresolved=1,
                  source_memory_ids=["wm-unresolved"], recurrence=1),
        candidate(formation_reason="RECURRENT_TOPIC", explicit_unresolved=0,
                  blocked_state=1, recurrence=1),
    )
    for index, item in enumerate(cases):
        folder = tmp_path / str(index)
        folder.mkdir()
        formation, store = evaluator(folder)
        assert formation.evaluate(replace(item, source_event_ids=[f"cycle-{index}"]),
                                  session_id="session")["status"] == "formed"
        assert len(store.orbits()) == 1


def test_recurrence_can_strengthen_existing_concrete_orbit_without_duplicate(tmp_path):
    formation, store = evaluator(tmp_path)
    first = formation.evaluate(candidate(), session_id="session")
    before = store.get_orbit(first["orbit_id"])
    recurrent = candidate(formation_reason="RECURRENT_TOPIC", explicit_unresolved=0,
        recurrence=1, source_event_ids=["cycle-2"])

    result = formation.evaluate(recurrent, session_id="session")

    assert result["status"] == "reactivated"
    assert len(store.orbits()) == 1
    assert store.get_orbit(first["orbit_id"]).activation > before.activation


def test_same_deficit_reactivates_one_orbit_and_persists(tmp_path):
    formation, store = evaluator(tmp_path, cooldown_seconds=3600)
    first = formation.evaluate(candidate(), session_id="session")
    original = store.get_orbit(first["orbit_id"])
    second = formation.evaluate(replace(candidate(), source_event_ids=["cycle-2"]),
                                session_id="session")
    reloaded = OrbitStore(store.path)
    updated = reloaded.get_orbit(first["orbit_id"])
    assert second["status"] == "reactivated"
    assert len(reloaded.orbits()) == 1
    assert updated.activation > original.activation
    assert updated.metadata["formation_source_ids"] == ["cycle-1", "cycle-2"]


def test_research_source_never_forms_orbit(tmp_path):
    formation, store = evaluator(tmp_path)
    result = formation.evaluate(candidate(source="research"), session_id="session")
    assert result["status"] == "suppressed_research_source"
    assert store.orbits() == []
    assert formation.debug()["suppressed_research_source"] == 1


def test_bounds_and_question_candidate_and_blocked_task_sources(tmp_path):
    formation, store = evaluator(tmp_path, max_new_per_session=1, max_open=1)
    source_orbit = store.create_orbit("Need input", "A blocked issue", importance=.9)
    question = store.create_question_candidate(source_orbit.orbit_id, "Which input is required?",
        importance=.9, reason="blocked")
    result = formation.from_question_candidate(question, source_orbit, session_id="session")
    assert result["status"] == "reactivated"
    blocked = formation.from_blocked_task(topic="another deficit", description="Task is blocked",
        task_id="task-1", orbit_id="orbit-other", session_id="session")
    assert blocked["status"] == "suppressed_bounds"


def test_observe_cycle_uses_validated_unresolved_item(tmp_path):
    formation, store = evaluator(tmp_path)
    snapshot, memory = snapshot_and_memory(tmp_path)
    item = memory.add("state", "Offen bleibt, welcher Resolver antwortet", snapshot.cycle_id,
                      kind="unresolved_question", importance=.7, topic="DNS resolver")
    results = formation.observe_cycle(source="conversation", user_text="normal", snapshot=snapshot,
        working_memory=memory, session_id="session")
    assert results[0]["status"] == "formed"
    orbit = store.orbits()[0]
    assert orbit.description == item.content
    assert orbit.metadata["formation_memory_ids"] == [item.id]


def test_orbit_formation_has_no_model_or_action_dependencies():
    import inspect
    import avacore.core.orbit_formation as module
    source = inspect.getsource(module)
    for forbidden in ("ModelRouter", "Ollama", "requests.", "httpx", "subprocess",
                      "telegram", "systemctl", "git "):
        assert forbidden not in source


def test_formation_defaults_are_safe(monkeypatch):
    monkeypatch.delenv("AVA_ORBIT_FORMATION_ENABLED", raising=False)
    settings = Settings()
    assert settings.orbit_formation_enabled is False
    assert settings.orbit_formation_threshold == .70
    assert settings.orbit_formation_max_new_per_session == 2
    assert settings.orbit_formation_max_open == 20


def test_semantic_signature_reactivates_same_orbit_and_reaches_research(tmp_path, monkeypatch):
    formation, store = evaluator(tmp_path)
    snapshot, memory = snapshot_and_memory(tmp_path,
        topic="langfristige Kohärenz", cycle="cycle-2")
    memory.add("user", "Wir wollen langfristige Kohärenz verbessern. Welche Schwäche ist relevant?",
               "cycle-1", topic="langfristige Kohärenz")
    formed = formation.observe_cycle(source="conversation",
        user_text=("Ich bin noch nicht überzeugt, dass dein Orbit-System langfristig relevante "
                   "Themen zuverlässig wieder aufgreift."), snapshot=snapshot,
        working_memory=memory, session_id="session")[0]
    orbit_before = store.get_orbit(formed["orbit_id"])
    monkeypatch.setattr("avacore.core.orbits.utc_now", lambda: "2030-01-01T00:00:00+00:00")

    changed = store.react(content=("Was hatten wir als mögliche Schwäche bei der langfristigen "
                                  "Kohärenz diskutiert?"), related_entities=[])

    assert [item.orbit_id for item in changed] == [orbit_before.orbit_id]
    assert len(store.orbits()) == 1
    orbit_after = store.get_orbit(orbit_before.orbit_id)
    assert orbit_after.activation > orbit_before.activation
    assert orbit_after.last_activated_at == "2030-01-01T00:00:00+00:00"
    assert orbit_after.metadata["last_reactivation_reason"] == "semantic_signature"
    assert orbit_after.metadata["last_reactivation_score"] >= .30
    assert orbit_after.metadata["last_reactivation_components"]["signature_overlap"] > 0

    research_memory = ResearchMemory(tmp_path / "research.json")
    research = ResearchService(research_memory, store,
        ResearchDriveConfig(enabled=True, threshold=.2))
    research.from_open_orbit(changed[0], event_id="cycle-3", session_id="session")
    assert research_memory.debug()["signals_seen"] == 1


def test_semantic_signature_is_bounded_and_normalizes_simple_inflections(tmp_path):
    formation, store = evaluator(tmp_path)
    long_context = [" ".join(f"term{index}" for index in range(40)) for _ in range(4)]
    formed = formation.evaluate(candidate(context_texts=long_context,
        context_topics=["one", "two", "three", "four"],
        related_entities=[f"entity:{index}" for index in range(12)],
        source_event_ids=[f"cycle-{index}" for index in range(12)]), session_id="session")
    signature = store.get_orbit(formed["orbit_id"]).metadata["semantic_signature"]
    assert len(signature["terms"]) <= 24
    assert len(signature["topics"]) <= 3
    assert len(signature["entities"]) <= 8
    assert len(signature["source_cycle_ids"]) <= 8

    from avacore.core.orbits import reactivation_terms
    assert reactivation_terms("langfristig langfristige langfristigen") == ["langfristig"]


def test_semantic_reactivation_rejects_unrelated_and_generic_topic_inputs(tmp_path):
    formation, store = evaluator(tmp_path)
    formed = formation.evaluate(candidate(context_texts=[
        "Langfristige Kohärenz und zuverlässige Wiederaufnahme relevanter Themen"
    ], context_topics=["avacore"]), session_id="session")
    orbit_id = formed["orbit_id"]
    before = store.get_orbit(orbit_id)

    assert store.react(content="Wie funktioniert die Kameraerkennung?", related_entities=[]) == []
    assert store.react(content="Heute sprechen wir über ein anderes System.", related_entities=[]) == []
    assert store.react(content="AvaCore", related_entities=[]) == []
    after = store.get_orbit(orbit_id)
    assert after.activation == before.activation
    assert "last_reactivation_score" not in after.metadata
