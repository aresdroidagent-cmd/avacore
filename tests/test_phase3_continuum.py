import json

import pytest

from avacore.core.continuum import ContinuumService, VisualObservation
from avacore.core.jspace import Continuum, ContinuumState, JSpace, JSpaceState
from avacore.core.cognitive_workspace import WorkingMemory
from avacore.channels.telegram.bot import CommandSpec, register_commands


def service(tmp_path, threshold=.78):
    return ContinuumService(tmp_path / "continuum.json", tmp_path / "workspace.json",
        tmp_path / "working.json", tmp_path / "history.json", tmp_path / "persons.json",
        confidence_threshold=threshold)


def test_continuum_names_are_compatibility_aliases():
    assert Continuum is JSpace
    assert ContinuumState is JSpaceState


def test_command_and_result_share_cycle_and_enter_memory(tmp_path):
    continuum = service(tmp_path)
    command = continuum.command(session_id="telegram:1", command="status", content="/status")
    result = continuum.command_result(session_id="telegram:1", command="status",
        content="Ava online", cycle_id=command.cycle_id)
    assert command.kind == "user_command"
    assert result.kind == "command_result"
    assert command.cycle_id == result.cycle_id
    memory = WorkingMemory(tmp_path / "working.json", session_id="telegram:1")
    assert [x.kind for x in memory.items[-2:]] == ["user_command", "command_result"]


def test_command_is_available_to_continuum_and_spotlight(tmp_path):
    continuum = service(tmp_path)
    continuum.command(session_id="telegram:1", command="bsp", content="/bsp")
    state = JSpaceState.load(tmp_path / "continuum.json")
    assert any(x.kind == "user_command" for x in state.items.values())
    workspace = json.loads((tmp_path / "workspace.json").read_text())["current"]
    assert any(x["kind"] == "user_command" for x in workspace["active_items"])


def test_visual_observation_deduplicates_stable_scene(tmp_path):
    continuum = service(tmp_path)
    observation = VisualObservation("desk scene", objects=["monitor"], confidence=.8)
    assert len(continuum.observe(observation)) == 1
    assert continuum.observe(observation) == []
    assert sum(x["kind"] == "visual_observation" for x in continuum.events()) == 1


def test_known_person_enter_stable_and_leave_events(tmp_path):
    continuum = service(tmp_path)
    continuum._write(tmp_path / "persons.json", {"persons": [{
        "person_id": "roger", "display_name": "Roger", "visual_identity": "roger",
        "voice_identity": None, "relationship_context": None,
        "current_presence": False, "current_location": None, "first_seen": None,
        "last_seen": None, "confidence": 0.0}]})
    seen = VisualObservation("Roger at desk", persons=[{"person_id":"roger", "confidence":.94, "location":"desk"}])
    first = continuum.observe(seen)
    assert {"visual_observation", "person_entered"} <= {x.kind for x in first}
    assert continuum.observe(seen) == []
    left = continuum.observe(VisualObservation("empty desk"))
    assert {"visual_observation", "person_left"} <= {x.kind for x in left}


@pytest.mark.parametrize("confidence", [.1, .77])
def test_low_confidence_never_claims_known_identity(tmp_path, confidence):
    continuum = service(tmp_path)
    continuum._write(tmp_path / "persons.json", {"persons": [{
        "person_id":"roger", "display_name":"Roger", "current_presence":False,
        "visual_identity":"roger", "voice_identity":None, "relationship_context":None,
        "current_location":None, "first_seen":None, "last_seen":None, "confidence":0.0}]})
    events = continuum.observe(VisualObservation("person", persons=[{"person_id":"roger", "confidence":confidence}]))
    assert not continuum.persons()["roger"].current_presence
    assert any(x.kind == "person_entered" and x.metadata["person_id"].startswith("unknown_person:") for x in events)


def test_unknown_person_remains_unknown(tmp_path):
    continuum = service(tmp_path)
    events = continuum.observe(VisualObservation("unknown person", persons=[{"person_id":"internet_name", "confidence":.99}]))
    assert any(x.kind == "person_entered" and x.metadata["person_id"].startswith("unknown_person:") for x in events)
    assert all(x.metadata.get("person_id") != "internet_name" for x in events)


def test_registry_rejects_duplicate_aliases():
    async def handler(*_): pass
    with pytest.raises(ValueError, match="duplicate"):
        register_commands([CommandSpec("one", handler, "one", aliases=("same",)),
                           CommandSpec("two", handler, "two", aliases=("same",))])


def test_explicit_telegram_and_vision_mapping_share_person_entity(tmp_path):
    continuum = ContinuumService(tmp_path / "continuum.json", tmp_path / "workspace.json",
        tmp_path / "working.json", tmp_path / "history.json", tmp_path / "persons.json",
        known_persons={"roger":"Roger"})
    command = continuum.command(session_id="telegram:1", command="status", content="/status", person_id="roger")
    vision = continuum.observe(VisualObservation("Roger present", persons=[{"person_id":"roger", "confidence":.95}]))
    assert command.related_entities == ["person:roger"]
    assert any(x.related_entities == ["person:roger"] for x in vision)
