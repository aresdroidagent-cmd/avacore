import json
from dataclasses import asdict

from avacore.core.continuum import ContinuumService, VisualObservation
from avacore.core.jspace import JSpaceState


def service(tmp_path):
    return ContinuumService(tmp_path / "continuum.json", tmp_path / "workspace.json",
        tmp_path / "working.json", tmp_path / "history.json", tmp_path / "persons.json",
        known_persons={"roger":"Roger"}, confidence_threshold=.8)


def test_repeated_frames_reuse_track_and_unknown_entity(tmp_path):
    continuum = service(tmp_path)
    frame = VisualObservation("person", persons=[{"track_id":"17", "confidence":.4}])
    continuum.observe(frame); continuum.observe(frame)
    tracks = [x for x in continuum.entities() if x["kind"] == "visual_track"]
    unknown = [x for x in continuum.entities() if x["kind"] == "person" and not x["known"]]
    assert len(tracks) == len(unknown) == 1
    assert tracks[0]["person_id"] == unknown[0]["person_id"]


def test_track_identity_resolution_is_conservative_and_emits_event(tmp_path):
    continuum = service(tmp_path)
    continuum.observe(VisualObservation("person", persons=[{"track_id":"17", "person_id":"roger", "confidence":.6}]))
    assert not continuum.persons()["roger"].current_presence
    events = continuum.observe(VisualObservation("Roger", persons=[{"track_id":"17", "person_id":"roger", "confidence":.95}]))
    assert continuum.persons()["roger"].current_presence
    assert any(x.kind == "identity_resolved" for x in events)
    assert any(x.predicate == "identified_as" and x.object_id == "person:roger" for x in continuum.relations())


def test_stable_track_and_relations_do_not_flood_events(tmp_path):
    continuum = service(tmp_path)
    frame = VisualObservation("Roger at desk", persons=[{"track_id":"17", "person_id":"roger", "confidence":.95, "location":"desk"}])
    continuum.observe(frame)
    before = len(continuum.events())
    assert continuum.observe(frame) == []
    assert len(continuum.events()) == before


def test_generic_relation_creation_and_update(tmp_path):
    continuum = service(tmp_path)
    first, created = continuum.relate("person:roger", "working_on", "project:avacore", confidence=.7, source="test")
    second, created_again = continuum.relate("person:roger", "working_on", "project:avacore", confidence=.9, source="test")
    assert created and not created_again
    assert first.id == second.id
    assert second.confidence == .9
    assert len([x for x in continuum.relations() if x.predicate == "working_on"]) == 1


def test_relation_removal_and_expiry(tmp_path):
    continuum = service(tmp_path)
    continuum.relate("person:roger", "working_on", "project:old", confidence=.8, source="test")
    assert continuum.remove_relation("person:roger", "working_on", "project:old")
    assert not continuum.remove_relation("person:roger", "working_on", "project:old")
    continuum.relate("person:roger", "related_to", "project:old", confidence=.8, source="test")
    graph = continuum._graph(); graph["relations"][0]["last_observed"] = "2000-01-01T00:00:00+00:00"
    continuum._write(continuum.persons_path, graph)
    assert continuum.expire_relations("2020-01-01T00:00:00+00:00") == 1


def test_location_change_replaces_relation_and_enters_continuum(tmp_path):
    continuum = service(tmp_path)
    continuum.observe(VisualObservation("desk", persons=[{"track_id":"17", "person_id":"roger", "confidence":.95, "location":"desk"}]))
    events = continuum.observe(VisualObservation("door", persons=[{"track_id":"17", "person_id":"roger", "confidence":.95, "location":"door"}]))
    assert any(x.kind == "location_changed" for x in events)
    assert not any(x.object_id == "location:desk" and x.predicate == "present_at" for x in continuum.relations())
    assert any(x.object_id == "location:door" and x.predicate == "present_at" for x in continuum.relations())
    state = JSpaceState.load(continuum.continuum_path)
    assert any(x.kind == "location_changed" for x in state.items.values())


def test_person_left_removes_presence_relation(tmp_path):
    continuum = service(tmp_path)
    continuum.observe(VisualObservation("desk", persons=[{"track_id":"17", "person_id":"roger", "confidence":.95, "location":"desk"}]))
    events = continuum.observe(VisualObservation("empty"))
    assert any(x.kind == "person_left" for x in events)
    assert any(x.kind == "relation_removed" for x in events)
    assert not any(x.predicate == "present_at" for x in continuum.relations())


def test_explicit_telegram_mapping_creates_same_person_relations(tmp_path):
    continuum = service(tmp_path)
    continuum.command(session_id="telegram:42", command="who", content="/who", person_id="roger")
    continuum.observe(VisualObservation("Roger", persons=[{"track_id":"17", "person_id":"roger", "confidence":.95}]))
    links = continuum.relations()
    assert any(x.subject_id == "person:roger" and x.predicate == "owns_session" for x in links)
    assert any(x.object_id == "person:roger" and x.predicate == "identified_as" for x in links)
    assert len([x for x in continuum.persons().values() if x.person_id == "roger"]) == 1


def test_relevant_person_event_can_enter_spotlight(tmp_path):
    continuum = service(tmp_path)
    continuum.observe(VisualObservation("Roger entered", persons=[{"track_id":"17", "person_id":"roger", "confidence":.99}]))
    workspace = json.loads(continuum.workspace_path.read_text())["current"]
    assert any("person:roger" in x.get("metadata", {}).get("related_entities", [])
               for x in workspace["active_items"])


def test_debug_entities_never_contain_embeddings(tmp_path):
    continuum = service(tmp_path)
    continuum.observe(VisualObservation("Roger", persons=[{"track_id":"17", "person_id":"roger", "confidence":.99}]))
    serialized = json.dumps({"entities":continuum.entities(), "relations":[asdict(x) for x in continuum.relations()]})
    assert "embedding" not in serialized.casefold()
