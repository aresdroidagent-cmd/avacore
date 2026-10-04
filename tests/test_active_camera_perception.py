from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from PIL import Image

from avacore.core.continuum import ContinuumService
from avacore.memory.sqlite_store import SQLiteStore
from avacore.vision.perception import CameraPerceptionService, _deduplicate_scene_persons


def continuum(tmp_path):
    return ContinuumService(tmp_path / "continuum.json", tmp_path / "workspace.json",
        tmp_path / "working.json", tmp_path / "history.json", tmp_path / "persons.json",
        known_persons={"roger":"Roger"}, confidence_threshold=.8)


def configuration(tmp_path, freshness=3):
    return SimpleNamespace(camera_enabled=True, camera_ip="camera.local", camera_user="user",
        camera_password="secret", camera_rtsp_path="/stream", camera_cache_dir=tmp_path,
        perception_freshness_seconds=freshness, perception_track_iou_threshold=.25,
        identity_enabled=True, person_recognition_enabled=True, known_persons={"roger":"Roger"},
        identity_dir=tmp_path / "identity", identity_model="test", identity_device="cpu",
        person_confidence_threshold=.8, identity_margin=.05, identity_top_k=3,
        identity_min_roger_votes=1, vision_enabled=True)


def frame(tmp_path):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (640, 480), "white").save(path)
    return path


def perception(tmp_path, detections, decisions=(), freshness=3, descriptions=()):
    source = frame(tmp_path); decision_iter = iter(decisions); description_iter = iter(descriptions)
    def recognize(**_):
        identity, confidence = next(decision_iter, ("unknown", .4))
        return SimpleNamespace(identity=identity, confidence=confidence, face_path="face.jpg",
                               top_label=identity, reason="test decision")
    return CameraPerceptionService(configuration(tmp_path, freshness), continuum(tmp_path),
        capture=lambda **_: source, detector=lambda _: list(detections), recognizer=recognize,
        describer=lambda *_args, **_kwargs: next(description_iter, ""))


def test_stale_request_captures_and_fresh_request_reuses(tmp_path):
    calls = {"capture":0}
    source = frame(tmp_path)
    service = CameraPerceptionService(configuration(tmp_path, 30), continuum(tmp_path),
        capture=lambda **_: (calls.__setitem__("capture", calls["capture"] + 1) or source),
        detector=lambda _: [], recognizer=lambda **_: None, describer=lambda *_a, **_k: "")
    first = service.request(reason="who_command")
    second = service.request(reason="who_command")
    assert not first.reused and second.reused
    assert calls["capture"] == 1


def test_forced_see_always_captures_and_requests_semantics(tmp_path):
    calls = {"description":0}; source = frame(tmp_path)
    service = CameraPerceptionService(configuration(tmp_path, 30), continuum(tmp_path),
        capture=lambda **_: source, detector=lambda _: [], recognizer=lambda **_: None,
        describer=lambda *_a, **_k: (calls.__setitem__("description", calls["description"] + 1) or "empty room"))
    service.request(reason="see_command", force=True, include_scene=True)
    service.request(reason="see_command", force=True, include_scene=True)
    assert calls["description"] == 2


def test_semantic_perception_preempts_before_vlm(tmp_path):
    order = []
    service = CameraPerceptionService(
        configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [],
        recognizer=lambda **_: None,
        vision_preflight=lambda: order.append("unload"),
        describer=lambda *_a, **_k: (order.append("vlm") or "empty room"),
    )
    service.request(reason="see_command", force=True, include_scene=True)
    assert order == ["unload", "vlm"]


def test_semantic_vlm_runs_inside_resource_lease(tmp_path):
    order = []

    @contextmanager
    def lease():
        order.append("lease-enter")
        try:
            yield
        finally:
            order.append("lease-exit")

    service = CameraPerceptionService(
        configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [],
        recognizer=lambda **_: None, vision_lease=lease,
        describer=lambda *_a, **_k: (order.append("vlm") or "empty room"),
    )
    service.request(reason="see_command", force=True, include_scene=True)
    assert order == ["lease-enter", "vlm", "lease-exit"]


def test_structured_perception_does_not_preempt_or_call_vlm(tmp_path):
    calls = []
    service = CameraPerceptionService(
        configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [],
        recognizer=lambda **_: None,
        vision_preflight=lambda: calls.append("unload"),
        describer=lambda *_a, **_k: calls.append("vlm"),
    )
    service.request(reason="who_command", force=True, include_scene=False)
    assert calls == []


def test_preflight_failure_does_not_abort_semantic_perception(tmp_path):
    calls = []
    service = CameraPerceptionService(
        configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [],
        recognizer=lambda **_: None,
        vision_preflight=lambda: (_ for _ in ()).throw(RuntimeError("Ollama offline")),
        describer=lambda *_a, **_k: (calls.append("vlm") or "empty room"),
    )
    result = service.request(reason="see_command", force=True, include_scene=True)
    assert result.scene_description == "empty room"
    assert calls == ["vlm"]


def test_who_path_does_not_require_semantic_model(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [], recognizer=lambda **_: None,
        describer=lambda *_a, **_k: pytest.fail("semantic model must not run"))
    result = service.request(reason="who_command")
    assert result.scene_description == ""
    assert result.persons == []


def test_visible_known_person_updates_canonical_entity(tmp_path):
    service = perception(tmp_path, [[10, 10, 100, 200]], [("roger", .96)])
    result = service.request(reason="who_command")
    assert result.identities_resolved == ["roger"]
    assert service.continuum.persons()["roger"].current_presence


def test_visible_person_without_face_remains_anonymous(tmp_path):
    service = perception(tmp_path, [[10, 10, 100, 200]], [])
    result = service.request(reason="who_command")
    assert len(result.persons) == 1 and result.persons[0]["person_id"] is None
    assert any(not person.known and person.current_presence for person in service.continuum.persons().values())


def test_two_people_get_distinct_tracks_and_known_unknown_coexist(tmp_path):
    service = perception(tmp_path, [[10, 10, 100, 200], [300, 10, 100, 200]],
                         [("roger", .96), ("unknown", .5)])
    result = service.request(reason="who_command")
    assert len(set(result.tracks_active)) == 2
    people = service.continuum.persons().values()
    assert any(x.person_id == "roger" and x.current_presence for x in people)
    assert any(not x.known and x.current_presence for x in people)


def test_repeated_boxes_reuse_tracks_without_event_flood(tmp_path):
    source = frame(tmp_path); decisions = iter([("unknown", .4), ("unknown", .4)])
    service = CameraPerceptionService(configuration(tmp_path, 0), continuum(tmp_path),
        capture=lambda **_: source, detector=lambda _: [[10, 10, 100, 200]],
        recognizer=lambda **_: SimpleNamespace(identity=next(decisions)[0], confidence=.4), describer=lambda *_a, **_k: "")
    first = service.request(reason="monitor", force=True)
    count = len(service.continuum.events())
    second = service.request(reason="monitor", force=True)
    assert first.tracks_active == second.tracks_active
    assert len(service.continuum.events()) == count


def test_empty_fresh_capture_expires_presence_and_clears_current_location(tmp_path):
    detections = iter([[[10, 10, 100, 200]], []]); source = frame(tmp_path)
    service = CameraPerceptionService(configuration(tmp_path, 0), continuum(tmp_path),
        capture=lambda **_: source, detector=lambda _: next(detections),
        recognizer=lambda **_: SimpleNamespace(identity="roger", confidence=.96), describer=lambda *_a, **_k: "")
    service.request(reason="who", force=True)
    service.request(reason="who", force=True)
    roger = service.continuum.persons()["roger"]
    assert not roger.current_presence and roger.current_location is None
    assert roger.last_location == "camera_view"


def test_scene_language_cannot_supply_identity(tmp_path):
    service = perception(tmp_path, [[10, 10, 100, 200]], [("unknown", .5)], descriptions=["Roger is here"])
    result = service.request(reason="see_command", force=True, include_scene=True)
    assert result.scene_description == "a person is here"
    assert result.identities_resolved == []
    assert not service.continuum.persons()["roger"].current_presence


def test_german_see_uses_english_vlm_then_translation(tmp_path):
    prompts = []
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [], recognizer=lambda **_: None,
        describer=lambda *_a, **kw: (prompts.append(kw["prompt"]) or
                                     "A person is standing next to a table."),
        translator=lambda text: "Eine Person steht neben einem Tisch.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert "Return the description in English" in prompts[0]
    assert result.description_en == "A person is standing next to a table."
    assert result.description_de == "Eine Person steht neben einem Tisch."
    assert result.scene_description == result.description_de
    assert result.vlm_prompt_language == result.vlm_output_language == "en"
    assert result.presentation_language == "de"
    assert (result.vision_model_calls, result.translation_model_calls,
            result.reasoning_model_calls) == (1, 1, 0)
    debug = service.state()
    assert debug["description_en_excerpt"] == result.description_en
    assert debug["description_de_excerpt"] == result.description_de
    assert debug["identity_source"] == "local_recognition"


def test_translation_preserves_uncertainty_and_does_not_invent_identity(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [], recognizer=lambda **_: None,
        describer=lambda *_a, **_kw: "There may be a small object on the table.",
        translator=lambda _text: "Auf dem Tisch könnte sich ein kleiner Gegenstand befinden.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert "könnte" in result.scene_description
    assert "Roger" not in result.scene_description


def test_local_identity_enrichment_happens_after_translation(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [[10, 10, 100, 200]],
        recognizer=lambda **_: SimpleNamespace(identity="roger", confidence=.96,
            face_path="face.jpg", top_label="roger", reason="test"),
        describer=lambda *_a, **_kw: "A person is sitting at a desk.",
        translator=lambda _text: "Eine Person sitzt an einem Schreibtisch.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert result.description_de == "Eine Person sitzt an einem Schreibtisch."
    assert result.scene_description == "Roger sitzt an einem Schreibtisch."
    assert result.identity_source == "local_recognition"


def test_man_wording_binds_single_fresh_local_identity(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [[10, 10, 100, 200]],
        recognizer=lambda **_: SimpleNamespace(identity="roger", confidence=.917,
            face_path="face.jpg", top_label="roger", reason="test"),
        describer=lambda *_a, **_kw: "A man is standing next to a sofa.",
        translator=lambda _text: "Ein Mann steht neben einem Sofa.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert result.scene_description == "Roger steht neben einem Sofa."
    assert result.identity_enriched is True
    person = result.last_scene_observation["persons"][0]
    assert person["person_entity_id"] == "person:roger"
    assert person["display_name"] == "Roger"
    assert person["binding_status"] == "confirmed_local"
    assert result.identity_binding_reason == "single_fresh_identity_action_bound"


def test_pronoun_sensitive_composition_uses_conservative_two_part_form(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [[10, 10, 100, 200]],
        recognizer=lambda **_: SimpleNamespace(identity="roger", confidence=.946,
            face_path="face.jpg", top_label="roger", reason="test"),
        describer=lambda *_a, **_kw: (
            "man holding phone with screen facing towards him while sitting next to couch."),
        translator=lambda _text: (
            "eine Person hält ein Telefon mit dem Bildschirm zum Gesicht gerichtet, "
            "während sie neben dem Sofa sitzt."))
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert result.scene_description.startswith("Roger ist sichtbar. Die Person hält")
    assert "Roger hält" not in result.scene_description
    assert "während sie" not in result.scene_description
    assert "während die Person" in result.scene_description
    assert result.identity_binding_reason == "single_fresh_identity_pronoun_safe"
    assert {"phone", "sofa"} <= set(result.objects)
    assert any(x["subject_id"] == "person:roger" and x["predicate"] == "HOLDING"
               and x["object_id"] == "phone" for x in result.relations)


def test_low_confidence_identity_is_not_enriched(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [[10, 10, 100, 200]],
        recognizer=lambda **_: SimpleNamespace(identity="roger", confidence=.5,
            face_path="face.jpg", top_label="roger", reason="below threshold"),
        describer=lambda *_a, **_kw: "A person is standing near a sofa.",
        translator=lambda _text: "Eine Person steht neben einem Sofa.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert result.identity_enriched is False
    assert result.persons[0]["person_entity_id"] is None
    assert result.scene_description.startswith("Eine Person")


def test_stale_track_identity_is_not_reused_for_new_see_frame(tmp_path):
    decisions = iter([("roger", .96), ("unknown", .4)])
    def recognize(**_):
        identity, confidence = next(decisions)
        return SimpleNamespace(identity=identity, confidence=confidence, face_path="face.jpg",
                               top_label=identity, reason="test")
    service = CameraPerceptionService(configuration(tmp_path, freshness=0), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [[10, 10, 100, 200]],
        recognizer=recognize, describer=lambda *_a, **_kw: "A person is near a sofa.",
        translator=lambda _text: "Eine Person steht neben einem Sofa.")
    first = service.request(reason="see_command", force=True, include_scene=True, scene_language="de")
    second = service.request(reason="see_command", force=True, include_scene=True, scene_language="de")
    assert first.identity_enriched is True
    assert second.identity_enriched is False
    assert second.scene_description.startswith("Eine Person")
    assert second.persons[0]["binding_status"] == "unbound"


def test_recent_spatial_unknown_track_hands_over_to_new_canonical_track(tmp_path):
    detections = iter([[[10, 10, 100, 200]], [[75, 10, 100, 200]]])
    decisions = iter([("unknown", .919), ("roger", .931)])
    def recognize(**_):
        identity, confidence = next(decisions)
        return SimpleNamespace(identity=identity, confidence=confidence, face_path="face.jpg",
                               top_label=identity, reason="test")
    service = CameraPerceptionService(configuration(tmp_path, freshness=5), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: next(detections),
        recognizer=recognize, describer=lambda *_a, **_kw: "A man is sitting near a sofa.",
        translator=lambda _text: "Ein Mann sitzt neben einem Sofa.")
    first = service.request(reason="idcheck", force=True, include_scene=False)
    second = service.request(reason="see_command", force=True, include_scene=True, scene_language="de")
    assert first.identities_resolved == []
    assert second.identities_resolved == ["roger"]
    assert second.track_handover_detected is True
    assert second.temporal_identity_bridge_used is False
    assert second.track_handover_from == first.tracks_active[0]
    assert second.track_handover_to == second.tracks_active[0]
    assert second.current_scene_unknown_count == 0
    assert second.historical_unknown_count >= 1
    assert second.scene_person_count == second.canonical_person_count == 1
    assert second.identity_enriched is True
    assert second.scene_description.startswith("Roger ")


def test_incompatible_boxes_prevent_track_handover(tmp_path):
    detections = iter([[[10, 10, 80, 180]], [[350, 10, 80, 180]]])
    decisions = iter([("unknown", .92), ("roger", .94)])
    def recognize(**_):
        identity, confidence = next(decisions)
        return SimpleNamespace(identity=identity, confidence=confidence, face_path="face.jpg",
                               top_label=identity, reason="test")
    service = CameraPerceptionService(configuration(tmp_path, freshness=5), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: next(detections), recognizer=recognize,
        describer=lambda *_a, **_kw: "A man is visible.",
        translator=lambda _text: "Ein Mann ist sichtbar.")
    service.request(reason="idcheck", force=True, include_scene=False)
    result = service.request(reason="see_command", force=True, include_scene=True, scene_language="de")
    assert result.track_handover_detected is False
    assert result.track_handover_from is None


def test_two_current_people_are_not_collapsed_by_handover(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path),
        detector=lambda _: [[10, 10, 100, 200], [300, 10, 100, 200]],
        recognizer=lambda **_: SimpleNamespace(identity="unknown", confidence=.4,
            face_path=None, top_label="unknown", reason="test"),
        describer=lambda *_a, **_kw: "Two people are visible.",
        translator=lambda _text: "Zwei Personen sind sichtbar.")
    result = service.request(reason="see_command", force=True, include_scene=True, scene_language="de")
    assert result.track_handover_detected is False
    assert result.scene_person_count == 2
    assert result.current_scene_unknown_count == 2


def test_scene_observation_binds_person_objects_and_explicit_relations(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [[10, 10, 100, 200]],
        recognizer=lambda **_: SimpleNamespace(identity="roger", confidence=.96,
            face_path="face.jpg", top_label="roger", reason="test"),
        describer=lambda *_a, **_kw: (
            "A man is standing near a sofa. A coffee cup is on the table."),
        translator=lambda _text: (
            "Ein Mann steht neben einem Sofa. Auf dem Tisch steht eine Kaffeetasse."))
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert set(result.objects) == {"sofa", "coffee_cup", "table"}
    assert {("person:roger", "NEAR", "sofa"), ("coffee_cup", "ON", "table")} == {
        (item["subject_id"], item["predicate"], item["object_id"]) for item in result.relations}
    stored = service.continuum._graph()["last_observation"]
    assert stored["objects"] == result.objects
    assert stored["relations"] == result.relations


def test_real_scene_objects_and_relations_are_normalized(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [[10, 10, 100, 200]],
        recognizer=lambda **_: SimpleNamespace(identity="roger", confidence=.94,
            face_path="face.jpg", top_label="roger", reason="test"),
        describer=lambda *_a, **_kw: (
            "man sitting on couch with coffee mug on table and papers on floor."),
        translator=lambda _text: (
            "Ein Mann sitzt auf dem Sofa mit einer Kaffeetasse auf dem Tisch "
            "und Papieren auf dem Boden."))
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert result.scene_description.startswith("Roger sitzt auf dem Sofa")
    assert set(result.objects) == {"sofa", "coffee_cup", "papers", "table", "floor"}
    triples = {(x["subject_id"], x["predicate"], x["object_id"]) for x in result.relations}
    assert ("person:roger", "SITTING_ON", "sofa") in triples
    assert ("coffee_cup", "ON", "table") in triples
    assert ("papers", "ON", "floor") in triples


def test_visible_cup_without_relation_does_not_invent_on_table(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [], recognizer=lambda **_: None,
        describer=lambda *_a, **_kw: "A cup is visible.",
        translator=lambda _text: "Eine Tasse ist sichtbar.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert result.objects == ["coffee_cup"]
    assert result.relations == []


def test_vlm_person_without_local_detection_materializes_anonymous_scene_person(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [], recognizer=lambda **_: None,
        describer=lambda *_a, **_kw: "man sitting on couch.",
        translator=lambda _text: "Ein Mann sitzt auf dem Sofa.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert result.tracks_active == []
    assert result.identities_resolved == []
    assert result.scene_person_count == 1
    assert result.canonical_person_count == 0
    assert len(result.persons) == 1
    anonymous = result.persons[0]
    assert anonymous["scene_person_id"].startswith(f"scene_person:{result.frame_id}:")
    assert anonymous["person_entity_id"] is None
    assert anonymous["binding_status"] == "anonymous_visual"
    assert anonymous["identity_source"] == "none"
    relation = next(x for x in result.relations if x["predicate"] == "SITTING_ON")
    assert relation["subject_id"] == anonymous["scene_person_id"]
    assert anonymous["scene_person_id"] not in service.continuum.persons()


def test_safe_canonical_binding_has_no_parallel_anonymous_person(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [[10, 10, 100, 200]],
        recognizer=lambda **_: SimpleNamespace(identity="roger", confidence=.95,
            face_path="face.jpg", top_label="roger", reason="test"),
        describer=lambda *_a, **_kw: "man sitting on couch.",
        translator=lambda _text: "Ein Mann sitzt auf dem Sofa.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert result.scene_person_count == result.canonical_person_count == 1
    assert len(result.persons) == 1
    assert result.persons[0]["person_entity_id"] == "person:roger"
    assert not any(person.get("binding_status") == "anonymous_visual" for person in result.persons)
    assert all(x["subject_id"] != f"scene_person:{result.frame_id}:1" for x in result.relations)


def test_ambiguous_drink_container_keeps_type_uncertainty_and_on_relation(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [], recognizer=lambda **_: None,
        describer=lambda *_a, **_kw: "A cup/glass/bottle is on the table.",
        translator=lambda _text: "Ein Trinkgefäß steht auf dem Tisch.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert "drink_container" in result.objects
    assert "coffee_cup" not in result.objects
    detail = next(x for x in result.object_details if x["object_id"] == "drink_container")
    assert detail["uncertain"] is True
    assert detail["raw_label"] == "cup/glass/bottle"
    assert detail["candidate_labels"] == ["coffee_cup", "glass", "bottle"]
    assert any(x["subject_id"] == "drink_container" and x["predicate"] == "ON"
               and x["object_id"] == "table" for x in result.relations)


def test_definite_coffee_mug_remains_coffee_cup(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [], recognizer=lambda **_: None,
        describer=lambda *_a, **_kw: "A coffee mug is on the table.",
        translator=lambda _text: "Eine Kaffeetasse steht auf dem Tisch.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert {"coffee_cup", "table"} <= set(result.objects)
    detail = next(x for x in result.object_details if x["object_id"] == "coffee_cup")
    assert detail["uncertain"] is False


def test_multiple_people_do_not_receive_ambiguous_scene_assignment(tmp_path):
    decisions = iter([("roger", .96), ("unknown", .4)])
    def recognize(**_):
        identity, confidence = next(decisions)
        return SimpleNamespace(identity=identity, confidence=confidence, face_path="face.jpg",
                               top_label=identity, reason="test")
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path),
        detector=lambda _: [[10, 10, 100, 200], [300, 10, 100, 200]], recognizer=recognize,
        describer=lambda *_a, **_kw: "A person is standing near a sofa. Another person is visible.",
        translator=lambda _text: "Eine Person steht neben einem Sofa. Eine weitere Person ist sichtbar.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert result.identity_enriched is False
    assert "Roger steht" not in result.scene_description
    assert not any(item["predicate"] == "NEAR" for item in result.relations)


def test_duplicate_tracks_for_same_identity_collapse_in_scene_only():
    people = [{"track_id":"camera_primary:21", "person_entity_id":"person:roger",
               "person_id":"roger"},
              {"track_id":"camera_primary:22", "person_entity_id":"person:roger",
               "person_id":"roger"}]
    result = _deduplicate_scene_persons(people)
    assert len(result) == 1
    assert result[0]["person_entity_id"] == "person:roger"


def test_repeated_scene_reuses_canonical_person_entity(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path, freshness=0), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [[10, 10, 100, 200]],
        recognizer=lambda **_: SimpleNamespace(identity="roger", confidence=.96,
            face_path="face.jpg", top_label="roger", reason="test"),
        describer=lambda *_a, **_kw: "A person is visible.",
        translator=lambda _text: "Eine Person ist sichtbar.")
    service.request(reason="see_command", force=True, include_scene=True, scene_language="de")
    service.request(reason="see_command", force=True, include_scene=True, scene_language="de")
    assert [person_id for person_id in service.continuum.persons() if person_id == "roger"] == ["roger"]
    assert len([relation for relation in service.continuum.relations()
                if relation.predicate == "identified_as" and relation.object_id == "person:roger"]) == 1


def test_single_see_does_not_create_long_term_memory_candidate(tmp_path):
    memory = SQLiteStore(tmp_path / "memory.db")
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [], recognizer=lambda **_: None,
        describer=lambda *_a, **_kw: "A cup is visible.",
        translator=lambda _text: "Eine Tasse ist sichtbar.")
    service.request(reason="see_command", force=True, include_scene=True, scene_language="de")
    assert memory.list_memory_items(status="candidate") == []


def test_unknown_person_remains_generic_after_translation(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [[10, 10, 100, 200]],
        recognizer=lambda **_: SimpleNamespace(identity="unknown", confidence=.4,
            face_path=None, top_label="unknown", reason="test"),
        describer=lambda *_a, **_kw: "A person is sitting at a desk.",
        translator=lambda _text: "Eine Person sitzt an einem Schreibtisch.")
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="de")
    assert result.scene_description.startswith("Eine Person")
    assert "Roger" not in result.scene_description


def test_english_see_skips_translation(tmp_path):
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [], recognizer=lambda **_: None,
        describer=lambda *_a, **_kw: "A cup is on the desk.",
        translator=lambda _text: pytest.fail("English presentation must not translate"))
    result = service.request(reason="see_command", force=True, include_scene=True,
                             scene_language="en")
    assert result.scene_description == "A cup is on the desk."
    assert (result.vision_model_calls, result.translation_model_calls,
            result.reasoning_model_calls) == (1, 0, 0)


def test_vision_and_translation_resource_leases_are_sequential(tmp_path):
    order = []
    @contextmanager
    def vision_lease():
        order.append("vision-enter")
        try: yield
        finally: order.append("vision-exit")
    @contextmanager
    def translation_lease():
        order.append("translation-enter")
        try: yield
        finally: order.append("translation-exit")
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: frame(tmp_path), detector=lambda _: [], recognizer=lambda **_: None,
        vision_lease=vision_lease, translation_lease=translation_lease,
        describer=lambda *_a, **_kw: (order.append("vlm") or "An empty room."),
        translator=lambda _text: (order.append("translate") or "Ein leerer Raum."))
    service.request(reason="see_command", force=True, include_scene=True, scene_language="de")
    assert order == ["vision-enter", "vlm", "vision-exit",
                     "translation-enter", "translate", "translation-exit"]


def test_perception_debug_state_is_structured_and_embedding_free(tmp_path):
    service = perception(tmp_path, [[10, 10, 100, 200]], [("roger", .96)])
    service.request(reason="idcheck", force=True)
    state = service.state()
    assert state["persons_detected"] == 1
    assert len(state["tracks_active"]) == 1
    assert state["identities_resolved"] == ["roger"]
    assert state["active_track_details"][0]["face_detected"] is True
    assert state["active_track_details"][0]["recognition_attempted"] is True
    assert state["active_track_details"][0]["recognition_candidate"] == "roger"
    assert state["active_track_details"][0]["recognition_confidence"] == .96
    assert "embedding" not in str(state).casefold()


@pytest.mark.anyio
async def test_who_command_refreshes_without_idcheck_and_uses_no_llm(monkeypatch):
    from avacore.channels.telegram import bot
    replies = []
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=42),
        effective_message=SimpleNamespace(reply_text=lambda text: _reply(replies, text)))
    class Response:
        ok = True; text = ""
        def __init__(self, data): self.data = data
        def json(self): return self.data
    calls = {"perception":0}
    async def post(url, **_): calls["perception"] += 1; return Response({})
    async def get(url, **_): return Response({"items":[{"display_name":"Roger", "known":True,
        "current_presence":True, "confidence":.96}]})
    monkeypatch.setattr(bot.http_client, "post", post); monkeypatch.setattr(bot.http_client, "get", get)
    await bot.who_cmd(update, SimpleNamespace())
    assert calls["perception"] == 1
    assert replies == ["Roger is currently present."]


@pytest.mark.anyio
async def test_see_uses_one_perception_call_and_api_managed_translation(monkeypatch, tmp_path):
    from avacore.channels.telegram import bot
    replies = []
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=42),
        effective_message=SimpleNamespace(
            reply_text=lambda text: _reply(replies, text),
            reply_photo=lambda **kwargs: _reply(replies, kwargs.get("caption", "")),
        ),
    )

    class Response:
        ok = True
        text = ""
        def json(self):
            image_path = tmp_path / "missing-camera-frame.jpg"
            return {"scene_description":"A person is by the sofa", "persons":[],
                    "identities_resolved":[], "image_path":str(image_path)}

    urls = []
    async def post(url, **kwargs):
        urls.append((url, kwargs.get("json") or {}))
        return Response()

    monkeypatch.setattr(bot.http_client, "post", post)
    await bot.active_camera_cmd(update, SimpleNamespace(chat_data={"reply_language":"de"}))
    assert len(urls) == 1
    assert urls[0][0].endswith("/perception/camera")
    assert urls[0][1]["scene_language"] == "de"
    assert not any(url.endswith("/reply") for url, _ in urls)


@pytest.mark.anyio
async def test_see_caption_uses_current_scene_unknown_count_only(monkeypatch, tmp_path):
    from avacore.channels.telegram import bot
    replies = []
    update = SimpleNamespace(effective_chat=SimpleNamespace(id=42),
        effective_message=SimpleNamespace(reply_text=lambda text: _reply(replies, text),
            reply_photo=lambda **kwargs: _reply(replies, kwargs.get("caption", ""))))
    class Response:
        ok = True; text = ""
        def json(self):
            return {"scene_description":"Roger sitzt neben einem Sofa.",
                "persons":[{"person_id":"roger"}, {"person_id":None}],
                "current_scene_unknown_count":0, "historical_unknown_count":1,
                "image_path":str(tmp_path / "missing.jpg")}
    async def post(*_args, **_kwargs): return Response()
    monkeypatch.setattr(bot.http_client, "post", post)
    await bot.active_camera_cmd(update, SimpleNamespace(chat_data={"reply_language":"de"}))
    assert "Unbekannte Personen" not in replies[0]


async def _reply(items, text):
    items.append(text)


def test_legacy_singleton_is_retired_and_never_active(tmp_path):
    service = perception(tmp_path, [], [])
    graph = service.continuum._graph()
    graph["tracks"] = {"camera_primary":{"person_id":"roger", "present":True, "confidence":.96}}
    graph["relations"] = [{"subject_id":"track:camera_primary", "predicate":"identified_as",
        "object_id":"person:roger", "confidence":.96, "source":"legacy",
        "first_observed":"2026-01-01T00:00:00+00:00", "last_observed":"2026-01-01T00:00:00+00:00", "metadata":{}}]
    service.continuum._write(service.continuum.persons_path, graph)
    state = service.state()
    stored = service.continuum._graph()
    assert state["tracks_active"] == [] and state["identities_resolved"] == []
    assert stored["tracks"]["camera_primary"]["legacy"] is True
    assert not any(x["subject_id"] == "track:camera_primary" for x in stored["relations"])
    assert stored["legacy_relations"][0]["object_id"] == "person:roger"


def test_recognition_failure_is_visible_in_track_diagnostics(tmp_path):
    source = frame(tmp_path)
    service = CameraPerceptionService(configuration(tmp_path), continuum(tmp_path),
        capture=lambda **_: source, detector=lambda _: [[10, 10, 100, 200]],
        recognizer=lambda **_: (_ for _ in ()).throw(RuntimeError("face model unavailable")),
        describer=lambda *_a, **_k: "")
    service.request(reason="who", force=True)
    detail = service.state()["active_track_details"][0]
    assert detail["recognition_attempted"] is True
    assert "face model unavailable" in detail["recognition_reason"]
    assert detail["identity_resolved"] is None


def test_canonical_duplicate_detection_preserves_tracks_but_not_extra_scene_person(tmp_path):
    service = perception(tmp_path, [[10, 10, 100, 200], [15, 10, 100, 200]],
                         [("roger", .96), ("roger", .95)],
                         descriptions=["A person is sitting on a sofa."])
    service.translator = lambda _: "Eine Person sitzt auf einem Sofa."
    result = service.request(reason="see_command", force=True, include_scene=True, scene_language="de")
    assert result.scene_person_count == result.canonical_person_count == 1
    assert result.current_scene_unknown_count == 0
    assert len(result.tracks_active) == result.persons_detected == 2
    graph = service.continuum._graph()
    assert all(graph["tracks"][key]["person_id"] == "roger" for key in result.tracks_active)
    assert len(graph["last_observation"]["persons"]) == 1
    assert service.state()["persons_detected"] == 2
    assert len(result.relations) == 1
    assert result.relations[0]["subject_id"] == "person:roger"
    assert (result.vision_model_calls, result.translation_model_calls, result.reasoning_model_calls) == (1, 1, 0)
    assert "Unbekannte Personen" not in result.scene_description


def test_true_current_second_person_survives_single_person_vlm_wording(tmp_path):
    service = perception(tmp_path, [[10, 10, 100, 200], [300, 10, 100, 200]],
                         [("roger", .96), ("unknown", .5)], descriptions=["A person is visible."])
    result = service.request(reason="see_command", force=True, include_scene=True)
    assert (result.scene_person_count, result.canonical_person_count, result.current_scene_unknown_count) == (2, 1, 1)
    assert not result.track_handover_detected


@pytest.mark.parametrize("age,expected", [(1, True), (30, False)])
def test_handover_uses_bounded_window_and_preserves_unknown_history(tmp_path, monkeypatch, age, expected):
    from datetime import datetime, timezone
    now = datetime(2026, 10, 4, tzinfo=timezone.utc).timestamp()
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.fromtimestamp(now, tz)
    monkeypatch.setattr("avacore.vision.perception.datetime", Clock)
    service = perception(tmp_path, [[10, 10, 100, 200]], [("unknown", .92), ("roger", .94)])
    first = service.request(reason="see_command", force=True, include_scene=True)
    now += age
    service.detector = lambda _: [[75, 10, 100, 200]]
    result = service.request(reason="see_command", force=True, include_scene=True)
    assert result.track_handover_detected is expected
    assert result.track_handover_identity == ("person:roger" if expected else None)
    assert result.current_scene_unknown_count == 0
    assert result.historical_unknown_count == 1
    old = service.continuum._graph()["tracks"][first.tracks_active[0]]
    assert not old["present"]
    assert old["person_id"].startswith("unknown_person:")
    state = service.state()
    for key in ("track_handover_detected", "track_handover_from", "track_handover_to",
                "track_handover_identity", "track_handover_reason", "current_scene_unknown_count",
                "historical_unknown_count", "scene_person_count", "canonical_person_count"):
        assert state[key] == getattr(result, key)


def test_disappeared_canonical_track_is_historical_even_if_person_remains(tmp_path):
    service = perception(tmp_path, [[10, 10, 100, 200]], [("roger", .96), ("roger", .96)])
    first = service.request(reason="see_command", force=True, include_scene=True)
    service.detector = lambda _: [[350, 10, 100, 200]]
    second = service.request(reason="see_command", force=True, include_scene=True)
    assert service.state()["tracks_active"] == second.tracks_active
    assert not service.continuum._graph()["tracks"][first.tracks_active[0]]["present"]
    assert second.scene_person_count == 1


def test_anonymous_relation_is_replaced_by_current_canonical_relation(tmp_path):
    service = perception(tmp_path, [], [("roger", .96)],
                         descriptions=["A person is sitting on a sofa."] * 2)
    first = service.request(reason="see_command", force=True, include_scene=True)
    assert first.relations[0]["subject_id"].startswith("scene_person:")
    service.detector = lambda _: [[10, 10, 100, 200]]
    second = service.request(reason="see_command", force=True, include_scene=True)
    assert len(second.relations) == 1
    assert second.relations[0]["subject_id"] == "person:roger"
    assert service.state()["last_scene_observation"]["relations"] == second.relations
    assert second.current_scene_unknown_count == 0


@pytest.mark.parametrize("reason", ["idcheck", "who_command"])
def test_non_scene_duplicate_identity_policy_and_model_calls_unchanged(tmp_path, reason):
    service = perception(tmp_path, [[10, 10, 100, 200], [15, 10, 100, 200]],
                         [("roger", .96), ("roger", .95)])
    result = service.request(reason=reason, force=True)
    assert [x["person_id"] for x in result.persons] == ["roger", None]
    assert (result.vision_model_calls, result.translation_model_calls, result.reasoning_model_calls) == (0, 0, 0)


def test_recently_disappeared_unknown_can_handover_after_empty_frame(tmp_path):
    service = perception(tmp_path, [[10, 10, 100, 200]], [("unknown", .92), ("roger", .94)])
    first = service.request(reason="see_command", force=True, include_scene=True)
    service.detector = lambda _: []
    service.request(reason="see_command", force=True, include_scene=True)
    service.detector = lambda _: [[75, 10, 100, 200]]
    result = service.request(reason="see_command", force=True, include_scene=True)
    assert result.track_handover_detected
    assert result.track_handover_from == first.tracks_active[0]


def test_same_center_with_incompatible_box_scale_is_not_continuity():
    from avacore.vision.perception import _track_continuity_score
    assert _track_continuity_score([0, 0, 100, 200], [40, 80, 20, 40]) == 0


def test_handover_requires_configured_identity_threshold(tmp_path):
    service = perception(tmp_path, [[10, 10, 100, 200]], [("unknown", .92), ("roger", .79)])
    service.request(reason="see_command", force=True, include_scene=True)
    service.detector = lambda _: [[75, 10, 100, 200]]
    result = service.request(reason="see_command", force=True, include_scene=True)
    assert not result.track_handover_detected
    assert result.canonical_person_count == 0
    assert result.current_scene_unknown_count == 1


def bridge_sequence(tmp_path, monkeypatch, *, age=16, description=None, language="de"):
    import avacore.vision.perception as module
    from datetime import datetime, timedelta, timezone
    clock = datetime(2026, 10, 4, 13, 4, 59, tzinfo=timezone.utc)
    monkeypatch.setattr(module, "_utc_now", lambda: clock.isoformat())
    service = perception(tmp_path, [[10, 10, 100, 200]], [("roger", .95)],
        descriptions=[description or "man holding phone next to couch with clutter on floor."])
    service.translator = lambda _: "Ein Mann hält das Telefon neben dem Sofa mit Unordnung auf dem Boden."
    first = service.request(reason="idcheck", force=True)
    original_person = service.continuum.persons()["roger"]
    memory_flags = []
    assimilate = service.continuum.assimilate
    def record_assimilation(*args, **kwargs):
        memory_flags.append(kwargs.get("memory", False))
        return assimilate(*args, **kwargs)
    monkeypatch.setattr(service.continuum, "assimilate", record_assimilation)
    service.bridge_memory_flags = memory_flags
    clock += timedelta(seconds=age)
    service.detector = lambda _: []
    result = service.request(reason="see_command", force=True, include_scene=True, scene_language=language)
    return service, first, result, original_person


def test_temporal_bridge_exact_real_sequence(tmp_path, monkeypatch):
    service, first, result, original = bridge_sequence(tmp_path, monkeypatch)
    assert first.captured_at == "2026-10-04T13:04:59+00:00"
    assert result.captured_at == "2026-10-04T13:05:15+00:00"
    assert result.temporal_identity_bridge_used
    assert result.temporal_identity_bridge_person == "person:roger"
    assert result.temporal_identity_bridge_age_seconds == 16
    assert result.temporal_identity_bridge_confidence == .95
    assert result.persons_detected == 0 and result.tracks_active == []
    assert result.identities_resolved == []
    assert (result.scene_person_count, result.canonical_person_count, result.current_scene_unknown_count) == (1, 1, 0)
    assert result.persons[0]["binding_status"] == "temporal_identity_bridge"
    assert result.identity_source == "local_recognition_history" and result.identity_enriched
    assert result.final_description.startswith("Roger hält das Telefon")
    assert (result.vision_model_calls, result.translation_model_calls, result.reasoning_model_calls) == (1, 1, 0)
    assert {(r["subject_id"], r["predicate"], r["object_id"]) for r in result.relations} >= {
        ("person:roger", "NEAR", "sofa"), ("person:roger", "HOLDING", "phone")}
    assert len(result.relations) == len({(r["subject_id"], r["predicate"], r["object_id"]) for r in result.relations})
    assert service.continuum.persons()["roger"] == original
    assert not any(service.bridge_memory_flags)
    for key in ("used", "person", "age_seconds", "confidence", "reason"):
        name = "temporal_identity_bridge_" + key
        assert service.state()[name] == getattr(result, name)


@pytest.mark.parametrize("age", [21, -1])
def test_temporal_bridge_expired_or_future(tmp_path, monkeypatch, age):
    _, _, result, _ = bridge_sequence(tmp_path, monkeypatch, age=age)
    assert not result.temporal_identity_bridge_used
    assert result.persons[0]["binding_status"] == "anonymous_visual"
    assert result.canonical_person_count == 0


@pytest.mark.parametrize("description", ["two men holding phones", "a man and a woman next to couch",
    "a person next to two people", "a man with another child"])
def test_temporal_bridge_multiple_visual_people(tmp_path, monkeypatch, description):
    _, _, result, _ = bridge_sequence(tmp_path, monkeypatch, description=description)
    assert not result.temporal_identity_bridge_used
    assert result.canonical_person_count == 0


def test_temporal_bridge_candidate_guards():
    from avacore.vision.perception import _temporal_identity_candidate
    tracks = {"a": {"person_id":"roger", "sensor_id":"camera_primary",
                     "last_seen":"2026-10-04T13:05:05Z", "confidence":.95}}
    def candidate():
        return _temporal_identity_candidate("a man holding phone", tracks,
            camera_id="camera_primary", captured_at="2026-10-04T13:05:15Z",
            window=20, threshold=.8, known_persons={"roger":"Roger", "alice":"Alice"})
    assert candidate()[0]["person_id"] == "roger"
    tracks["b"] = {**tracks["a"], "person_id":"alice", "last_seen":"2026-10-04T13:05:10Z"}
    assert candidate() == (None, "competing_recent_canonical_identities")
    del tracks["b"]
    tracks["a"]["sensor_id"] = "other_camera"
    assert candidate()[0] is None
    tracks["a"]["sensor_id"] = "camera_primary"
    tracks["a"]["confidence"] = .79
    assert candidate()[0] is None


def test_temporal_bridge_capture_time_and_no_renewal(tmp_path, monkeypatch):
    import avacore.vision.perception as module
    service, _, result, original = bridge_sequence(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_utc_now", lambda: "2026-10-04T13:05:20Z")
    def describe(*args, **kwargs):
        monkeypatch.setattr(module, "_utc_now", lambda: "2026-10-04T13:06:20Z")
        return "a man holding phone"
    service.describer = describe
    expired = service.request(reason="see_command", force=True, include_scene=True)
    assert not expired.temporal_identity_bridge_used
    assert service.continuum.persons()["roger"].last_seen == original.last_seen
    monkeypatch.setattr(module, "_utc_now", lambda: "2026-10-04T13:05:15Z")
    delayed = service.request(reason="see_command", force=True, include_scene=True)
    assert delayed.temporal_identity_bridge_used
    assert delayed.temporal_identity_bridge_age_seconds == 16


def test_current_local_recognition_does_not_use_temporal_bridge(tmp_path):
    service = perception(tmp_path, [[10, 10, 100, 200]], [("roger", .95)],
                         descriptions=["a man holding phone"])
    result = service.request(reason="see_command", force=True, include_scene=True)
    assert result.persons[0]["binding_status"] == "confirmed_local"
    assert not result.temporal_identity_bridge_used


def test_temporal_bridge_competing_identities_on_same_track(tmp_path, monkeypatch):
    import avacore.vision.perception as module
    clock = "2026-10-04T13:05:05Z"
    monkeypatch.setattr(module, "_utc_now", lambda: clock)
    service = perception(tmp_path, [[10, 10, 100, 200]], [("roger", .95), ("alice", .95)],
                         descriptions=["a man holding phone"])
    service.settings.known_persons["alice"] = "Alice"
    # Register Alice in the existing canonical registry through its normal setup.
    service.continuum = ContinuumService(tmp_path / "continuum.json", tmp_path / "workspace.json",
        tmp_path / "working.json", tmp_path / "history.json", tmp_path / "persons.json",
        known_persons={"roger":"Roger", "alice":"Alice"}, confidence_threshold=.8)
    service.request(reason="idcheck", force=True)
    clock = "2026-10-04T13:05:10Z"
    service.request(reason="idcheck", force=True)
    clock = "2026-10-04T13:05:15Z"
    service.detector = lambda _: []
    result = service.request(reason="see_command", force=True, include_scene=True)
    assert not result.temporal_identity_bridge_used
    assert result.temporal_identity_bridge_reason == "competing_recent_canonical_identities"
    assert result.canonical_person_count == 0


@pytest.mark.parametrize("description", ["water bottle", "bottle", "Flasche", "Wasserflasche"])
def test_bottle_normalization(description):
    from avacore.vision.perception import _extract_scene_structure
    objects, relations, details = _extract_scene_structure(description, [], frame_id="test", camera_id="camera_primary")
    assert objects == ["bottle"]
    assert not relations
    assert details[0]["uncertain"] is False


@pytest.mark.parametrize("description", ["water bottle on table", "bottle on table",
    "A water bottle is on the table.", "Flasche auf dem Tisch", "Wasserflasche auf dem Tisch",
    "Roger sitzt neben dem Sofa mit einer Wasserflasche auf dem Tisch."])
def test_bottle_explicit_on_table(description):
    from avacore.vision.perception import _extract_scene_structure
    objects, relations, _ = _extract_scene_structure(description, [], frame_id="test", camera_id="camera_primary")
    assert "bottle" in objects and "table" in objects
    assert [(r["subject_id"], r["predicate"], r["object_id"]) for r in relations] == [("bottle", "ON", "table")]


@pytest.mark.parametrize("description", ["a bottle is visible", "a bottle is visible near a table",
    "a bottle is not on the table", "Eine Wasserflasche ist neben dem Tisch sichtbar."])
def test_bottle_does_not_infer_on_table(description):
    from avacore.vision.perception import _extract_scene_structure
    objects, relations, _ = _extract_scene_structure(description, [], frame_id="test", camera_id="camera_primary")
    assert "bottle" in objects
    assert not relations


def test_bottle_cup_glass_uncertainty_is_preserved():
    from avacore.vision.perception import _extract_scene_structure
    objects, relations, details = _extract_scene_structure("bottle/cup/glass on table", [],
        frame_id="test", camera_id="camera_primary")
    assert objects == ["drink_container", "table"]
    assert details[0]["uncertain"] is True
    assert details[0]["candidate_labels"] == ["bottle", "coffee_cup", "glass"]
    assert [(r["subject_id"], r["predicate"], r["object_id"]) for r in relations] == [("drink_container", "ON", "table")]
