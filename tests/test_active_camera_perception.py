from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from avacore.core.continuum import ContinuumService
from avacore.vision.perception import CameraPerceptionService


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
