from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import logging
import re
import uuid
from typing import Any, Callable

import cv2
from PIL import Image

from avacore.core.continuum import ContinuumService, VisualObservation
from avacore.tools.camera_rtsp import build_rtsp_url, capture_rtsp_snapshot, crop_camera_overlay
from avacore.tools.identity_rag import recognize_face_image
from avacore.vision.describe import camera_scene_prompt, describe_image_with_smolvlm


_logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _age_seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        return max(0.0, (datetime.now(timezone.utc) - datetime.fromisoformat(timestamp)).total_seconds())
    except (TypeError, ValueError):
        return None


def _iou(first: list[int], second: list[int]) -> float:
    ax, ay, aw, ah = first; bx, by, bw, bh = second
    left, top, right, bottom = max(ax, bx), max(ay, by), min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _track_continuity_score(first: list[int], second: list[int]) -> float:
    """Bounded spatial continuity from overlap or normalized center proximity."""
    overlap = _iou(first, second)
    ax, ay, aw, ah = first; bx, by, bw, bh = second
    if min(aw, ah, bw, bh) <= 0 or min(aw * ah, bw * bh) / max(aw * ah, bw * bh) < .5:
        return 0.0
    scale_x, scale_y = max(1.0, float(max(aw, bw))), max(1.0, float(max(ah, bh)))
    dx = abs((ax + aw / 2) - (bx + bw / 2)) / scale_x
    dy = abs((ay + ah / 2) - (by + bh / 2)) / scale_y
    proximity = max(0.0, 1.0 - ((dx * dx + dy * dy) ** .5))
    return round(max(overlap, proximity), 3)


def detect_people(image_path: Path) -> list[list[int]]:
    """Local person detection. Face boxes are fallback person evidence, not identity."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"could not read camera frame: {image_path}")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    detected = [[int(x), int(y), int(w), int(h)] for x, y, w, h in boxes]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    for x, y, w, h in detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(28, 28)):
        # Expand a detected face to an approximate person region. This preserves
        # the required detection-before-recognition separation.
        person = [max(0, int(x - w)), max(0, int(y - h)), int(min(frame.shape[1], w * 3)), int(min(frame.shape[0], h * 5))]
        if not any(_iou(person, existing) > .2 for existing in detected):
            detected.append(person)
    return detected


@dataclass
class PerceptionResult:
    captured_at: str
    perceived_at: str
    image_path: str
    scene_image_path: str
    persons: list[dict[str, Any]]
    tracks_active: list[str]
    identities_resolved: list[str]
    scene_description: str = ""
    scene_description_at: str | None = None
    reused: bool = False
    reason: str = "request"
    description_en: str = ""
    description_de: str = ""
    vlm_prompt_language: str = "en"
    vlm_output_language: str = "en"
    presentation_language: str = "en"
    vision_model_calls: int = 0
    translation_model_calls: int = 0
    reasoning_model_calls: int = 0
    identity_source: str = "local_recognition"
    frame_id: str = ""
    camera_id: str = "camera_primary"
    identity_enriched: bool = False
    identity_binding_reason: str = "no_confirmed_local_identity"
    final_description: str = ""
    scene_person_count: int = 0
    canonical_person_count: int = 0
    objects: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    last_scene_observation: dict[str, Any] = field(default_factory=dict)
    object_details: list[dict[str, Any]] = field(default_factory=list)
    track_handover_detected: bool = False
    track_handover_from: str | None = None
    track_handover_to: str | None = None
    track_handover_identity: str | None = None
    track_handover_reason: str | None = None
    track_continuity_score: float = 0.0
    current_scene_unknown_count: int = 0
    historical_unknown_count: int = 0
    persons_detected: int = 0
    temporal_identity_bridge_used: bool = False
    temporal_identity_bridge_person: str | None = None
    temporal_identity_bridge_age_seconds: float | None = None
    temporal_identity_bridge_confidence: float | None = None
    temporal_identity_bridge_reason: str = "not_needed"


_VISIBLE_OBJECT_PATTERNS = {
    "phone": r"\b(?:phone|smartphone|mobile phone)\b",
    "sofa": r"\b(?:sofa|couch)\b",
    "coffee_cup": r"\b(?:coffee\s+cup|coffee\s+mug|cup|mug)\b",
    "bottle": r"\b(?:water\s+bottle|bottle|wasserflasche|flasche)\b",
    "papers": r"\b(?:paper|papers|documents?)\b",
    "table": r"\b(?:table|desk|tisch)\b",
    "floor": r"\b(?:floor|ground)\b",
}


def _deduplicate_scene_persons(persons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one scene person per canonical identity and distinct anonymous tracks."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for person in persons:
        key = str(person.get("person_entity_id") or f"track:{person.get('track_id')}")
        if key in seen:
            continue
        seen.add(key)
        result.append(person)
    return result


def _compose_single_person_description(description: str, display_name: str, *,
                                       language: str) -> tuple[str, str]:
    """Bind a canonical subject to its action without gendering the identity."""
    text = " ".join((description or "").split()).strip()
    if not text:
        visible = f"{display_name} ist sichtbar." if language == "de" else f"{display_name} is visible."
        return visible, "single_fresh_identity_no_description"
    if language == "de":
        # This composition runs only for one canonical scene person. Reuse its
        # name rather than introducing generic subjects or inferred gender.
        pronoun_sensitive = bool(re.search(
            r"\b(?:er|sie|ihn|ihm|sein(?:e|en|em|er|es)?|ihr(?:e|en|em|er|es)?)\b",
            text, flags=re.IGNORECASE))
        text = re.sub(r"\b(?:mit|in)\s+(?:sein|ihr)(?:er|en|e)\s+Hand\b",
                      "in der Hand", text, flags=re.IGNORECASE)
        genitive_name = display_name + ("'" if display_name[-1:].casefold() in "sxzß" else "s")
        text = re.sub(r"\b(?:sein|ihr)(?:e|en|em|er|es)?\b(?=\s+[A-ZÄÖÜ])",
                      lambda _: genitive_name, text)
        text = re.sub(r"\b(?:ein|eine|der|die)\s+(?:Person|Frau|Mann)\b",
                      lambda _: display_name, text, flags=re.IGNORECASE)
        text = re.sub(
            r"(^|[.!?]\s+|\b(?:während|wobei|obwohl|als|und|aber)\s+)(?:er|sie)\b",
            lambda m: m.group(1) + display_name, text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:ihn|ihm)\b", lambda _: display_name,
                      text, flags=re.IGNORECASE)
        if display_name not in text:
            return f"{display_name} ist sichtbar. {text}", "single_fresh_identity_conservative"
        reason = ("single_fresh_identity_pronoun_safe" if pronoun_sensitive else
                  "single_fresh_identity_action_bound")
        return text, reason
    match = re.match(r"^(?:(?:a|the)\s+)?(?:person|man|woman|individual)\s+(.+)$",
                     text, flags=re.IGNORECASE)
    if not match:
        return f"{display_name} is visible. {text}", "single_fresh_identity_conservative"
    action = match.group(1)
    if re.search(r"\b(?:he|she|him|her|his|hers)\b", action, flags=re.IGNORECASE):
        action = re.sub(r"\b(while|whereas|although|as)\s+(?:he|she)\b",
                        r"\1 the person", action, flags=re.IGNORECASE)
        return (f"{display_name} is visible. The person {action}",
                "single_fresh_identity_pronoun_safe")
    return f"{display_name} {action}", "single_fresh_identity_action_bound"


def _anonymous_scene_person(description_en: str, *, frame_id: str,
                            camera_id: str) -> dict[str, Any] | None:
    if not re.search(r"\b(?:person|man|woman|individual)\b", description_en, flags=re.IGNORECASE):
        return None
    scene_person_id = f"scene_person:{frame_id}:1"
    return {"scene_person_id":scene_person_id, "track_id":scene_person_id,
            "person_id":None, "person_entity_id":None, "display_name":None,
            "confidence":.5, "recognition_confidence":None,
            "identity_status":"anonymous_visual", "binding_status":"anonymous_visual",
            "identity_source":"none", "frame_id":frame_id, "camera_id":camera_id,
            "sensor_id":camera_id, "require_fresh_identity":True}


def _temporal_identity_candidate(description: str, tracks: dict[str, Any], *,
                                 camera_id: str, captured_at: str, window: float,
                                 threshold: float, known_persons: dict[str, str]
                                 ) -> tuple[dict[str, Any] | None, str]:
    """A short recognition prior, never refreshed by a visual-only binding."""
    if window <= 0:
        return None, "bridge_disabled"
    # Require an explicit singular visual subject, rejecting plural/count evidence.
    subjects = re.findall(r"\b(?:person|man|woman|individual|boy|girl|child|baby)\b",
                          description, flags=re.IGNORECASE)
    if (len(subjects) != 1 or re.search(
            r"\b(?:people|persons|men|women|individuals|boys|girls|children|babies|"
            r"couple|crowd|group|two|three|four|several|multiple|another|second)\b|\b[2-9]\b",
            description, flags=re.IGNORECASE)):
        return None, "visual_person_count_not_exactly_one"
    candidates: dict[str, dict[str, Any]] = {}
    for track in tracks.values():
        person_id = track.get("person_id")
        if person_id not in known_persons or track.get("sensor_id") != camera_id:
            continue
        recognition = track.get("recognition") or {}
        if recognition and (not recognition.get("recognition_attempted") or
                            recognition.get("recognition_candidate") != person_id):
            continue
        try:
            age = (datetime.fromisoformat(captured_at.replace("Z", "+00:00")) -
                   datetime.fromisoformat(track["last_seen"].replace("Z", "+00:00"))).total_seconds()
            confidence = float(recognition.get("recognition_confidence", track.get("confidence", 0)))
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= age <= window and threshold <= confidence <= 1):
            continue
        candidate = {"person_id":person_id, "age":age, "confidence":confidence}
        if person_id not in candidates or age < candidates[person_id]["age"]:
            candidates[person_id] = candidate
    if len(candidates) > 1:
        return None, "competing_recent_canonical_identities"
    if not candidates:
        return None, "no_fresh_secure_same_camera_identity"
    return next(iter(candidates.values())), "single_recent_secure_same_camera_identity"


def _extract_scene_structure(description_en: str, persons: list[dict[str, Any]], *,
                             frame_id: str, camera_id: str
                             ) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract a tiny bounded scene projection only from explicit VLM wording."""
    text = " ".join((description_en or "").casefold().split())
    ambiguous_match = re.search(
        r"\b(?:cup|mug|glass|bottle)(?:\s*(?:/|\bor\b)\s*(?:cup|mug|glass|bottle)){1,3}\b",
        text)
    objects = [name for name, pattern in _VISIBLE_OBJECT_PATTERNS.items()
               if re.search(pattern, text) and not (name in {"coffee_cup", "bottle"} and ambiguous_match)][:12]
    object_details: list[dict[str, Any]] = []
    if ambiguous_match:
        raw_label = ambiguous_match.group(0)
        candidates = []
        for token in re.findall(r"cup|mug|glass|bottle", raw_label):
            label = "coffee_cup" if token in {"cup", "mug"} else token
            if label not in candidates:
                candidates.append(label)
        objects.insert(0, "drink_container")
        object_details.append({"object_id":"drink_container", "raw_label":raw_label,
            "normalized_label":"drink_container", "candidate_labels":candidates[:4],
            "uncertain":True})
    for object_id in objects:
        if object_id != "drink_container":
            object_details.append({"object_id":object_id, "raw_label":object_id,
                "normalized_label":object_id, "candidate_labels":[object_id], "uncertain":False})
    relations: list[dict[str, Any]] = []
    sole_person = persons[0] if len(persons) == 1 else None
    mentions_person = bool(re.search(r"\b(?:person|man|woman|individual)\b", text))
    person_subject = ((sole_person or {}).get("person_entity_id") or
                      (f"scene_person:{frame_id}:1" if mentions_person and len(persons) <= 1 else None))
    def add(subject: str, predicate: str, object_id: str, confidence: float = .85) -> None:
        relations.append({"subject_id":subject, "predicate":predicate, "object_id":object_id,
                          "confidence":confidence, "source":"vlm_visible_relation",
                          "frame_id":frame_id, "camera_id":camera_id})
    if (person_subject and "sofa" in objects and
            re.search(r"\b(?:person|man|woman|individual)\b[^.]{0,100}\b(?:next to|near|beside)\b[^.]{0,40}\b(?:sofa|couch)\b", text)):
        add(person_subject, "NEAR", "sofa", .8)
    if (person_subject and "sofa" in objects and
            re.search(r"\b(?:person|man|woman|individual)\b[^.]{0,80}\b(?:sitting|seated|sits)\s+on\b[^.]{0,30}\b(?:a\s+|the\s+)?(?:sofa|couch)\b", text)):
        add(person_subject, "SITTING_ON", "sofa")
    if (person_subject and "phone" in objects and
            re.search(r"\b(?:person|man|woman|individual)\b[^.]{0,60}\b(?:holding|holds)\b[^.]{0,25}\b(?:a\s+|the\s+)?(?:phone|smartphone|mobile phone)\b", text)):
        add(person_subject, "HOLDING", "phone")
    for object_id, object_pattern in (("coffee_cup", r"(?:coffee\s+cup|coffee\s+mug|cup|mug)"),
                                      ("papers", r"(?:paper|papers|documents?)")):
        if object_id not in objects:
            continue
        for surface_id, surface_pattern in (("table", r"(?:table|desk)"), ("floor", r"(?:floor|ground)")):
            if surface_id in objects and re.search(
                    rf"\b{object_pattern}\b[^.]{{0,45}}\b(?:is|are|lies?|rests?|sits?|stands?)?\s*on\b[^.]{{0,25}}\b(?:the\s+)?{surface_pattern}\b", text):
                add(object_id, "ON", surface_id)
    if "bottle" in objects and "table" in objects and re.search(
            r"\b(?:(?:water\s+)?bottle\s+(?:(?:is|lies|rests|sits|stands)\s+)?"
            r"on\s+(?:(?:a|the)\s+)?(?:table|desk)|"
            r"(?:wasserflasche|flasche)\s+(?:(?:ist|liegt|steht)\s+)?"
            r"auf\s+(?:(?:dem|einem)\s+)?tisch)\b", text):
        add("bottle", "ON", "table")
    if ambiguous_match:
        for surface_id, surface_pattern in (("table", r"(?:table|desk)"), ("floor", r"(?:floor|ground)")):
            if surface_id in objects and re.search(
                    rf"{re.escape(ambiguous_match.group(0))}[^.]{{0,30}}\bon\b[^.]{{0,20}}\b(?:the\s+)?{surface_pattern}\b",
                    text):
                add("drink_container", "ON", surface_id)
    return objects[:12], relations[:12], object_details[:12]


class CameraPerceptionService:
    """Shared, request-driven camera perception independent of any adapter."""

    def __init__(self, settings: Any, continuum: ContinuumService, *,
                 capture: Callable[..., Path] = capture_rtsp_snapshot,
                 detector: Callable[[Path], list[list[int]]] = detect_people,
                 recognizer: Callable[..., Any] = recognize_face_image,
                 describer: Callable[..., str] = describe_image_with_smolvlm,
                 translator: Callable[[str], str] | None = None,
                 vision_preflight: Callable[[], Any] | None = None,
                 vision_lease: Callable[[], Any] | None = None,
                 translation_lease: Callable[[], Any] | None = None):
        self.settings, self.continuum = settings, continuum
        self.capture, self.detector, self.recognizer, self.describer = capture, detector, recognizer, describer
        self.vision_preflight = vision_preflight
        self.vision_lease = vision_lease
        self.translator = translator
        self.translation_lease = translation_lease

    def _retire_legacy_singleton(self) -> dict[str, Any]:
        graph = self.continuum._graph()
        legacy = graph.get("tracks", {}).get("camera_primary")
        legacy_relations = [x for x in graph.get("relations", []) if x.get("subject_id") == "track:camera_primary"]
        retire_track = bool(legacy and not legacy.get("legacy"))
        if retire_track:
            legacy["present"] = False
            legacy["legacy"] = True
        if legacy_relations:
            graph.setdefault("legacy_relations", []).extend(
                [{**x, "retired_at": _utc_now()} for x in legacy_relations]
            )
            graph["relations"] = [x for x in graph.get("relations", [])
                                  if x.get("subject_id") != "track:camera_primary"]
        if retire_track or legacy_relations:
            self.continuum._write(self.continuum.persons_path, graph)
        return graph

    def state(self) -> dict[str, Any]:
        graph = self._retire_legacy_singleton()
        perception = dict(graph.get("perception") or {})
        age = _age_seconds(perception.get("last_perception"))
        perception["fresh"] = age is not None and age <= self.settings.perception_freshness_seconds
        perception["age_seconds"] = age
        for key in ("used", "person", "age_seconds", "confidence", "reason"):
            name = f"temporal_identity_bridge_{key}"
            perception.setdefault(name, PerceptionResult.__dataclass_fields__[name].default)
        perception.setdefault("persons_detected", len(perception.get("persons") or []))
        perception["capture_timestamp"] = perception.get("last_capture") or perception.get("captured_at")
        perception["tracks_active"] = [key for key, value in graph.get("tracks", {}).items()
                                        if key.startswith("camera_primary:") and value.get("present")]
        perception["identities_resolved"] = sorted({value.get("person_id") for value in graph.get("tracks", {}).values()
                                                      if value.get("present") and value.get("track_id") and
                                                      value.get("person_id") in self.settings.known_persons})
        perception["active_track_details"] = [{"track_id": key,
            "person_bbox": value.get("bounding_box"), **dict(value.get("recognition") or {}),
            "identity_resolved": value.get("person_id") if value.get("person_id") in self.settings.known_persons else None}
            for key, value in graph.get("tracks", {}).items()
            if key.startswith("camera_primary:") and value.get("present")]
        perception["description_en_excerpt"] = str(perception.get("description_en") or "")[:240]
        perception["description_de_excerpt"] = str(perception.get("description_de") or "")[:240]
        return perception

    def request(self, *, reason: str, force: bool = False, include_scene: bool = False,
                session_id: str = "perception:camera", scene_language: str = "en") -> PerceptionResult:
        cached = self.state()
        if cached.get("fresh") and not force and (not include_scene or cached.get("scene_description")):
            return PerceptionResult(**{key: cached.get(key) for key in PerceptionResult.__dataclass_fields__
                                      if key not in {"reused", "reason"}}, reused=True, reason=reason)
        if not self.settings.camera_enabled or not self.settings.camera_ip:
            raise RuntimeError("camera perception not configured")
        captured_at = _utc_now()
        frame_id = f"frame_{uuid.uuid4().hex}"
        camera_id = "camera_primary"
        url = build_rtsp_url(self.settings.camera_user, self.settings.camera_password,
                             self.settings.camera_ip, self.settings.camera_rtsp_path)
        image_path = self.capture(url=url, output_dir=self.settings.camera_cache_dir,
                                  camera_name="perception-camera")
        try:
            scene_path = crop_camera_overlay(image_path)
        except Exception:
            scene_path = image_path
        boxes = self.detector(scene_path)
        graph = self.continuum._graph(); tracks = graph.get("tracks", {})
        recognition_history = dict(graph.get("local_recognition_history") or {})
        # Preserve secure observations even if the same track later changes identity.
        for track in tracks.values():
            person_id = track.get("person_id")
            if person_id in self.settings.known_persons and track.get("sensor_id") == camera_id:
                history_key = f"{camera_id}:{person_id}"
                recognition_history.setdefault(history_key, dict(track))
        active = {key: value for key, value in tracks.items() if value.get("present") and value.get("sensor_id", "camera_primary") == "camera_primary"}
        used: set[str] = set(); evidence: list[dict[str, Any]] = []
        resolved_in_frame: set[str] = set()
        next_id = int(graph.get("next_track_id", 1))
        image = Image.open(scene_path).convert("RGB")
        for box in boxes:
            matches = sorted(((_iou(box, value.get("bounding_box") or [0, 0, 0, 0]), key)
                              for key, value in active.items() if key not in used), reverse=True)
            if matches and matches[0][0] >= self.settings.perception_track_iou_threshold:
                track_id = matches[0][1]
            else:
                track_id = f"camera_primary:{next_id}"; next_id += 1
            used.add(track_id)
            x, y, width, height = [int(value) for value in box]
            x, y = max(0, x), max(0, y)
            right, bottom = min(image.width, x + max(1, width)), min(image.height, y + max(1, height))
            box = [x, y, max(1, right - x), max(1, bottom - y)]
            width, height = box[2], box[3]
            crop_path = self.settings.camera_cache_dir / f"{Path(image_path).stem}-{track_id.replace(':', '-')}.jpg"
            image.crop((x, y, x + width, y + height)).save(crop_path, quality=92)
            person_id, confidence = None, .5
            recognition = {"face_detected":False, "recognition_attempted":False,
                           "recognition_candidate":None, "recognition_confidence":None,
                           "recognition_reason":"recognition disabled"}
            if self.settings.identity_enabled and self.settings.person_recognition_enabled:
                recognition["recognition_attempted"] = True
                try:
                    decision = self.recognizer(image_path=crop_path, identity_dir=self.settings.identity_dir,
                        model_name=self.settings.identity_model, device=self.settings.identity_device,
                        threshold=self.settings.person_confidence_threshold, margin_threshold=self.settings.identity_margin,
                        top_k=self.settings.identity_top_k, min_roger_votes=self.settings.identity_min_roger_votes)
                    confidence = decision.confidence
                    person_id = (decision.identity if decision.identity in self.settings.known_persons and
                                 confidence >= self.settings.person_confidence_threshold else None)
                    recognition.update({"face_detected":bool(getattr(decision, "face_path", None)),
                        "recognition_candidate":getattr(decision, "top_label", decision.identity),
                        "recognition_confidence":decision.confidence,
                        "recognition_reason":getattr(decision, "reason", "recognition completed")})
                    if person_id in resolved_in_frame and not include_scene:
                        person_id = None
                    elif person_id:
                        resolved_in_frame.add(person_id)
                except Exception as exc:
                    recognition["recognition_reason"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            display_name = self.settings.known_persons.get(person_id) if person_id else None
            evidence.append({"track_id":track_id, "person_id":person_id,
                             "person_entity_id":f"person:{person_id}" if person_id else None,
                             "display_name":display_name, "confidence":confidence,
                             "recognition_confidence":confidence,
                             "identity_status":"confirmed_local" if person_id else "unresolved",
                             "binding_status":"confirmed_local" if person_id else "unbound",
                             "require_fresh_identity":bool(include_scene),
                             "frame_id":frame_id, "camera_id":camera_id,
                             "location":"camera_view", "bounding_box":box,
                             "sensor_id":"camera_primary", "recognition":recognition})
        track_handover_detected = False
        track_handover_from = track_handover_to = track_handover_reason = None
        track_handover_identity = None
        track_continuity_score = 0.0
        if len(boxes) == len(evidence) == 1 and evidence[0].get("person_entity_id"):
            current_track = evidence[0]["track_id"]
            current_box = evidence[0]["bounding_box"]
            freshness_window = float(getattr(self.settings, "perception_handover_max_age_seconds", 3.0))
            candidates = []
            for old_track, old in tracks.items():
                old_person = str(old.get("person_id") or "")
                age = _age_seconds(old.get("last_seen"))
                old_box = old.get("bounding_box") or []
                if (current_track in tracks or old_track == current_track or old_track in used or
                        old.get("persons_detected", cached.get("persons_detected")) != 1 or
                        old.get("handover_to") or
                        not old_person.startswith("unknown_person:") or
                        old.get("sensor_id", camera_id) != camera_id or
                        age is None or age > freshness_window or len(old_box) != 4):
                    continue
                score = _track_continuity_score(old_box, current_box)
                if score >= .35:
                    candidates.append((score, old_track))
            if len(candidates) == 1:
                track_continuity_score, track_handover_from = max(candidates)
                track_handover_to = current_track
                track_handover_reason = "single_current_person_recent_spatial_unknown_to_canonical"
                track_handover_detected = True
                track_handover_identity = evidence[0]["person_entity_id"]
                tracks[track_handover_from]["present"] = False
                tracks[track_handover_from]["handover_to"] = track_handover_to
                tracks[track_handover_from]["retired_reason"] = "track_handover"
                self.continuum._write(self.continuum.persons_path, graph)
        description = ""
        description_en = ""
        description_de = ""
        vision_model_calls = 0
        translation_model_calls = 0
        identity_enriched = False
        identity_binding_reason = "no_visible_local_person" if not evidence else "no_confirmed_local_identity"
        objects: list[str] = []
        relations: list[dict[str, Any]] = []
        object_details: list[dict[str, Any]] = []
        scene_persons = _deduplicate_scene_persons(evidence)
        description_at = None
        bridge = {"temporal_identity_bridge_used":False,
                  "temporal_identity_bridge_person":None,
                  "temporal_identity_bridge_age_seconds":None,
                  "temporal_identity_bridge_confidence":None,
                  "temporal_identity_bridge_reason":"not_needed"}
        if include_scene and self.settings.vision_enabled:
            lease = self.vision_lease() if self.vision_lease is not None else nullcontext()
            with lease:
                if self.vision_preflight is not None:
                    try:
                        self.vision_preflight()
                    except Exception:
                        # Compatibility hook for callers predating the central
                        # ResourceCoordinator.
                        _logger.warning("Vision resource preflight failed", exc_info=True)
                description_en = self.describer(
                    scene_path, mode="camera", prompt=camera_scene_prompt("en")
                ) or ""
                vision_model_calls = 1
            # The VLM and translation worker are never identity authorities.
            for display_name in self.settings.known_persons.values():
                description_en = re.sub(rf"\b{re.escape(display_name)}\b", "a person",
                                        description_en, flags=re.IGNORECASE)
            if not scene_persons:
                anonymous = _anonymous_scene_person(
                    description_en, frame_id=frame_id, camera_id=camera_id)
                if anonymous:
                    scene_persons = [anonymous]
                    identity_binding_reason = "anonymous_visual_person"
            if not boxes and not used and len(scene_persons) == 1:
                candidate, bridge_reason = _temporal_identity_candidate(
                    description_en, {**recognition_history, **tracks}, camera_id=camera_id, captured_at=captured_at,
                    window=getattr(self.settings, "perception_identity_bridge_seconds", 20.0),
                    threshold=self.settings.person_confidence_threshold,
                    known_persons=self.settings.known_persons)
                bridge["temporal_identity_bridge_reason"] = bridge_reason
                if candidate:
                    person_id = candidate["person_id"]
                    scene_persons[0].update({"person_id":person_id,
                        "person_entity_id":f"person:{person_id}",
                        "display_name":self.settings.known_persons[person_id],
                        "binding_status":"temporal_identity_bridge",
                        "identity_status":"temporal_identity_bridge",
                        "identity_source":"local_recognition_history"})
                    bridge.update({"temporal_identity_bridge_used":True,
                        "temporal_identity_bridge_person":f"person:{person_id}",
                        "temporal_identity_bridge_age_seconds":candidate["age"],
                        "temporal_identity_bridge_confidence":candidate["confidence"]})
            presentation_language = "de" if scene_language.strip().lower().startswith("de") else "en"
            if presentation_language == "de" and description_en and self.translator is not None:
                lease = self.translation_lease() if self.translation_lease is not None else nullcontext()
                with lease:
                    description_de = self.translator(description_en) or ""
                    translation_model_calls = 1
                for display_name in self.settings.known_persons.values():
                    description_de = re.sub(rf"\b{re.escape(display_name)}\b", "eine Person",
                                            description_de, flags=re.IGNORECASE)
            description = description_de if presentation_language == "de" else description_en
            resolved_names = [self.settings.known_persons[x["person_id"]]
                              for x in scene_persons if x.get("person_id") in self.settings.known_persons]
            if len(scene_persons) == 1 and len(resolved_names) == 1:
                description, identity_binding_reason = _compose_single_person_description(
                    description, resolved_names[0], language=presentation_language)
                identity_enriched = True
            elif len(scene_persons) > 1:
                identity_binding_reason = "multiple_scene_persons_ambiguous"
            objects, relations, object_details = _extract_scene_structure(
                description_en, scene_persons, frame_id=frame_id, camera_id=camera_id)
            description_at = _utc_now()
        perceived_at = _utc_now()
        identity_source = "local_recognition_history" if bridge["temporal_identity_bridge_used"] else "local_recognition"
        scene_observation = {"frame_id":frame_id, "camera_id":camera_id,
            "captured_at":captured_at, "perceived_at":perceived_at,
            "description_en":description_en, "description_de":description_de,
            "final_description":description, "identity_enriched":identity_enriched,
            "identity_binding_reason":identity_binding_reason,
            "identity_source":identity_source, "persons":scene_persons, **bridge,
            "scene_person_count":len(scene_persons),
            "canonical_person_count":len({x["person_entity_id"] for x in scene_persons
                                           if x.get("person_entity_id")}),
            "track_handover_detected":track_handover_detected,
            "track_handover_from":track_handover_from, "track_handover_to":track_handover_to,
            "track_handover_identity":track_handover_identity,
            "track_handover_reason":track_handover_reason,
            "persons_detected":len(boxes),
            "current_scene_unknown_count":sum(not x.get("person_entity_id") for x in scene_persons),
            "track_continuity_score":track_continuity_score,
            "objects":objects, "object_details":object_details, "relations":relations}
        self.continuum.observe(VisualObservation(description or f"Camera perception: {len(evidence)} person(s)",
            persons=scene_persons, objects=objects, relations=relations, confidence=.8,
            timestamp=perceived_at), session_id=session_id, track_evidence=evidence or None)
        graph = self.continuum._graph()
        current_scene_unknown_count = sum(1 for item in scene_persons
                                          if not item.get("person_entity_id"))
        historical_unknown_count = sum(1 for person in self.continuum.persons().values()
                                       if not person.known and not person.current_presence)
        # Track presence is frame-local, independent of canonical person presence.
        for key, track in graph.get("tracks", {}).items():
            if track.get("sensor_id", camera_id) == camera_id:
                track["present"] = key in used
        for item in evidence:
            track = graph.setdefault("tracks", {}).setdefault(item["track_id"], {})
            track.update({"track_id":item["track_id"], "sensor_id":"camera_primary",
                          "first_seen":track.get("first_seen") or perceived_at,
                          "last_seen":perceived_at, "present":True,
                          "bounding_box":item["bounding_box"], "persons_detected":len(boxes)})
        result = PerceptionResult(captured_at, perceived_at, str(image_path), str(scene_path), scene_persons,
            [x["track_id"] for x in evidence], sorted({x["person_id"] for x in evidence if x["person_id"]}),
            description, description_at, False, reason, description_en, description_de,
            "en", "en", "de" if scene_language.strip().lower().startswith("de") else "en",
            vision_model_calls, translation_model_calls, 0, "local_recognition",
            frame_id, camera_id, identity_enriched, identity_binding_reason, description,
            len(scene_persons), len({x["person_entity_id"] for x in scene_persons
                                    if x.get("person_entity_id")}),
            objects, relations, scene_observation)
        result.identity_source = identity_source
        for key, value in bridge.items():
            setattr(result, key, value)
        result.persons_detected = len(boxes)
        result.object_details = object_details
        result.track_handover_detected = track_handover_detected
        result.track_handover_from = track_handover_from
        result.track_handover_to = track_handover_to
        result.track_handover_identity = track_handover_identity
        result.track_handover_reason = track_handover_reason
        result.track_continuity_score = track_continuity_score
        result.current_scene_unknown_count = current_scene_unknown_count
        result.historical_unknown_count = historical_unknown_count
        for item in evidence:
            if item.get("person_entity_id"):
                recognition_history[f"{camera_id}:{item['person_id']}"] = {
                    "person_id":item["person_id"], "sensor_id":camera_id,
                    "last_seen":perceived_at, "confidence":item["confidence"],
                    "recognition":dict(item["recognition"])}
        graph["local_recognition_history"] = recognition_history
        graph["next_track_id"] = next_id
        graph["perception"] = {"last_capture":captured_at, "last_perception":perceived_at, **asdict(result)}
        self.continuum._write(self.continuum.persons_path, graph)
        return result
