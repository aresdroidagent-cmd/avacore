from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
import logging
import re
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


class CameraPerceptionService:
    """Shared, request-driven camera perception independent of any adapter."""

    def __init__(self, settings: Any, continuum: ContinuumService, *,
                 capture: Callable[..., Path] = capture_rtsp_snapshot,
                 detector: Callable[[Path], list[list[int]]] = detect_people,
                 recognizer: Callable[..., Any] = recognize_face_image,
                 describer: Callable[..., str] = describe_image_with_smolvlm,
                 vision_preflight: Callable[[], Any] | None = None,
                 vision_lease: Callable[[], Any] | None = None):
        self.settings, self.continuum = settings, continuum
        self.capture, self.detector, self.recognizer, self.describer = capture, detector, recognizer, describer
        self.vision_preflight = vision_preflight
        self.vision_lease = vision_lease

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
        perception["persons_detected"] = len(perception.get("persons") or [])
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
                    person_id = decision.identity if decision.identity in self.settings.known_persons else None
                    confidence = decision.confidence
                    recognition.update({"face_detected":bool(getattr(decision, "face_path", None)),
                        "recognition_candidate":getattr(decision, "top_label", decision.identity),
                        "recognition_confidence":decision.confidence,
                        "recognition_reason":getattr(decision, "reason", "recognition completed")})
                    if person_id in resolved_in_frame:
                        person_id = None
                    elif person_id:
                        resolved_in_frame.add(person_id)
                except Exception as exc:
                    recognition["recognition_reason"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            evidence.append({"track_id":track_id, "person_id":person_id, "confidence":confidence,
                             "location":"camera_view", "bounding_box":box,
                             "sensor_id":"camera_primary", "recognition":recognition})
        description = ""
        description_at = None
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
                description = self.describer(
                    scene_path, mode="camera", prompt=camera_scene_prompt(scene_language)
                ) or ""
            resolved = {x["person_id"] for x in evidence if x.get("person_id")}
            for person_id, display_name in self.settings.known_persons.items():
                if person_id not in resolved:
                    description = re.sub(rf"\b{re.escape(display_name)}\b", "a person", description,
                                         flags=re.IGNORECASE)
            description_at = _utc_now()
        perceived_at = _utc_now()
        self.continuum.observe(VisualObservation(description or f"Camera perception: {len(evidence)} person(s)",
            persons=evidence, confidence=.8), session_id=session_id)
        graph = self.continuum._graph()
        for item in evidence:
            track = graph.get("tracks", {}).get(item["track_id"], {})
            track.update({"track_id":item["track_id"], "sensor_id":"camera_primary",
                          "first_seen":track.get("first_seen") or perceived_at,
                          "last_seen":perceived_at, "present":True,
                          "bounding_box":item["bounding_box"]})
        result = PerceptionResult(captured_at, perceived_at, str(image_path), str(scene_path), evidence,
            [x["track_id"] for x in evidence], sorted({x["person_id"] for x in evidence if x["person_id"]}),
            description, description_at, False, reason)
        graph["next_track_id"] = next_id
        graph["perception"] = {"last_capture":captured_at, "last_perception":perceived_at, **asdict(result)}
        self.continuum._write(self.continuum.persons_path, graph)
        return result
