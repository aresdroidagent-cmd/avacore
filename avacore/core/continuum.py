from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avacore.core.cognitive_workspace import WorkingMemory, run_workspace_cycle
from avacore.core.jspace import ContinuumState, clamp, infer_jspace_tags


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CognitiveEvent:
    source: str
    kind: str
    content: str
    session_id: str
    cycle_id: str = field(default_factory=lambda: f"cy_{uuid.uuid4().hex}")
    id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex}")
    timestamp: str = field(default_factory=now)
    activation: float = .5
    salience: float = .5
    confidence: float = .5
    related_entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualObservation:
    scene_description: str
    persons: list[dict[str, Any]] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = .5
    timestamp: str = field(default_factory=now)


@dataclass
class PersonEntity:
    person_id: str
    display_name: str
    visual_identity: str | None = None
    voice_identity: str | None = None
    relationship_context: str | None = None
    current_presence: bool = False
    current_location: str | None = None
    last_location: str | None = None
    first_seen: str | None = None
    last_seen: str | None = None
    confidence: float = 0.0
    known: bool = True
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def entity_id(self) -> str:
        return f"person:{self.person_id}"


@dataclass
class EntityRelation:
    subject_id: str
    predicate: str
    object_id: str
    confidence: float
    source: str
    first_observed: str = field(default_factory=now)
    last_observed: str = field(default_factory=now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.subject_id}|{self.predicate}|{self.object_id}"


class IdentityResolver:
    """Resolve only enrolled identities or continuity of an existing track."""

    def __init__(self, known_persons: dict[str, str], threshold: float):
        self.known_persons = known_persons
        self.threshold = threshold

    def resolve(self, evidence: dict[str, Any], tracks: dict[str, dict[str, Any]]) -> tuple[str, bool]:
        track_id = str(evidence.get("track_id") or "").strip()
        candidate = str(evidence.get("person_id") or "").strip()
        confidence = clamp(evidence.get("confidence", 0))
        if candidate in self.known_persons and confidence >= self.threshold:
            return candidate, True
        prior = tracks.get(track_id, {}).get("person_id") if track_id else None
        if prior:
            return str(prior), False
        return f"unknown_person:{track_id or 'untracked'}", False


class ContinuumService:
    """Small persistence/assimilation layer over the Phase-2 J-Space store."""

    def __init__(self, continuum_path: Path | str, workspace_path: Path | str,
                 working_memory_path: Path | str, history_path: Path | str,
                 persons_path: Path | str, *, history_limit: int = 200,
                 confidence_threshold: float = .78, event_cooldown: float = 10.0,
                 known_persons: dict[str, str] | None = None):
        self.continuum_path, self.workspace_path = Path(continuum_path), Path(workspace_path)
        self.working_memory_path, self.history_path = Path(working_memory_path), Path(history_path)
        self.persons_path = Path(persons_path)
        self.history_limit = history_limit
        self.confidence_threshold = confidence_threshold
        self.event_cooldown = event_cooldown
        self.known_persons = dict(known_persons or {})

    @staticmethod
    def _read(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return default

    @staticmethod
    def _write(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def events(self) -> list[dict[str, Any]]:
        return list(self._read(self.history_path, {}).get("events", []))

    def assimilate(self, event: CognitiveEvent, *, memory: bool = True) -> CognitiveEvent:
        state = ContinuumState.load(self.continuum_path)
        state.inject(event.source, event.kind, event.content, infer_jspace_tags(event.content),
                     activation_boost=event.activation, priority=event.salience,
                     persistence=.45, confidence=event.confidence,
                     relevance=event.activation, recency=1.0,
                     continuity=.9, metadata={**event.metadata, "event_id": event.id,
                     "cycle_id": event.cycle_id, "session_id": event.session_id,
                     "related_entities": event.related_entities})
        state.save(self.continuum_path)
        history = self.events() + [asdict(event)]
        self._write(self.history_path, {"version": 1, "events": history[-self.history_limit:]})
        if memory:
            wm = WorkingMemory(self.working_memory_path, session_id=event.session_id)
            wm.add("user" if event.kind == "user_command" else "system", event.content,
                   event.cycle_id, kind=event.kind, importance=event.salience)
            wm.save()
        # Use the Phase-2 activation/competition path; this does not invoke an
        # LLM and does not force the event into the selected subset.
        run_workspace_cycle(
            jspace_path=self.continuum_path, workspace_path=self.workspace_path,
            stimulus="", trigger=event.kind, cycle_id=event.cycle_id,
            candidates=[{"source": event.source, "kind": event.kind,
                         "content": event.content, "activation": event.activation,
                         "priority": event.salience, "confidence": event.confidence,
                         "continuity": .9, "metadata": {**event.metadata,
                         "event_id": event.id, "cycle_id": event.cycle_id,
                         "session_id": event.session_id}}],
            session_id=event.session_id,
        )
        return event

    def command(self, *, session_id: str, command: str, content: str,
                cycle_id: str | None = None, person_id: str | None = None) -> CognitiveEvent:
        cycle_id = cycle_id or f"cy_{uuid.uuid4().hex}"
        if person_id in self.known_persons:
            self.relate(f"person:{person_id}", "speaking_via", "channel:telegram",
                        confidence=1.0, source="explicit_mapping", session_id=session_id,
                        cycle_id=cycle_id)
            self.relate(f"person:{person_id}", "owns_session", f"session:{session_id}",
                        confidence=1.0, source="explicit_mapping", session_id=session_id,
                        cycle_id=cycle_id)
        return self.assimilate(CognitiveEvent("telegram", "user_command", content,
            session_id, cycle_id=cycle_id, activation=1.0,
            salience=1.0, confidence=1.0,
            related_entities=[f"person:{person_id}"] if person_id in self.known_persons else [],
            metadata={"command": command, **({"person_id": person_id} if person_id in self.known_persons else {})}))

    def command_result(self, *, session_id: str, command: str, content: str,
                       cycle_id: str, status: str = "success") -> CognitiveEvent:
        return self.assimilate(CognitiveEvent("command", "command_result", content,
            session_id, cycle_id=cycle_id, activation=.9, salience=.8, confidence=1.0,
            metadata={"command": command, "status": status}))

    def persons(self) -> dict[str, PersonEntity]:
        raw = self._read(self.persons_path, {}).get("persons", [])
        people = {x["person_id"]: PersonEntity(**x) for x in raw}
        for person_id, display_name in self.known_persons.items():
            people.setdefault(person_id, PersonEntity(person_id, display_name, visual_identity=person_id))
        return people

    def _graph(self) -> dict[str, Any]:
        return self._read(self.persons_path, {"version": 2, "persons": [], "tracks": {}, "relations": []})

    def relations(self) -> list[EntityRelation]:
        return [EntityRelation(**x) for x in self._graph().get("relations", [])]

    def relate(self, subject_id: str, predicate: str, object_id: str, *, confidence: float,
               source: str, session_id: str = "system", cycle_id: str | None = None,
               metadata: dict[str, Any] | None = None) -> tuple[EntityRelation, bool]:
        graph = self._graph()
        relations = {x.id: x for x in self.relations()}
        relation, created = self._upsert_relation(relations, subject_id, predicate, object_id,
                                                  confidence, source, now(), metadata)
        graph["relations"] = [asdict(x) for x in relations.values()]
        self._write(self.persons_path, graph)
        if created:
            self.assimilate(CognitiveEvent(source, "new_relation",
                f"{subject_id} {predicate} {object_id}", session_id,
                cycle_id=cycle_id or f"cy_{uuid.uuid4().hex}", activation=.5,
                salience=.4, confidence=confidence, related_entities=[subject_id, object_id],
                metadata={"predicate": predicate, **(metadata or {})}), memory=False)
        return relation, created

    def entities(self) -> list[dict[str, Any]]:
        graph = self._graph()
        persons = [{"id": x.entity_id, "kind": "person", **asdict(x)} for x in self.persons().values()]
        tracks = [{"id": f"track:{key}", "kind": "visual_track", **value}
                  for key, value in graph.get("tracks", {}).items()]
        other: dict[str, dict[str, Any]] = {}
        for relation in self.relations():
            for entity_id in (relation.subject_id, relation.object_id):
                if not entity_id.startswith(("person:", "track:")):
                    other.setdefault(entity_id, {"id": entity_id, "kind": entity_id.split(":", 1)[0]})
        return persons + tracks + list(other.values())

    @staticmethod
    def _upsert_relation(relations: dict[str, EntityRelation], subject_id: str, predicate: str,
                         object_id: str, confidence: float, source: str, timestamp: str,
                         metadata: dict[str, Any] | None = None) -> tuple[EntityRelation, bool]:
        relation = EntityRelation(subject_id, predicate, object_id, clamp(confidence), source,
                                  timestamp, timestamp, metadata or {})
        existing = relations.get(relation.id)
        if existing:
            existing.last_observed = timestamp
            existing.confidence = max(existing.confidence, relation.confidence)
            existing.metadata.update(relation.metadata)
            return existing, False
        relations[relation.id] = relation
        return relation, True

    def remove_relation(self, subject_id: str, predicate: str, object_id: str,
                        *, session_id: str = "system", reason: str = "removed") -> bool:
        graph = self._graph()
        relations = {x.id: x for x in self.relations()}
        relation_id = f"{subject_id}|{predicate}|{object_id}"
        relation = relations.pop(relation_id, None)
        if not relation:
            return False
        graph["relations"] = [asdict(x) for x in relations.values()]
        self._write(self.persons_path, graph)
        self.assimilate(CognitiveEvent("system", "relation_removed",
            f"{subject_id} no longer {predicate} {object_id}", session_id,
            activation=.55, salience=.5, confidence=relation.confidence,
            related_entities=[subject_id, object_id], metadata={"predicate": predicate, "reason": reason}), memory=False)
        return True

    def expire_relations(self, before: str, *, predicates: set[str] | None = None) -> int:
        expired = [x for x in self.relations() if x.last_observed < before and
                   (predicates is None or x.predicate in predicates)]
        for relation in expired:
            self.remove_relation(relation.subject_id, relation.predicate, relation.object_id, reason="expired")
        return len(expired)

    def observe(self, observation: VisualObservation, *, session_id: str = "vision") -> list[CognitiveEvent]:
        graph = self._graph()
        people = self.persons()
        tracks = dict(graph.get("tracks") or {})
        relations = {x.id: x for x in self.relations()}
        resolver = IdentityResolver({key: value.display_name for key, value in people.items() if value.known},
                                    self.confidence_threshold)
        previous = {key for key, value in people.items() if value.current_presence}
        current: set[str] = set()
        related: list[str] = []
        relation_events: list[CognitiveEvent] = []
        for index, evidence in enumerate(observation.persons):
            confidence = clamp(evidence.get("confidence", 0))
            track_id = str(evidence.get("track_id") or f"frame_person_{index}")
            previous_person = tracks.get(track_id, {}).get("person_id")
            person_id, newly_resolved = resolver.resolve({**evidence, "track_id": track_id}, tracks)
            if person_id not in people:
                people[person_id] = PersonEntity(person_id, "Unknown person", known=False)
            current.add(person_id); related.append(people[person_id].entity_id)
            person = people[person_id]
            previous_location = person.current_location
            person.current_presence, person.last_seen = True, observation.timestamp
            person.first_seen = person.first_seen or observation.timestamp
            person.current_location = evidence.get("location")
            if person.current_location:
                person.last_location = person.current_location
            person.confidence = confidence
            tracks[track_id] = {**tracks.get(track_id, {}), "person_id": person_id,
                                "last_seen": observation.timestamp, "present": True,
                                "confidence": confidence,
                                **({"bounding_box": evidence["bounding_box"]} if evidence.get("bounding_box") else {}),
                                **({"sensor_id": evidence["sensor_id"]} if evidence.get("sensor_id") else {}),
                                **({"recognition": evidence["recognition"]} if evidence.get("recognition") else {})}
            _, created = self._upsert_relation(relations, f"track:{track_id}", "seen_by", "sensor:camera",
                                               confidence, "vision", observation.timestamp)
            if created:
                relation_events.append(CognitiveEvent("vision", "new_relation", f"track:{track_id} seen_by sensor:camera",
                    session_id, activation=.45, salience=.35, confidence=confidence,
                    related_entities=[f"track:{track_id}", "sensor:camera"], metadata={"predicate":"seen_by"}))
            if person.known:
                _, created = self._upsert_relation(relations, f"track:{track_id}", "identified_as", person.entity_id,
                                                   confidence, "local_identity", observation.timestamp)
                if newly_resolved and previous_person != person_id:
                    relation_events.append(CognitiveEvent("vision", "identity_resolved",
                        f"track:{track_id} identified as {person.display_name}", session_id,
                        activation=.9, salience=.85, confidence=confidence,
                        related_entities=[f"track:{track_id}", person.entity_id], metadata={"person_id":person_id}))
            if person.current_location:
                if previous_location and previous_location != person.current_location:
                    relations.pop(f"{person.entity_id}|present_at|location:{previous_location}", None)
                    relation_events.append(CognitiveEvent("vision", "location_changed",
                        f"{person.display_name} moved from {previous_location} to {person.current_location}",
                        session_id, activation=.75, salience=.7, confidence=confidence,
                        related_entities=[person.entity_id, f"location:{person.current_location}"],
                        metadata={"person_id":person_id, "from":previous_location, "to":person.current_location}))
                self._upsert_relation(relations, person.entity_id, "present_at", f"location:{person.current_location}",
                                      confidence, "vision", observation.timestamp)
        events: list[CognitiveEvent] = []
        signature = {"scene": observation.scene_description.strip().casefold(),
                     "persons": sorted(current), "objects": sorted(set(observation.objects))}
        prior = self._read(self.persons_path, {}).get("last_signature")
        changed = signature != prior
        if changed:
            events.append(self.assimilate(CognitiveEvent("vision", "visual_observation",
                observation.scene_description or "Visual observation", session_id,
                activation=.65, salience=.55, confidence=observation.confidence,
                related_entities=related, metadata={"persons": observation.persons,
                "objects": observation.objects, "relations": observation.relations}), memory=False))
        for person_id in current - previous:
            if not people[person_id].known:
                events.append(self.assimilate(CognitiveEvent("vision", "new_unknown_person",
                    "An unknown person entered", session_id, activation=.85,
                    salience=.8, confidence=people[person_id].confidence,
                    related_entities=[people[person_id].entity_id], metadata={"person_id": person_id}), memory=True))
            events.append(self.assimilate(CognitiveEvent("vision", "person_entered",
                f"{people[person_id].display_name} entered", session_id, activation=.9,
                salience=.9, confidence=people[person_id].confidence,
                related_entities=[f"person:{person_id}"], metadata={"person_id": person_id}), memory=True))
        for person_id in previous - current:
            people[person_id].current_presence = False
            if people[person_id].current_location:
                location_id = f"location:{people[person_id].current_location}"
                removed = relations.pop(f"{people[person_id].entity_id}|present_at|{location_id}", None)
                if removed:
                    relation_events.append(CognitiveEvent("vision", "relation_removed",
                        f"{people[person_id].display_name} no longer present at {people[person_id].current_location}",
                        session_id, activation=.6, salience=.55, confidence=removed.confidence,
                        related_entities=[people[person_id].entity_id, location_id],
                        metadata={"predicate":"present_at", "reason":"person_left"}))
            people[person_id].last_location = people[person_id].current_location or people[person_id].last_location
            people[person_id].current_location = None
            for track in tracks.values():
                if track.get("person_id") == person_id:
                    track["present"] = False
            events.append(self.assimilate(CognitiveEvent("vision", "person_left",
                f"{people[person_id].display_name} left", session_id, activation=.85,
                salience=.85, confidence=1.0, related_entities=[f"person:{person_id}"],
                metadata={"person_id": person_id}), memory=True))
        for event in relation_events:
            events.append(self.assimilate(event, memory=event.kind == "identity_resolved"))
        graph.update({"version": 2, "last_observation": asdict(observation),
            "last_signature": signature, "persons": [asdict(x) for x in people.values()],
            "tracks": tracks, "relations": [asdict(x) for x in relations.values()]})
        self._write(self.persons_path, graph)
        return events

    def summary(self) -> dict[str, Any]:
        state = ContinuumState.load(self.continuum_path)
        workspace = self._read(self.workspace_path, {}).get("current") or {}
        return {"enabled": True, "entities": len(state.items),
                "active": sum(x.activation >= .2 for x in state.items.values()),
                "workspace": len(workspace.get("active_items") or []),
                "vision_entities": sum(x.source == "vision" for x in state.items.values()),
                "persons_active": sum(x.current_presence for x in self.persons().values()),
                "updated_at": state.updated_at, "top_items": [asdict(x) for x in state.top_items(12)]}
