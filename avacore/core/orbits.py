from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avacore.core.jspace import clamp


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tokens(text: str) -> set[str]:
    return {x for x in re.findall(r"[\wäöüß-]{3,}", (text or "").casefold())}


@dataclass
class CognitiveOrbit:
    orbit_id: str
    title: str
    description: str
    status: str = "open"
    importance: float = .5
    activation: float = .05
    baseline_activation: float = .05
    created_at: str = field(default_factory=utc_now)
    last_activated_at: str = field(default_factory=utc_now)
    last_progress_at: str | None = None
    unresolved_questions: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    related_memories: list[str] = field(default_factory=list)
    related_tasks: list[str] = field(default_factory=list)
    progress: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveTask:
    task_id: str
    orbit_id: str
    title: str
    objective: str
    task_type: str
    status: str = "pending"
    priority: float = .5
    expected_cost: float = .2
    risk_level: str = "low"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    required_capabilities: list[str] = field(default_factory=list)
    result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionCandidate:
    question_id: str
    orbit_id: str
    question: str
    importance: float
    reason: str
    created_at: str = field(default_factory=utc_now)
    already_asked: bool = False
    delivery_enabled: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class OrbitStore:
    VALID_STATUSES = {"open", "active", "blocked", "resolved", "archived"}
    TASK_TYPES = {"inspect", "recall", "analyze", "research", "vision", "code", "test", "review", "ask_user"}

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            data = {}
        return {"version": 1, "last_drive_at": data.get("last_drive_at"),
                "orbits": list(data.get("orbits") or []), "tasks": list(data.get("tasks") or []),
                "questions": list(data.get("questions") or [])}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def orbits(self) -> list[CognitiveOrbit]:
        return [CognitiveOrbit(**x) for x in self._load()["orbits"]]

    def tasks(self) -> list[CognitiveTask]:
        return [CognitiveTask(**x) for x in self._load()["tasks"]]

    def questions(self) -> list[QuestionCandidate]:
        return [QuestionCandidate(**x) for x in self._load()["questions"]]

    def create_orbit(self, title: str, description: str, *, importance: float = .5,
                     baseline_activation: float = .05, related_entities: list[str] | None = None,
                     metadata: dict[str, Any] | None = None) -> CognitiveOrbit:
        data = self._load(); baseline = clamp(baseline_activation, 0, .3)
        orbit = CognitiveOrbit(f"orbit_{uuid.uuid4().hex}", title.strip(), description.strip(),
            importance=clamp(importance), activation=baseline, baseline_activation=baseline,
            related_entities=list(dict.fromkeys(related_entities or [])), metadata=dict(metadata or {}))
        if not orbit.title or not orbit.description:
            raise ValueError("orbit title and description are required")
        data["orbits"].append(asdict(orbit)); self._save(data)
        return orbit

    def _update_orbit(self, orbit: CognitiveOrbit) -> None:
        data = self._load()
        data["orbits"] = [asdict(orbit) if x["orbit_id"] == orbit.orbit_id else x for x in data["orbits"]]
        self._save(data)

    def get_orbit(self, orbit_id: str) -> CognitiveOrbit:
        orbit = next((x for x in self.orbits() if x.orbit_id == orbit_id), None)
        if not orbit: raise KeyError(orbit_id)
        return orbit

    def decay(self, factor: float = .85) -> None:
        data = self._load(); updated = []
        for orbit in self.orbits():
            if orbit.status in {"open", "active", "blocked"}:
                orbit.activation = clamp(orbit.baseline_activation +
                    max(0, orbit.activation - orbit.baseline_activation) * clamp(factor))
            else:
                orbit.activation = clamp(orbit.activation * .25)
            updated.append(asdict(orbit))
        data["orbits"] = updated; self._save(data)

    def react(self, *, content: str, related_entities: list[str], amount: float = .3) -> list[CognitiveOrbit]:
        data = self._load(); changed: list[CognitiveOrbit] = []; output = []
        event_tokens = _tokens(content); entities = set(related_entities)
        for orbit in self.orbits():
            if orbit.status not in {"open", "active", "blocked"}:
                output.append(asdict(orbit)); continue
            entity_match = bool(entities & set(orbit.related_entities))
            orbit_tokens = _tokens(f"{orbit.title} {orbit.description}")
            lexical = len(event_tokens & orbit_tokens) / max(1, min(len(orbit_tokens), 8))
            relevance = max(1.0 if entity_match else 0.0, lexical)
            if relevance >= .2:
                orbit.activation = clamp(max(orbit.baseline_activation, orbit.activation) +
                                         min(.35, amount * relevance))
                orbit.last_activated_at = utc_now(); changed.append(orbit)
            output.append(asdict(orbit))
        data["orbits"] = output; self._save(data)
        return changed

    def candidates(self, minimum: float = .2) -> list[dict[str, Any]]:
        return [{"source":"orbit", "kind":"cognitive_orbit", "content":f"{x.title}: {x.description}",
                 "activation":x.activation, "relevance":x.activation,
                 "priority":clamp(x.importance * .55), "persistence":.8, "confidence":1.0,
                 "recency":x.activation, "continuity":x.activation,
                 "source_ref":f"orbit:{x.orbit_id}", "metadata":{"orbit_id":x.orbit_id,
                 "status":x.status, "replace_activation":True}}
                for x in self.orbits() if x.status in {"open", "active", "blocked"} and x.activation >= minimum]

    def add_hypothesis(self, orbit_id: str, hypothesis: str) -> CognitiveOrbit:
        orbit = self.get_orbit(orbit_id)
        if hypothesis.strip() and hypothesis.strip() not in orbit.hypotheses: orbit.hypotheses.append(hypothesis.strip())
        self._update_orbit(orbit); return orbit

    def add_question(self, orbit_id: str, question: str) -> CognitiveOrbit:
        orbit = self.get_orbit(orbit_id)
        if question.strip() and question.strip() not in orbit.unresolved_questions: orbit.unresolved_questions.append(question.strip())
        self._update_orbit(orbit); return orbit

    def record_progress(self, orbit_id: str, note: str, *, kind: str = "progress") -> CognitiveOrbit:
        orbit = self.get_orbit(orbit_id); timestamp = utc_now()
        orbit.progress.append({"timestamp":timestamp, "kind":kind, "note":note.strip()})
        orbit.last_progress_at = timestamp; orbit.activation = clamp(max(orbit.activation, orbit.baseline_activation) + .08)
        self._update_orbit(orbit); return orbit

    def resolve(self, orbit_id: str, note: str = "") -> CognitiveOrbit:
        orbit = self.get_orbit(orbit_id); orbit.status = "resolved"; orbit.activation = 0.0
        if note: orbit.progress.append({"timestamp":utc_now(), "kind":"resolution", "note":note})
        self._update_orbit(orbit); return orbit

    def reopen(self, orbit_id: str) -> CognitiveOrbit:
        orbit = self.get_orbit(orbit_id); orbit.status = "open"; orbit.activation = orbit.baseline_activation
        orbit.last_activated_at = utc_now(); self._update_orbit(orbit); return orbit

    def create_task(self, orbit_id: str, title: str, objective: str, task_type: str, *,
                    priority: float = .5, expected_cost: float = .2, risk_level: str = "low",
                    required_capabilities: list[str] | None = None) -> CognitiveTask | None:
        if task_type not in self.TASK_TYPES: raise ValueError("invalid task type")
        self.get_orbit(orbit_id); data = self._load()
        signature = (orbit_id, task_type, " ".join(objective.casefold().split()))
        for item in self.tasks():
            if item.status in {"pending", "in_progress", "blocked"} and (item.orbit_id, item.task_type, " ".join(item.objective.casefold().split())) == signature:
                return None
        task = CognitiveTask(f"task_{uuid.uuid4().hex}", orbit_id, title.strip(), objective.strip(), task_type,
            priority=clamp(priority), expected_cost=clamp(expected_cost), risk_level=risk_level,
            required_capabilities=list(required_capabilities or []))
        data["tasks"].append(asdict(task))
        data["orbits"] = [{**x, "related_tasks":list(dict.fromkeys(list(x.get("related_tasks") or []) + [task.task_id]))}
                          if x["orbit_id"] == orbit_id else x for x in data["orbits"]]
        self._save(data); return task

    def create_question_candidate(self, orbit_id: str, question: str, *, importance: float,
                                  reason: str) -> QuestionCandidate | None:
        self.get_orbit(orbit_id); data = self._load(); normalized = " ".join(question.casefold().split())
        if any(x["orbit_id"] == orbit_id and " ".join(x["question"].casefold().split()) == normalized for x in data["questions"]):
            return None
        candidate = QuestionCandidate(f"question_{uuid.uuid4().hex}", orbit_id, question.strip(),
                                      clamp(importance), reason.strip())
        data["questions"].append(asdict(candidate)); self._save(data); return candidate

    def run_task_drive(self, *, enabled: bool, minimum_interval_seconds: int,
                       max_tasks: int, priority_threshold: float, timestamp: datetime | None = None) -> list[CognitiveTask]:
        if not enabled: return []
        data = self._load(); current = timestamp or datetime.now(timezone.utc)
        if data.get("last_drive_at"):
            previous = datetime.fromisoformat(data["last_drive_at"])
            if (current - previous).total_seconds() < minimum_interval_seconds: return []
        created: list[CognitiveTask] = []
        ranked = sorted((x for x in self.orbits() if x.status in {"open", "active", "blocked"}),
                        key=lambda x: -(x.importance * .65 + x.activation * .35))
        for orbit in ranked:
            score = clamp(orbit.importance * .65 + orbit.activation * .35)
            if score < priority_threshold or len(created) >= max(0, max_tasks): continue
            task_type = str(orbit.metadata.get("next_action_type") or ("ask_user" if orbit.status == "blocked" and orbit.unresolved_questions else "inspect"))
            if task_type not in self.TASK_TYPES:
                task_type = "inspect"
            objective = str(orbit.metadata.get("next_action") or
                            (orbit.unresolved_questions[0] if task_type == "ask_user" and orbit.unresolved_questions else f"Inspect current state of {orbit.title}"))
            task = self.create_task(orbit.orbit_id, f"Next action: {orbit.title}", objective, task_type,
                                    priority=score, expected_cost=clamp(orbit.metadata.get("expected_cost", .2)),
                                    risk_level=str(orbit.metadata.get("risk_level", "low")))
            if task:
                created.append(task)
                if task_type == "ask_user":
                    self.create_question_candidate(orbit.orbit_id, objective, importance=orbit.importance,
                                                   reason="Orbit is blocked on user information")
        data = self._load(); data["last_drive_at"] = current.isoformat(timespec="seconds"); self._save(data)
        return created
