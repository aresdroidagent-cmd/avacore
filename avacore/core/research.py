from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from avacore.core.jspace import clamp
from avacore.core.orbits import CognitiveOrbit, OrbitStore, QuestionCandidate


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SUPPORTED_SIGNAL_TYPES = {
    "QUESTION_CANDIDATE", "OPEN_ORBIT", "REPEATED_UNCERTAINTY",
    "CONTRADICTION", "TASK_FAILURE", "SELF_OBSERVATION",
}


@dataclass(frozen=True)
class ResearchSignal:
    signal_id: str
    signal_type: str
    topic: str
    created_at: str = field(default_factory=utc_now)
    source_event_ids: list[str] = field(default_factory=list)
    source_orbit_ids: list[str] = field(default_factory=list)
    source_question_candidate_id: str | None = None
    question_text: str | None = None
    importance: float = .5
    uncertainty: float = .5
    recurrence: float = .0
    contradiction: float = .0
    novelty: float = .5
    expected_information_gain: float | None = None
    estimated_cost: float = .2


@dataclass(frozen=True)
class ResearchDriveConfig:
    enabled: bool = False
    threshold: float = .65
    max_open_questions: int = 20
    max_new_per_session: int = 3
    dedupe_window_seconds: int = 86400
    weights: dict[str, float] = field(default_factory=lambda: {
        "uncertainty":.25, "recurrence":.20, "contradiction":.20,
        "importance":.20, "novelty":.10, "expected_information_gain":.05,
    })
    cost_penalty: float = .25


@dataclass
class ResearchQuestion:
    question_id: str
    text: str
    origin: str
    topic: str
    fingerprint: str
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    source_event_ids: list[str] = field(default_factory=list)
    source_orbit_ids: list[str] = field(default_factory=list)
    source_question_candidate_ids: list[str] = field(default_factory=list)
    importance: float = .5
    uncertainty: float = .5
    recurrence: float = .0
    contradiction: float = .0
    novelty: float = .5
    expected_information_gain: float = .25
    estimated_cost: float = .2
    priority: float = .5
    status: str = "NEW"
    hypothesis_count: int = 0
    evidence_count: int = 0
    confidence: float = .0
    reactivation_count: int = 0
    score_components: dict[str, float] = field(default_factory=dict)
    threshold: float = .65
    trigger: str = "OPEN_ORBIT"
    last_dedupe_decision: str = "created"
    observed_origins: list[str] = field(default_factory=list)


@dataclass
class ResearchOrbit:
    research_question_id: str
    cognitive_orbit_id: str
    activation: float
    importance: float
    uncertainty: float
    recurrence: float
    evidence_strength: float = .0
    last_activation: str = field(default_factory=utc_now)
    research_status: str = "NEW"


class ResearchMemory:
    """Separate persistent namespace for unresolved research, never confirmed knowledge."""

    METRICS = ("signals_seen", "questions_created", "questions_reactivated",
        "questions_deduplicated", "suppressed_threshold", "suppressed_limits",
        "suppressed_disabled", "suppressed_cooldown")

    def __init__(self, path: Path | str, *, signal_history_limit: int = 200):
        self.path = Path(path)
        self.signal_history_limit = max(1, signal_history_limit)

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            data = {}
        metrics = {key:int((data.get("metrics") or {}).get(key, 0)) for key in self.METRICS}
        return {"version":1, "questions":list(data.get("questions") or []),
                "orbits":list(data.get("orbits") or []),
                "signals":list(data.get("signals") or []), "metrics":metrics,
                "session_counts":dict(data.get("session_counts") or {})}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def questions(self, limit: int = 100) -> list[ResearchQuestion]:
        items = [ResearchQuestion(**item) for item in self._load()["questions"]]
        bounded = max(0, min(limit, 500))
        return items[-bounded:] if bounded else []

    def orbits(self, limit: int = 100) -> list[ResearchOrbit]:
        items = [ResearchOrbit(**item) for item in self._load()["orbits"]]
        bounded = max(0, min(limit, 500))
        return items[-bounded:] if bounded else []

    def record_signal(self, signal: ResearchSignal) -> None:
        data = self._load(); data["metrics"]["signals_seen"] += 1
        data["signals"] = (data["signals"] + [asdict(signal)])[-self.signal_history_limit:]
        self._save(data)

    def signal_occurrences(self, signal: ResearchSignal) -> int:
        topic = normalize_topic(signal.topic)
        orbit = signal.source_orbit_ids[0] if signal.source_orbit_ids else ""
        return sum(normalize_topic(item.get("topic", "")) == topic and
                   item.get("signal_type") == signal.signal_type and
                   ((item.get("source_orbit_ids") or [""])[0] == orbit)
                   for item in self._load()["signals"])

    def increment(self, metric: str) -> None:
        data = self._load()
        if metric not in data["metrics"]: raise ValueError("unknown research metric")
        data["metrics"][metric] += 1; self._save(data)

    def session_count(self, session_id: str) -> int:
        return int(self._load()["session_counts"].get(session_id, 0))

    def save_question(self, question: ResearchQuestion, orbit: ResearchOrbit,
                      *, new: bool, session_id: str) -> None:
        data = self._load()
        found = any(item["question_id"] == question.question_id for item in data["questions"])
        data["questions"] = [asdict(question) if item["question_id"] == question.question_id else item
                             for item in data["questions"]]
        if not found: data["questions"].append(asdict(question))
        linked = any(item["research_question_id"] == question.question_id for item in data["orbits"])
        data["orbits"] = [asdict(orbit) if item["research_question_id"] == question.question_id else item
                          for item in data["orbits"]]
        if not linked: data["orbits"].append(asdict(orbit))
        if new:
            data["session_counts"][session_id] = int(data["session_counts"].get(session_id, 0)) + 1
            data["metrics"]["questions_created"] += 1
            if len(data["session_counts"]) > 100:
                data["session_counts"] = dict(list(data["session_counts"].items())[-100:])
        self._save(data)

    def debug(self) -> dict[str, Any]:
        data = self._load(); open_statuses = {"NEW", "ACTIVE", "DORMANT"}
        return {**data["metrics"],
                "open_questions":sum(item.get("status") in open_statuses for item in data["questions"]),
                "active_research_orbits":sum(item.get("research_status") in open_statuses
                                               for item in data["orbits"])}


class ResearchDrive:
    def __init__(self, config: ResearchDriveConfig): self.config = config

    def score(self, signal: ResearchSignal) -> tuple[float, dict[str, float]]:
        values = {"uncertainty":clamp(signal.uncertainty), "recurrence":clamp(signal.recurrence),
            "contradiction":clamp(signal.contradiction), "importance":clamp(signal.importance),
            "novelty":clamp(signal.novelty),
            "expected_information_gain":clamp(signal.expected_information_gain
                if signal.expected_information_gain is not None
                else clamp(signal.uncertainty) * clamp(signal.importance))}
        base = sum(clamp(self.config.weights.get(key, 0)) * value for key, value in values.items())
        priority = clamp(base * (1 - clamp(signal.estimated_cost) * clamp(self.config.cost_penalty)))
        return priority, values

    def accepts(self, signal: ResearchSignal) -> tuple[bool, float, dict[str, float]]:
        priority, components = self.score(signal)
        return self.config.enabled and priority >= clamp(self.config.threshold), priority, components


def normalize_topic(topic: str) -> str:
    return " ".join(re.findall(r"[\wäöüß-]+", topic.casefold()))[:240]


def research_fingerprint(signal: ResearchSignal) -> str:
    orbit_identity = ",".join(sorted(set(signal.source_orbit_ids))) or "topic-only"
    raw = f"{normalize_topic(signal.topic)}|{orbit_identity}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def deterministic_question(signal: ResearchSignal) -> str:
    if signal.question_text: return " ".join(signal.question_text.split())[:600]
    topic = " ".join(signal.topic.split())[:240]
    templates = {"REPEATED_UNCERTAINTY":f"Why does uncertainty about {topic} recur across interactions?",
        "CONTRADICTION":f"What explains the unresolved contradiction concerning {topic}?",
        "OPEN_ORBIT":f"Why has the cognitive orbit concerning {topic} remained unresolved?",
        "TASK_FAILURE":f"What remains unresolved after the failed task concerning {topic}?",
        "SELF_OBSERVATION":f"What should be understood about the observed system behavior concerning {topic}?"}
    return templates.get(signal.signal_type, f"What remains unresolved concerning {topic}?")


class ResearchService:
    def __init__(self, memory: ResearchMemory, orbit_store: OrbitStore,
                 config: ResearchDriveConfig, *, now: Callable[[], str] = utc_now):
        self.memory, self.orbit_store = memory, orbit_store
        self.config, self.drive, self.now = config, ResearchDrive(config), now

    def evaluate(self, signal: ResearchSignal, *, session_id: str,
                 orbit_already_reactivated: bool = False) -> dict[str, Any]:
        if signal.signal_type not in SUPPORTED_SIGNAL_TYPES: raise ValueError("unsupported research signal")
        self.memory.record_signal(signal)
        occurrences = self.memory.signal_occurrences(signal)
        if occurrences > 1:
            signal = replace(signal, recurrence=clamp(max(
                signal.recurrence, min(1.0, (occurrences - 1) * .2))))
        accepted, priority, components = self.drive.accepts(signal)
        if not self.config.enabled:
            self.memory.increment("suppressed_disabled")
            return {"status":"suppressed_disabled", "priority":priority, "components":components}
        if not accepted:
            self.memory.increment("suppressed_threshold")
            return {"status":"suppressed_threshold", "priority":priority, "components":components}
        fingerprint = research_fingerprint(signal); timestamp = self.now()
        existing = next((item for item in self.memory.questions()
                         if item.fingerprint == fingerprint and item.status in {"NEW", "ACTIVE", "DORMANT"}), None)
        if existing:
            existing.recurrence = clamp(max(existing.recurrence, signal.recurrence) + .1)
            existing.updated_at = timestamp; existing.reactivation_count += 1
            existing.source_event_ids = list(dict.fromkeys(existing.source_event_ids + signal.source_event_ids))
            existing.source_orbit_ids = list(dict.fromkeys(existing.source_orbit_ids + signal.source_orbit_ids))
            if signal.source_question_candidate_id:
                existing.source_question_candidate_ids = list(dict.fromkeys(
                    existing.source_question_candidate_ids + [signal.source_question_candidate_id]))
            existing.observed_origins = list(dict.fromkeys(
                existing.observed_origins + [existing.origin, signal.signal_type]))
            link = next(item for item in self.memory.orbits() if item.research_question_id == existing.question_id)
            orbit = (self.orbit_store.get_orbit(link.cognitive_orbit_id)
                     if orbit_already_reactivated else
                     self.orbit_store.activate(link.cognitive_orbit_id,
                         min(.3, .1 + signal.recurrence * .15)))
            link.activation, link.recurrence = orbit.activation, existing.recurrence
            link.last_activation, link.research_status = timestamp, "ACTIVE"; existing.status = "ACTIVE"
            existing.last_dedupe_decision = "reactivated_existing_fingerprint"
            self.memory.save_question(existing, link, new=False, session_id=session_id)
            self.memory.increment("questions_reactivated"); self.memory.increment("questions_deduplicated")
            return {"status":"reactivated", "question":existing, "orbit":link,
                    "priority":priority, "components":components, "deduplicated":True}
        if (self.memory.debug()["open_questions"] >= max(0, self.config.max_open_questions) or
                self.memory.session_count(session_id) >= max(0, self.config.max_new_per_session)):
            self.memory.increment("suppressed_limits")
            return {"status":"suppressed_limits", "priority":priority, "components":components}
        closed = next((item for item in self.memory.questions() if item.fingerprint == fingerprint), None)
        if closed:
            age = datetime.fromisoformat(timestamp) - datetime.fromisoformat(closed.updated_at)
            if age.total_seconds() < max(0, self.config.dedupe_window_seconds):
                self.memory.increment("suppressed_cooldown")
                return {"status":"suppressed_cooldown", "priority":priority, "components":components}
        source_orbit = next((orbit for orbit_id in signal.source_orbit_ids for orbit in self.orbit_store.orbits()
                             if orbit.orbit_id == orbit_id), None)
        cognitive_orbit = source_orbit or self.orbit_store.create_orbit(
            signal.topic, deterministic_question(signal), importance=signal.importance,
            baseline_activation=min(.12, max(.03, priority * .12)), metadata={
                "kind":"research", "research_fingerprint":fingerprint})
        question = ResearchQuestion(f"research_{uuid.uuid4().hex}", deterministic_question(signal),
            signal.signal_type, signal.topic, fingerprint, created_at=timestamp, updated_at=timestamp,
            source_event_ids=list(dict.fromkeys(signal.source_event_ids)),
            source_orbit_ids=list(dict.fromkeys(signal.source_orbit_ids)),
            source_question_candidate_ids=([signal.source_question_candidate_id]
                if signal.source_question_candidate_id else []), importance=clamp(signal.importance),
            uncertainty=clamp(signal.uncertainty), recurrence=clamp(signal.recurrence),
            contradiction=clamp(signal.contradiction), novelty=clamp(signal.novelty),
            expected_information_gain=components["expected_information_gain"],
            estimated_cost=clamp(signal.estimated_cost), priority=priority,
            score_components=components, threshold=clamp(self.config.threshold),
            trigger=signal.signal_type, observed_origins=[signal.signal_type])
        link = ResearchOrbit(question.question_id, cognitive_orbit.orbit_id, cognitive_orbit.activation,
            question.importance, question.uncertainty, question.recurrence, last_activation=timestamp)
        self.memory.save_question(question, link, new=True, session_id=session_id)
        return {"status":"created", "question":question, "orbit":link,
                "priority":priority, "components":components, "deduplicated":False}

    def from_question_candidate(self, candidate: QuestionCandidate,
                                orbit: CognitiveOrbit, *, session_id: str) -> dict[str, Any]:
        return self.evaluate(ResearchSignal(f"signal_{uuid.uuid4().hex}", "QUESTION_CANDIDATE", orbit.title,
            source_orbit_ids=[orbit.orbit_id], source_question_candidate_id=candidate.question_id,
            question_text=candidate.question, importance=candidate.importance,
            uncertainty=clamp(candidate.metadata.get("uncertainty", .8)),
            recurrence=clamp(candidate.metadata.get("recurrence", .4)), novelty=.7,
            estimated_cost=clamp(candidate.metadata.get("estimated_cost", .2))), session_id=session_id)

    def from_open_orbit(self, orbit: CognitiveOrbit, *, event_id: str,
                        session_id: str) -> dict[str, Any]:
        signal_type = str(orbit.metadata.get("research_signal_type", "OPEN_ORBIT")).upper()
        if signal_type not in SUPPORTED_SIGNAL_TYPES: signal_type = "OPEN_ORBIT"
        return self.evaluate(ResearchSignal(f"signal_{uuid.uuid4().hex}", signal_type, orbit.title,
            source_event_ids=[event_id], source_orbit_ids=[orbit.orbit_id],
            question_text=(orbit.unresolved_questions[0] if orbit.unresolved_questions else None),
            importance=orbit.importance, uncertainty=clamp(orbit.metadata.get("uncertainty", .6)),
            recurrence=clamp(orbit.metadata.get("recurrence", .3)),
            contradiction=clamp(orbit.metadata.get("contradiction", 0)),
            novelty=clamp(orbit.metadata.get("novelty", .5)),
            estimated_cost=clamp(orbit.metadata.get("expected_cost", .2))),
            session_id=session_id, orbit_already_reactivated=True)
