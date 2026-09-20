from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from avacore.core.cognitive_workspace import WorkingMemory, WorkspaceSnapshot
from avacore.core.jspace import clamp
from avacore.core.orbits import (
    CognitiveOrbit,
    OrbitStore,
    QuestionCandidate,
    semantic_signature,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_orbit_topic(value: str) -> str:
    return " ".join(re.findall(r"[\wäöüß-]+", (value or "").casefold()))[:240]


def orbit_deficit_fingerprint(topic: str, stable_source_ids: list[str] | None = None) -> str:
    identity = ",".join(sorted(set(stable_source_ids or []))) or "topic-only"
    raw = f"{normalize_orbit_topic(topic)}|{identity}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


_CONCERN_PATTERNS = (
    re.compile(r"\b(?:ich bin noch nicht überzeugt|das bleibt unklar|das funktioniert noch nicht|"
               r"das ist noch offen|hier haben wir ein problem|das scheint eine schwäche zu sein)\b", re.I),
    re.compile(r"\b(?:i(?:'m| am) not convinced|this remains unclear|this still does not work|"
               r"this is still open|we have a problem here|this seems to be a weakness)\b", re.I),
)


def has_explicit_concern(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in _CONCERN_PATTERNS)


def topic_from_text(text: str) -> str:
    normalized = " ".join((text or "").split()).strip()
    return normalized[:240]


@dataclass(frozen=True)
class OrbitFormationConfig:
    enabled: bool = False
    threshold: float = .70
    max_new_per_session: int = 2
    max_open: int = 20
    cooldown_seconds: int = 3600
    weights: dict[str, float] = field(default_factory=lambda: {
        "importance":.25, "uncertainty":.20, "recurrence":.20,
        "persistence":.15, "explicit_unresolved":.10, "explicit_concern":.10,
    })
    blocked_boost: float = .10


@dataclass(frozen=True)
class OrbitFormationCandidate:
    topic: str
    title: str
    description: str
    source_event_ids: list[str] = field(default_factory=list)
    source_memory_ids: list[str] = field(default_factory=list)
    source_question_candidate_ids: list[str] = field(default_factory=list)
    stable_source_ids: list[str] = field(default_factory=list)
    context_texts: list[str] = field(default_factory=list)
    context_topics: list[str] = field(default_factory=list)
    context_cycle_ids: list[str] = field(default_factory=list)
    related_entities: list[str] = field(default_factory=list)
    importance: float = .5
    uncertainty: float = .5
    recurrence: float = .0
    persistence: float = .5
    novelty: float = .5
    explicit_unresolved: float = .0
    explicit_concern: float = .0
    blocked_state: float = .0
    formation_reason: str = "RECURRENT_TOPIC"
    source: str = "conversation"


class EpistemicSalienceEvaluator:
    METRICS = ("formation_candidates_seen", "orbits_formed", "orbits_reactivated_instead",
               "suppressed_salience", "suppressed_bounds", "suppressed_research_source",
               "suppressed_disabled", "suppressed_cooldown",
               "suppressed_missing_epistemic_anchor")

    def __init__(self, store: OrbitStore, config: OrbitFormationConfig,
                 *, now: Callable[[], str] = utc_now):
        self.store, self.config, self.now = store, config, now

    def score(self, candidate: OrbitFormationCandidate) -> tuple[float, dict[str, float]]:
        components = {name:clamp(getattr(candidate, name)) for name in (
            "importance", "uncertainty", "recurrence", "persistence", "novelty",
            "explicit_unresolved", "explicit_concern", "blocked_state")}
        base = sum(clamp(self.config.weights.get(name, 0)) * components[name]
                   for name in self.config.weights)
        return clamp(base + clamp(self.config.blocked_boost) * components["blocked_state"]), components

    def _state(self) -> dict[str, Any]:
        state = self.store.formation_state()
        metrics = {name:int((state.get("metrics") or {}).get(name, 0)) for name in self.METRICS}
        return {"metrics":metrics, "session_counts":dict(state.get("session_counts") or {}),
                "observations":dict(state.get("observations") or {}),
                "last_candidates":list(state.get("last_candidates") or [])[-50:]}

    def _save(self, state: dict[str, Any]) -> None:
        state["last_candidates"] = state["last_candidates"][-50:]
        if len(state["session_counts"]) > 100:
            state["session_counts"] = dict(list(state["session_counts"].items())[-100:])
        self.store.save_formation_state(state)

    def evaluate(self, candidate: OrbitFormationCandidate, *, session_id: str) -> dict[str, Any]:
        state = self._state(); state["metrics"]["formation_candidates_seen"] += 1
        score, components = self.score(candidate)
        fingerprint = orbit_deficit_fingerprint(candidate.topic, candidate.stable_source_ids)
        timestamp = self.now()
        diagnostic = {"timestamp":timestamp, "fingerprint":fingerprint,
            "reason":candidate.formation_reason, "score":score, "components":components,
            "source_event_ids":candidate.source_event_ids}
        if candidate.source == "research":
            state["metrics"]["suppressed_research_source"] += 1
            diagnostic["status"] = "suppressed_research_source"
            state["last_candidates"].append(diagnostic); self._save(state); return diagnostic
        if not self.config.enabled:
            state["metrics"]["suppressed_disabled"] += 1
            diagnostic["status"] = "suppressed_disabled"
            state["last_candidates"].append(diagnostic); self._save(state); return diagnostic
        existing = next((orbit for orbit in self.store.orbits()
            if orbit.status in {"open", "active", "blocked"} and
            (orbit.orbit_id in candidate.stable_source_ids or
             orbit.metadata.get("formation_fingerprint") == fingerprint or
             (not candidate.stable_source_ids and normalize_orbit_topic(orbit.title) ==
              normalize_orbit_topic(candidate.title)))), None)
        epistemic_anchor = bool(candidate.explicit_unresolved or candidate.explicit_concern or
            candidate.blocked_state or candidate.source_question_candidate_ids or
            (candidate.source_memory_ids and candidate.explicit_unresolved))
        if candidate.formation_reason == "RECURRENT_TOPIC" and not epistemic_anchor and not existing:
            state["metrics"]["suppressed_missing_epistemic_anchor"] += 1
            diagnostic["status"] = "suppressed_missing_epistemic_anchor"
            diagnostic["epistemic_anchor"] = False
            state["last_candidates"].append(diagnostic); self._save(state); return diagnostic
        if score < clamp(self.config.threshold):
            state["metrics"]["suppressed_salience"] += 1
            diagnostic["status"] = "suppressed_salience"
            state["last_candidates"].append(diagnostic); self._save(state); return diagnostic

        if existing:
            last = existing.metadata.get("formation_last_seen")
            previous_sources = set(existing.metadata.get("formation_source_ids") or [])
            if last and candidate.source_event_ids and set(candidate.source_event_ids) <= previous_sources:
                age = datetime.fromisoformat(timestamp) - datetime.fromisoformat(last)
                if age.total_seconds() < max(0, self.config.cooldown_seconds):
                    state["metrics"]["suppressed_cooldown"] += 1
                    diagnostic["status"] = "suppressed_cooldown"
                    state["last_candidates"].append(diagnostic); self._save(state); return diagnostic
            existing.activation = clamp(max(existing.activation, existing.baseline_activation) + .15)
            existing.last_activated_at = timestamp
            existing.metadata["formation_recurrence"] = clamp(
                float(existing.metadata.get("formation_recurrence", 0)) + .1)
            existing.metadata["formation_last_seen"] = timestamp
            existing.metadata["semantic_signature"] = semantic_signature(
                texts=[candidate.title, candidate.description] + candidate.context_texts[:3],
                topics=[candidate.topic] + candidate.context_topics,
                entities=list(dict.fromkeys(existing.related_entities + candidate.related_entities)),
                source_cycle_ids=(list(existing.metadata.get("formation_source_ids") or []) +
                    candidate.source_event_ids + candidate.context_cycle_ids))
            for key, values in (("formation_source_ids", candidate.source_event_ids),
                                ("formation_memory_ids", candidate.source_memory_ids),
                                ("formation_question_candidate_ids", candidate.source_question_candidate_ids)):
                existing.metadata[key] = list(dict.fromkeys(list(existing.metadata.get(key) or []) + values))
            self.store._update_orbit(existing)
            state["metrics"]["orbits_reactivated_instead"] += 1
            diagnostic.update({"status":"reactivated", "orbit_id":existing.orbit_id})
            state["last_candidates"].append(diagnostic); self._save(state); return diagnostic

        open_count = sum(orbit.status in {"open", "active", "blocked"} for orbit in self.store.orbits())
        if (open_count >= max(0, self.config.max_open) or
                int(state["session_counts"].get(session_id, 0)) >= max(0, self.config.max_new_per_session)):
            state["metrics"]["suppressed_bounds"] += 1
            diagnostic["status"] = "suppressed_bounds"
            state["last_candidates"].append(diagnostic); self._save(state); return diagnostic
        metadata = {"origin":"formed_from_workspace", "formation_fingerprint":fingerprint,
            "formation_reason":candidate.formation_reason, "formation_score":score,
            "formation_components":components, "formation_source_ids":candidate.source_event_ids,
            "formation_memory_ids":candidate.source_memory_ids,
            "formation_question_candidate_ids":candidate.source_question_candidate_ids,
            "formation_recurrence":candidate.recurrence, "formation_last_seen":timestamp,
            "semantic_signature":semantic_signature(
                texts=[candidate.title, candidate.description] + candidate.context_texts[:3],
                topics=[candidate.topic] + candidate.context_topics,
                entities=candidate.related_entities,
                source_cycle_ids=candidate.source_event_ids + candidate.context_cycle_ids)}
        orbit = self.store.create_orbit(candidate.title, candidate.description,
            importance=candidate.importance, baseline_activation=min(.15, max(.05, score * .15)),
            related_entities=candidate.related_entities, metadata=metadata)
        state["metrics"]["orbits_formed"] += 1
        state["session_counts"][session_id] = int(state["session_counts"].get(session_id, 0)) + 1
        diagnostic.update({"status":"formed", "orbit_id":orbit.orbit_id})
        state["last_candidates"].append(diagnostic); self._save(state); return diagnostic

    def observe_cycle(self, *, source: str, user_text: str, snapshot: WorkspaceSnapshot,
                      working_memory: WorkingMemory, session_id: str) -> list[dict[str, Any]]:
        candidates: list[OrbitFormationCandidate] = []
        prior_items = [item for item in reversed(working_memory.items)
                       if item.cycle_id != snapshot.cycle_id][:3]
        context_texts = [item.content[:300] for item in prior_items]
        context_topics = [item.topic for item in prior_items if item.topic]
        context_cycle_ids = [item.cycle_id for item in prior_items]
        cycle_unresolved = [item for item in working_memory.items
            if item.cycle_id == snapshot.cycle_id and item.kind == "unresolved_question"]
        for item in cycle_unresolved:
            topic = item.topic or topic_from_text(item.content)
            candidates.append(OrbitFormationCandidate(topic, topic, item.content,
                source_event_ids=[snapshot.cycle_id], source_memory_ids=[item.id],
                context_texts=context_texts, context_topics=context_topics,
                context_cycle_ids=context_cycle_ids,
                importance=1, uncertainty=1, persistence=1, explicit_unresolved=1,
                formation_reason="EXPLICIT_UNRESOLVED", source=source))
        if has_explicit_concern(user_text):
            topic = topic_from_text(user_text)
            candidates.append(OrbitFormationCandidate(topic, topic, topic,
                source_event_ids=[snapshot.cycle_id], importance=1, uncertainty=1,
                context_texts=context_texts, context_topics=context_topics,
                context_cycle_ids=context_cycle_ids,
                persistence=1, explicit_concern=1, formation_reason="EXPLICIT_CONCERN",
                source=source))
        topic = snapshot.active_topic or working_memory.current_topic
        if topic and not candidates:
            state = self._state(); key = orbit_deficit_fingerprint(topic)
            observation = dict(state["observations"].get(key) or {})
            count = int(observation.get("count", 0)) + 1
            state["observations"][key] = {"count":count, "last_seen":self.now()}
            self._save(state)
            if count >= 3:
                recurrence = clamp((count - 1) / 2)
                candidates.append(OrbitFormationCandidate(topic, topic,
                    f"Recurrent unresolved topic: {topic}", source_event_ids=[snapshot.cycle_id],
                    importance=.6, uncertainty=.2, recurrence=recurrence, persistence=.8,
                    formation_reason="RECURRENT_TOPIC", source=source))
        return [self.evaluate(candidate, session_id=session_id) for candidate in candidates]

    def from_question_candidate(self, candidate: QuestionCandidate, orbit: CognitiveOrbit,
                                *, session_id: str) -> dict[str, Any]:
        return self.evaluate(OrbitFormationCandidate(orbit.title, orbit.title, candidate.question,
            source_question_candidate_ids=[candidate.question_id], stable_source_ids=[orbit.orbit_id],
            related_entities=orbit.related_entities,
            importance=1, uncertainty=1, persistence=1, explicit_unresolved=1,
            formation_reason="QUESTION_CANDIDATE"), session_id=session_id)

    def from_blocked_task(self, *, topic: str, description: str, task_id: str,
                          orbit_id: str, session_id: str) -> dict[str, Any]:
        return self.evaluate(OrbitFormationCandidate(topic, topic, description,
            stable_source_ids=[orbit_id], importance=.9, uncertainty=.9, recurrence=.5,
            persistence=1, blocked_state=1, formation_reason="BLOCKED_TASK",
            source_event_ids=[task_id]), session_id=session_id)

    def debug(self) -> dict[str, Any]:
        state = self._state()
        return {"enabled":self.config.enabled, "threshold":self.config.threshold,
                **state["metrics"], "recent_candidates":state["last_candidates"][-20:]}
