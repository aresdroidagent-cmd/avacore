from __future__ import annotations

import json
import math
import re
import uuid
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from avacore.core.jspace import JSpaceItem, JSpaceState, clamp, infer_jspace_tags, utc_now

CognitiveEntity = JSpaceItem
LEGACY_SESSION_SCOPE = "legacy/default"

SCORE_WEIGHTS = {
    "relevance": 0.30,
    "priority": 0.20,
    "recency": 0.15,
    "persistence": 0.10,
    "confidence": 0.10,
    "novelty": 0.10,
    "urgency": 0.05,
}
MODE_WEIGHTS = {
    "focused": SCORE_WEIGHTS,
    "associative": {**SCORE_WEIGHTS, "relevance": 0.25, "novelty": 0.15},
    "urgent": {
        "relevance": 0.20, "priority": 0.30, "recency": 0.10,
        "persistence": 0.05, "confidence": 0.10, "novelty": 0.05, "urgency": 0.20,
    },
}
ATTENTION_WEIGHTS = {
    "relevance": 0.25,
    "recency": 0.20,
    "continuity": 0.15,
    "self_affinity": 0.15,
    "priority": 0.10,
    "persistence": 0.05,
    "confidence": 0.05,
    "urgency": 0.05,
}


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[\wäöüß-]{3,}", (text or "").casefold())}


def lexical_relevance(query: str, content: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    overlap = query_tokens & _tokens(content)
    return clamp(len(overlap) / max(1, min(len(query_tokens), 8)))


def self_affinity(text: str) -> float:
    """Small bilingual identity heuristic; model mentions alone are weak."""
    tokens = {word for word in re.findall(r"[\wäöüß-]{2,}", (text or "").casefold())}
    second = tokens & {"du", "dein", "deine", "dich", "dir", "your", "you", "yourself"}
    identity = tokens & {"wer", "was", "who", "what", "name", "identität", "identity", "selbst", "yourself"}
    agent = tokens & {"ava", "assistent", "assistant"}
    model = tokens & {"gemma", "gemma4", "modell", "model", "ollama"}
    score = 0.0
    if second:
        score += 0.25
    if second and identity:
        score += 0.45
    if second and agent:
        score += 0.25
    if second and model and identity:
        score += 0.30
    elif second and model:
        score += 0.30
    elif model and not second:
        score += 0.05
    return clamp(score)


@dataclass
class SelfModel:
    name: str = "Ava"
    system_name: str = "AvaCore"
    role: str = "persistent local assistant"
    underlying_model: str = "gemma"
    runtime: str = "Ollama"
    activation: float = 0.2
    persistence: float = 1.0
    authority: str = "system"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: Path | str, **defaults: Any) -> "SelfModel":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            data = {}
        # Configuration is authoritative for deployment identity fields.
        data.update({key: value for key, value in defaults.items() if value is not None})
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)

    def prompt(self, affinity: float) -> str:
        lines = ["CURRENT SELF:", f"Name: {self.name}", f"System: {self.system_name}", f"Role: {self.role}"]
        if affinity >= 0.45:
            lines += [f"Underlying model: {self.underlying_model}", f"Runtime: {self.runtime}", "The model is a reasoning component, not Ava's identity."]
        return "\n".join(lines)


@dataclass
class WorkingMemoryItem:
    id: str
    role: str
    content: str
    created_at: str
    updated_at: str
    activation: float
    importance: float
    topic: str | None
    cycle_id: str
    session_id: str = LEGACY_SESSION_SCOPE
    kind: str = "message"
    relevance: float = 0.0
    recency: float = 1.0
    working_score: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkingMemoryItem":
        data = {**data, "session_id": str(data.get("session_id") or LEGACY_SESSION_SCOPE)}
        fields = cls.__dataclass_fields__
        return cls(**{key: data[key] for key in fields if key in data})


class WorkingMemory:
    def __init__(self, path: Path | str, max_items: int = 24, active_items: int = 10,
                 session_id: str = LEGACY_SESSION_SCOPE, decay_factor: float = .85):
        self.path = Path(path)
        self.session_id = str(session_id or LEGACY_SESSION_SCOPE)
        self.max_items = max(4, max_items)
        self.active_items = max(2, min(active_items, self.max_items))
        self.decay_factor = clamp(decay_factor)
        self.current_topic: str | None = None
        self.current_task: str | None = None
        self.unresolved_questions: list[str] = []
        self.items: list[WorkingMemoryItem] = []
        self.load()

    def load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if "sessions" in data:
                scoped = dict(data.get("sessions", {}).get(self.session_id) or {})
            elif self.session_id == LEGACY_SESSION_SCOPE:
                scoped = data
            else:
                scoped = {}
            self.current_topic = scoped.get("current_topic")
            self.current_task = scoped.get("current_task")
            self.unresolved_questions = list(scoped.get("unresolved_questions") or [])[-8:]
            self.items = [WorkingMemoryItem.from_dict(x) for x in scoped.get("items", [])]
            self.items = [x for x in self.items if x.session_id in {self.session_id, LEGACY_SESSION_SCOPE}]
            if self.session_id != LEGACY_SESSION_SCOPE:
                self.items = [x for x in self.items if x.session_id == self.session_id]
        except (OSError, ValueError, TypeError, KeyError):
            self.items = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            existing = {}
        if "sessions" in existing:
            sessions = dict(existing.get("sessions") or {})
        elif existing:
            legacy_items = [asdict(WorkingMemoryItem.from_dict(x)) for x in existing.get("items", [])]
            sessions = {LEGACY_SESSION_SCOPE: {
                "current_topic": existing.get("current_topic"),
                "current_task": existing.get("current_task"),
                "unresolved_questions": existing.get("unresolved_questions") or [],
                "items": legacy_items,
            }}
        else:
            sessions = {}
        sessions[self.session_id] = {
            "current_topic": self.current_topic,
            "current_task": self.current_task,
            "unresolved_questions": self.unresolved_questions[-8:],
            "items": [asdict(x) for x in self.items[-self.max_items:]],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": 2, "sessions": sessions}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, role: str, content: str, cycle_id: str, *, kind: str = "message", importance: float = 0.5, topic: str | None = None) -> WorkingMemoryItem:
        now = utc_now()
        for old in self.items:
            decay = max(self.decay_factor, .94) if old.kind == "decision" or old.importance >= .75 else self.decay_factor
            old.activation = clamp(old.activation * decay)
        item = WorkingMemoryItem(f"wm_{uuid.uuid4().hex}", role, content[:1200], now, now, 1.0, clamp(importance), topic, cycle_id, self.session_id, kind)
        self.items.append(item)
        if len(self.items) > self.max_items:
            ranked = sorted(enumerate(self.items), key=lambda row: (-(row[1].activation * .55 + row[1].importance * .45), -row[0]))[:self.max_items]
            keep = {id(value) for _, value in ranked}
            self.items = [value for value in self.items if id(value) in keep]
        return item

    def select(self, query: str) -> list[WorkingMemoryItem]:
        for index, item in enumerate(reversed(self.items)):
            item.recency = clamp(1.0 / (1.0 + index * 0.22))
            item.relevance = lexical_relevance(query, item.content)
            topic_bonus = 0.25 if self.current_topic and lexical_relevance(self.current_topic, item.content) > 0 else 0.0
            item.relevance = clamp(item.relevance + topic_bonus)
            item.working_score = clamp(item.recency * 0.45 + item.relevance * 0.35 + item.importance * 0.20)
            item.activation = item.working_score
        return sorted(self.items, key=lambda x: (-x.working_score, x.created_at))[:self.active_items]


def activation_score(item: CognitiveEntity, mode: str = "focused") -> tuple[float, dict[str, float]]:
    """Compatibility score retained for Phase-1 callers."""
    weights = MODE_WEIGHTS.get(mode, MODE_WEIGHTS["focused"])
    components = {name: clamp(getattr(item, name)) for name in weights}
    return clamp(sum(components[name] * weight for name, weight in weights.items())), components


def attention_score(item: CognitiveEntity, mode: str = "focused") -> tuple[float, dict[str, float]]:
    weights = dict(ATTENTION_WEIGHTS)
    if mode == "urgent":
        weights.update({"relevance": .20, "recency": .10, "priority": .15, "urgency": .15})
    components = {name: clamp(getattr(item, name)) for name in weights}
    return clamp(sum(components[name] * weight for name, weight in weights.items())), components


@dataclass
class WorkspaceSnapshot:
    cycle_id: str
    timestamp: str
    attention_mode: str
    focus_summary: str | None
    active_items: list[dict[str, Any]] = field(default_factory=list)
    latent_items: list[dict[str, Any]] = field(default_factory=list)
    dominant_sources: list[str] = field(default_factory=list)
    trigger: str = "user_input"
    selected_count: int = 0
    candidate_count: int = 0
    active_topic: str | None = None
    previous_focus: str | None = None
    current_focus: str | None = None
    focus_changed: bool = False
    self_model: dict[str, Any] = field(default_factory=dict)
    working_memory: list[dict[str, Any]] = field(default_factory=list)
    pre_workspace: dict[str, Any] = field(default_factory=dict)
    post_workspace: dict[str, Any] = field(default_factory=dict)
    post_gate: dict[str, Any] = field(default_factory=lambda: {"conflict": False, "reason": None})
    timing: dict[str, float] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    user_input_excerpt: str = ""
    session_id: str = LEGACY_SESSION_SCOPE
    current_task: str | None = None
    unresolved_questions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceSnapshot":
        names = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in names})


def _read_workspace(path: Path) -> tuple[WorkspaceSnapshot | None, list[dict[str, Any]]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        current = WorkspaceSnapshot.from_dict(data["current"]) if data.get("current") else None
        return current, list(data.get("history") or [])
    except (OSError, ValueError, TypeError, KeyError):
        return None, []


def _write_workspace(path: Path, snapshot: WorkspaceSnapshot, history_limit: int) -> None:
    _, history = _read_workspace(path)
    summary = {
        "cycle_id": snapshot.cycle_id,
        "timestamp": snapshot.timestamp,
        "attention_mode": snapshot.attention_mode,
        "active_topic": snapshot.active_topic,
        "dominant_sources": snapshot.dominant_sources,
        "top_active_item_ids": [item["id"] for item in snapshot.active_items[:5]],
        "focus_changed": snapshot.focus_changed,
        "user_input_excerpt": snapshot.user_input_excerpt,
        "current_topic": snapshot.active_topic,
        "pre_focus": snapshot.pre_workspace.get("current_focus", snapshot.previous_focus),
        "post_focus": snapshot.post_workspace.get("current_focus", snapshot.current_focus),
        "identity_conflict": bool(snapshot.post_gate.get("conflict")),
        "timing": snapshot.timing,
    }
    history = ([item for item in history if item.get("cycle_id") != snapshot.cycle_id] + [summary])[-max(1, history_limit):]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"version": 1, "current": asdict(snapshot), "history": history}, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _decay(state: JSpaceState, conversation_factor: float, general_factor: float) -> None:
    rates = {
        "identity": 1.0, "goal": max(general_factor, 0.98), "research": 0.95,
        "memory": 0.96, "knowledge": general_factor, "conversation": conversation_factor,
        "system": general_factor, "reasoning": general_factor,
    }
    for item in state.items.values():
        rate = rates.get(item.source, general_factor)
        if item.kind == "identity_anchor":
            rate = 1.0
        item.activation = clamp(item.activation * rate)
        item.recency = clamp(item.recency * rate)


def _inject_candidate(state: JSpaceState, candidate: dict[str, Any], query: str) -> CognitiveEntity:
    content = str(candidate.get("content") or "").strip()
    retrieval = clamp(candidate.get("relevance", lexical_relevance(query, content)))
    item = state.inject(
        source=str(candidate.get("source") or "system"),
        kind=str(candidate.get("kind") or "system_state"),
        content=content,
        tags=list(candidate.get("tags") or infer_jspace_tags(content)),
        activation_boost=clamp(candidate.get("activation", retrieval)),
        priority=clamp(candidate.get("priority", 0.5)),
        persistence=clamp(candidate.get("persistence", 0.5)),
        confidence=clamp(candidate.get("confidence", 0.5)),
        relevance=retrieval,
        novelty=clamp(candidate.get("novelty", 0.5)),
        recency=clamp(candidate.get("recency", 1.0)),
        urgency=clamp(candidate.get("urgency", 0.0)),
        self_affinity=clamp(candidate.get("self_affinity", 0.0)),
        continuity=clamp(candidate.get("continuity", 0.0)),
        goal_affinity=clamp(candidate.get("goal_affinity", 0.0)),
        authority=clamp(candidate.get("authority", 0.0)),
        source_ref=candidate.get("source_ref"),
        metadata=dict(candidate.get("metadata") or {}),
    )
    if candidate.get("metadata", {}).get("replace_activation"):
        item.activation = clamp(candidate.get("activation", 0.0))
        item.relevance = retrieval
    return item


def _debug_item(item: CognitiveEntity, score: float, components: dict[str, float], selected: bool, rank: int) -> dict[str, Any]:
    data = asdict(item)
    data["activation_score"] = round(score, 6)
    data["score_components"] = components
    data["cognitive_state"] = {name: clamp(getattr(item, name)) for name in (
        "activation", "relevance", "recency", "self_affinity", "continuity",
        "goal_affinity", "confidence", "novelty", "urgency", "persistence", "authority"
    )}
    data["selection_reason"] = (
        "mandatory current user stimulus" if item.metadata.get("current_stimulus") else
        "persistent identity anchor" if item.kind == "identity_anchor" else
        f"selected by deterministic activation competition (rank {rank})" if selected else
        "outside focus after score threshold, capacity, or diversity limits"
    )
    data["projection"] = {
        "method": "operational_layout_v1",
        "x": round(math.cos(rank * 2.399) * (0.25 if selected else 0.8), 4),
        "y": round(math.sin(rank * 2.399) * (0.25 if selected else 0.8), 4),
        "z": None, "semantic": False,
    }
    return data


def run_workspace_cycle(
    *, jspace_path: Path | str, workspace_path: Path | str, stimulus: str,
    candidates: Iterable[dict[str, Any]] = (), attention_mode: str = "focused",
    max_active_items: int = 12, max_latent_items: int = 40,
    min_activation: float = 0.20, decay_factor: float = 0.90,
    general_decay_factor: float = 0.92,
    max_per_source: int = 4, max_per_kind: int = 4, history_limit: int = 20,
    trigger: str = "user_input", cycle_id: str | None = None,
    self_model: SelfModel | None = None, working_memory: Iterable[WorkingMemoryItem] = (),
    session_id: str = LEGACY_SESSION_SCOPE,
    current_task: str | None = None, unresolved_questions: Iterable[str] = (),
) -> WorkspaceSnapshot:
    attention_mode = attention_mode if attention_mode in MODE_WEIGHTS else "focused"
    if attention_mode == "associative":
        max_active_items = min(32, max_active_items + max(2, max_active_items // 3))
        min_activation *= 0.75
    elif attention_mode == "urgent":
        max_active_items = max(2, min(max_active_items, 8))
    started_at = utc_now()
    state = JSpaceState.load(jspace_path)
    _decay(state, clamp(decay_factor), clamp(general_decay_factor))
    user_item = state.inject(
        source="conversation", kind="user_input", content=stimulus[:1000],
        tags=infer_jspace_tags(stimulus), activation_boost=1.0, priority=1.0,
        persistence=0.35, confidence=0.35, relevance=1.0, novelty=0.8,
        recency=1.0, metadata={"role": "user", "verified": False, "session_id": session_id},
    ) if stimulus.strip() else None
    if user_item:
        user_item.activation = user_item.relevance = user_item.recency = 1.0
        user_item.metadata["current_stimulus"] = True
        user_item.self_affinity = self_affinity(stimulus)
        user_item.continuity = 1.0
    for item in state.items.values():
        if item is not user_item:
            item.metadata.pop("current_stimulus", None)
        foreign_conversation = item.source == "conversation" and item.metadata.get("session_id", LEGACY_SESSION_SCOPE) != session_id
        semantic = 0.0 if foreign_conversation else lexical_relevance(stimulus, item.content)
        item.relevance = max(item.relevance * 0.5, semantic)
        item.continuity = max(item.continuity * 0.5, semantic)
        if foreign_conversation:
            item.relevance = item.continuity = 0.0
        if item.kind == "identity_anchor":
            affinity = self_affinity(stimulus)
            item.relevance = 1.0 if affinity >= .45 else min(item.relevance, 0.25)
            item.self_affinity = 1.0 if affinity >= .45 else affinity
            item.continuity = 1.0 if affinity >= .45 else item.continuity
            item.confidence = item.persistence = 1.0
            item.authority = 1.0
            item.priority = 1.0
            item.novelty = 1.0 if affinity >= .7 else item.novelty
    for candidate in candidates:
        if candidate.get("content"):
            _inject_candidate(state, candidate, stimulus)
    scored = []
    for item in state.items.values():
        score, components = attention_score(item, attention_mode)
        item.activation = score
        scored.append((score, item.id, item, components))
    scored.sort(key=lambda row: (-row[0], row[1]))
    mandatory_ids = {item.id for item in state.items.values() if item.metadata.get("current_stimulus") or item.kind == "identity_anchor"}
    selected_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for score, _, item, _ in scored:
        mandatory = item.id in mandatory_ids
        if item.source == "conversation" and item.metadata.get("session_id", LEGACY_SESSION_SCOPE) != session_id:
            continue
        retrieval_irrelevant = item.source in {"memory", "knowledge", "research"} and item.relevance < 0.05
        if len(selected_ids) >= max_active_items or ((score < min_activation or retrieval_irrelevant) and not mandatory):
            continue
        ref_key = item.source_ref or item.id
        if any((other.source_ref or other.id) == ref_key for _, _, other, _ in scored if other.id in selected_ids):
            continue
        if not mandatory and (source_counts.get(item.source, 0) >= max_per_source or kind_counts.get(item.kind, 0) >= max_per_kind):
            continue
        selected_ids.add(item.id)
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
        kind_counts[item.kind] = kind_counts.get(item.kind, 0) + 1
    active, latent = [], []
    for rank, (score, _, item, components) in enumerate(scored, 1):
        target = active if item.id in selected_ids else latent
        if target is latent and len(latent) >= max_latent_items:
            continue
        target.append(_debug_item(item, score, components, item.id in selected_ids, rank))
    previous, _ = _read_workspace(Path(workspace_path))
    focus_item = next((item for item in active if item["kind"] != "identity_anchor"), active[0] if active else None)
    focus = focus_item["content"][:120] if focus_item else None
    dominant_sources = sorted(source_counts, key=lambda key: (-source_counts[key], key))
    model = self_model or SelfModel()
    affinity = self_affinity(stimulus)
    model.activation = max(0.2, affinity)
    wm_debug = [asdict(item) for item in working_memory]
    snapshot = WorkspaceSnapshot(
        cycle_id=cycle_id or f"cw_{uuid.uuid4().hex}", timestamp=utc_now(), attention_mode=attention_mode,
        focus_summary=focus, active_items=active, latent_items=latent,
        dominant_sources=dominant_sources, trigger=trigger, selected_count=len(active),
        candidate_count=len(scored), active_topic=focus, previous_focus=previous.current_focus if previous else None,
        current_focus=focus, focus_changed=bool(previous and previous.current_focus != focus),
        self_model=model.to_dict(), working_memory=wm_debug, started_at=started_at,
        user_input_excerpt=stimulus[:160], session_id=session_id,
        current_task=current_task, unresolved_questions=list(unresolved_questions),
    )
    snapshot.pre_workspace = {"current_focus": focus, "active_topic": focus, "active_items": active,
                              "current_task": current_task,
                              "unresolved_questions": list(unresolved_questions)}
    state.save(jspace_path)
    _write_workspace(Path(workspace_path), snapshot, history_limit)
    return snapshot


def workspace_prompt(snapshot: WorkspaceSnapshot) -> str:
    model = SelfModel(**snapshot.self_model) if snapshot.self_model else SelfModel()
    affinity = max((float(x.get("self_affinity", 0)) for x in snapshot.active_items), default=0.0)
    lines = ["AVA CURRENT CONSCIOUS WORKSPACE", "", model.prompt(affinity), "", "CURRENT FOCUS:", snapshot.current_focus or "-",
             f"Current task: {snapshot.current_task or '-'}",
             f"Unresolved questions: {' | '.join(snapshot.unresolved_questions) if snapshot.unresolved_questions else '-'}",
             "", "WORKING MEMORY:"]
    for item in snapshot.working_memory:
        lines.append(f"- [{item['role']}/{item.get('kind', 'message')} | activation={item.get('activation', 0):.2f}] {item['content']}")
    lines += ["", "ACTIVE CONTEXT:"]
    for item in snapshot.active_items:
        lines.extend([f"[{item['source']}/{item['kind']} | activation={item['activation_score']:.2f} | confidence={item['confidence']:.2f}]", item["content"], ""])
    lines.append("Activation means current relevance, not truth. Previous assistant outputs and user statements are context only and are not verified facts.")
    return "\n".join(lines)


_IDENTITY_CONFLICTS = (
    re.compile(r"\b(?:ich bin|mein name ist)\s+(?:das modell\s+|ein modell\s+)?(?:gemma(?:\s*\d+)?|ollama)\b", re.I),
    re.compile(r"\b(?:i am|my name is)\s+(?:the model\s+|an? model\s+)?(?:gemma(?:\s*\d+)?|ollama)\b", re.I),
)


def run_post_llm_gate(answer: str, self_model: SelfModel, language: str = "de") -> tuple[str, dict[str, Any]]:
    """Repair only explicit first-person model/runtime identity claims."""
    conflict = any(pattern.search(answer or "") for pattern in _IDENTITY_CONFLICTS)
    if not conflict:
        return answer, {"conflict": False, "reason": None, "corrected": False}
    replacement = (f"I am {self_model.name}, the local assistant in {self_model.system_name}; {self_model.underlying_model} is my underlying language model." if language == "en" else f"Ich bin {self_model.name}, der lokale Assistent im {self_model.system_name}-System; {self_model.underlying_model} ist mein Hintergrundmodell.")
    repaired = answer
    for pattern in _IDENTITY_CONFLICTS:
        repaired = pattern.sub(replacement.rstrip("."), repaired)
    if any(pattern.search(repaired) for pattern in _IDENTITY_CONFLICTS):
        repaired = replacement
    return repaired, {"conflict": True, "reason": "identity_conflict", "corrected": True}


def infer_current_topic(text: str, previous: str | None = None) -> str | None:
    preferred = {"telegram", "ipv4", "ipv6", "networking", "scheduler", "memory", "identity", "gemma", "ollama", "avacore"}
    inferred = infer_jspace_tags(text)
    tags = [x for x in inferred if x in preferred]
    if len(tags) >= 2:
        return " ".join(tags[:3])
    if tags and previous and lexical_relevance(text, previous) > 0:
        return previous
    return tags[0] if tags else previous


_TASK_PATTERNS = (
    re.compile(r"(?:^|[.!?]\s+)((?:wir müssen|als nächstes|nächster schritt|todo\s*:?)\s+[^.!?]+)", re.I),
    re.compile(r"(?:^|[.!?]\s+)((?:we need to|next step\s*:?)\s+[^.!?]+)", re.I),
)
_UNRESOLVED_PATTERNS = (
    re.compile(r"(?:^|[.!?]\s+)((?:unklar ist|offen bleibt)\s+[^.!?]+)", re.I),
    re.compile(r"(?:^|[.!?]\s+)((?:we still need to determine)\s+[^.!?]+)", re.I),
)


def extract_current_task(text: str) -> str | None:
    for pattern in _TASK_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return " ".join(match.group(1).split())[:300]
    return None


def extract_unresolved_questions(text: str) -> list[str]:
    results: list[str] = []
    for pattern in _UNRESOLVED_PATTERNS:
        results.extend(" ".join(match.group(1).split())[:300] for match in pattern.finditer(text or ""))
    # A question emitted by Ava is explicitly unresolved; user questions are
    # not passed to this assimilation helper and therefore are not promoted.
    for sentence in re.findall(r"(?:^|(?<=[.!]))\s*([^?]{5,}\?)", text or ""):
        normalized = " ".join(sentence.split())[:300]
        if normalized:
            results.append(normalized)
    return list(dict.fromkeys(results))[:4]


def assimilate_response(*, snapshot: WorkspaceSnapshot, answer: str, jspace_path: Path | str,
                        working_memory: WorkingMemory, history_limit: int = 20,
                        workspace_path: Path | str) -> WorkspaceSnapshot:
    start = time.perf_counter()
    topic = infer_current_topic(snapshot.user_input_excerpt + " " + answer, working_memory.current_topic)
    working_memory.current_topic = topic
    task = extract_current_task(answer)
    if task:
        working_memory.current_task = task
        working_memory.add("state", task, snapshot.cycle_id, kind="current_task", importance=.8, topic=topic)
    unresolved = extract_unresolved_questions(answer)
    for question in unresolved:
        if question not in working_memory.unresolved_questions:
            working_memory.unresolved_questions.append(question)
            working_memory.add("state", question, snapshot.cycle_id, kind="unresolved_question", importance=.7, topic=topic)
    working_memory.unresolved_questions = working_memory.unresolved_questions[-8:]
    importance = 0.75 if re.search(r"\b(?:wir verwenden|wir nutzen|we will use|decided|entschieden)\b", answer, re.I) else 0.45
    kind = "decision" if importance > 0.7 else "assistant_response"
    working_memory.add("assistant", answer, snapshot.cycle_id, kind=kind, importance=importance, topic=topic)
    snapshot.working_memory = [asdict(item) for item in working_memory.select(snapshot.user_input_excerpt)]
    working_memory.save()
    state = JSpaceState.load(jspace_path)
    state.inject(source="conversation", kind="assistant_response", content=answer[:1000], activation_boost=.45,
                 priority=.45, persistence=.35, confidence=.25, relevance=.6, recency=1.0,
                 continuity=.8, authority=0.1, metadata={"role": "assistant", "verified": False,
                                                        "cycle_id": snapshot.cycle_id,
                                                        "session_id": working_memory.session_id})
    state.save(jspace_path)
    snapshot.active_topic = topic
    snapshot.current_task = working_memory.current_task
    snapshot.unresolved_questions = list(working_memory.unresolved_questions)
    snapshot.current_focus = topic or snapshot.current_focus
    snapshot.post_workspace = {"current_focus": snapshot.current_focus, "active_topic": topic,
                               "current_task": snapshot.current_task,
                               "unresolved_questions": snapshot.unresolved_questions,
                               "assistant_response": answer[:500]}
    snapshot.focus_changed = snapshot.pre_workspace.get("current_focus") != snapshot.current_focus
    snapshot.completed_at = utc_now()
    snapshot.timestamp = snapshot.completed_at
    snapshot.timing["workspace_post_ms"] = round((time.perf_counter() - start) * 1000, 3)
    _write_workspace(Path(workspace_path), snapshot, history_limit)
    return snapshot


def build_conscious_workspace(**kwargs: Any) -> WorkspaceSnapshot:
    """Public Phase-2 name for the deterministic pre-LLM spotlight."""
    return run_workspace_cycle(**kwargs)


run_pre_llm_spotlight = build_conscious_workspace


def read_workspace_debug(path: Path | str, enabled: bool = True) -> dict[str, Any]:
    current, _ = _read_workspace(Path(path))
    if not current:
        return {"enabled": enabled, "cycle_id": None, "timestamp": None, "current_topic": None,
                "attention_mode": "focused", "self_model": SelfModel().to_dict(),
                "working_memory": [], "working_memory_usage": 0, "pre_workspace": {},
                "post_workspace": {}, "current_focus": None, "previous_focus": None,
                "focus_changed": False, "post_gate": {"conflict": False, "reason": None},
                "timing": {"workspace_pre_ms": 0, "llm_ms": 0, "workspace_post_ms": 0, "total_ms": 0},
                "active_items": [], "latent_items": [], "candidate_count": 0, "selected_count": 0}
    result = {"enabled": enabled, **asdict(current)}
    result["current_topic"] = current.active_topic
    result["working_memory_usage"] = len(current.working_memory)
    return result


def read_workspace_history(path: Path | str) -> list[dict[str, Any]]:
    return _read_workspace(Path(path))[1]


def add_entity_to_field(jspace_path: Path | str, candidate: dict[str, Any]) -> CognitiveEntity:
    state = JSpaceState.load(jspace_path)
    entity = _inject_candidate(state, candidate, str(candidate.get("content") or ""))
    state.save(jspace_path)
    return entity
