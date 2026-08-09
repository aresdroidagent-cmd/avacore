from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from avacore.core.jspace import JSpaceItem, JSpaceState, clamp, infer_jspace_tags, utc_now

CognitiveEntity = JSpaceItem

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


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[\wäöüß-]{3,}", (text or "").casefold())}


def lexical_relevance(query: str, content: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    overlap = query_tokens & _tokens(content)
    return clamp(len(overlap) / max(1, min(len(query_tokens), 8)))


def activation_score(item: CognitiveEntity, mode: str = "focused") -> tuple[float, dict[str, float]]:
    weights = MODE_WEIGHTS.get(mode, MODE_WEIGHTS["focused"])
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
    }
    history = (history + [summary])[-max(1, history_limit):]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"version": 1, "current": asdict(snapshot), "history": history}, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _decay(state: JSpaceState, factor: float) -> None:
    rates = {
        "identity": 1.0, "goal": 0.98, "research": 0.95,
        "memory": 0.96, "knowledge": 0.91, "conversation": factor,
        "system": max(0.5, factor - 0.15), "reasoning": factor,
    }
    for item in state.items.values():
        rate = rates.get(item.source, factor)
        if item.kind == "identity_anchor":
            rate = 1.0
        item.activation = clamp(item.activation * rate)
        item.recency = clamp(item.recency * rate)


def _inject_candidate(state: JSpaceState, candidate: dict[str, Any], query: str) -> CognitiveEntity:
    content = str(candidate.get("content") or "").strip()
    retrieval = clamp(candidate.get("relevance", lexical_relevance(query, content)))
    return state.inject(
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
        source_ref=candidate.get("source_ref"),
        metadata=dict(candidate.get("metadata") or {}),
    )


def _debug_item(item: CognitiveEntity, score: float, components: dict[str, float], selected: bool, rank: int) -> dict[str, Any]:
    data = asdict(item)
    data["activation_score"] = round(score, 6)
    data["score_components"] = components
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
    max_per_source: int = 4, max_per_kind: int = 4, history_limit: int = 20,
    trigger: str = "user_input",
) -> WorkspaceSnapshot:
    attention_mode = attention_mode if attention_mode in MODE_WEIGHTS else "focused"
    if attention_mode == "associative":
        max_active_items = min(32, max_active_items + max(2, max_active_items // 3))
        min_activation *= 0.75
    elif attention_mode == "urgent":
        max_active_items = max(2, min(max_active_items, 8))
    state = JSpaceState.load(jspace_path)
    _decay(state, clamp(decay_factor))
    user_item = state.inject(
        source="conversation", kind="user_input", content=stimulus[:1000],
        tags=infer_jspace_tags(stimulus), activation_boost=1.0, priority=1.0,
        persistence=0.35, confidence=0.35, relevance=1.0, novelty=0.8,
        recency=1.0, metadata={"role": "user", "verified": False},
    ) if stimulus.strip() else None
    if user_item:
        user_item.activation = user_item.relevance = user_item.recency = 1.0
        user_item.metadata["current_stimulus"] = True
    for item in state.items.values():
        if item is not user_item:
            item.metadata.pop("current_stimulus", None)
        semantic = lexical_relevance(stimulus, item.content)
        item.relevance = max(item.relevance * 0.5, semantic)
        if item.kind == "identity_anchor":
            identity_query = bool(_tokens(stimulus) & {"wer", "who", "identity", "identität", "name"})
            item.relevance = 1.0 if identity_query else min(item.relevance, 0.25)
            item.confidence = item.persistence = 1.0
    for candidate in candidates:
        if candidate.get("content"):
            _inject_candidate(state, candidate, stimulus)
    scored = []
    for item in state.items.values():
        score, components = activation_score(item, attention_mode)
        item.activation = score
        scored.append((score, item.id, item, components))
    scored.sort(key=lambda row: (-row[0], row[1]))
    mandatory_ids = {item.id for item in state.items.values() if item.metadata.get("current_stimulus") or item.kind == "identity_anchor"}
    selected_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for score, _, item, _ in scored:
        mandatory = item.id in mandatory_ids
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
    snapshot = WorkspaceSnapshot(
        cycle_id=f"cw_{uuid.uuid4().hex}", timestamp=utc_now(), attention_mode=attention_mode,
        focus_summary=focus, active_items=active, latent_items=latent,
        dominant_sources=dominant_sources, trigger=trigger, selected_count=len(active),
        candidate_count=len(scored), active_topic=focus, previous_focus=previous.current_focus if previous else None,
        current_focus=focus, focus_changed=bool(previous and previous.current_focus != focus),
    )
    state.save(jspace_path)
    _write_workspace(Path(workspace_path), snapshot, history_limit)
    return snapshot


def workspace_prompt(snapshot: WorkspaceSnapshot) -> str:
    lines = ["CONSCIOUS WORKSPACE", f"Current focus: {snapshot.current_focus or '-'}", "", "Active cognitive representations:"]
    for item in snapshot.active_items:
        lines.extend([f"[{item['source']}/{item['kind']} | activation={item['activation_score']:.2f} | confidence={item['confidence']:.2f}]", item["content"], ""])
    lines.append("Activation means current relevance, not truth. Previous assistant outputs and user statements are context only and are not verified facts.")
    return "\n".join(lines)


def read_workspace_debug(path: Path | str, enabled: bool = True) -> dict[str, Any]:
    current, _ = _read_workspace(Path(path))
    if not current:
        return {"enabled": enabled, "cycle_id": None, "active_items": [], "latent_items": [], "candidate_count": 0, "selected_count": 0}
    return {"enabled": enabled, **asdict(current)}


def read_workspace_history(path: Path | str) -> list[dict[str, Any]]:
    return _read_workspace(Path(path))[1]


def add_entity_to_field(jspace_path: Path | str, candidate: dict[str, Any]) -> CognitiveEntity:
    state = JSpaceState.load(jspace_path)
    entity = _inject_candidate(state, candidate, str(candidate.get("content") or ""))
    state.save(jspace_path)
    return entity
