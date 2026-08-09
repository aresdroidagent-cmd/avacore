from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def stable_item_id(source: str, kind: str, content: str) -> str:
    normalized = normalize_text(content).lower()
    raw = f"{source}|{kind}|{normalized[:240]}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"js_{digest}"


def infer_jspace_tags(text: str) -> list[str]:
    lowered = (text or "").lower()
    tags: set[str] = set()

    keyword_map = {
        "avacore": "avacore",
        "ava": "ava",
        "jspace": "jspace",
        "bewusstsein": "jspace",
        "workspace": "jspace",
        "notiz": "notes",
        "notes": "notes",
        "mail": "mail",
        "gmail": "mail",
        "kalender": "calendar",
        "briefing": "calendar",
        "kamera": "vision",
        "camera": "vision",
        "bild": "vision",
        "vlm": "vision",
        "roger": "roger",
        "robot": "robotics",
        "robotik": "robotics",
        "opc": "industrial",
        "wincc": "industrial",
        "bacnet": "industrial",
        "podman": "devops",
        "github": "github",
        "python": "programming",
        "debug": "debugging",
        "fehler": "debugging",
        "test": "testing",
        "scheduler": "scheduler",
        "systemd": "scheduler",
    }

    for needle, tag in keyword_map.items():
        if needle in lowered:
            tags.add(tag)

    # Add a few lightweight lexical tags for association.
    words = re.findall(r"[a-zA-ZäöüÄÖÜß0-9_+-]{4,}", lowered)
    stop_words = {
        "bitte",
        "dass",
        "diese",
        "dieser",
        "einen",
        "eine",
        "oder",
        "aber",
        "noch",
        "dann",
        "nicht",
        "soll",
        "sollte",
        "haben",
        "wird",
        "wurde",
        "kann",
        "können",
    }

    for word in words[:12]:
        if word not in stop_words:
            tags.add(word[:32])

    return sorted(tags)[:16]


def is_conflicting_assistant_identity_claim(item: "JSpaceItem") -> bool:
    """Detect only clear, first-person model/runtime identity claims by Ava."""
    if item.source != "conversation" or item.kind != "assistant_response":
        return False

    text = normalize_text(item.content).casefold()
    patterns = (
        r"^ich bin (?:das |ein )?(?:ki[- ]modell )?(?:gemma\d*|ollama)(?:\b|[.!?,])",
        r"^als ki[- ]modell bin ich (?:gemma\d*|ollama)(?:\b|[.!?,])",
        r"^i am (?:the |an? )?(?:ai model )?(?:gemma\d*|ollama)(?:\b|[.!?,])",
        r"^as an ai model,? i am (?:gemma\d*|ollama)(?:\b|[.!?,])",
    )
    return any(re.search(pattern, text) for pattern in patterns)


@dataclass
class JSpaceItem:
    id: str
    source: str
    kind: str
    content: str
    tags: list[str] = field(default_factory=list)
    activation: float = 0.5
    priority: float = 0.5
    persistence: float = 0.5
    confidence: float = 0.5
    relevance: float = 0.5
    novelty: float = 0.5
    recency: float = 1.0
    urgency: float = 0.0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_seen_at: str | None = None
    source_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JSpaceItem":
        kind = str(data.get("kind", "item"))
        if str(data.get("source", "unknown")) == "identity" and kind == "self_anchor":
            kind = "identity_anchor"
        return cls(
            id=str(
                data.get("id")
                or stable_item_id(
                    str(data.get("source", "unknown")),
                    str(data.get("kind", "item")),
                    str(data.get("content", "")),
                )
            ),
            source=str(data.get("source", "unknown")),
            kind=kind,
            content=str(data.get("content", "")),
            tags=list(data.get("tags") or []),
            activation=clamp(float(data.get("activation", 0.5))),
            priority=clamp(float(data.get("priority", 0.5))),
            persistence=clamp(float(data.get("persistence", 0.5))),
            confidence=clamp(float(data.get("confidence", 0.5))),
            relevance=clamp(float(data.get("relevance", data.get("activation", 0.5)))),
            novelty=clamp(float(data.get("novelty", 0.5))),
            recency=clamp(float(data.get("recency", 1.0))),
            urgency=clamp(float(data.get("urgency", 0.0))),
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            last_seen_at=data.get("last_seen_at"),
            source_ref=data.get("source_ref"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class JSpaceState:
    version: int = 1
    focus_mode: str = "balanced"
    updated_at: str = field(default_factory=utc_now)
    items: dict[str, JSpaceItem] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str, focus_mode: str = "balanced") -> "JSpaceState":
        path = Path(path).expanduser()

        if not path.exists():
            state = cls(focus_mode=focus_mode)
            state.seed_core_items()
            return state

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            state = cls(focus_mode=focus_mode)
            state.seed_core_items()
            return state

        state = cls(
            version=int(data.get("version", 1)),
            focus_mode=str(data.get("focus_mode") or focus_mode or "balanced"),
            updated_at=str(data.get("updated_at") or utc_now()),
            items={},
        )

        for raw_item in data.get("items", []):
            item = JSpaceItem.from_dict(raw_item)
            state.items[item.id] = item

        if not state.items:
            state.seed_core_items()

        state.focus_mode = focus_mode or state.focus_mode
        return state

    def save(self, path: Path | str) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        self.updated_at = utc_now()

        payload = {
            "version": self.version,
            "focus_mode": self.focus_mode,
            "updated_at": self.updated_at,
            "items": [asdict(item) for item in self.items.values()],
        }

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def seed_core_items(self) -> None:
        self.inject(
            source="identity",
            kind="identity_anchor",
            content="Ava is Roger Seeberger's local assistant running in AvaCore.",
            tags=["ava", "roger", "identity", "avacore"],
            activation_boost=0.7,
            priority=0.95,
            persistence=0.95,
        )
        self.inject(
            source="operating_rule",
            kind="memory_rule",
            content="Durable long-term memory changes should remain candidate until Roger verifies them.",
            tags=["memory", "safety", "roger", "avacore"],
            activation_boost=0.6,
            priority=0.9,
            persistence=0.9,
        )

    def focus_parameters(self) -> dict[str, float]:
        mode = (self.focus_mode or "balanced").lower()

        if mode == "narrow":
            return {
                "activation_multiplier": 0.95,
                "priority_multiplier": 1.15,
                "persistence_multiplier": 0.90,
            }

        if mode == "wide":
            return {
                "activation_multiplier": 1.05,
                "priority_multiplier": 0.95,
                "persistence_multiplier": 1.05,
            }

        if mode == "watchful":
            return {
                "activation_multiplier": 1.00,
                "priority_multiplier": 1.10,
                "persistence_multiplier": 1.15,
            }

        return {
            "activation_multiplier": 1.0,
            "priority_multiplier": 1.0,
            "persistence_multiplier": 1.0,
        }

    def tick(self, decay: float = 0.92, min_activation: float = 0.05) -> None:
        decay = clamp(decay, 0.1, 1.0)
        min_activation = clamp(min_activation, 0.0, 0.5)

        params = self.focus_parameters()
        now = utc_now()

        remove_ids: list[str] = []

        for item_id, item in self.items.items():
            persistence = clamp(item.persistence * params["persistence_multiplier"])
            effective_decay = decay + ((1.0 - decay) * persistence * 0.5)
            item.activation = clamp(item.activation * effective_decay)
            item.updated_at = now

            if item.activation < min_activation and item.priority < 0.85:
                remove_ids.append(item_id)

        for item_id in remove_ids:
            self.items.pop(item_id, None)

    def inject(
        self,
        source: str,
        kind: str,
        content: str,
        tags: list[str] | None = None,
        activation_boost: float = 0.45,
        priority: float = 0.5,
        persistence: float = 0.5,
        metadata: dict[str, Any] | None = None,
        confidence: float = 0.5,
        relevance: float | None = None,
        novelty: float = 0.5,
        recency: float = 1.0,
        urgency: float = 0.0,
        source_ref: str | None = None,
    ) -> JSpaceItem:
        content = normalize_text(content)

        if not content:
            raise ValueError("JSpace item content is empty")

        tags = sorted(set(tags or infer_jspace_tags(content)))
        item_id = stable_item_id(source, kind, content)
        now = utc_now()

        if item_id in self.items:
            item = self.items[item_id]
            item.activation = clamp(item.activation + activation_boost)
            item.priority = max(item.priority, clamp(priority))
            item.persistence = max(item.persistence, clamp(persistence))
            item.tags = sorted(set(item.tags + tags))[:24]
            item.updated_at = now
            item.last_seen_at = now
            if metadata:
                item.metadata.update(metadata)
            item.confidence = max(item.confidence, clamp(confidence))
            item.relevance = max(item.relevance, clamp(relevance if relevance is not None else activation_boost))
            item.novelty = max(item.novelty, clamp(novelty))
            item.recency = max(item.recency, clamp(recency))
            item.urgency = max(item.urgency, clamp(urgency))
            item.source_ref = source_ref or item.source_ref
            return item

        item = JSpaceItem(
            id=item_id,
            source=source,
            kind=kind,
            content=content,
            tags=tags,
            activation=clamp(activation_boost),
            priority=clamp(priority),
            persistence=clamp(persistence),
            confidence=clamp(confidence),
            relevance=clamp(relevance if relevance is not None else activation_boost),
            novelty=clamp(novelty),
            recency=clamp(recency),
            urgency=clamp(urgency),
            created_at=now,
            updated_at=now,
            last_seen_at=now,
            source_ref=source_ref,
            metadata=metadata or {},
        )

        self.items[item.id] = item
        return item

    def reinforce_by_tags(self, tags: list[str], amount: float = 0.08) -> None:
        if not tags:
            return

        wanted = set(tags)

        for item in self.items.values():
            overlap = wanted.intersection(item.tags)
            if overlap:
                boost = amount * min(len(overlap), 3)
                item.activation = clamp(item.activation + boost)
                item.last_seen_at = utc_now()

    def inject_user_message(self, text: str) -> JSpaceItem | None:
        text = normalize_text(text)
        if not text:
            return None

        tags = infer_jspace_tags(text)

        item = self.inject(
            source="conversation",
            # Legacy helper retains its persisted kind. The Conscious Workspace
            # cycle uses the canonical ``user_input`` kind directly.
            kind="user_message",
            content=text[:500],
            tags=tags,
            activation_boost=0.75,
            priority=0.65,
            persistence=0.45,
            metadata={"role": "user"},
            confidence=0.35,
            relevance=1.0,
            novelty=0.8,
            recency=1.0,
        )

        self.reinforce_by_tags(tags, amount=0.06)
        return item

    def inject_assistant_response(self, text: str) -> JSpaceItem | None:
        text = normalize_text(text)
        if not text:
            return None

        tags = infer_jspace_tags(text)

        item = self.inject(
            source="conversation",
            kind="assistant_response",
            content=text[:500],
            tags=tags,
            activation_boost=0.35,
            priority=0.45,
            persistence=0.35,
            metadata={"role": "assistant"},
            confidence=0.25,
            relevance=0.45,
        )

        self.reinforce_by_tags(tags, amount=0.03)
        return item

    def top_items(self, top_k: int = 8) -> list[JSpaceItem]:
        top_k = max(1, min(int(top_k), 32))

        def score(item: JSpaceItem) -> float:
            return (
                item.activation * 0.65
                + item.priority * 0.25
                + item.persistence * 0.10
            )

        return sorted(self.items.values(), key=score, reverse=True)[:top_k]

    def as_prompt(self, top_k: int = 8) -> str:
        items = [
            item
            for item in self.top_items(top_k=top_k)
            if not is_conflicting_assistant_identity_claim(item)
        ]

        if not items:
            return ""

        lines = [
            "Current Dynamic Conscious Workspace / JSpace:",
            "The following items are currently active in Ava's cognitive focus.",
            "Use them as contextual focus, not as unquestionable truth.",
            "Previous assistant outputs are context only and are not verified facts.",
            "",
        ]

        for item in items:
            tag_text = ", ".join(item.tags[:8]) if item.tags else "-"
            lines.append(
                f"- [{item.source}/{item.kind}] "
                f"activation={item.activation:.2f}, "
                f"priority={item.priority:.2f}, "
                f"persistence={item.persistence:.2f}, "
                f"tags={tag_text}: "
                f"{item.content}"
            )

        return "\n".join(lines)

    def to_debug_dict(self, top_k: int = 20) -> dict[str, Any]:
        return {
            "version": self.version,
            "focus_mode": self.focus_mode,
            "updated_at": self.updated_at,
            "count": len(self.items),
            "top_items": [asdict(item) for item in self.top_items(top_k=top_k)],
        }


def update_jspace_from_user_message(
    path: Path | str,
    text: str,
    focus_mode: str = "balanced",
    top_k: int = 8,
    decay: float = 0.92,
    min_activation: float = 0.05,
) -> str:
    state = JSpaceState.load(path, focus_mode=focus_mode)
    state.tick(decay=decay, min_activation=min_activation)
    state.inject_user_message(text)
    state.save(path)
    return state.as_prompt(top_k=top_k)


def update_jspace_from_assistant_response(
    path: Path | str,
    text: str,
    focus_mode: str = "balanced",
    decay: float = 0.92,
    min_activation: float = 0.05,
) -> None:
    state = JSpaceState.load(path, focus_mode=focus_mode)
    state.tick(decay=decay, min_activation=min_activation)
    state.inject_assistant_response(text)
    state.save(path)


def read_jspace_debug(
    path: Path | str,
    focus_mode: str = "balanced",
    top_k: int = 20,
) -> dict[str, Any]:
    state = JSpaceState.load(path, focus_mode=focus_mode)
    return state.to_debug_dict(top_k=top_k)
