from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

QUEUE_STATUSES = {
    "candidate",
    "pending",
    "running",
    "completed",
    "failed",
    "dismissed",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_question(question: str) -> str:
    normalized = " ".join((question or "").strip().lower().split())
    return re.sub(r"[^\wäöüß]+", " ", normalized, flags=re.UNICODE).strip()


def stable_research_id(question: str) -> str:
    digest = hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()[:16]
    return f"research_{digest}"


@dataclass
class ResearchTopic:
    id: str
    title: str
    question: str
    origin: str = "jspace"
    origin_item_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)
    status: str = "candidate"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_researched_at: str | None = None
    attempts: int = 0
    result_memory_id: int | None = None
    error: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchTopic:
        question = str(data.get("question") or "").strip()
        status = str(data.get("status") or "candidate")
        if status not in QUEUE_STATUSES:
            status = "candidate"
        components = {
            key: max(0.0, min(1.0, float(value)))
            for key, value in dict(data.get("score_components") or {}).items()
        }
        return cls(
            id=str(data.get("id") or stable_research_id(question)),
            title=str(data.get("title") or question[:80]).strip(),
            question=question,
            origin=str(data.get("origin") or "jspace"),
            origin_item_ids=list(data.get("origin_item_ids") or []),
            tags=list(data.get("tags") or []),
            score=max(0.0, min(1.0, float(data.get("score", 0.0)))),
            score_components=components,
            status=status,
            created_at=str(data.get("created_at") or utc_now()),
            updated_at=str(data.get("updated_at") or utc_now()),
            last_researched_at=data.get("last_researched_at"),
            attempts=max(0, int(data.get("attempts", 0))),
            result_memory_id=data.get("result_memory_id"),
            error=str(data.get("error") or ""),
        )


@dataclass
class ResearchRun:
    topic_id: str
    status: str
    started_at: str
    finished_at: str
    memory_id: int | None = None
    error: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchRun:
        return cls(
            topic_id=str(data.get("topic_id") or ""),
            status=str(data.get("status") or "failed"),
            started_at=str(data.get("started_at") or utc_now()),
            finished_at=str(data.get("finished_at") or utc_now()),
            memory_id=data.get("memory_id"),
            error=str(data.get("error") or ""),
        )


@dataclass
class ResearchQueue:
    version: int = 1
    updated_at: str = field(default_factory=utc_now)
    topics: dict[str, ResearchTopic] = field(default_factory=dict)
    run_history: list[ResearchRun] = field(default_factory=list)
    history_limit: int = 100

    @classmethod
    def load(cls, path: Path | str) -> ResearchQueue:
        path = Path(path).expanduser()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return cls()
        queue = cls(
            version=int(data.get("version", 1)),
            updated_at=str(data.get("updated_at") or utc_now()),
            history_limit=max(10, min(500, int(data.get("history_limit", 100)))),
        )
        for raw_topic in data.get("topics", []):
            topic = ResearchTopic.from_dict(raw_topic)
            if topic.question:
                queue.topics[topic.id] = topic
        queue.run_history = [
            ResearchRun.from_dict(item) for item in data.get("run_history", [])
        ][-queue.history_limit :]
        return queue

    def save(self, path: Path | str) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = utc_now()
        payload = {
            "version": self.version,
            "updated_at": self.updated_at,
            "history_limit": self.history_limit,
            "topics": [asdict(topic) for topic in self.topics.values()],
            "run_history": [asdict(run) for run in self.run_history[-self.history_limit :]],
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def add_or_update(self, topic: ResearchTopic) -> tuple[ResearchTopic, bool]:
        normalized = normalize_question(topic.question)
        existing = next(
            (
                item
                for item in self.topics.values()
                if normalize_question(item.question) == normalized
            ),
            None,
        )
        if existing:
            existing.origin_item_ids = sorted(
                set(existing.origin_item_ids + topic.origin_item_ids)
            )
            existing.tags = sorted(set(existing.tags + topic.tags))
            existing.score = max(existing.score, topic.score)
            if topic.score >= existing.score:
                existing.score_components = dict(topic.score_components)
                existing.title = topic.title
            existing.updated_at = utc_now()
            return existing, False
        self.topics[topic.id] = topic
        return topic, True

    def get(self, topic_id: str) -> ResearchTopic | None:
        return self.topics.get(topic_id)

    def question_in_cooldown(
        self,
        question: str,
        cooldown_hours: int,
        now: datetime | None = None,
    ) -> bool:
        normalized = normalize_question(question)
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        cooldown = timedelta(hours=max(0, cooldown_hours))
        for topic in self.topics.values():
            if normalize_question(topic.question) != normalized:
                continue
            researched_at = parse_timestamp(topic.last_researched_at)
            if researched_at and current - researched_at < cooldown:
                return True
        return False

    def candidates(
        self,
        min_score: float,
        cooldown_hours: int,
        now: datetime | None = None,
    ) -> list[ResearchTopic]:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        cooldown = timedelta(hours=max(0, cooldown_hours))
        eligible: list[ResearchTopic] = []
        for topic in self.topics.values():
            if topic.status not in {"candidate", "pending", "failed"}:
                continue
            if topic.score < min_score:
                continue
            researched_at = parse_timestamp(topic.last_researched_at)
            if researched_at and now - researched_at < cooldown:
                continue
            eligible.append(topic)
        return sorted(eligible, key=lambda item: (-item.score, item.created_at, item.id))

    def runs_today(self, now: datetime | None = None) -> int:
        today = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date()
        return sum(
            1
            for run in self.run_history
            if (started := parse_timestamp(run.started_at)) is not None
            and started.date() == today
        )

    def append_run(self, run: ResearchRun) -> None:
        self.run_history.append(run)
        self.run_history = self.run_history[-self.history_limit :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "count": len(self.topics),
            "topics": [asdict(topic) for topic in self.topics.values()],
            "run_history": [asdict(run) for run in self.run_history],
        }
