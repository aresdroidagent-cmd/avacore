from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from avacore.core.jspace import JSpaceState, clamp
from avacore.core.cognitive_workspace import read_workspace_debug, run_workspace_cycle
from avacore.core.research_curiosity import derive_topics, is_researchable_item
from avacore.core.research_queue import (
    ResearchQueue,
    ResearchRun,
    ResearchTopic,
    utc_now,
)
from avacore.tools.web_research import (
    ResearchSource,
    build_research_context,
    collect_research_sources,
    serialize_sources,
)

SourceCollector = Callable[..., list[ResearchSource]]
TelegramSender = Callable[[str, str, str], None]


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> None:
    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=30,
    )
    response.raise_for_status()


class AutonomousResearchService:
    def __init__(
        self,
        *,
        settings: Any,
        memory_store: Any,
        backend: Any,
        source_collector: SourceCollector = collect_research_sources,
        telegram_sender: TelegramSender = send_telegram_message,
        ensure_backend: Callable[[], None] | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.memory_store = memory_store
        self.backend = backend
        self.source_collector = source_collector
        self.telegram_sender = telegram_sender
        self.ensure_backend = ensure_backend or (lambda: None)
        self.now = now or (lambda: datetime.now(timezone.utc))

    @property
    def queue_path(self) -> Path:
        return Path(self.settings.research_queue_path).expanduser()

    def load_queue(self) -> ResearchQueue:
        return ResearchQueue.load(self.queue_path)

    def derive(self) -> dict[str, Any]:
        if not self.settings.research_enabled or self.settings.auto_research == "off":
            return {"ok": True, "status": "disabled", "created": 0, "topics": []}
        if not self.settings.jspace_enabled:
            return {
                "ok": True,
                "status": "idle",
                "created": 0,
                "topics": [],
                "message": "JSpace is disabled",
            }

        queue = self.load_queue()
        state = JSpaceState.load(
            self.settings.jspace_path,
            focus_mode=self.settings.jspace_focus_mode,
        )
        jspace_items = dict(state.items)
        for queued_topic in queue.topics.values():
            if queued_topic.origin != "jspace":
                continue
            if queued_topic.status not in {"candidate", "pending", "failed"}:
                continue
            origin_items = [
                jspace_items[item_id]
                for item_id in queued_topic.origin_item_ids
                if item_id in jspace_items
            ]
            if origin_items and not any(
                is_researchable_item(item) for item in origin_items
            ):
                queued_topic.status = "dismissed"
                queued_topic.updated_at = utc_now()
                queued_topic.error = "filtered as non-researchable internal context"

        workspace = read_workspace_debug(getattr(self.settings, "workspace_path", "./data/state/conscious_workspace.json"))
        focused_ids = [item["id"] for item in workspace.get("active_items", [])]
        focused_items = [state.items[item_id] for item_id in focused_ids if item_id in state.items]
        derived = derive_topics(
            focused_items or state.top_items(top_k=max(self.settings.jspace_top_k, 16)),
            curiosity_weight=self.settings.research_curiosity_weight,
        )
        created = 0
        accepted: list[ResearchTopic] = []
        for topic in derived:
            if queue.question_in_cooldown(
                topic.question,
                cooldown_hours=self.settings.research_cooldown_hours,
                now=self.now(),
            ):
                continue
            stored, is_new = queue.add_or_update(topic)
            created += int(is_new)
            accepted.append(stored)
            state.inject(
                source="research",
                kind="research_question",
                content=stored.question,
                tags=stored.tags,
                activation_boost=stored.score,
                priority=stored.score,
                persistence=0.6,
                confidence=0.3,
                relevance=stored.score,
                urgency=stored.score_components.get("urgency", 0.0),
                source_ref=stored.id,
                metadata={"topic_id": stored.id, "status": stored.status, "uncertainty": 1.0},
            )
        if accepted:
            state.save(self.settings.jspace_path)
        queue.save(self.queue_path)
        return {
            "ok": True,
            "status": "idle" if not accepted else "approval_required",
            "created": created,
            "topics": [asdict(topic) for topic in accepted],
        }

    def dismiss(self, topic_id: str) -> dict[str, Any]:
        queue = self.load_queue()
        topic = queue.get(topic_id)
        if not topic:
            return {"ok": False, "status": "idle", "topic_id": topic_id}
        topic.status = "dismissed"
        topic.updated_at = utc_now()
        topic.error = ""
        queue.save(self.queue_path)
        return {"ok": True, "status": "dismissed", "topic": asdict(topic)}

    def run_next(self) -> dict[str, Any]:
        if not self.settings.research_enabled or self.settings.auto_research == "off":
            return {"ok": True, "status": "disabled"}

        derive_result = self.derive()
        queue = self.load_queue()
        if self.settings.auto_research == "ask":
            candidates = queue.candidates(
                min_score=self.settings.research_min_score,
                cooldown_hours=self.settings.research_cooldown_hours,
                now=self.now(),
            )
            return {
                "ok": True,
                "status": "approval_required" if candidates else "idle",
                "derived": derive_result.get("created", 0),
                "candidate": asdict(candidates[0]) if candidates else None,
            }

        if self.settings.auto_research != "bounded":
            return {"ok": True, "status": "disabled"}

        if queue.runs_today(now=self.now()) >= self.settings.research_max_runs_per_day:
            return {"ok": True, "status": "budget_exhausted"}

        candidates = queue.candidates(
            min_score=self.settings.research_min_score,
            cooldown_hours=self.settings.research_cooldown_hours,
            now=self.now(),
        )
        if not candidates:
            return {
                "ok": True,
                "status": "idle",
                "derived": derive_result.get("created", 0),
            }

        # Phase one intentionally executes exactly one topic even if the setting is higher.
        return self._run_topic(queue, candidates[0])

    def _run_topic(
        self,
        queue: ResearchQueue,
        topic: ResearchTopic,
    ) -> dict[str, Any]:
        started_at = self.now().astimezone(timezone.utc).isoformat(timespec="seconds")
        topic.status = "running"
        topic.attempts += 1
        topic.last_researched_at = started_at
        topic.updated_at = started_at
        topic.error = ""
        run_record = ResearchRun(
            topic_id=topic.id,
            status="running",
            started_at=started_at,
            finished_at=started_at,
        )
        queue.append_run(run_record)
        queue.save(self.queue_path)

        try:
            sources = self.source_collector(
                query=topic.question,
                max_results=self.settings.research_max_sources_per_topic,
                max_chars_per_source=5000,
            )
            readable = [source for source in sources if source.ok and source.text.strip()]
            if not readable:
                raise RuntimeError("no successful readable research sources")

            context = build_research_context(topic.question, readable)
            self.ensure_backend()
            answer = self.backend.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Du bist Ava, ein lokaler Recherche-Assistent. Nutze nur "
                            "die bereitgestellten lesbaren Quellen. Antworte auf Deutsch. "
                            "Trenne Erkenntnisse, Unsicherheiten und Projektrelevanz. "
                            "Erfinde keine Fakten."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{context}\n\n"
                            "Erstelle eine kurze Zusammenfassung, die wichtigsten "
                            "Erkenntnisse, offene Unsicherheiten und die Relevanz für "
                            "AvaCore beziehungsweise Rogers aktive Projekte."
                        ),
                    },
                ]
            ).strip()
            if not answer:
                raise RuntimeError("Ollama returned an empty research summary")

            memory_id = self._get_or_create_memory(topic, answer, readable)
            topic.status = "completed"
            topic.result_memory_id = memory_id
            topic.updated_at = utc_now()
            topic.error = ""

            self._activate_finding(topic, answer)
            notification_error = self._notify_if_needed(topic, answer)

            finished_at = utc_now()
            run_record.status = "completed"
            run_record.finished_at = finished_at
            run_record.memory_id = memory_id
            queue.save(self.queue_path)
            return {
                "ok": True,
                "status": "completed",
                "topic": asdict(topic),
                "answer": answer,
                "sources": serialize_sources(readable),
                "memory_id": memory_id,
                "memory_status": "candidate",
                "notification_error": notification_error,
            }
        except Exception as exc:  # noqa: BLE001 - scheduler must persist any run failure
            error = str(exc)
            topic.status = "failed"
            topic.updated_at = utc_now()
            topic.error = error
            run_record.status = "failed"
            run_record.finished_at = utc_now()
            run_record.error = error
            queue.save(self.queue_path)
            return {
                "ok": False,
                "status": "failed",
                "topic": asdict(topic),
                "error": error,
            }

    def _get_or_create_memory(
        self,
        topic: ResearchTopic,
        answer: str,
        sources: list[ResearchSource],
    ) -> int:
        existing = self.memory_store.list_memory_items(
            memory_type="research_lead",
            scope="user",
            limit=500,
        )
        duplicate = next(
            (
                item
                for item in existing
                if item.get("source_type") == "autonomous_research"
                and item.get("source_ref") == topic.id
            ),
            None,
        )
        if duplicate:
            return int(duplicate["id"])

        source_lines = "\n".join(
            f"- {source.title}: {source.url}" for source in sources
        )
        relevance = ", ".join(topic.tags[:8]) or "aktiver JSpace-Kontext"
        content = (
            f"Recherchefrage:\n{topic.question}\n\n"
            f"Zusammenfassung und wichtigste Erkenntnisse:\n{answer}\n\n"
            "Offene Unsicherheiten:\n"
            "Siehe die in der Zusammenfassung ausdrücklich genannten Unsicherheiten; "
            "die Quellen müssen vor dauerhafter Übernahme geprüft werden.\n\n"
            f"Relevanz für AvaCore / Rogers aktive Projekte:\n{relevance}\n\n"
            f"Verwendete Quellen:\n{source_lines}"
        )
        return self.memory_store.create_memory_item(
            scope="user",
            title=f"Autonomous Research: {topic.title[:80]}",
            content=content,
            memory_type="research_lead",
            status="candidate",
            source_type="autonomous_research",
            source_ref=topic.id,
            confidence=min(0.8, max(0.4, topic.score)),
            importance=max(1, min(5, round(topic.score * 5))),
            tags=",".join(sorted(set(["research", "autonomous"] + topic.tags))),
            created_from_user_text=topic.question,
            created_from_assistant_text=answer,
        )

    def _activate_finding(self, topic: ResearchTopic, answer: str) -> None:
        if not self.settings.jspace_enabled:
            return
        state = JSpaceState.load(
            self.settings.jspace_path,
            focus_mode=self.settings.jspace_focus_mode,
        )
        state.inject(
            source="research",
            kind="research_finding",
            content=f"{topic.title}: {answer[:500]}",
            tags=sorted(set(["research"] + topic.tags)),
            activation_boost=clamp(0.35 + topic.score * 0.45),
            priority=clamp(0.30 + topic.score * 0.60),
            persistence=clamp(0.25 + topic.score * 0.55),
            metadata={
                "research_topic_id": topic.id,
                "topic_id": topic.id,
                "memory_id": topic.result_memory_id,
                "score": topic.score,
                "confidence": min(0.8, max(0.4, topic.score)),
                "uncertainties": True,
            },
            confidence=min(0.8, max(0.4, topic.score)),
            relevance=topic.score,
            source_ref=topic.id,
        )
        state.save(self.settings.jspace_path)
        if hasattr(self.settings, "workspace_path"):
            run_workspace_cycle(
                jspace_path=self.settings.jspace_path,
                workspace_path=self.settings.workspace_path,
                stimulus=topic.question,
                candidates=[],
                attention_mode=self.settings.workspace_default_mode,
                max_active_items=self.settings.workspace_max_active_items,
                max_latent_items=self.settings.workspace_max_latent_items,
                min_activation=self.settings.workspace_min_activation,
                decay_factor=self.settings.workspace_decay_factor,
                max_per_source=self.settings.workspace_max_per_source,
                max_per_kind=self.settings.workspace_max_per_kind,
                history_limit=self.settings.workspace_history_limit,
                trigger="autonomous_research_completion",
            )

    def _notify_if_needed(self, topic: ResearchTopic, answer: str) -> str:
        if topic.score < self.settings.research_notify_score:
            return ""
        token = (self.settings.telegram_bot_token or "").strip()
        chat_id = (self.settings.telegram_allowed_chat_id or "").strip()
        if not token or not chat_id:
            return ""
        relevance = ", ".join(topic.tags[:5]) or "aktiver Projektkontext"
        message = (
            "🔎 Ava Research\n\n"
            f"Thema:\n{topic.title}\n\n"
            f"Wichtigste Erkenntnis:\n{answer[:700]}\n\n"
            f"Warum relevant:\n{relevance}\n\n"
            "Der vollständige Eintrag wurde als Memory-Kandidat gespeichert."
        )
        try:
            self.telegram_sender(token, chat_id, message)
        except Exception as exc:  # noqa: BLE001 - notification must not fail research
            return str(exc)
        return ""
