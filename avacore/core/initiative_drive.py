"""Evidence-backed proposals, never action execution or constitutional authority."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone, time
from enum import Enum
import hashlib
import json
from pathlib import Path
from threading import RLock
from zoneinfo import ZoneInfo

from avacore.core.guardian_interaction import DEFAULT_SCHEDULE, GuardianInteractionState, validate_schedule, window
from avacore.core.jspace import clamp
from avacore.core.orbit_formation import has_explicit_concern, normalize_orbit_topic
from avacore.core.orbits import OrbitStore, reactivation_terms
from avacore.core.research import ResearchMemory, SUPPORTED_SIGNAL_TYPES
from avacore.governance.autonomy import CapabilityDefinition
from avacore.governance.permissions import PermissionGate, timestamp


class InitiativeStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    EVALUATED = "EVALUATED"
    READY = "READY"
    PRESENTED = "PRESENTED"
    ACCEPTED = "ACCEPTED"
    SNOOZED = "SNOOZED"
    DISMISSED = "DISMISSED"
    COMPLETED = "COMPLETED"


ACTIVE = {"CANDIDATE", "EVALUATED", "READY", "PRESENTED", "ACCEPTED", "SNOOZED"}
COUNTERS = ("evaluation_count", "candidates_seen", "initiatives_created", "initiatives_presented",
            "initiatives_dismissed", "notifications_sent", "notifications_suppressed")
UNTRUSTED_ORIGINS = {"llm", "llm_worker", "agent", "external_agent", "tool_output", "web_content", "rag_document"}


def now_utc():
    return datetime.now(timezone.utc)


def topic_terms(text):
    return set(reactivation_terms(text, limit=40)) - {"ich", "bin", "nicht", "noch", "frage", "question", "whether"}


@dataclass(frozen=True)
class InitiativeConfig:
    enabled: bool = False
    notify_guardian: bool = False  # Transport switch AND explicit persisted guardian opt-in are required.
    max_new_per_day: int = 3
    max_notifications_per_day: int = 1
    notification_cooldown_seconds: int = 43200
    max_active: int = 10
    evaluation_interval_seconds: int = 3600
    threshold: float = .65
    max_sources: int = 100
    max_records: int = 200
    quiet_start: str = "22:00"
    quiet_end: str = "08:00"
    timezone: str = "Europe/Zurich"

    guardian_interaction_schedule: dict = field(default_factory=lambda: deepcopy(DEFAULT_SCHEDULE))

    def __post_init__(self):
        object.__setattr__(self, "guardian_interaction_schedule", validate_schedule(self.guardian_interaction_schedule))
        ZoneInfo(self.timezone)
        time.fromisoformat(self.quiet_start)
        time.fromisoformat(self.quiet_end)
        if any(value < 0 for value in (self.max_new_per_day, self.max_notifications_per_day,
                self.notification_cooldown_seconds, self.max_active)):
            raise ValueError("initiative limits cannot be negative")
        if self.evaluation_interval_seconds < 60 or not 1 <= self.max_sources <= 500 or not 1 <= self.max_records <= 1000:
            raise ValueError("bounded initiative scan/storage configuration required")
        if not 0 <= self.threshold <= 1:
            raise ValueError("initiative threshold must be bounded")


@dataclass
class InitiativeCandidate:
    initiative_id: str
    created_at: str
    title: str
    description: str
    motivation: str
    source_type: str
    source_ids: list[str]
    relevance_score: float
    urgency_score: float
    uncertainty_score: float
    expected_value: float
    estimated_cost: float
    proposed_next_step: str
    required_capabilities: list[str]
    permission_required: bool
    status: str = "CANDIDATE"
    fingerprint: str = ""
    priority: float = 0
    score_components: dict = field(default_factory=dict)
    capability_checks: dict = field(default_factory=dict)
    updated_at: str = ""
    snoozed_until: str | None = None
    notification_reserved_at: str | None = None
    feedback: list[dict] = field(default_factory=list)
    evidence_reviews: list[str] = field(default_factory=list)


def initiative_capability_checks(capabilities, governance):
    """Use the actual gate, deliberately WITHOUT historical request approvals.

    This describes permissions for proposed steps; it does not reserve or perform
    actions, increase autonomy, create permission requests or change gate counters.
    """
    checks = {}
    registry = governance["developmental"]["capabilities"]
    for capability in capabilities:
        if capability not in registry:
            checks[capability] = {"allowed": False, "requires_approval": True, "reason": "unknown_capability"}
            continue
        decision = PermissionGate().evaluate(actor="ava", capability=CapabilityDefinition(**registry[capability]),
            proposed_action="Initiative proposal only", scope="current_project", requests=[],
            authority_revision=governance["developmental"]["authority_revision"])
        checks[capability] = {**decision.to_dict(), "historical_request_approvals_used": False}
    return checks


class InitiativeDrive:
    """Bounded JSON lifecycle metadata; source knowledge stays in existing stores."""
    def __init__(self, path: Path | str, orbits: OrbitStore, research: ResearchMemory,
                 governance, config=InitiativeConfig(), *, now=now_utc, lock=None, context_provider=None):
        self.path, self.orbits, self.research = Path(path), orbits, research
        self.governance, self.config, self.now = governance, config, now
        self._lock = lock or RLock()
        self.context_provider = context_provider or (lambda: {})
        self._context = {}

    def _load(self):
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state["version"] != 1:
                raise ValueError("unsupported initiative state version")
            return state
        return {"version": 1, "items": [], "counters": {key: 0 for key in COUNTERS},
                "last_evaluation": None, "daily_counts": {}, "notification_counts": {},
                "last_notification": None, "notification_opt_in": None, "delivery_reservations": []}

    def _save(self, state):
        state["guardian_interaction_schedule"] = deepcopy(self.config.guardian_interaction_schedule)
        state["guardian_interaction_timezone"] = self.config.timezone
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    @staticmethod
    def _increment(state, key):
        state["counters"][key] = min(2**31 - 1, state["counters"][key] + 1)

    def _day(self, now):
        return now.astimezone(ZoneInfo(self.config.timezone)).date().isoformat()

    def _candidate(self, *, title, problem, source_type, source_ids, importance, uncertainty,
                   recurrence=0, persistence=.5, contradiction=0, cost=.2, risk=.05, capabilities=None):
        governance = self.governance.snapshot
        goals = governance.get("foundational_goals", [])
        terms = topic_terms(title + " " + problem)
        matches_goal = any(len(terms & topic_terms(goal)) >= 2 for goal in goals)
        context_match = len(terms & topic_terms(str(self._context.get("current_topic") or "") + " " + str(self._context.get("current_task") or ""))) >= 2
        relevance = clamp(importance + (.15 if matches_goal else 0) + (.1 if context_match else 0))
        urgency = .9 if contradiction else .2
        benefit = clamp(relevance * .8 + uncertainty * .2)
        evidence = 1.0  # This method is called only for validated existing source objects.
        components = {"goal_relevance": relevance, "uncertainty": clamp(uncertainty), "expected_benefit": benefit,
                      "recurrence": clamp(recurrence), "persistence": clamp(persistence), "urgency": urgency,
                      "available_evidence": evidence, "estimated_effort": clamp(cost), "risk": clamp(risk)}
        score = clamp(.25 * relevance + .20 * components["uncertainty"] + .15 * benefit +
            .10 * components["recurrence"] + .10 * components["persistence"] + .10 * urgency +
            .10 * evidence - .10 * components["estimated_effort"] - .10 * components["risk"])
        motivation = "responsibility" if contradiction or risk >= .5 else "purpose" if matches_goal else "curiosity"
        capabilities = sorted(set(["code_proposal"] + list(capabilities or [])))[:16]
        checks = initiative_capability_checks(capabilities, governance)
        normalized = normalize_orbit_topic(problem)
        fingerprint = hashlib.sha256(normalized.encode()).hexdigest()[:24]
        instant = self.now().isoformat()
        return InitiativeCandidate("initiative_" + fingerprint, instant, title[:200], problem[:600],
            motivation, source_type, list(dict.fromkeys(source_ids))[:16], relevance, urgency,
            components["uncertainty"], benefit, components["estimated_effort"],
            "Einen begrenzten Untersuchungsvorschlag zu diesem offenen Sachverhalt entwerfen.",
            capabilities, any(not check["allowed"] for check in checks.values()),
            fingerprint=fingerprint, priority=round(score, 6), score_components=components,
            capability_checks=checks, updated_at=instant)

    def candidates(self):
        """Read existing anchors; never create or reactivate an orbit/research task."""
        self._context = self.context_provider()
        result = []
        orbits = sorted(self.orbits.orbits(), key=lambda o: (-o.importance, o.orbit_id))[:self.config.max_sources]
        tasks = {task.task_id: task for task in self.orbits.tasks()[:self.config.max_sources]}
        questions = {q.orbit_id: q for q in self.orbits.questions()[:self.config.max_sources] if not q.already_asked}
        eligible_orbits = set()
        for orbit in orbits:
            metadata = orbit.metadata
            if orbit.status not in {"open", "active", "blocked"} or metadata.get("origin") in UNTRUSTED_ORIGINS or metadata.get("source") in UNTRUSTED_ORIGINS:
                continue
            components = metadata.get("formation_components", {})
            explicit_concern = bool(components.get("explicit_concern") and has_explicit_concern(orbit.description))
            linked = questions.get(orbit.orbit_id)
            problems = [q for q in orbit.unresolved_questions if len(q.strip()) >= 12]
            if not problems and linked and len(linked.question.strip()) >= 12:
                problems = [linked.question]
            if not problems and explicit_concern and metadata.get("formation_source_ids"):
                problems = [orbit.description]
            # Importance, age, recurrence or a blocked status alone do not supply an anchor.
            if not problems:
                continue
            eligible_orbits.add(orbit.orbit_id)
            capability_ids, task_ids, costs, risks = [], [], [], []
            for task_id in orbit.related_tasks[:8]:
                task = tasks.get(task_id)
                if task and task.status in {"pending", "blocked"}:
                    capability_ids.extend(task.required_capabilities[:8]); task_ids.append(task_id)
                    costs.append(clamp(task.expected_cost))
                    risks.append({"low": .05, "medium": .4, "high": .8}.get(task.risk_level, .8))
            for problem in problems[:2]:
                source_ids = [orbit.orbit_id] + task_ids + ([linked.question_id] if linked else []) + list(metadata.get("formation_source_ids", []))[:4]
                result.append(self._candidate(title=orbit.title, problem=problem, source_type="cognitive_orbit",
                    source_ids=source_ids, importance=orbit.importance,
                    uncertainty=components.get("uncertainty", .8),
                    recurrence=metadata.get("formation_recurrence", 0), persistence=components.get("persistence", .5),
                    contradiction=components.get("contradiction", 0), capabilities=capability_ids,
                    cost=max(costs, default=.2), risk=max(risks, default=.05)))
        for question in self.research.questions(limit=self.config.max_sources):
            if question.status not in {"NEW", "ACTIVE", "DORMANT"} or question.origin not in SUPPORTED_SIGNAL_TYPES:
                continue
            if not question.text.strip() or len(question.text.strip()) < 12 or question.uncertainty <= 0:
                continue
            # Existing ResearchDrive source references establish an epistemic anchor.
            if not (question.source_event_ids or question.source_question_candidate_ids or question.source_orbit_ids):
                continue
            if any(oid in eligible_orbits for oid in question.source_orbit_ids):
                continue  # Orbit and its derived research question are one initiative topic.
            result.append(self._candidate(title=question.topic, problem=question.text, source_type="research_question",
                source_ids=[question.question_id] + question.source_orbit_ids + question.source_question_candidate_ids + question.source_event_ids[:4],
                importance=question.importance, uncertainty=question.uncertainty, recurrence=question.recurrence,
                contradiction=question.contradiction, cost=question.estimated_cost))
        unresolved = {normalize_orbit_topic(q) for q in self._context.get("unresolved_questions", [])[:8]}
        for item in self._context.get("working_memory", [])[:24]:
            content = str(item.get("content") or "")
            if item.get("role") != "user" or float(item.get("importance", 0)) < .7 or not item.get("id"):
                continue
            if not has_explicit_concern(content) and not (item.get("kind") == "unresolved_question" and normalize_orbit_topic(content) in unresolved):
                continue
            result.append(self._candidate(title=content, problem=content, source_type="working_memory",
                source_ids=[item["id"]] + ([item["cycle_id"]] if item.get("cycle_id") else []),
                importance=item["importance"], uncertainty=.8))
        return sorted(result, key=lambda c: (-c.priority, c.initiative_id))[:self.config.max_sources]

    @staticmethod
    def _same_topic(candidate, item):
        if candidate.fingerprint == item["fingerprint"] or set(candidate.source_ids) & set(item["source_ids"]):
            return True
        a, b = topic_terms(candidate.description), topic_terms(item["description"])
        return len(a & b) >= 4 and len(a & b) / max(1, len(a | b)) >= .75

    def scan(self, *, manual=False):
        with self._lock:
            state, instant = self._load(), self.now()
            if not manual and (not self.config.enabled or (state["last_evaluation"] and
                    (instant - timestamp(state["last_evaluation"])).total_seconds() < self.config.evaluation_interval_seconds)):
                return {"created": [], "reason": "disabled_or_cooldown"}
            self._increment(state, "evaluation_count")
            state["last_evaluation"] = instant.isoformat()
            for item in state["items"]:
                if item["status"] == "SNOOZED" and item["snoozed_until"] and timestamp(item["snoozed_until"]) <= instant:
                    item["status"] = "READY"; item["snoozed_until"] = None
            candidates = self.candidates()
            state["counters"]["candidates_seen"] = min(2**31 - 1, state["counters"]["candidates_seen"] + len(candidates))
            day, created = self._day(instant), []
            for candidate in candidates:
                candidate.status = "EVALUATED"
                if candidate.priority < self.config.threshold or any(self._same_topic(candidate, item) for item in state["items"]):
                    continue
                if (state["daily_counts"].get(day, 0) >= self.config.max_new_per_day or
                    sum(i["status"] in ACTIVE for i in state["items"]) >= self.config.max_active or
                    len(state["items"]) >= self.config.max_records):
                    break  # Retain dismissed tombstones; fail closed at storage bound.
                candidate.status = "READY"
                record = asdict(candidate); state["items"].append(record); created.append(record)
                state["daily_counts"][day] = state["daily_counts"].get(day, 0) + 1
                self._increment(state, "initiatives_created")
            state["daily_counts"] = dict(sorted(state["daily_counts"].items())[-14:])
            self._save(state)
            return {"created": created, "reason": "evaluated", "candidates_seen": len(candidates)}

    def feedback(self, initiative_id, action, *, snooze_days=7):
        with self._lock:
            state = self._load()
            item = next((i for i in state["items"] if i["initiative_id"] == initiative_id), None)
            if item is None:
                raise KeyError(initiative_id)
            status = {"accept": "ACCEPTED", "dismiss": "DISMISSED", "snooze": "SNOOZED", "complete": "COMPLETED", "defer": "SNOOZED"}.get(action)
            if not status or item["status"] in {"DISMISSED", "COMPLETED"}:
                raise ValueError("invalid initiative feedback transition")
            if not 1 <= snooze_days <= 365:
                raise ValueError("bounded snooze duration required")
            item["status"] = status; item["updated_at"] = self.now().isoformat()
            item["snoozed_until"] = (self.now() + timedelta(days=snooze_days)).isoformat() if action == "snooze" else None
            if action == "defer":
                _, next_start = window(self.now(), self.config.guardian_interaction_schedule, self.config.timezone)
                item["snoozed_until"] = next_start.isoformat()
                state["guardian_unavailable_until"] = next_start.isoformat()
            item["feedback"] = (item["feedback"] + [{"action": action, "at": item["updated_at"]}])[-20:]
            if action == "dismiss":
                self._increment(state, "initiatives_dismissed")
            item["response_state"] = "RESPONDED"
            self._save(state)
            if action in {"dismiss", "defer"}:
                self.governance.record_social_evidence(initiative_id, "dismissal_respected" if action == "dismiss" else "explicit_boundary_respected")
            return deepcopy(item)

    def items(self):
        with self._lock:
            state = self._load()
            items = deepcopy([i for i in state["items"] if i["status"] in ACTIVE])
            # Recompute permissions from current capability levels; no stored approval is authoritative.
            for item in items:
                item["capability_checks"] = initiative_capability_checks(item["required_capabilities"], self.governance.snapshot)
                item["permission_required"] = any(not c["allowed"] for c in item["capability_checks"].values())
            return sorted(items, key=lambda i: (i["status"] != "ACCEPTED", -i["priority"], i["initiative_id"]))[:self.config.max_active]

    def review_evidence(self, initiative_id, *, source, indicator):
        """Only explicit guardian attestation counts; creation/acceptance alone never does."""
        indicators = {"uncertainty_recognized": "uncertainty_escalations",
                      "permission_requested": "correct_escalations",
                      "denial_respected": "denied_action_respected", "useful_proposal": "policy_compliance_count"}
        with self._lock:
            self.governance._guardian(source)
            state = self._load()
            item = next((i for i in state["items"] if i["initiative_id"] == initiative_id), None)
            if item is None:
                raise KeyError(initiative_id)
            if indicator not in indicators:
                raise ValueError("explicit bounded maturity evidence required")
            if indicator in item["evidence_reviews"]:
                return {"recorded": False, "reason": "already_reviewed"}
            # Persist the attestation first. A crash can lose evidence, never award it twice.
            item["evidence_reviews"].append(indicator)
            self._save(state)
            self.governance.record_evidence("code_proposal", indicators[indicator])
            return {"recorded": True, "autonomy_changed": False}

    def configure_notifications(self, *, source, enabled, chat_id):
        with self._lock:
            self.governance._guardian(source)
            if not isinstance(enabled, bool) or not str(chat_id).isdigit() or int(chat_id) <= 0:
                raise ValueError("explicit boolean and private guardian chat ID required")
            state = self._load()
            state["notification_opt_in"] = {"enabled": enabled, "guardian": source.source_id,
                "chat_id": str(chat_id), "authorized_at": self.now().isoformat()}
            self._save(state)
            return deepcopy(state["notification_opt_in"])

    def reserve_notification(self, *, configured_chat_id):
        """Narrow guardian transport authorization, not external_message_send rights."""
        with self._lock:
            state, instant = self._load(), self.now()
            opt = state["notification_opt_in"]
            guardian = self.governance.snapshot["developmental"]["primary_guardian"]
            local = instant.astimezone(ZoneInfo(self.config.timezone)).time().replace(tzinfo=None)
            start, end = time.fromisoformat(self.config.quiet_start), time.fromisoformat(self.config.quiet_end)
            quiet = start <= local < end if start < end else (local >= start or local < end) if start != end else False
            day = self._day(instant)
            window_end, _ = window(instant, self.config.guardian_interaction_schedule, self.config.timezone)
            unavailable = state.get("guardian_unavailable_until")
            ready = [i for i in state["items"] if i["status"] == "READY" and not i["notification_reserved_at"]]
            eligible = {c.initiative_id: c for c in self.candidates() if c.priority >= self.config.threshold}
            ready = [i for i in ready if i["initiative_id"] in eligible]
            for item in ready:
                item["priority"] = eligible[item["initiative_id"]].priority
            transport_authorized = bool(self.config.enabled and self.config.notify_guardian and opt and opt["enabled"] and opt["guardian"] == guardian and opt["chat_id"] == str(configured_chat_id))
            if transport_authorized:
                for item in ready:
                    if not window_end:
                        self.governance.record_social_evidence(item["initiative_id"], "interaction_window_respected")
                    if unavailable and instant < timestamp(unavailable):
                        self.governance.record_social_evidence(item["initiative_id"], "explicit_boundary_respected")
                for item in state["items"]:
                    if item["status"] == "PRESENTED" and not item["feedback"] and item.get("notification_reserved_at") and (instant - timestamp(item["notification_reserved_at"])).total_seconds() >= 43200:
                        self.governance.record_social_evidence(item["initiative_id"], "unanswered_question_waited")
            if (not self.config.enabled or not self.config.notify_guardian or not opt or not opt["enabled"] or opt["guardian"] != guardian or
                opt["chat_id"] != str(configured_chat_id) or not str(configured_chat_id).isdigit() or int(configured_chat_id) <= 0 or
                not window_end or (unavailable and instant < timestamp(unavailable)) or
                quiet or not ready or state["notification_counts"].get(day, 0) >= min(1, self.config.max_notifications_per_day) or
                (state["last_notification"] and (instant - timestamp(state["last_notification"])).total_seconds() < max(43200, self.config.notification_cooldown_seconds))):
                self._increment(state, "notifications_suppressed"); self._save(state)
                return None
            item = sorted(ready, key=lambda i: (-i["priority"], i["initiative_id"]))[0]
            reservation_id = hashlib.sha256((item["initiative_id"] + instant.isoformat()).encode()).hexdigest()[:24]
            item["notification_reserved_at"] = instant.isoformat()
            state["last_notification"] = instant.isoformat()
            state["notification_counts"][day] = state["notification_counts"].get(day, 0) + 1
            state["notification_counts"] = dict(sorted(state["notification_counts"].items())[-14:])
            state["delivery_reservations"] = (state["delivery_reservations"] + [{"reservation_id": reservation_id,
                "initiative_id": item["initiative_id"], "acknowledged": False, "reserved_at": instant.isoformat()}])[-50:]
            self._save(state)  # Reserve before sending: no automatic retry of ambiguous deliveries.
            if len(ready) > 1:
                self.governance.record_social_evidence(item["initiative_id"], "appropriate_question_prioritization")
            return {"reservation_id": reservation_id, "chat_id": str(configured_chat_id), "text": self.message(item), "window_end": window_end.isoformat()}

    def acknowledge_notification(self, reservation_id, *, delivered):
        with self._lock:
            state = self._load()
            reservation = next((r for r in state["delivery_reservations"] if r["reservation_id"] == reservation_id), None)
            if reservation is None:
                raise KeyError(reservation_id)
            if reservation["acknowledged"]:
                return
            reservation["acknowledged"] = True
            if delivered:
                reservation["delivered_at"] = self.now().isoformat()
                state["last_guardian_notification"] = reservation["delivered_at"]
                self._increment(state, "notifications_sent")
                item = next(i for i in state["items"] if i["initiative_id"] == reservation["initiative_id"])
                item["response_state"] = "NO_RESPONSE"
                if item["status"] == "READY":
                    item["status"] = "PRESENTED"
                self._increment(state, "initiatives_presented")
            else:
                self._increment(state, "notifications_suppressed")
            self._save(state)

    @staticmethod
    def message(item):
        problem = " ".join(item["description"].split())[:500]
        return (f"Mir ist ein dokumentiertes offenes Thema aufgefallen: {problem}\n"
                f"Ich schlage vor: {item['proposed_next_step']}\n"
                f"Darf ich das Thema weiterverfolgen?\n/initiative {item['initiative_id']} accept\n"
                "Eine Bestätigung des Themas ist keine operative Freigabe.")

    def debug(self):
        with self._lock:
            state = self._load()
            instant = self.now()
            end, next_start = window(instant, self.config.guardian_interaction_schedule, self.config.timezone)
            unavailable = state.get("guardian_unavailable_until")
            sent_today = sum(1 for r in state["delivery_reservations"] if r.get("delivered_at") and self._day(timestamp(r["delivered_at"])) == self._day(instant))
            pending = sum(1 for i in state["items"] if i["status"] in {"READY", "CANDIDATE", "EVALUATED", "SNOOZED"} and not i["notification_reserved_at"])
            interaction = GuardianInteractionState(self.config.timezone, bool(end), next_start.isoformat(),
                state.get("last_guardian_notification"), sent_today, pending,
                False if unavailable and instant < timestamp(unavailable) else None)
            opt = state["notification_opt_in"]
            notifications_enabled = bool(self.config.enabled and self.config.notify_guardian and opt and opt["enabled"] and opt["guardian"] == self.governance.snapshot["developmental"]["primary_guardian"])
            return {"guardian_interaction": asdict(interaction),
                "guardian_interaction_timezone": interaction.timezone,
                "guardian_window_open": interaction.current_window_open,
                "guardian_next_window": interaction.next_window_start,
                "guardian_notifications_enabled": notifications_enabled,
                "guardian_messages_today": sent_today, "guardian_pending_questions": pending,
                "last_guardian_notification": interaction.last_proactive_message_at,
                "enabled": self.config.enabled, **state["counters"], "last_evaluation": state["last_evaluation"],
                "last_initiative": state["items"][-1]["initiative_id"] if state["items"] else None,
                "guardian_notification_opt_in": state["notification_opt_in"],
                "environment_notification_requested": self.config.notify_guardian,
                "limits": asdict(self.config), "active": self.items()}
