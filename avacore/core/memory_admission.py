"""Deterministic admission policy for durable review-memory candidates."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class MemoryClass(str, Enum):
    EPHEMERAL = "EPHEMERAL"
    SESSION_RELEVANT = "SESSION_RELEVANT"
    CANDIDATE = "CANDIDATE"


@dataclass(frozen=True)
class MemoryAdmissionDecision:
    admit: bool
    reason: str
    source_type: str
    memory_class: MemoryClass
    volatility: str
    confidence: float
    durability: float
    evidence_quality: float
    temporal_scope: str
    source_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def debug(self) -> dict[str, Any]:
        return {"admit": self.admit, "reason": self.reason,
                "source_type": self.source_type, "memory_class": self.memory_class.value,
                "volatility": self.volatility, "confidence": self.confidence,
                "durability": self.durability, "evidence_quality": self.evidence_quality,
                "temporal_scope": self.temporal_scope, "source_ids": list(self.source_ids),
                "metadata": dict(self.metadata)}


class MemoryAdmissionPolicy:
    """Hard guards first, followed by conservative durability rules."""

    def research(self, *, research_ok: bool, temporal_scope: str, response_mode: str,
                 plan_compliance: bool, fallback_used: bool, required_fact_coverage: float,
                 recommendation_confidence: str, evidence_conflict: bool,
                 temporal_claim_conflict: bool, search_failed: bool,
                 facts: Iterable[dict[str, Any]], source_ids: Iterable[str]) -> MemoryAdmissionDecision:
        facts = list(facts)
        source_ids = tuple(dict.fromkeys(source_ids))[:8]
        volatilities = [str(f.get("volatility", "MEDIUM")).upper() for f in facts]
        volatility = ("HIGH" if "HIGH" in volatilities else "MEDIUM" if "MEDIUM" in volatilities else "LOW")
        durable = [f for f in facts if str(f.get("volatility", "MEDIUM")).upper() == "LOW" or
                   str(f.get("temporal_basis", "")) in {"VALIDITY_INTERVAL", "TIMELESS_FACT"}]
        confidence = round(sum(float(f.get("confidence", 0)) for f in facts) / len(facts), 3) if facts else 0.0
        evidence_quality = round(min(1.0, required_fact_coverage) * confidence, 3)
        durability = (0.9 if facts and len(durable) == len(facts) else
                      0.65 if durable else 0.2 if volatility == "HIGH" else 0.45)
        base = {"source_type": "research", "volatility": volatility, "confidence": confidence,
                "durability": durability, "evidence_quality": evidence_quality,
                "temporal_scope": temporal_scope, "source_ids": source_ids,
                "metadata": {"fallback_used": fallback_used,
                             "recommendation_confidence": recommendation_confidence,
                             "required_fact_coverage": required_fact_coverage}}
        if search_failed or not research_ok:
            return MemoryAdmissionDecision(False, "search_failure", memory_class=MemoryClass.SESSION_RELEVANT,
                                           **base)
        if not facts:
            return MemoryAdmissionDecision(False, "insufficient_evidence", memory_class=MemoryClass.SESSION_RELEVANT,
                                           **base)
        if response_mode == "insufficient_current_evidence":
            return MemoryAdmissionDecision(False, "insufficient_evidence", memory_class=MemoryClass.SESSION_RELEVANT,
                                           **base)
        if temporal_claim_conflict:
            return MemoryAdmissionDecision(False, "temporal_claim_conflict", memory_class=MemoryClass.SESSION_RELEVANT,
                                           **base)
        if evidence_conflict:
            return MemoryAdmissionDecision(False, "unresolved_evidence_conflict",
                                           memory_class=MemoryClass.SESSION_RELEVANT, **base)
        if recommendation_confidence in {"LOW", "INSUFFICIENT"}:
            return MemoryAdmissionDecision(False, "incomplete_recommendation",
                                           memory_class=MemoryClass.SESSION_RELEVANT, **base)
        if not plan_compliance and not durable:
            return MemoryAdmissionDecision(False, "noncompliant_without_durable_facts",
                                           memory_class=MemoryClass.SESSION_RELEVANT, **base)
        if volatility == "HIGH":
            return MemoryAdmissionDecision(False, "high_volatility", memory_class=MemoryClass.EPHEMERAL, **base)
        if temporal_scope == "TIMELESS" and volatility == "LOW" and evidence_quality >= .35:
            return MemoryAdmissionDecision(True, "durable_validated_research",
                                           memory_class=MemoryClass.CANDIDATE, **base)
        if durability >= .8 and evidence_quality >= .35:
            return MemoryAdmissionDecision(True, "durable_validated_facts",
                                           memory_class=MemoryClass.CANDIDATE, **base)
        return MemoryAdmissionDecision(False, "session_relevant", memory_class=MemoryClass.SESSION_RELEVANT, **base)

    def conversation(self, text: str, *, decision_detected: bool = False,
                     structured_candidate: bool = False) -> MemoryAdmissionDecision:
        lower = text.casefold()
        requested = any(marker in lower for marker in ("merk dir", "behalte das", "remember this"))
        high = any(marker in lower for marker in ("wetter", "temperatur", "aktienkurs", "live score"))
        volatility = "HIGH" if high else "LOW"
        eligible = requested or decision_detected or structured_candidate
        reason = ("user_requested" if requested else "durable_decision" if decision_detected else
                  "structured_conversation_memory" if structured_candidate else "no_memory_intent")
        return MemoryAdmissionDecision(eligible, reason, "user_explicit" if requested else "decision",
                                       MemoryClass.CANDIDATE if eligible else MemoryClass.SESSION_RELEVANT,
                                       volatility, .55, .75 if not high else .25, .55, "CURRENT" if high else "TIMELESS",
                                       metadata={"user_requested": requested, "decision_detected": decision_detected})


class MemoryAdmissionObservability:
    def __init__(self) -> None:
        self.last_decision: dict[str, Any] = {}
        self.counters: Counter[str] = Counter()

    def record(self, decision: MemoryAdmissionDecision, *, duplicate: bool = False,
               persisted: bool | None = None) -> None:
        self.counters["candidates_considered"] += 1
        if duplicate:
            self.counters["suppressed_duplicate"] += 1
        elif decision.admit and persisted is not False:
            self.counters["candidates_admitted"] += 1
        elif decision.memory_class == MemoryClass.EPHEMERAL:
            self.counters["suppressed_ephemeral"] += 1
        elif decision.reason in {"insufficient_evidence", "incomplete_recommendation",
                                 "noncompliant_without_durable_facts"}:
            self.counters["suppressed_insufficient_evidence"] += 1
        elif decision.reason == "search_failure":
            self.counters["suppressed_search_failure"] += 1
        elif "conflict" in decision.reason:
            self.counters["suppressed_conflict"] += 1
        self.last_decision = decision.debug() | {"duplicate": duplicate}

    def debug(self) -> dict[str, Any]:
        keys = ("candidates_considered", "candidates_admitted", "suppressed_ephemeral",
                "suppressed_insufficient_evidence", "suppressed_search_failure",
                "suppressed_conflict", "suppressed_duplicate")
        return {"last_decision": dict(self.last_decision),
                **{key: self.counters[key] for key in keys}}
