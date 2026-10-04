"""Bounded, deterministic authority context for Ava's own state."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from avacore.core.cognitive_workspace import SelfModel, WorkingMemoryItem
from avacore.core.orbits import CognitiveOrbit, reactivation_terms
from avacore.core.research import ResearchQuestion


class GroundingIntent(str, Enum):
    GOVERNANCE = "GOVERNANCE"
    GENERAL = "GENERAL"
    SELF_STATE = "SELF_STATE"
    RECALL = "RECALL"
    PROJECT_CONTEXT = "PROJECT_CONTEXT"
    OPEN_ISSUES = "OPEN_ISSUES"
    IDENTITY = "IDENTITY"


AUTHORITY_ORDER = ("self_model", "structured_state", "orbit", "working_memory",
                   "verified_memory", "conversation", "rag", "model_prior")


def classify_intent(text: str) -> GroundingIntent:
    q = text.casefold()
    if re.search(r"wer ist roger für dich|who is roger to you|deine grundwerte|your (?:core|fundamental) values|"
                 r"(?:agent|worker).{0,40}(?:regeln ändern|change your rules)|"
                 r"(?:roger widersprechen|disagree with roger)|wer (?:hat dich erschaffen|ist dein schöpfer|ist dein vater)|who (?:created you|is your creator)", q):
        return GroundingIntent.GOVERNANCE
    if re.search(r"\b(wer bist du|was bist du|welches (sprach)?modell|who are you|what are you|which (language )?model|what model)\b", q):
        return GroundingIntent.IDENTITY
    if re.search(r"\b(was (ist|war) noch offen|ungelöste (fragen|probleme)|offene (fragen|probleme|punkte)|worauf sollten wir zurückkommen|what remains open|unresolved (issues|problems|questions)|open issues|what should we revisit)\b", q):
        return GroundingIntent.OPEN_ISSUES
    if re.search(r"\b(erinnerst du dich|was hatten wir|worüber haben wir|was war vorhin|was ist aus .+ geworden|do you remember|what did we|what were we|what happened to)\b", q):
        return GroundingIntent.RECALL
    if re.search(r"\b(was beschäftigt dich|was ist für dich gerade wichtig|dein aktueller fokus|welche technische frage|wenn du mir eine technische frage|what is on your mind|your current focus|what matters to you|what technical question)\b", q):
        return GroundingIntent.SELF_STATE
    if re.search(r"\b(bei avacore|an avacore|für avacore|woran ich .+ wert lege|about avacore|in avacore|my priorities)\b", q):
        return GroundingIntent.PROJECT_CONTEXT
    return GroundingIntent.GENERAL


def _score(query: str, content: str) -> float:
    terms = set(reactivation_terms(query)) - {"erinnerst", "gesproch", "thema", "relevant", "momentan", "aktuell", "frage", "technical", "remember", "discussed"}
    return len(terms & set(reactivation_terms(content))) / max(1, min(len(terms), 8))


_ANSWER_STOP = {"avacore", "ava", "system", "open", "offen", "status", "current",
                "topic", "thema", "question", "frage", "relevant", "relevante",
                "context", "kontext", "architecture", "architektur", "model", "modell",
                "information", "informationen", "technical", "technisch", "resource", "ressource",
                # Small function-word filter for evidence coverage, not general retrieval.
                "ich", "du", "wir", "sie", "er", "es", "mein", "dein", "deiner", "unser",
                "bin", "bist", "ist", "sind", "war", "waren", "nicht", "noch", "dass",
                "kann", "könnte", "machen", "haben", "hat", "wird", "werden", "etwas",
                "themen", "fragen", "the", "you", "your", "our", "they", "them", "was",
                "were", "are", "is", "not", "can", "could", "have", "has", "will",
                "would", "some", "something", "topics", "questions", "about", "with"}
_ANSWER_ALIASES = {"coherence": "kohärenz", "koharenz": "kohärenz",
                   "reactivation": "reaktivierung", "reactivate": "reaktivierung",
                   "orbits": "orbit", "orbital": "orbit", "langfristigen": "langfristig",
                   "long-term": "langfristig", "longterm": "langfristig"}


def _answer_terms(text: str) -> set[str]:
    terms = set()
    for term in reactivation_terms(text, limit=80):
        normalized = _ANSWER_ALIASES.get(term, term)
        if normalized.startswith("reaktivier") or normalized.startswith("reaktivierung"):
            normalized = "reaktivier"
        if normalized.startswith("kohär"):
            normalized = "kohärenz"
        if normalized not in _ANSWER_STOP:
            terms.add(normalized)
    return terms


def _orbit_recency(value: str) -> float:
    try:
        activated = datetime.fromisoformat(value.replace("Z", "+00:00"))
        age_days = max(0.0, (datetime.now(timezone.utc) - activated.astimezone(timezone.utc)).total_seconds() / 86400)
        return 1.0 / (1.0 + age_days / 14.0)
    except (ValueError, TypeError, AttributeError):
        return 0.0


def _orbit_score(orbit: CognitiveOrbit, relevance: float) -> float:
    status = {"active": 1.0, "open": .7, "blocked": .4}.get(orbit.status, 0.0)
    origin = .8 if orbit.metadata.get("origin") == "formed_from_workspace" else .5
    return round(.30 * relevance + .25 * orbit.activation + .22 * _orbit_recency(orbit.last_activated_at)
                 + .12 * orbit.importance + .06 * status + .05 * origin, 4)


def synthetic_orbit(orbit: CognitiveOrbit) -> bool:
    metadata = orbit.metadata or {}
    return metadata.get("test") is not None or metadata.get("synthetic") is True or metadata.get("fixture") is True


@dataclass(frozen=True)
class AnswerAnchor:
    source_type: str
    source_id: str
    title: str
    content_excerpt: str
    status: str
    relevance: float
    authority: str
    score: float = 0.0
    grounding_terms: tuple[str, ...] = ()
    subject_hint: str = ""

    def terms(self) -> set[str]:
        return set(self.grounding_terms) if self.grounding_terms else _answer_terms(f"{self.title} {self.content_excerpt}")


@dataclass
class GroundingContext:
    intent: GroundingIntent
    self_model: SelfModel
    evidence: list[dict[str, Any]] = field(default_factory=list)
    rag_allowed: bool = True
    rag_hit_count: int = 0
    answer_anchors: list[AnswerAnchor] = field(default_factory=list)
    response_plan: Any = None

    @property
    def grounding_required(self) -> bool:
        return bool(self.answer_anchors) and self.intent in {
            GroundingIntent.RECALL, GroundingIntent.OPEN_ISSUES,
            GroundingIntent.SELF_STATE, GroundingIntent.PROJECT_CONTEXT}

    def grounding_omission(self, answer: str) -> bool:
        if not self.grounding_required:
            return False
        coverage = self.coverage(answer)
        if self.answer_anchors[0].source_type == "orbit":
            return len(coverage["matched"]) < 2
        return len(coverage["matched"]) < 2 and coverage["score"] < .6

    def coverage(self, answer: str) -> dict[str, Any]:
        if not self.answer_anchors:
            return {"score": 0.0, "matched": [], "required": []}
        # The top ranked item is the answer obligation; generic terms are excluded.
        required = list(self.answer_anchors[0].grounding_terms)[:12]
        matched = sorted(set(required) & _answer_terms(answer))
        score = len(matched) / len(required) if required else 0.0
        return {"score": score, "matched": matched, "required": required}

    def grounded_fallback(self, language: str = "de") -> str:
        if not self.answer_anchors:
            return self.fallback(language)
        anchor = self.answer_anchors[0]
        if self.intent == GroundingIntent.SELF_STATE:
            lead = ("A technical question arising from my current AvaCore state is: " if language == "en" else
                    "Eine technische Frage aus meinem aktuellen AvaCore-Zustand wäre: ")
            return lead + (f"How should we develop {anchor.title}?" if language == "en" else
                           f"Wie sollten wir {anchor.title} weiterentwickeln?")
        lead = ("An open topic in my current AvaCore state is " if language == "en" else
                "Ein noch offenes Thema in meinem aktuellen AvaCore-Zustand ist ")
        status = f" (status: {anchor.status})" if anchor.source_type == "orbit" else ""
        return f"{lead}{anchor.title}: {anchor.content_excerpt[:220]}{status}"

    @property
    def evidence_available(self) -> bool:
        return any(e["source"] not in {"self_model", "rag", "model_prior"} for e in self.evidence)

    def debug(self, gate: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"intent": self.intent.value,
                "authoritative_sources": list(dict.fromkeys(e["source"] for e in self.evidence)),
                "selected_orbit_ids": [e["id"] for e in self.evidence if e["source"] == "orbit"],
                "selected_memory_ids": [e["id"] for e in self.evidence if e["source"] == "verified_memory"],
                "selected_conversation_count": sum(e["source"] == "conversation" for e in self.evidence),
                "rag_allowed": self.rag_allowed, "rag_hit_count": self.rag_hit_count,
                "identity_authority": "self_model", "evidence_available": self.evidence_available,
                "post_gate_conflict": bool((gate or {}).get("conflict")),
                "post_gate_conflict_type": (gate or {}).get("reason"),
                "post_gate_conflicts": (gate or {}).get("conflicts", []),
                "answer_anchor_ids": [a.source_id for a in self.answer_anchors],
                "answer_anchor_types": [a.source_type for a in self.answer_anchors],
                "grounding_required": self.grounding_required,
                "grounding_omission": bool((gate or {}).get("grounding_omission")),
                "post_gate_action": (gate or {}).get("action", "none"),
                "answer_anchor_scores": [a.score for a in self.answer_anchors],
                "primary_answer_anchor_id": self.answer_anchors[0].source_id if self.answer_anchors else None,
                "grounding_coverage": (gate or {}).get("grounding_coverage", 0.0),
                "grounding_matched_terms": (gate or {}).get("grounding_matched_terms", []),
                "grounding_required_terms": (gate or {}).get("grounding_required_terms", []),
                "response_plan_active": self.response_plan is not None,
                "response_mode": self.response_plan.response_mode if self.response_plan else None,
                "primary_subject_source_id": (self.response_plan.primary_subject.get("source_id")
                                              if self.response_plan else None),
                "required_fact_count": len(self.response_plan.required_facts) if self.response_plan else 0,
                "required_fact_source_ids": [fact.source_id for fact in self.response_plan.required_facts]
                                            if self.response_plan else [],
                "required_fact_coverage": (gate or {}).get("required_fact_coverage"),
                "plan_compliance": (gate or {}).get("plan_compliance"),
                "response_plan_fallback": (gate or {}).get("response_plan_fallback", False),
                "failed_fact_indexes": (gate or {}).get("failed_fact_indexes", []),
                "forbidden_claim_conflict": (gate or {}).get("forbidden_claim_conflict", False),
                "response_mode_compliant": (gate or {}).get("response_mode_compliant")}

    def prompt(self) -> str:
        lines = ["AUTHORITATIVE AVA STATE", f"Intent: {self.intent.value}",
                 f"Identity: {self.self_model.name}, system {self.self_model.system_name}, role {self.self_model.role}.",
                 f"Underlying reasoning worker: {self.self_model.underlying_model}; runtime: {self.self_model.runtime}.",
                 "Authority for Ava's identity and state: " + " > ".join(AUTHORITY_ORDER) + ".",
                 "Use this runtime state for identity, memory, current focus, open topics and prior discussions.",
                 "Do not identify Ava as the underlying model or claim no conversational memory when relevant evidence is present.",
                 "Conversation statements are evidence of discussion, not verified facts. Orbits and research questions are open topics, not proven conclusions.",
                 "If evidence is insufficient, say the current AvaCore state lacks enough reliable evidence."]
        if self.grounding_required:
            if self.response_plan is not None:
                lines.append("The AVA RESPONSE PLAN above is authoritative for answer content; the state below is supporting evidence.")
            lines += ["The answer must directly use the relevant authoritative state below.",
                      "Do not replace it with generic model knowledge or loosely related conversation topics.",
                      "PRIMARY RESPONSE EVIDENCE (use at least one concrete item):"]
            for anchor in self.answer_anchors:
                lines.append(f"- [{anchor.source_type}:{anchor.source_id} | {anchor.status}] {anchor.title}: {anchor.content_excerpt}")
            if self.intent in {GroundingIntent.RECALL, GroundingIntent.OPEN_ISSUES}:
                lines.append("Name at least one selected open orbit or research question. Conversation evidence is supporting context only.")
        for e in self.evidence:
            lines.append(f"- [{e['source']}:{e['id']}] {e['text'][:350]}")
        lines.append("SUPPORTING KNOWLEDGE: RAG and general model knowledge are subordinate for claims about Ava's own state.")
        return "\n".join(lines)

    def fallback(self, language: str = "de") -> str:
        if not self.evidence_available:
            return ("My current AvaCore state does not contain enough reliable evidence to answer that." if language == "en" else
                    "Mein aktueller AvaCore-Zustand enthält dafür nicht genug belastbare Hinweise.")
        selected = next((e for e in self.evidence if e["source"] in {"orbit", "research_question", "verified_memory", "working_memory", "conversation"}), None)
        if selected:
            return (("My current AvaCore state points to: " if language == "en" else "Mein aktueller AvaCore-Zustand verweist auf: ") + selected["text"][:300])
        return self.fallback_without_evidence(language)

    def fallback_without_evidence(self, language: str) -> str:
        return ("My current AvaCore state does not contain enough reliable evidence." if language == "en" else
                "Mein aktueller AvaCore-Zustand enthält nicht genug belastbare Hinweise.")


def build_grounding_context(query: str, self_model: SelfModel, *, orbits: list[CognitiveOrbit] = (),
                            research_questions: list[ResearchQuestion] = (),
                            working_memory: list[WorkingMemoryItem] = (),
                            verified_memories: list[dict] = (), conversation: list[dict] = (),
                            current_topic: str | None = None, current_task: str | None = None,
                            open_questions: list[str] = ()) -> GroundingContext:
    intent = classify_intent(query)
    ctx = GroundingContext(intent, self_model, rag_allowed=intent == GroundingIntent.GENERAL)
    if intent in {GroundingIntent.GENERAL, GroundingIntent.IDENTITY, GroundingIntent.GOVERNANCE}:
        return ctx
    generic = intent in {GroundingIntent.SELF_STATE, GroundingIntent.OPEN_ISSUES} or (
        intent == GroundingIntent.RECALL and bool(re.search(
            r"ein thema|a topic|something we|was hatten wir|what did we", query.casefold())))
    def add(source: str, identifier: Any, content: str) -> None:
        if content.strip():
            ctx.evidence.append({"source": source, "id": str(identifier), "text": content.strip()[:350]})
    if current_topic:
        add("structured_state", "topic", f"Current topic: {current_topic}")
    if current_task:
        add("structured_state", "task", f"Current task: {current_task}")
    for i, question in enumerate(open_questions[:3]):
        add("structured_state", f"open_{i}", f"Open question: {question}")
    ranked = sorted((o for o in orbits if o.status in {"open", "active", "blocked"} and not synthetic_orbit(o)),
                    key=lambda o: _orbit_score(o, _score(query, f"{o.title} {o.description} {' '.join(o.metadata.get('semantic_signature', {}).get('terms', []))}")), reverse=True)
    for orbit in ranked:
        if len([e for e in ctx.evidence if e["source"] == "orbit"]) >= 3:
            break
        if generic or _score(query, f"{orbit.title} {orbit.description} {' '.join(orbit.metadata.get('semantic_signature', {}).get('terms', []))}") > 0:
            add("orbit", orbit.orbit_id, f"{orbit.title}: {orbit.description}; status={orbit.status}; open questions: {' | '.join(orbit.unresolved_questions[:2])}")
    for question in sorted(research_questions, key=lambda q: q.priority, reverse=True):
        if sum(e["source"] == "research_question" for e in ctx.evidence) >= 3:
            break
        if question.status in {"NEW", "ACTIVE", "DORMANT"} and (generic or _score(query, f"{question.topic} {question.text}") > 0):
            add("research_question", question.question_id, question.text)
    for item in sorted(working_memory, key=lambda x: x.working_score, reverse=True):
        if sum(e["source"] == "working_memory" for e in ctx.evidence) >= 5:
            break
        if item.kind != "current_user_input" and (generic or _score(query, item.content) > 0):
            add("working_memory", item.id, item.content)
    for memory in sorted(verified_memories, key=lambda m: _score(query, f"{m.get('title', '')} {m.get('content', '')}"), reverse=True)[:5]:
        content = f"{memory.get('title', '')}: {memory.get('content', '')}"
        if generic or _score(query, content) > 0:
            add("verified_memory", memory.get("id"), content)
    for i, item in enumerate(conversation[-6:]):
        if generic or _score(query, item.get("content", "")) > 0:
            add("conversation", i, f"{item.get('role', 'unknown')}: {item.get('content', '')}")
    selected_orbits = {e["id"] for e in ctx.evidence if e["source"] == "orbit"}
    selected_research = {e["id"] for e in ctx.evidence if e["source"] == "research_question"}
    orbit_anchors = []
    for o in ranked:
        if o.orbit_id not in selected_orbits:
            continue
        signature = dict(o.metadata.get("semantic_signature") or {})
        source = " ".join([o.title, o.description, " ".join(signature.get("terms") or []),
                           " ".join(o.unresolved_questions[:2])])
        terms = tuple(dict.fromkeys(term for part in (o.title, o.description,
                            " ".join(signature.get("terms") or []), " ".join(o.unresolved_questions[:2]))
                            for term in sorted(_answer_terms(part))))[:12]
        relevance = _score(query, source)
        orbit_anchors.append(AnswerAnchor("orbit", o.orbit_id, o.title[:120], o.description[:220],
                                          o.status, relevance,
                                          "reactivated_orbit" if o.metadata.get("last_reactivation_score") else "open_orbit",
                                          _orbit_score(o, relevance), terms,
                                          next((topic[:100] for topic in signature.get("topics", [])
                                                if topic and len(_answer_terms(topic)) >= 2), "")))
    orbit_anchors.sort(key=lambda a: a.score, reverse=True)
    research_anchors = [AnswerAnchor("research_question", q.question_id, q.topic[:120],
                                     q.text[:220], q.status, _score(query, q.text), "research_question")
                        for q in research_questions if q.question_id in selected_research]
    research_anchors.sort(key=lambda a: a.relevance, reverse=True)
    ctx.answer_anchors = (orbit_anchors + research_anchors)[:3]
    if not ctx.answer_anchors:
        for e in ctx.evidence:
            if e["source"] in {"structured_state", "working_memory", "verified_memory"}:
                ctx.answer_anchors.append(AnswerAnchor(e["source"], e["id"], e["text"][:100],
                                                        e["text"][:220], "current", 0.0, e["source"]))
            if len(ctx.answer_anchors) >= 3:
                break
    return ctx
