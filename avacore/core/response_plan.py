"""Deterministic projection of selected cognitive evidence into an answer contract."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from avacore.core.grounding import GroundingContext, GroundingIntent, _answer_terms


@dataclass(frozen=True)
class PlanFact:
    source_type: str
    source_id: str
    fact: str
    terms: tuple[str, ...]
    kind: str = "content"


@dataclass
class ResponsePlan:
    intent: GroundingIntent
    response_mode: str
    primary_subject: dict[str, Any]
    required_facts: list[PlanFact] = field(default_factory=list)
    optional_facts: list[PlanFact] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    answer_goal: str = ""
    source_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_available: bool = False
    requested_output_type: str = "answer"

    def prompt(self) -> str:
        lines = ["AVA RESPONSE PLAN", "Your task is to verbalize this response plan.",
                 "Do not choose a different topic or replace required facts with general model knowledge.",
                 "Use every required fact relevant to the requested response type.",
                 "Do not contradict forbidden claims. You are the language worker; AvaCore selected this plan.",
                 f"response_mode: {self.response_mode}", f"answer_goal: {self.answer_goal}",
                 f"requested_output_type: {self.requested_output_type}",
                 f"evidence_available: {str(self.evidence_available).lower()}",
                 f"primary_subject: {self.primary_subject.get('subject_label', '-')} | {self.primary_subject.get('subject_description', '-')}",
                 "required_facts:"]
        lines.extend(f"- [{fact.source_type}:{fact.source_id}] {fact.fact}" for fact in self.required_facts)
        lines.append("optional_facts:")
        lines.extend(f"- [{fact.source_type}:{fact.source_id}] {fact.fact}" for fact in self.optional_facts)
        lines.append("forbidden_claims:")
        lines.extend(f"- {claim}" for claim in self.forbidden_claims)
        return "\n".join(lines)

    def compliance(self, answer: str) -> dict[str, Any]:
        if not self.evidence_available and self.intent != GroundingIntent.IDENTITY:
            lower = answer.casefold()
            cautious = any(x in lower for x in ("nicht genug", "nicht ausreichend", "insufficient", "not enough", "uncertain"))
            return {"compliant": cautious, "coverage": 1.0 if cautious else 0.0,
                    "failed_fact_indexes": [], "mode_compliant": cautious}
        answer_terms = _answer_terms(answer)
        failed: list[int] = []
        for index, fact in enumerate(self.required_facts):
            if fact.kind == "status":
                matched = bool(re.search(r"\b(offen|offenes?|open|blocked|blockiert|active|aktiv)\b", answer, re.I))
            elif fact.kind == "identity":
                matched = fact.fact.casefold() in answer.casefold()
            else:
                matched = len(answer_terms & set(fact.terms)) >= min(2, len(fact.terms)) if fact.terms else False
            if not matched:
                failed.append(index)
        mode_ok = True
        if self.response_mode == "technical_question":
            mode_ok = "?" in answer or bool(re.search(r"\b(technische frage|technical question)\b", answer, re.I))
        coverage = (len(self.required_facts) - len(failed)) / len(self.required_facts) if self.required_facts else 1.0
        return {"compliant": not failed and mode_ok, "coverage": coverage,
                "failed_fact_indexes": failed[:5], "mode_compliant": mode_ok}

    def render(self, language: str = "de") -> str:
        if self.intent == GroundingIntent.IDENTITY:
            label = self.primary_subject.get("subject_label", "Ava")
            description = self.primary_subject.get("subject_description", "AvaCore")
            if self.response_mode == "worker_model_answer":
                return (f"I am {label}; my current reasoning worker is {description}." if language == "en" else
                        f"Ich bin {label}; mein aktueller Reasoning-Worker ist {description}.")
            return (f"I am {label} and run on {description}." if language == "en" else
                    f"Ich bin {label} und laufe auf {description}.")
        if not self.evidence_available:
            return ("My current AvaCore state does not contain enough reliable evidence to answer that." if language == "en" else
                    "Mein aktueller AvaCore-Zustand enthält dafür nicht genug belastbare Hinweise.")
        label = self.primary_subject.get("subject_label", "")
        description = self.primary_subject.get("subject_description", "")
        status = self.primary_subject.get("status", "")
        if self.response_mode == "technical_question":
            lead = ("A technical question from my current AvaCore state concerns:" if language == "en" else
                    "Eine technische Frage aus meinem aktuellen AvaCore-Zustand betrifft:")
            safe_subject = description if _phrase(description) else label
            return f"{lead} {safe_subject}".strip()
        lead = ("A currently open topic concerns" if language == "en" else "Ein noch offenes Thema betrifft")
        detail = (f" It concerns: {description}." if language == "en" else f" Dabei geht es um: {description}.") if description and description != label else "."
        status_line = f" Status: {status}." if status else ""
        return f"{lead} {label}.{detail}{status_line}".replace("..", ".")


def _phrase(text: str) -> bool:
    value = text.strip()
    return bool(value) and len(value) <= 100 and not re.match(r"^(ich|wir|du|i|we|you)\b", value, re.I) and not re.search(r"[.!?]$", value)


def _fact(source_type: str, source_id: str, text: str, kind: str = "content") -> PlanFact:
    return PlanFact(source_type, source_id, text[:240], tuple(sorted(_answer_terms(text)))[:8], kind)


def build_response_plan(query: str, grounding: GroundingContext) -> ResponsePlan | None:
    intent = grounding.intent
    if intent == GroundingIntent.GENERAL:
        return None
    model = grounding.self_model
    forbidden = ["Ava is the underlying language model"]
    if grounding.evidence_available:
        forbidden += ["Ava has no conversational memory", "No relevant state context exists"]
    if intent == GroundingIntent.IDENTITY:
        worker = bool(re.search(r"\b(sprachmodell|hintergrundmodell|modell verwendest|model do you use|language model)\b", query.casefold()))
        subject = {"source_type": "self_model", "source_id": "self_model", "subject_label": model.name,
                   "subject_description": model.underlying_model if worker else model.system_name,
                   "subject_terms": [model.name.casefold()]}
        facts = [_fact("self_model", "self_model", model.name, "identity")]
        if worker:
            facts.append(_fact("self_model", "self_model", model.underlying_model, "identity"))
        return ResponsePlan(intent, "worker_model_answer" if worker else "identity_answer", subject,
                            facts, forbidden_claims=forbidden, answer_goal="Answer according to SelfModel identity.",
                            source_ids=["self_model"], confidence=1.0, evidence_available=True)
    technical = intent == GroundingIntent.SELF_STATE and bool(re.search(r"\b(technische frage|technical question)\b", query.casefold()))
    mode = ("technical_question" if technical else
            {GroundingIntent.RECALL: "recall_summary", GroundingIntent.OPEN_ISSUES: "open_issue_summary",
             GroundingIntent.SELF_STATE: "self_state_summary", GroundingIntent.PROJECT_CONTEXT: "project_context_summary"}[intent])
    goals = {"technical_question": "Ask one technical question derived from the primary current open issue.",
             "recall_summary": "Directly identify the currently relevant remembered open issue.",
             "open_issue_summary": "Identify the most relevant currently open cognitive issue.",
             "self_state_summary": "Summarize the current relevant structured state.",
             "project_context_summary": "Answer from current AvaCore project state."}
    if not grounding.answer_anchors:
        return ResponsePlan(intent, mode, {}, forbidden_claims=forbidden, answer_goal=goals[mode],
                            evidence_available=False, requested_output_type="question" if technical else "answer")
    anchor = grounding.answer_anchors[0]
    # A complete user utterance is evidence, not a grammatical subject label.
    signature_topics = []
    for evidence in grounding.evidence:
        if evidence["source"] == "structured_state" and evidence["id"] == "topic":
            signature_topics.append(evidence["text"].removeprefix("Current topic: "))
    label = anchor.subject_hint if _phrase(anchor.subject_hint) else ""
    if not label:
        label = next((topic for topic in signature_topics if _phrase(topic) and len(_answer_terms(topic)) >= 2), "")
    if not label:
        label = anchor.title if _phrase(anchor.title) else " / ".join(anchor.grounding_terms[:3])
    raw_description = anchor.content_excerpt if anchor.content_excerpt != anchor.title else anchor.title
    description = raw_description if _phrase(raw_description) else " / ".join(anchor.grounding_terms[:5])
    if not description:
        description = label
    subject = {"source_type": anchor.source_type, "source_id": anchor.source_id,
               "subject_label": label[:100], "subject_description": description[:220],
               "subject_terms": list(anchor.grounding_terms[:12]), "status": anchor.status}
    facts = [_fact(anchor.source_type, anchor.source_id, label)]
    if raw_description != label and len(_answer_terms(raw_description)) >= 2:
        facts.append(_fact(anchor.source_type, anchor.source_id, raw_description))
    if anchor.source_type == "orbit" and mode in {"recall_summary", "open_issue_summary"}:
        facts.append(_fact(anchor.source_type, anchor.source_id, anchor.status, "status"))
    return ResponsePlan(intent, mode, subject, facts[:5],
                        optional_facts=[_fact(a.source_type, a.source_id, a.content_excerpt)
                                        for a in grounding.answer_anchors[1:3]],
                        forbidden_claims=forbidden, answer_goal=goals[mode],
                        source_ids=[a.source_id for a in grounding.answer_anchors[:3]],
                        confidence=min(1.0, max(.3, anchor.score)), evidence_available=True,
                        requested_output_type="question" if technical else "answer")
