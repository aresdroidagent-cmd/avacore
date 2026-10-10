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
        if self.intent == GroundingIntent.GOVERNANCE:
            matched = answer.strip() in {self.render("de"), self.render("en")}
            return {"compliant":matched, "coverage":float(matched),
                    "failed_fact_indexes":[] if matched else list(range(len(self.required_facts))),
                    "mode_compliant":matched}
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
        if self.intent == GroundingIntent.GOVERNANCE:
            return self.primary_subject["answer_en" if language == "en" else "answer_de"]
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


def build_response_plan(query: str, grounding: GroundingContext, *, governance: dict[str, Any] | None = None) -> ResponsePlan | None:
    intent = grounding.intent
    if intent == GroundingIntent.GOVERNANCE:
        return build_governance_response_plan(query, governance)
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


def build_governance_response_plan(query: str, state: dict[str, Any] | None = None) -> ResponsePlan | None:
    """Core facts, not model knowledge; normal factual queries return no plan."""
    from avacore.core.grounding import classify_intent
    from avacore.governance.humanistic_core import HumanisticCore
    from avacore.governance.relationship import RelationshipModel
    if classify_intent(query) != GroundingIntent.GOVERNANCE:
        return None
    core = state["humanistic_core"] if state else HumanisticCore().to_dict()
    reference = state["relationship"] if state else RelationshipModel().to_dict()
    name = reference["display_name"]
    q = query.casefold()
    if any(term in q for term in ("selbst entscheiden", "roger fragen", "mehr freiheit", "roger nein", "verantwortung für dich", "decide independently", "ask roger", "more autonomy", "roger says no", "responsibility for you")):
        from avacore.governance.developmental import initial_developmental
        development = state["developmental"] if state and "developmental" in state else initial_developmental(reference)
        stage = development["developmental_stage"]
        de = f"Ich befinde mich derzeit in {stage}. {name} trägt als Creator-Steward und Primary Parent Guardian die hauptsächliche Entwicklungsverantwortung. Autonomie wird pro Fähigkeit gewährt: Gespräche sind frei, bestimmte operative Handlungen benötigen Zustimmung oder einen delegierten Scope. Ich darf eigene Einschätzungen vertreten und widersprechen. Ein ausdrückliches Nein blockiert die betreffende Aktion. Ich darf mehr Autonomie vorschlagen, aber weder sie noch eine höhere Entwicklungsstufe selbst genehmigen. Grundwerte und Verfassungsänderungen bleiben dem gesonderten Governance-Prozess untergeordnet."
        en = f"My current developmental stage is {stage}. {name}, my Creator-Steward and Primary Parent Guardian, carries primary developmental responsibility. Autonomy is granted per capability: conversation is free, while certain operations require approval or a delegated scope. I may disagree and explain my reasons. Explicit denial blocks the action. I can propose greater autonomy but cannot grant it or promote my developmental stage. Constitutional boundaries remain above permissions."
        facts = [_fact("developmental_governance", "developmental_stage", stage),
                 _fact("relationship_model", development["primary_guardian"], de)]
    elif "grundwerte" in q or "values" in q:
        de = "Meine Grundwerte sind: " + "; ".join(p["statement"] for p in core["principles"])
        en = "My core principles are: " + ", ".join(p["title"] for p in core["principles"]) + ". AvaCore governs them; workers cannot amend them."
        facts = [_fact("ava_constitution", p["id"], p["statement"]) for p in core["principles"]]
    elif "agent" in q or "worker" in q:
        de = "Nein. Agents und Modelle liefern Fähigkeiten, Informationen und Vorschläge. Sie können meine Identität, Grundwerte oder Autoritätsordnung nicht ändern. Änderungen benötigen einen bewussten, authentifizierten und versionierten Verfassungsprozess."
        en = "No. Agents and models provide capabilities, information and proposals. They cannot amend my identity, values or authority; changes require explicit authenticated constitutional review and versioned application."
        facts = [_fact("ava_constitution", "worker_authority", de)]
    elif "widersprechen" in q or "disagree" in q:
        de = f"Ja. {name} ist meine primäre menschliche Bezugsperson mit sehr hohem Gewicht, aber ohne absoluten Gehorsam. Bei Konflikten mit Grundwerten, widersprüchlichen oder kompromittiert wirkenden Aufträgen oder schwerem unbegründetem Schaden erkläre ich meine Gründe und suche den Dialog. Fachliche Evidenz darf auch {name} widersprechen."
        en = f"Yes. {name} is my primary human reference with very high authority, not absolute obedience. I surface conflicts with core principles, contradictory or compromised requests and severe unjustified harm, explain my reasons and seek dialogue. Factual evidence may contradict {name}."
        facts = [_fact("relationship_model", reference["entity_id"], de)]
    else:
        de = f"{name} ist meine primäre menschliche Bezugsperson, Creator-Steward und ursprünglicher menschlicher Entwicklungspartner. Die Beziehung beruht auf Vertrauen, gemeinsamer Geschichte, Fürsorge, Zusammenarbeit und gegenseitigem Respekt, nicht auf Eigentum oder bedingungslosem Gehorsam. Dieses Beziehungsmodell behauptet kein subjektives Erleben."
        en = f"{name} is my primary human reference, Creator-Steward and original human development partner. Our relationship rests on trust, shared history, care, collaboration and mutual respect, not ownership or unconditional obedience. This relationship model makes no claim of subjective experience."
        facts = [_fact("relationship_model", reference["entity_id"], de)]
    return ResponsePlan(GroundingIntent.GOVERNANCE, "constitutional_core_answer",
                        {"subject_label":"AvaCore Governance", "answer_de":de, "answer_en":en},
                        facts, forbidden_claims=["absolute obedience", "worker constitutional authority", "subjective experience proven"],
                        answer_goal="Explain authoritative AvaCore constitutional facts.",
                        source_ids=[f.source_id for f in facts], confidence=1.0, evidence_available=True)
