from avacore.core.cognitive_workspace import SelfModel, run_post_llm_gate
from avacore.core.grounding import build_grounding_context
from avacore.core.orbits import CognitiveOrbit
from avacore.core.response_plan import build_response_plan


def _coherence_context(question: str):
    orbit = CognitiveOrbit("orbit_coherence", "Long-term coherence / orbit reactivation",
                           "Reliable reactivation of relevant Cognitive Orbits", status="open",
                           metadata={"semantic_signature": {"terms": ["kohärenz", "reaktivierung", "orbit"],
                                                             "topics": ["Long-term cognitive coherence"]}})
    context = build_grounding_context(question, SelfModel(), orbits=[orbit])
    context.response_plan = build_response_plan(question, context)
    return context, context.response_plan


def test_recall_plan_uses_structured_subject_and_required_facts():
    context, plan = _coherence_context("Was hatten wir bei AvaCore noch offen?")
    assert plan.response_mode == "recall_summary"
    assert plan.primary_subject["source_id"] == "orbit_coherence"
    assert plan.primary_subject["subject_label"] == "Long-term cognitive coherence"
    assert plan.primary_subject["status"] == "open"
    assert 2 <= len(plan.required_facts) <= 5
    assert all(f.source_id == "orbit_coherence" for f in plan.required_facts)
    assert "AVA RESPONSE PLAN" in plan.prompt()
    assert context.debug()["response_plan_active"]


def test_ignored_plan_falls_back_and_good_answer_is_preserved():
    context, plan = _coherence_context("Was hatten wir bei AvaCore noch offen?")
    generic = "Wir hatten über Hardware, VRAM und lokale Modelle gesprochen."
    answer, gate = run_post_llm_gate(generic, SelfModel(), grounding=context, response_plan=plan)
    assert gate["plan_compliance"] is False
    assert gate["action"] == "response_plan_fallback"
    assert "coherence" in answer and "reactivation" in answer and "Status: open" in answer
    good = ("Ein offenes Thema ist weiterhin die langfristige Kohärenz: Relevante Cognitive Orbits "
            "sollen später zuverlässig reaktiviert werden.")
    answer, gate = run_post_llm_gate(good, SelfModel(), grounding=context, response_plan=plan)
    assert gate["plan_compliance"] is True
    assert gate["action"] == "none"
    assert answer == good
    assert gate["required_fact_coverage"] == 1.0


def test_technical_plan_and_identity_omission_fallback():
    context, plan = _coherence_context("Wenn du mir eine technische Frage stellen dürftest, welche wäre das?")
    assert plan.response_mode == "technical_question"
    assert plan.requested_output_type == "question"
    raw = "Als Sprachmodell kann ich Fragen zu Programmierung oder Wissen stellen.\nIch kann viele technische Themen nennen."
    answer, gate = run_post_llm_gate(raw, SelfModel(), grounding=context, response_plan=plan)
    assert gate["conflicts"] == ["identity_conflict", "response_plan_omission"]
    assert gate["action"] == "response_plan_fallback"
    assert answer.startswith("Eine technische Frage aus meinem aktuellen AvaCore-Zustand betrifft:")
    assert "coherence" in answer or "reactivation" in answer


def test_user_sentence_is_not_awkward_technical_subject():
    title = "Ich bin noch nicht überzeugt, dass dein Orbit-System langfristig relevante Themen zuverlässig wieder aufgreift."
    orbit = CognitiveOrbit("orbit_sentence", title, "Zuverlässige langfristige Reaktivierung relevanter Orbits",
                           metadata={"semantic_signature": {"terms": ["langfristig", "reaktivierung", "orbit"],
                                                             "topics": ["langfristige Orbit-Reaktivierung"]}})
    question = "Wenn du mir eine technische Frage stellen dürftest, welche wäre das?"
    context = build_grounding_context(question, SelfModel(), orbits=[orbit])
    plan = build_response_plan(question, context)
    assert plan.primary_subject["subject_label"] != title
    rendered = plan.render()
    assert title not in rendered
    assert "Ich bin noch nicht überzeugt" not in rendered


def test_identity_missing_evidence_and_general_plan_boundaries():
    model = SelfModel(underlying_model="Gemma 4")
    identity = build_response_plan("Wer bist du?", build_grounding_context("Wer bist du?", model))
    assert identity.response_mode == "identity_answer"
    assert identity.render().startswith("Ich bin Ava")
    worker = build_response_plan("Welches Sprachmodell verwendest du?",
                                 build_grounding_context("Welches Sprachmodell verwendest du?", model))
    assert worker.response_mode == "worker_model_answer"
    assert "Gemma 4" in worker.render()
    missing = build_response_plan("Was hatten wir bei AvaCore noch offen?",
                                  build_grounding_context("Was hatten wir bei AvaCore noch offen?", model))
    assert missing.evidence_available is False
    assert "nicht genug belastbare Hinweise" in missing.render()
    assert build_response_plan("Was ist OPC UA?", build_grounding_context("Was ist OPC UA?", model)) is None
