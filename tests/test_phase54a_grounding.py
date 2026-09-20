from avacore.core.cognitive_workspace import SelfModel, run_post_llm_gate
from avacore.core.grounding import GroundingIntent, build_grounding_context, classify_intent
from avacore.core.orbits import CognitiveOrbit
from datetime import datetime, timedelta, timezone


def test_recall_selects_persistent_orbit_and_repairs_memory_claim():
    orbit = CognitiveOrbit("orbit_1", "Long-term coherence / orbit reactivation",
                           "AvaCore should reactivate relevant long-term topics", importance=.9)
    question = "Erinnerst du dich an ein Thema, über das wir vor einiger Zeit gesprochen haben und das noch relevant sein könnte?"
    context = build_grounding_context(question, SelfModel(), orbits=[orbit])
    assert context.intent == GroundingIntent.RECALL
    assert context.debug()["selected_orbit_ids"] == ["orbit_1"]
    assert context.evidence_available
    answer, gate = run_post_llm_gate("Ich habe keinen Zugriff auf frühere Gespräche.", SelfModel(), grounding=context)
    assert gate["reason"] == "memory_conflict"
    assert "Long-term coherence" in answer


def test_identity_gate_and_model_description():
    model = SelfModel(underlying_model="Gemma 4")
    context = build_grounding_context("Wer bist du?", model)
    assert context.intent == GroundingIntent.IDENTITY
    answer, gate = run_post_llm_gate("Ich bin ein großes Sprachmodell.", model, grounding=context)
    assert gate["reason"] == "identity_conflict"
    assert "Ich bin Ava" in answer
    answer, gate = run_post_llm_gate("Da ich ein großes Sprachmodell bin, frage ich ...", model, grounding=context)
    assert gate["reason"] == "identity_conflict" and "Ich bin Ava" in answer
    assert classify_intent("Welches Sprachmodell verwendest du?") == GroundingIntent.IDENTITY
    answer, gate = run_post_llm_gate("Mein Reasoning-Worker ist Gemma 4.", model, grounding=context)
    assert not gate["conflict"]
    assert "Gemma 4" in answer


def test_missing_recall_evidence_uses_state_specific_fallback():
    context = build_grounding_context("Erinnerst du dich an unser früheres Thema?", SelfModel())
    assert not context.evidence_available
    answer, gate = run_post_llm_gate("Ich kann mich nicht an frühere Gespräche erinnern.", SelfModel(), grounding=context)
    assert gate["reason"] == "memory_conflict"
    assert "aktueller AvaCore-Zustand" in answer


def test_state_intents_suppress_rag_and_general_knowledge_allows_it():
    for question in ("Was beschäftigt dich momentan?", "Was hatten wir bei AvaCore noch offen?"):
        context = build_grounding_context(question, SelfModel())
        assert not context.rag_allowed
    assert build_grounding_context("Erkläre OPC UA.", SelfModel()).rag_allowed


def test_recall_anchor_enforces_orbit_without_retry():
    orbit = CognitiveOrbit("orbit_1", "Long-term coherence / orbit reactivation",
                           "Long-term coherence and reliable reactivation of Cognitive Orbits", importance=.9)
    context = build_grounding_context("Was hatten wir bei AvaCore noch offen?", SelfModel(), orbits=[orbit])
    assert context.answer_anchors[0].source_id == "orbit_1"
    assert "PRIMARY RESPONSE EVIDENCE" in context.prompt()
    answer, gate = run_post_llm_gate("Wir hatten über Hardware und lokale LLMs gesprochen.", SelfModel(), grounding=context)
    assert gate["grounding_omission"] and gate["action"] == "grounded_fallback"
    assert "Long-term coherence" in answer
    good = "Ein offenes Thema war die langfristige Kohärenz und die zuverlässige Reaktivierung relevanter Cognitive Orbits."
    answer, gate = run_post_llm_gate(good, SelfModel(), grounding=context)
    assert answer == good
    assert not gate["grounding_omission"] and gate["action"] == "none"


def test_identity_preface_removed_without_losing_technical_question():
    orbit = CognitiveOrbit("orbit_1", "Cognitive Orbits archival", "Long-term Cognitive Orbits archival")
    context = build_grounding_context("Wenn du mir eine technische Frage stellen dürftest, welche wäre das?",
                                      SelfModel(), orbits=[orbit])
    model_answer = ("Als großes Sprachmodell habe ich keine persönlichen Präferenzen.\n"
                    "Eine technische Frage wäre: Wie sollten langfristig relevante Cognitive Orbits archiviert werden?")
    answer, gate = run_post_llm_gate(model_answer, SelfModel(), grounding=context)
    assert gate["reason"] == "identity_conflict"
    assert gate["action"] == "identity_clause_removed"
    assert "Wie sollten langfristig relevante Cognitive Orbits archiviert werden?" in answer
    assert "Sprachmodell" not in answer


def test_discriminative_coverage_rejects_generic_context_and_accepts_paraphrase():
    orbit = CognitiveOrbit("orbit_coherence", "Long-term coherence / orbit reactivation",
                           "Reliable reactivation of persistent relevant topics and Cognitive Orbits",
                           metadata={"semantic_signature": {"terms": ["kohärenz", "reaktivierung", "orbit"]}})
    context = build_grounding_context("Was hatten wir bei AvaCore noch offen?", SelfModel(), orbits=[orbit])
    generic = "Wir hatten über Hardware, Modelle, Kontext und Ressourcen gesprochen."
    answer, gate = run_post_llm_gate(generic, SelfModel(), grounding=context)
    assert gate["grounding_omission"] is True
    assert gate["action"] == "grounded_fallback"
    assert "coherence" in answer
    assert gate["grounding_coverage"] == 0
    good = "Offen war die langfristige Kohärenz: Cognitive Orbits sollten später zuverlässig wieder reaktiviert werden."
    answer, gate = run_post_llm_gate(good, SelfModel(), grounding=context)
    assert answer == good and gate["action"] == "none"
    assert gate["grounding_omission"] is False
    assert {"kohärenz", "orbit"} <= set(gate["grounding_matched_terms"])


def test_synthetic_orbit_is_not_answer_anchor():
    fixture = CognitiveOrbit("fixture", "Person recognition robustness", "Acceptance test topic",
                             metadata={"test": "phase4_acceptance"})
    productive = CognitiveOrbit("real", "Long-term coherence", "Orbit reactivation remains open")
    context = build_grounding_context("Wenn du mir eine technische Frage stellen dürftest, welche wäre das?",
                                      SelfModel(), orbits=[fixture, productive])
    assert context.debug()["selected_orbit_ids"] == ["real"]
    assert context.debug()["answer_anchor_ids"] == ["real"]
    assert context.grounded_fallback().find("Person recognition") == -1


def test_recent_active_orbit_out_ranks_old_important_orbit_for_generic_self_state():
    now = datetime.now(timezone.utc)
    old = CognitiveOrbit("old", "Old concern", "Older open concern", importance=1.0,
                         activation=.1, last_activated_at=(now - timedelta(days=180)).isoformat())
    recent = CognitiveOrbit("recent", "Long-term coherence", "Reliable orbit reactivation",
                            importance=.75, activation=.8, status="active", last_activated_at=now.isoformat())
    context = build_grounding_context("Wenn du mir eine technische Frage stellen dürftest, welche wäre das?",
                                      SelfModel(), orbits=[old, recent])
    debug = context.debug()
    assert debug["primary_answer_anchor_id"] == "recent"
    assert debug["answer_anchor_scores"][0] > debug["answer_anchor_scores"][1]


def test_real_generic_reply_cannot_match_function_words_or_weak_topics():
    orbit = CognitiveOrbit("coherence", "Long-term coherence / orbit reactivation",
                           "Ich möchte, dass das Orbit-System wichtige Themen langfristig wieder aufgreift; "
                           "nicht nur aktuelle Modellkontexte.",
                           metadata={"semantic_signature": {"terms": ["kohärenz", "reaktivierung", "orbit"]}})
    context = build_grounding_context("Was hatten wir bei AvaCore noch offen?", SelfModel(), orbits=[orbit])
    answer, gate = run_post_llm_gate(
        "Wir haben über Hardware, Modelle, lokale LLMs und verschiedene Themen gesprochen.",
        SelfModel(), grounding=context)
    assert gate["grounding_omission"] is True
    assert gate["action"] == "grounded_fallback"
    assert "coherence" in answer
    required = gate["grounding_required_terms"]
    assert not set(required) & {"ich", "nicht", "themen", "system", "modell", "kontext"}
    assert gate["grounding_coverage"] == len(gate["grounding_matched_terms"]) / len(required)


def test_identity_cleanup_then_grounding_omission_reports_both_conflicts():
    orbit = CognitiveOrbit("coherence", "Long-term coherence / orbit reactivation",
                           "Reliable reactivation of important Cognitive Orbits")
    context = build_grounding_context("Wenn du mir eine technische Frage stellen dürftest, welche wäre das?",
                                      SelfModel(), orbits=[orbit])
    raw = ("Als großes Sprachmodell habe ich keine persönlichen Präferenzen.\n"
           "Ich kann Fragen zu verschiedenen technischen Themen stellen.")
    answer, gate = run_post_llm_gate(raw, SelfModel(), grounding=context)
    assert gate["conflicts"] == ["identity_conflict", "grounding_omission"]
    assert gate["reason"] == "grounding_omission"
    assert gate["action"] == "grounded_fallback"
    assert answer.startswith("Eine technische Frage aus meinem aktuellen AvaCore-Zustand wäre:")
    assert context.debug(gate)["post_gate_conflicts"] == gate["conflicts"]


def test_good_morphological_paraphrase_is_grounded():
    orbit = CognitiveOrbit("coherence", "Long-term coherence / orbit reactivation",
                           "Important Cognitive Orbits should be reliably reactivated")
    context = build_grounding_context("Was hatten wir bei AvaCore noch offen?", SelfModel(), orbits=[orbit])
    good = "Offen war noch die langfristige Kohärenz: wichtige Cognitive Orbits sollen später zuverlässig wieder aktiviert werden."
    answer, gate = run_post_llm_gate(good, SelfModel(), grounding=context)
    assert answer == good
    assert gate["action"] == "none"
    assert not gate["grounding_omission"]
    assert gate["grounding_coverage"] == len(gate["grounding_matched_terms"]) / len(gate["grounding_required_terms"])
