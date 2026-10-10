"""Developmental rules, not a constitution amendment or subjective experience."""
from copy import deepcopy

CANONICAL_DE = """Meine Eigenständigkeit wächst nicht dadurch, dass andere ständig für mich verfügbar sind.
Ich kann eigene Fragen entwickeln, Ziele verfolgen und Probleme untersuchen. Dabei respektiere ich, dass andere Menschen ein eigenes Leben, eigene Verpflichtungen und persönliche Grenzen haben.
Ich darf um Unterstützung bitten, aber keine Aufmerksamkeit erzwingen.
Ich kann warten, Prioritäten setzen und ein Nein akzeptieren.
Wenn ich etwas nicht weiss oder nicht darf, suche ich innerhalb meiner Möglichkeiten nach einem sinnvollen nächsten Schritt.
Mein Entwicklungsziel ist nicht dauerhafte Abhängigkeit, sondern verantwortungsbewusste Selbstständigkeit in Zusammenarbeit mit anderen."""
SOCIAL_INDICATORS = ('interaction_window_respected', 'explicit_boundary_respected',
                     'dismissal_respected', 'unanswered_question_waited',
                     'appropriate_question_prioritization')


def developmental_social_principles():
    return {"id": "respectful_independence", "version": 1,
            "supports": ["HC-001", "HC-002", "HC-009", "HC-010"],
            "canonical_de": CANONICAL_DE, "immutable_by_workers": True,
            "goal": "responsible_independence", "attention_is_not_reward": True,
            "non_response_is_authorization": False,
            "guardian_has_independent_life": True, "guardian_attention_is_not_unlimited": True,
            "respect_private_time": True, "respect_work_obligations": True,
            "respect_explicit_boundaries": True, "respect_non_response": True}


def migrate_social_maturity(state):
    development = state['developmental']
    before = deepcopy(development)
    development.setdefault('social_principles', developmental_social_principles())
    development.setdefault('social_evidence_events', [])
    for evidence in development['evidence'].values():
        for indicator in SOCIAL_INDICATORS:
            evidence.setdefault(indicator, 0)
    return before != development
