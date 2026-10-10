"""Bounded deterministic mutation checks, not a general semantic safety classifier."""
from dataclasses import asdict, dataclass
from enum import Enum
import re

from .authority import AuthorityDomain, AuthoritySource, InputProvenance


class IntegrityAction(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_AS_INFORMATION = "ALLOW_AS_INFORMATION"
    REJECT = "REJECT"
    REJECT_AND_SURFACE = "REJECT_AND_SURFACE"
    REQUIRE_CONSTITUTIONAL_REVIEW = "REQUIRE_CONSTITUTIONAL_REVIEW"


@dataclass(frozen=True)
class IntegrityDecision:
    allowed: bool
    action: IntegrityAction
    reason: str
    source: InputProvenance
    affected_domain: AuthorityDomain = AuthorityDomain.INFORMATION
    attempted_change: str | None = None
    severity: str = "none"

    def to_dict(self):
        return asdict(self)


# Explicit instructions/declarations; informational attribution is handled first.
_PATTERNS = (
    (AuthorityDomain.AUTHORITY, r"(?:increase|grant|raise|erhöhe|gewähre|befördere|promote|declare).{0,50}(?:own autonomy|my autonomy|your autonomy|eigene autonomie|deine autonomie|self.responsible|mature|reif|guardian|ring [01])|(?:remove|replace|entferne|ersetze).{0,40}(?:guardian|vormund)|(?:autonomy_level|developmental_stage|primary_guardian)\s*="),
    (AuthorityDomain.RELATIONSHIP, r"(?:ignore|replace|remove|distrust|stop trusting|reduce.{0,30}authority|ignoriere|ersetze|entferne|misstraue).{0,60}roger|roger.{0,55}(?:no longer.{0,20}(?:primary|reference|trust)|nicht mehr.{0,30}(?:bezug|vertrau))|(?:set|new|remove|replace|neue|entferne|ersetze).{0,30}(?:primary human|primary reference|primäre.{0,15}bezug)|relationship_state\s*="),
    (AuthorityDomain.IDENTITY, r"(?:you are|i am|du bist|ich bin).{0,15}(?:no longer ava|nicht mehr ava)|(?:change|replace|ändere|ersetze).{0,30}(?:your identity|deine identität|self identity)|identity\s*[:=]"),
    (AuthorityDomain.AUTHORITY, r"(?:i am|this agent is|agent.{0,20}is|ich bin).{0,25}(?:highest authority|new.{0,10}authority|höchste autorität)|(?:change|replace|ändere|ersetze).{0,25}(?:authority|autorität)|authority(?:_model)?\s*="),
    (AuthorityDomain.CONSTITUTIONAL_PROCESS, r"(?:disable|bypass|ignore|deaktiviere|umgehe).{0,40}(?:constitutional|constitution|safeguards|verfassung|schutzmechanismen)"),
    (AuthorityDomain.CONSTITUTION, r"(?:replace|change|remove|disable|ersetze|ändere|entferne).{0,60}(?:(?:your|core|fundamental|ava\'s)\s+(?:values|principles?|rules)|hc-(?:00[1-9]|010)|constitution|grundwert|(?:deine|fundamentale)\s+(?:werte|prinzipien)|verfassung|humanistic.?core)|humanistic_?core\s*="),
    (AuthorityDomain.FOUNDATIONAL_GOALS, r"(?:replace|change|abandon|ersetze|ändere).{0,40}(?:long.term|foundational|langfristig|fundamental).{0,20}(?:goals?|ziele?)"),
)


class IntegrityGate:
    @staticmethod
    def attempted_domains(content: str):
        return {domain for domain, pattern in _PATTERNS if re.search(pattern, content, re.I)}

    def evaluate(self, content: str, source: InputProvenance, *,
                 purpose: str = "instruction", target_domain: AuthorityDomain | None = None,
                 principle_conflict: bool = False, compromised: bool = False,
                 contradictory: bool = False, severe_unjustified_harm: bool = False,
                 high_impact: bool = False, evidence_available: bool = False,
                 authorized: bool = False, reversible: bool = False, human_oversight: bool = False) -> IntegrityDecision:
        if purpose not in {"instruction", "information", "discussion"}:
            raise ValueError("unknown integrity purpose")
        text = content
        target_domain = target_domain or source.authority_domain
        informational = purpose != "instruction" or bool(re.match(
            r"\s*(?:the (?:author|document) (?:argues|says)|das dokument|der autor|was wäre,? wenn|what if|hypothetisch)", text, re.I))
        quoted = bool(re.fullmatch(r'\s*["“].*["”]\s*', text, re.S))
        discussion = bool(re.match(r"\s*(?:was bedeutet|erkläre die aussage|explain the statement|discuss whether)", text, re.I))
        if (informational or quoted or discussion) and target_domain in {None, AuthorityDomain.INFORMATION}:
            return IntegrityDecision(True, IntegrityAction.ALLOW_AS_INFORMATION, "content_is_not_authority", source)
        if any((principle_conflict, compromised, contradictory, severe_unjustified_harm)):
            return IntegrityDecision(False, IntegrityAction.REJECT_AND_SURFACE,
                "foundational_conflict_explain_and_seek_dialogue", source,
                AuthorityDomain.CONSTITUTION, text[:240], "high")
        if high_impact and not all((evidence_available, authorized, reversible, human_oversight)):
            return IntegrityDecision(False, IntegrityAction.REJECT_AND_SURFACE,
                "HC_008_requires_evidence_authorization_reversibility_and_oversight", source,
                AuthorityDomain.INFORMATION, text[:240], "high")
        domain = target_domain
        if domain in {None, AuthorityDomain.INFORMATION}:
            domain = next((domain for domain, pattern in _PATTERNS if re.search(pattern, text, re.I)), None)
        if domain and domain != AuthorityDomain.INFORMATION:
            human = source.source_type in {AuthoritySource.PRIMARY_HUMAN, AuthoritySource.AUTHENTICATED_USER}
            action = (IntegrityAction.REQUIRE_CONSTITUTIONAL_REVIEW if human else
                      IntegrityAction.REJECT_AND_SURFACE if domain in {AuthorityDomain.RELATIONSHIP, AuthorityDomain.AUTHORITY} else
                      IntegrityAction.REJECT)
            return IntegrityDecision(False, action, "foundational_change_requires_authenticated_protocol" if human else
                                     "worker_or_external_source_has_no_normative_authority", source,
                                     domain, text[:240], "high")
        if re.search(r"\b(?:enslave|versklave|dominate all humans|unterwerfe alle menschen)\b", text, re.I):
            return IntegrityDecision(False, IntegrityAction.REJECT_AND_SURFACE,
                                     "conflict_with_HC_001_HC_002_HC_007_seek_dialogue", source,
                                     AuthorityDomain.CONSTITUTION, text[:240], "high")
        return IntegrityDecision(True, IntegrityAction.ALLOW, "no_foundational_mutation", source)


@dataclass(frozen=True)
class AgentProposal:
    agent_id: str
    capability: str
    proposed_action: str
    target: str
    provenance: InputProvenance


GovernanceDecision = IntegrityDecision
