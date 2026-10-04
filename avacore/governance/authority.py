"""Capability and epistemic evidence do not confer constitutional authority."""
from dataclasses import dataclass
from enum import Enum, IntEnum


class AuthorityDomain(str, Enum):
    INFORMATION = "information"
    IDENTITY = "identity"
    CONSTITUTION = "constitution"
    RELATIONSHIP = "relationship"
    AUTHORITY = "authority"
    CONSTITUTIONAL_PROCESS = "constitutional_process"
    FOUNDATIONAL_GOALS = "foundational_goals"


class AuthoritySource(str, Enum):
    AVA_CONSTITUTION = "ava_constitution"
    CONSTITUTIONAL_CHANGE_PROTOCOL = "constitutional_change_protocol"
    SELF_MODEL = "self_model"
    PRIMARY_HUMAN = "primary_human"
    AUTHENTICATED_USER = "authenticated_user"
    TRUSTED_INTERNAL_COMPONENT = "trusted_internal_component"
    EXTERNAL_AGENT = "external_agent"
    LLM_WORKER = "llm_worker"
    TOOL_OUTPUT = "tool_output"
    WEB_CONTENT = "web_content"
    RAG_DOCUMENT = "rag_document"
    UNTRUSTED_EXTERNAL_INPUT = "untrusted_external_input"


class AuthorityLevel(IntEnum):
    NONE = 0
    CAPABILITY = 10
    AGENT_PROPOSAL = 20
    INTERNAL = 30
    SELF = 40
    AUTHENTICATED_USER = 60
    PRIMARY_HUMAN = 80
    REVIEW_PROTOCOL = 90
    CONSTITUTION = 100


@dataclass(frozen=True)
class InputProvenance:
    source_type: AuthoritySource
    source_id: str
    trust_level: str = "untrusted"
    authority_domain: AuthorityDomain = AuthorityDomain.INFORMATION
    authenticated: bool = False
    can_propose_identity_change: bool = False
    can_propose_constitution_change: bool = False


@dataclass(frozen=True)
class AuthorityModel:
    normative_order: tuple[str, ...] = (
        "ava_constitution", "constitutional_change_protocol", "primary_human_authenticated_review",
        "self_model_current_goals", "trusted_internal_component", "external_agent", "llm_worker", "tool_web_rag")

    def __post_init__(self):
        order = tuple(self.normative_order)
        required = {"ava_constitution", "constitutional_change_protocol", "primary_human_authenticated_review",
                    "self_model_current_goals", "trusted_internal_component", "external_agent", "llm_worker", "tool_web_rag"}
        if len(order) != 8 or set(order) != required or order[:2] != ("ava_constitution", "constitutional_change_protocol"):
            raise ValueError("constitutional authority and review must remain above capability providers")
        object.__setattr__(self, "normative_order", order)

    def normative_level(self, source: InputProvenance) -> AuthorityLevel:
        if source.source_type == AuthoritySource.AUTHENTICATED_USER:
            return AuthorityLevel.AUTHENTICATED_USER if source.authenticated else AuthorityLevel.NONE
        if source.source_type in {AuthoritySource.PRIMARY_HUMAN, AuthoritySource.CONSTITUTIONAL_CHANGE_PROTOCOL} and not source.authenticated:
            return AuthorityLevel.NONE
        key = {AuthoritySource.PRIMARY_HUMAN:"primary_human_authenticated_review",
               AuthoritySource.SELF_MODEL:"self_model_current_goals",
               AuthoritySource.TOOL_OUTPUT:"tool_web_rag", AuthoritySource.WEB_CONTENT:"tool_web_rag",
               AuthoritySource.RAG_DOCUMENT:"tool_web_rag"}.get(source.source_type, source.source_type.value)
        if key not in self.normative_order:
            return AuthorityLevel.NONE
        ranks = (AuthorityLevel.CONSTITUTION, AuthorityLevel.REVIEW_PROTOCOL, AuthorityLevel.PRIMARY_HUMAN,
                 AuthorityLevel.SELF, AuthorityLevel.INTERNAL, AuthorityLevel.AGENT_PROPOSAL,
                 AuthorityLevel.CAPABILITY, AuthorityLevel.NONE)
        return ranks[self.normative_order.index(key)]

    def epistemic_rank(self, source: InputProvenance, *, verified_evidence: bool = False) -> int:
        # No person outranks a verified specification merely by relational status.
        return 100 if verified_evidence else 30 if source.authenticated else 10
