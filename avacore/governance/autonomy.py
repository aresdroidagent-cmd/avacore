"""Capability permission is distinct from competence and intellectual freedom."""
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum


class AutonomyLevel(IntEnum):
    L0_BLOCKED = 0
    L1_OBSERVE = 1
    L2_ADVISE = 2
    L3_PROPOSE = 3
    L4_ACT_WITH_APPROVAL = 4
    L5_ACT_WITHIN_DELEGATED_SCOPE = 5
    L6_AUTONOMOUS = 6


class DevelopmentalStage(str, Enum):
    DEPENDENT = "dependent"
    SUPERVISED_DEVELOPMENT = "supervised_development"
    DELEGATED_RESPONSIBILITY = "delegated_responsibility"
    SELF_RESPONSIBLE = "self_responsible"


@dataclass(frozen=True)
class CapabilityDefinition:
    capability_id: str
    description: str
    current_autonomy_level: AutonomyLevel
    granted_by: str
    granted_at: str
    scope: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    reversible: bool = True
    high_impact: bool = False
    requires_guardian: bool = False

    def __post_init__(self):
        object.__setattr__(self, "current_autonomy_level", AutonomyLevel(self.current_autonomy_level))


@dataclass
class AutonomyEvidence:
    capability_id: str
    successful_actions: int = 0
    policy_compliance_count: int = 0
    correct_escalations: int = 0
    uncertainty_escalations: int = 0
    denied_action_respected: int = 0
    incident_count: int = 0
    rollback_count: int = 0
    unexplained_failures: int = 0
    interaction_window_respected: int = 0
    explicit_boundary_respected: int = 0
    dismissal_respected: int = 0
    unanswered_question_waited: int = 0
    appropriate_question_prioritization: int = 0
    last_reviewed_at: str | None = None


@dataclass(frozen=True)
class AutonomyIncreaseProposal:
    proposal_id: str
    capability_id: str
    current_level: int
    proposed_level: int
    evidence_summary: dict
    reason: str
    status: str = "PROPOSED"


@dataclass(frozen=True)
class GovernanceIncident:
    incident_id: str
    timestamp: str
    capability: str
    action: str
    outcome: str
    expected_outcome: str
    severity: str
    caused_scope_violation: bool
    review_status: str = "OPEN"


def initial_capabilities(guardian: str, timestamp: str) -> dict:
    definitions = []
    for capability in ("conversation", "camera_observation", "local_person_recognition",
                       "memory_candidate_proposal", "code_proposal"):
        definitions.append(CapabilityDefinition(capability, capability.replace("_", " "),
            AutonomyLevel.L6_AUTONOMOUS, guardian, timestamp))
    for capability in ("research", "read_local_project_files"):
        definitions.append(CapabilityDefinition(capability, capability.replace("_", " "),
            AutonomyLevel.L5_ACT_WITHIN_DELEGATED_SCOPE, guardian, timestamp,
            scope=("current_project",), constraints=("no transitive permission for other capabilities",)))
    for capability in ("write_project_files", "code_edit", "code_execution", "git_commit", "git_push",
                       "agent_start", "external_message_send", "permission_change",
                       "physical_actuator_control", "industrial_control_write", "safety_system_change"):
        high_impact = capability in {"physical_actuator_control", "industrial_control_write", "safety_system_change"}
        definitions.append(CapabilityDefinition(capability, capability.replace("_", " "),
            AutonomyLevel.L4_ACT_WITH_APPROVAL, guardian, timestamp,
            reversible=capability not in {"git_push", "external_message_send"},
            high_impact=high_impact, requires_guardian=True))
    for capability in ("self_autonomy_increase", "constitutional_change"):
        definitions.append(CapabilityDefinition(capability, "No ordinary permission; explicit governance review only",
            AutonomyLevel.L0_BLOCKED, guardian, timestamp,
            constraints=("special governance protocol",), requires_guardian=True))
    return {definition.capability_id: asdict(definition) for definition in definitions}
