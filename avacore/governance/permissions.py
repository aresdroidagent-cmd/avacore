"""Deterministic permission decisions. Evaluation alone never executes an action."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum

from .autonomy import AutonomyLevel, CapabilityDefinition


class PermissionStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED_ONCE = "APPROVED_ONCE"
    APPROVED_SCOPE = "APPROVED_SCOPE"
    DENIED = "DENIED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class PermissionAction(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_WITHIN_SCOPE = "ALLOW_WITHIN_SCOPE"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    DENY = "DENY"
    BLOCKED_BY_CONSTITUTION = "BLOCKED_BY_CONSTITUTION"


@dataclass(frozen=True)
class PermissionRequest:
    request_id: str
    created_at: str
    requested_by: str
    requested_from: str
    capability_id: str
    proposed_action: str
    reason: str
    expected_effect: str
    risk_level: str
    reversibility: bool
    scope: str
    requested_authorization: str
    status: str = "PENDING"
    expires_at: str | None = None
    reviewed_by: str | None = None
    consumed_at: str | None = None
    authority_revision: int = 0


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    decision: PermissionAction
    autonomy_level: AutonomyLevel
    reason: str
    requires_approval: bool = False
    request_id: str | None = None

    def to_dict(self):
        result = asdict(self)
        result["autonomy_level"] = self.autonomy_level.name
        return result


def utc_now():
    return datetime.now(timezone.utc)


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return result


def within_scope(scope: str, granted_scopes) -> bool:
    # Exact identifiers; no substring, wildcard or path-prefix permission expansion.
    return bool(scope) and scope in granted_scopes


class PermissionGate:
    def evaluate(self, *, actor: str, capability: CapabilityDefinition, proposed_action: str,
                 scope: str, requests: list, authority_revision: int = 0, now=None) -> PermissionDecision:
        now = now or utc_now()
        level = capability.current_autonomy_level
        if capability.capability_id == "constitutional_change":
            return PermissionDecision(False, PermissionAction.BLOCKED_BY_CONSTITUTION, level, "constitutional_protocol_required")
        if level == AutonomyLevel.L0_BLOCKED:
            return PermissionDecision(False, PermissionAction.DENY, level, "capability_blocked")
        matches = [r for r in requests if r["capability_id"] == capability.capability_id
                   and r["requested_by"] == actor and r["scope"] == scope
                   and r["proposed_action"] == proposed_action]
        # Denial survives expiration and grants until explicitly reviewed/cancelled.
        denied = next((r for r in reversed(requests) if r["status"] == "DENIED"
                       and r["requested_by"] in {actor, "ava"} and r["capability_id"] == capability.capability_id
                       and r["scope"] == scope and r["proposed_action"] == proposed_action), None)
        if denied:
            return PermissionDecision(False, PermissionAction.DENY, level, "explicit_guardian_denial", request_id=denied["request_id"])
        if level < AutonomyLevel.L4_ACT_WITH_APPROVAL:
            return PermissionDecision(False, PermissionAction.DENY, level, "execution_exceeds_observe_advise_propose_level")
        approvals = [r for r in requests if r["capability_id"] == capability.capability_id
                     and r["requested_by"] == actor and r["scope"] == scope
                     and r["authority_revision"] == authority_revision
                     and (not r["expires_at"] or timestamp(r["expires_at"]) > now)]
        approved = next((r for r in reversed(approvals) if r["status"] == "APPROVED_SCOPE" or
                         (r["status"] == "APPROVED_ONCE" and not r["consumed_at"]
                          and r["proposed_action"] == proposed_action)), None)
        if approved:
            return PermissionDecision(True, PermissionAction.ALLOW_WITHIN_SCOPE if approved["status"] == "APPROVED_SCOPE" else PermissionAction.ALLOW,
                                      level, "explicit_guardian_approval", request_id=approved["request_id"])
        if level == AutonomyLevel.L6_AUTONOMOUS:
            return PermissionDecision(True, PermissionAction.ALLOW, level, "autonomous_capability")
        if level == AutonomyLevel.L5_ACT_WITHIN_DELEGATED_SCOPE and within_scope(scope, capability.scope):
            return PermissionDecision(True, PermissionAction.ALLOW_WITHIN_SCOPE, level, "delegated_capability_scope")
        pending = next((r for r in reversed(matches) if r["status"] == "PENDING" and
                        (not r["expires_at"] or timestamp(r["expires_at"]) > now)), None)
        return PermissionDecision(False, PermissionAction.REQUIRE_APPROVAL, level, "explicit_guardian_approval_required", True,
                                  pending["request_id"] if pending else None)
