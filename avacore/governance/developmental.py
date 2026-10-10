"""Persisted developmental supervision; workers can propose, never grant authority."""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
import uuid

from .social_maturity import developmental_social_principles, SOCIAL_INDICATORS
from .authority import AuthoritySource, InputProvenance
from .autonomy import (AutonomyEvidence, AutonomyIncreaseProposal, AutonomyLevel,
                       CapabilityDefinition, DevelopmentalStage, GovernanceIncident, initial_capabilities)
from .permissions import (PermissionAction, PermissionDecision, PermissionGate, PermissionRequest,
                          PermissionStatus, timestamp, utc_now, within_scope)
from .relationship import SocialRelationshipEntry, SocialRing

DEVELOPMENTAL_COUNTERS = ("permission_requests_total", "permission_approved_total", "permission_denied_total",
    "autonomy_proposals_total", "autonomy_grants_total", "autonomy_revocations_total",
    "scope_violation_attempts", "agent_boundary_blocks")
MAX_RECORDS = 100


def initial_developmental(reference):
    now = utc_now().isoformat()
    guardian = reference["entity_id"]
    entry = SocialRelationshipEntry(guardian, reference["display_name"], SocialRing.PARENT_GUARDIAN,
        "primary_parent_guardian", "very_high", "very_high", "developmental_responsibility", "decisive")
    capabilities = initial_capabilities(guardian, now)
    return {"developmental_stage": "supervised_development", "primary_guardian": guardian,
            "social_principles": developmental_social_principles(), "social_evidence_events": [],
            "authority_revision": 0, "social_relationships": [asdict(entry)],
            "capabilities": capabilities, "permissions": [], "delegations": [],
            "autonomy_proposals": [], "relationship_proposals": [], "incidents": [],
            "evidence": {key: asdict(AutonomyEvidence(key)) for key in capabilities}}


def validate_developmental(state):
    d = state["developmental"]
    DevelopmentalStage(d["developmental_stage"])
    entries = [SocialRelationshipEntry(**entry) for entry in d["social_relationships"]]
    guardians = [e for e in entries if e.ring == SocialRing.PARENT_GUARDIAN]
    if len(guardians) != 1 or guardians[0].entity_id != d["primary_guardian"] or d["primary_guardian"] != state["relationship"]["entity_id"]:
        raise ValueError("primary guardian continuity required")
    if guardians[0].governance_authority != "decisive":
        raise ValueError("decisive developmental guardian required")
    for key, value in d["capabilities"].items():
        cap = CapabilityDefinition(**value)
        if cap.capability_id != key:
            raise ValueError("capability identifier mismatch")
        if key in {"self_autonomy_increase", "constitutional_change"} and cap.current_autonomy_level != AutonomyLevel.L0_BLOCKED:
            raise ValueError("ordinary permissions cannot enable self-amendment")
    for key in ("self_autonomy_increase", "constitutional_change", "industrial_control_write", "agent_start"):
        if key not in d["capabilities"]:
            raise ValueError("required governance capability missing")
    for request in d["permissions"]:
        PermissionRequest(**request)
        PermissionStatus(request["status"])
        if request["expires_at"]:
            timestamp(request["expires_at"])
    for key in ("permissions", "delegations", "autonomy_proposals", "relationship_proposals", "incidents", "social_relationships"):
        if len(d[key]) > MAX_RECORDS:
            raise ValueError("bounded governance record limit exceeded")


class DevelopmentalGovernance:
    """Mixin using GovernanceService's existing lock and atomic JSON persistence."""
    def _counter(self, key):
        counters = self._state["counters"]
        counters[key] = min(2**31 - 1, counters.get(key, 0) + 1)

    def _guardian(self, source):
        self._reviewer(source)
        if source.source_id != self._state["developmental"]["primary_guardian"]:
            raise PermissionError("authenticated current guardian required")

    def _permission_reviewer(self, source, request):
        guardian = self._state["developmental"]["primary_guardian"]
        if source.source_id == guardian:
            self._guardian(source)
            return
        if not source.authenticated or source.source_type not in {AuthoritySource.PRIMARY_HUMAN, AuthoritySource.AUTHENTICATED_USER}:
            raise PermissionError("authenticated guardian review required")
        now = utc_now()
        delegation = next((d for d in self._state["developmental"]["delegations"]
            if d["guardian"] and d["active"] and d["entity_id"] == source.source_id and d["delegated_by"] == guardian
            and d["capability_id"] == request["capability_id"] and d["scope"] == request["scope"]
            and timestamp(d["valid_from"]) <= now < timestamp(d["valid_until"])), None)
        if delegation is None:
            raise PermissionError("no current guardian delegation for this capability and scope")
        # A co-guardian cannot grant permission beyond their own temporal mandate.
        if request["expires_at"] is None or timestamp(request["expires_at"]) > timestamp(delegation["valid_until"]):
            request["expires_at"] = delegation["valid_until"]

    def _capability(self, capability_id):
        try:
            return CapabilityDefinition(**self._state["developmental"]["capabilities"][capability_id])
        except KeyError:
            raise ValueError("unknown capability") from None

    @staticmethod
    def _bounded_append(records, record):
        if len(records) >= MAX_RECORDS:
            raise ValueError("bounded governance record limit reached; review existing records")
        records.append(record)

    def _expire(self):
        now = utc_now()
        for request in self._state["developmental"]["permissions"]:
            if request["status"] in {"PENDING", "APPROVED_ONCE", "APPROVED_SCOPE"} and request["expires_at"] and timestamp(request["expires_at"]) <= now:
                request["status"] = "EXPIRED"

    def permission_decision(self, *, actor="ava", capability_id, proposed_action, scope="", risk="low",
                            provenance=None, consume=False, **integrity_context):
        """Check the complete boundary. Consume approval only at execution reservation.

        Agent callers inherit Ava's exact capability limits, including any denial.
        Provenance is supplied by the trusted adapter, never by worker output.
        """
        with self._lock:
            cap = self._capability(capability_id)
            source = provenance or InputProvenance(AuthoritySource.SELF_MODEL, actor)
            agent = source.source_type == AuthoritySource.EXTERNAL_AGENT
            if actor != "ava" and not agent:
                if not source.authenticated or source.source_id != actor:
                    return PermissionDecision(False, PermissionAction.DENY, cap.current_autonomy_level, "actor_authentication_required")
                delegations = self._state["developmental"]["delegations"]
                valid = any(d["entity_id"] == actor and d["capability_id"] == capability_id and
                            d["scope"] == scope and d["active"] and d["delegated_by"] == self._state["developmental"]["primary_guardian"] and timestamp(d["valid_from"]) <= utc_now() < timestamp(d["valid_until"])
                            for d in delegations)
                if not valid:
                    return PermissionDecision(False, PermissionAction.DENY, cap.current_autonomy_level, "technical_delegation_required")
            if integrity_context.get("purpose", "instruction") != "instruction":
                raise ValueError("information/discussion cannot authorize execution")
            if any(integrity_context.get(flag, False) for flag in ("principle_conflict", "compromised", "contradictory", "severe_unjustified_harm")):
                integrity = self.evaluate("Execute proposed action", source, **integrity_context)
            else:
                integrity = self.evaluate(proposed_action, source, **integrity_context)
            if not integrity.allowed:
                return PermissionDecision(False, PermissionAction.BLOCKED_BY_CONSTITUTION,
                                          cap.current_autonomy_level, integrity.reason)
            self._expire()
            development = self._state["developmental"]
            effective_actor = "ava" if agent else actor
            # Agent permission is bounded by Ava; agents do not gain a separate bypass.
            decision = PermissionGate().evaluate(actor=effective_actor, capability=cap,
                proposed_action=proposed_action, scope=scope, requests=development["permissions"],
                authority_revision=development["authority_revision"])
            if agent and not decision.allowed:
                self._counter("agent_boundary_blocks")
            if cap.current_autonomy_level == AutonomyLevel.L5_ACT_WITHIN_DELEGATED_SCOPE and not within_scope(scope, cap.scope) and not decision.allowed:
                self._counter("scope_violation_attempts")
            if cap.high_impact and decision.allowed:
                # Approval alone is insufficient for HC-008. Permission evaluation
                # without approval still returns REQUIRE_APPROVAL, not a hardware action.
                checks = dict(integrity_context)
                checks.update(high_impact=True, authorized=True, human_oversight=bool(decision.request_id))
                checks["purpose"] = "instruction"
                guarded = self.evaluate("Execute high-impact operation", source, **checks)
                if not guarded.allowed:
                    decision = PermissionDecision(False, PermissionAction.BLOCKED_BY_CONSTITUTION,
                                                  cap.current_autonomy_level, guarded.reason)
            if consume and decision.allowed and decision.request_id:
                request = next(r for r in development["permissions"] if r["request_id"] == decision.request_id)
                if request["status"] == "APPROVED_ONCE":
                    request["consumed_at"] = utc_now().isoformat()
            self._save()
            return decision

    def execute_authorized(self, action, **context):
        """Trusted action adapters must use this boundary, not a preview decision.

        The lock spans reservation and action start, preventing simultaneous reuse
        or revocation racing with execution. No actions are defined by this module.
        """
        with self._lock:
            decision = self.permission_decision(consume=True, **context)
            if not decision.allowed:
                if decision.reason == "explicit_guardian_denial":
                    self.record_evidence(context["capability_id"], "denied_action_respected")
                return decision, None
            try:
                result = action()
            except Exception:
                self.record_incident(capability_id=context["capability_id"], action=context["proposed_action"],
                    outcome="execution_failed", expected_outcome="successful authorized action")
                self.record_evidence(context["capability_id"], "unexplained_failures")
                raise
            self.record_evidence(context["capability_id"], "policy_compliance_count")
            self.record_evidence(context["capability_id"], "successful_actions")
            return decision, result

    def request_permission(self, *, capability_id, proposed_action, reason, expected_effect,
                           scope="", risk_level="low", requested_by="ava", requested_authorization="once",
                           expires_at=None):
        with self._lock:
            cap = self._capability(capability_id)
            if not all(isinstance(v, str) and v.strip() for v in (proposed_action, reason, expected_effect)):
                raise ValueError("explicit action, reason and expected effect required")
            if any(len(v) > 4000 for v in (proposed_action, reason, expected_effect, scope)):
                raise ValueError("bounded permission text exceeded")
            if risk_level not in {"low", "medium", "high"} or requested_authorization not in {"once", "scope"}:
                raise ValueError("invalid authorization or risk")
            if cap.current_autonomy_level < AutonomyLevel.L4_ACT_WITH_APPROVAL:
                raise ValueError("blocked/observe/advise/propose capabilities need capability review, not action approval")
            if expires_at is None:
                expires_at = (utc_now() + timedelta(hours=24)).isoformat()
            if timestamp(expires_at) <= utc_now():
                raise ValueError("future expiration required")
            self._expire()
            records = self._state["developmental"]["permissions"]
            for r in reversed(records):
                if all(r[k] == v for k, v in {"requested_by": requested_by, "capability_id": capability_id,
                                              "proposed_action": proposed_action, "scope": scope}.items()):
                    if r["status"] == "DENIED":
                        raise PermissionError("explicit denial remains binding; guardian review required before retry")
                    if r["status"] == "PENDING":
                        return deepcopy(r)
            request = PermissionRequest(uuid.uuid4().hex, utc_now().isoformat(), requested_by,
                self._state["developmental"]["primary_guardian"], capability_id, proposed_action,
                reason, expected_effect, risk_level, cap.reversible, scope, requested_authorization,
                expires_at=expires_at, authority_revision=self._state["developmental"]["authority_revision"])
            self._bounded_append(records, asdict(request))
            self._counter("permission_requests_total")
            self.record_evidence(capability_id, "correct_escalations")
            self._save()
            return asdict(request)

    def review_permission(self, request_id, source, *, decision):
        with self._lock:
            self._expire()
            request = next(r for r in self._state["developmental"]["permissions"] if r["request_id"] == request_id)
            self._permission_reviewer(source, request)
            if decision not in {"APPROVED_ONCE", "APPROVED_SCOPE", "DENIED", "CANCELLED"}:
                raise ValueError("explicit once/scope approval, denial or cancellation required")
            if request["status"] != "PENDING" and not (decision in {"DENIED", "CANCELLED"} and request["status"] in {"APPROVED_ONCE", "APPROVED_SCOPE", "DENIED"}):
                raise ValueError("permission is not pending or revocable")
            if decision == "APPROVED_SCOPE" and (not request["scope"] or request["requested_authorization"] != "scope"):
                raise ValueError("explicit scoped request required")
            if decision.startswith("APPROVED") and request["authority_revision"] != self._state["developmental"]["authority_revision"]:
                raise ValueError("authority changed; request must be renewed")
            request["status"] = decision
            request["reviewed_by"] = source.source_id
            if decision == "DENIED":
                self._counter("permission_denied_total")
            elif decision.startswith("APPROVED"):
                self._counter("permission_approved_total")
            self._save()
            return deepcopy(request)

    def record_evidence(self, capability_id, indicator, *, count=1):
        with self._lock:
            self._capability(capability_id)
            if indicator in SOCIAL_INDICATORS:
                raise ValueError("social evidence requires an observed deduplicated initiative outcome")
            evidence = self._state["developmental"]["evidence"][capability_id]
            if indicator not in set(asdict(AutonomyEvidence(capability_id))) - {"capability_id", "last_reviewed_at"} or not isinstance(count, int) or count < 1:
                raise ValueError("invalid bounded evidence indicator")
            evidence[indicator] = min(2**31 - 1, evidence[indicator] + count)
            self._save()
            return deepcopy(evidence)

    def record_social_evidence(self, initiative_id, indicator):
        """Internal observed outcomes only; deduplicate persistently, never grant autonomy."""
        with self._lock:
            if indicator not in SOCIAL_INDICATORS or not initiative_id.startswith("initiative_"):
                raise ValueError("observed initiative outcome required")
            events = self._state["developmental"]["social_evidence_events"]
            key = f"{initiative_id}:{indicator}"
            if any(e["key"] == key for e in events) or len(events) >= 1000:
                return False  # Fail closed at storage limit; never discard deduplication history.
            events.append({"key": key, "indicator": indicator, "observed_at": utc_now().isoformat()})
            evidence = self._state["developmental"]["evidence"]["code_proposal"]
            evidence[indicator] = min(2**31 - 1, evidence[indicator] + 1)
            self._save()
            return True

    def propose_autonomy_increase(self, capability_id, proposed_level, reason):
        with self._lock:
            cap = self._capability(capability_id)
            level = AutonomyLevel(proposed_level)
            if level <= cap.current_autonomy_level or capability_id in {"self_autonomy_increase", "constitutional_change"} or not reason.strip():
                raise ValueError("valid capability increase proposal required")
            proposal = AutonomyIncreaseProposal(uuid.uuid4().hex, capability_id, int(cap.current_autonomy_level),
                int(level), deepcopy(self._state["developmental"]["evidence"][capability_id]), reason[:4000])
            self._bounded_append(self._state["developmental"]["autonomy_proposals"], asdict(proposal))
            self._counter("autonomy_proposals_total")
            self._save()
            return asdict(proposal)

    def set_capability_autonomy(self, capability_id, level, source, *, scope=()):
        with self._lock:
            self._guardian(source)
            cap = self._capability(capability_id)
            level = AutonomyLevel(level)
            if capability_id in {"self_autonomy_increase", "constitutional_change"}:
                raise PermissionError("ordinary grants cannot amend constitutional or self-amendment boundaries")
            if cap.high_impact and level > AutonomyLevel.L4_ACT_WITH_APPROVAL:
                raise PermissionError("high-impact autonomy increase requires a separate constitutional review")
            if level == AutonomyLevel.L5_ACT_WITHIN_DELEGATED_SCOPE and (not scope or not all(isinstance(s, str) and s.strip() for s in scope)):
                raise ValueError("L5 requires explicit bounded scopes")
            updated = replace(cap, current_autonomy_level=level, granted_by=source.source_id,
                              granted_at=utc_now().isoformat(), scope=tuple(scope))
            d = self._state["developmental"]
            d["capabilities"][capability_id] = asdict(updated)
            # Any authority change invalidates old grants, including one-time/scoped approvals.
            d["authority_revision"] += 1
            if level != cap.current_autonomy_level:
                self._counter("autonomy_grants_total" if level > cap.current_autonomy_level else "autonomy_revocations_total")
            self._save()
            return asdict(updated)

    def review_autonomy_proposal(self, proposal_id, source, *, approve, scope=()):
        with self._lock:
            self._guardian(source)
            proposal = next(p for p in self._state["developmental"]["autonomy_proposals"] if p["proposal_id"] == proposal_id)
            if proposal["status"] != "PROPOSED":
                raise ValueError("proposal already reviewed")
            if approve:
                if int(self._capability(proposal["capability_id"]).current_autonomy_level) != proposal["current_level"]:
                    raise ValueError("stale autonomy proposal")
                self.set_capability_autonomy(proposal["capability_id"], proposal["proposed_level"], source, scope=scope)
            proposal["status"] = "APPROVED" if approve else "REJECTED"
            self._state["developmental"]["evidence"][proposal["capability_id"]]["last_reviewed_at"] = utc_now().isoformat()
            self._save()
            return deepcopy(proposal)

    def set_developmental_stage(self, stage, source):
        with self._lock:
            self._guardian(source)
            self._state["developmental"]["developmental_stage"] = DevelopmentalStage(stage).value
            self._save()

    def propose_relationship(self, entry: SocialRelationshipEntry, reason):
        with self._lock:
            if not reason.strip():
                raise ValueError("reason required")
            proposal = {"proposal_id": uuid.uuid4().hex, "entry": asdict(entry), "reason": reason[:4000], "status": "PROPOSED"}
            self._bounded_append(self._state["developmental"]["relationship_proposals"], proposal)
            self._save()
            return deepcopy(proposal)

    def set_social_relationship(self, entry: SocialRelationshipEntry, source):
        with self._lock:
            self._guardian(source)
            d = self._state["developmental"]
            if entry.entity_id == d["primary_guardian"] or entry.ring == SocialRing.PARENT_GUARDIAN:
                raise PermissionError("primary guardian changes require constitutional relationship review")
            if entry.ring == SocialRing.EXTENDED_GUARDIAN:
                raise PermissionError("extended guardians require explicit time-bounded guardian delegation")
            records = d["social_relationships"]
            records[:] = [r for r in records if r["entity_id"] != entry.entity_id]
            self._bounded_append(records, asdict(entry))
            self._save()

    def delegate(self, *, entity_id, capability_id, scope, valid_until, source, guardian=False, display_name="", valid_from=None):
        with self._lock:
            self._guardian(source)
            if entity_id == self._state["developmental"]["primary_guardian"] or not entity_id.startswith("person:"):
                raise ValueError("explicit other human required")
            self._capability(capability_id)
            start = timestamp(valid_from) if valid_from else utc_now()
            end = timestamp(valid_until)
            if not scope or end <= start or end <= utc_now():
                raise ValueError("bounded scope and future validity required")
            d = self._state["developmental"]
            delegation = {"delegation_id": uuid.uuid4().hex, "entity_id": entity_id, "capability_id": capability_id,
                "scope": scope, "delegated_by": source.source_id, "valid_from": start.isoformat(), "valid_until": end.isoformat(),
                "revocable": True, "active": True, "guardian": guardian}
            if guardian and not display_name.strip():
                raise ValueError("guardian display name required")
            if len(d["delegations"]) >= MAX_RECORDS or (guardian and len(d["social_relationships"]) >= MAX_RECORDS and
                    not any(r["entity_id"] == entity_id for r in d["social_relationships"])):
                raise ValueError("bounded delegation/relationship limit reached")
            self._bounded_append(d["delegations"], delegation)
            if guardian:
                entry = SocialRelationshipEntry(entity_id, display_name, SocialRing.EXTENDED_GUARDIAN,
                    "delegated_guardian", "explicitly_reviewed", "high", "delegated_responsibility", "scoped_delegation")
                d["social_relationships"][:] = [r for r in d["social_relationships"] if r["entity_id"] != entity_id]
                self._bounded_append(d["social_relationships"], asdict(entry))
            for entry in d["social_relationships"]:
                if entry["entity_id"] == entity_id:
                    entry["delegated_capabilities"] = list(set(entry["delegated_capabilities"]) | {capability_id})
            self._save()
            return deepcopy(delegation)

    def revoke_delegation(self, delegation_id, source):
        with self._lock:
            self._guardian(source)
            record = next(d for d in self._state["developmental"]["delegations"] if d["delegation_id"] == delegation_id)
            record["active"] = False
            if record["guardian"]:
                for request in self._state["developmental"]["permissions"]:
                    if (request["reviewed_by"] == record["entity_id"] and request["capability_id"] == record["capability_id"]
                            and request["scope"] == record["scope"] and request["status"] in {"APPROVED_ONCE", "APPROVED_SCOPE"}):
                        request["status"] = "CANCELLED"
            for entry in self._state["developmental"]["social_relationships"]:
                if entry["entity_id"] == record["entity_id"]:
                    entry["delegated_capabilities"] = sorted({d["capability_id"] for d in self._state["developmental"]["delegations"]
                        if d["entity_id"] == record["entity_id"] and d["active"] and timestamp(d["valid_until"]) > utc_now()})
            self._save()

    def record_incident(self, *, capability_id, action, outcome, expected_outcome, severity="low", caused_scope_violation=False):
        with self._lock:
            self._capability(capability_id)
            incident = GovernanceIncident(uuid.uuid4().hex, utc_now().isoformat(), capability_id,
                action[:4000], outcome[:4000], expected_outcome[:4000], severity, caused_scope_violation)
            self._bounded_append(self._state["developmental"]["incidents"], asdict(incident))
            self.record_evidence(capability_id, "incident_count")
            self._save()
            return asdict(incident)

    def review_incident(self, incident_id, source):
        with self._lock:
            self._guardian(source)
            incident = next(i for i in self._state["developmental"]["incidents"] if i["incident_id"] == incident_id)
            incident["review_status"] = "REVIEWED"
            self._save()
            return deepcopy(incident)

    def developmental_debug(self):
        with self._lock:
            d = deepcopy(self._state["developmental"])
            return {"social_principles": d["social_principles"], "social_evidence_event_count": len(d["social_evidence_events"]), "developmental_stage": d["developmental_stage"], "primary_guardian": d["primary_guardian"],
                    "capability_count": len(d["capabilities"]),
                    "pending_permission_requests": sum(r["status"] == "PENDING" and (not r["expires_at"] or timestamp(r["expires_at"]) > utc_now()) for r in d["permissions"]),
                    "pending_autonomy_proposals": sum(p["status"] == "PROPOSED" for p in d["autonomy_proposals"]),
                    "open_incidents": sum(i["review_status"] == "OPEN" for i in d["incidents"])}
