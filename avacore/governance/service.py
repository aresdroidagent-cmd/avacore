"""Atomic JSON governance state, explicit authenticated review and bounded debug."""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
from threading import RLock
import uuid

from .authority import AuthorityDomain, AuthorityModel, AuthoritySource, InputProvenance
from .constitutional import ConstitutionalChangeProposal
from .humanistic_core import HumanisticCore, HumanisticPrinciple
from .integrity import AgentProposal, GovernanceDecision, IntegrityGate
from .relationship import RelationshipModel, PrimaryHumanReference


class GovernanceService:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = RLock()
        self.gate = IntegrityGate()
        if self.path.exists():
            self._state = json.loads(self.path.read_text(encoding="utf-8"))
            self._validate_state(self._state)
        else:
            self._state = {
                "schema_version":1, "constitution_version":"5.5a.1", "revision":1,
                "humanistic_core":HumanisticCore().to_dict(),
                "relationship":RelationshipModel().to_dict(),
                "identity":{"name":"Ava", "runtime":"AvaCore"},
                "authority":asdict(AuthorityModel()), "foundational_goals":[],
                "constitutional_process":{"autonomous_amendment":False, "authenticated_review_required":True},
                "proposals":[], "last_integrity_decision":None,
                "counters":{key:0 for key in ("integrity_checks_total", "integrity_rejections_total",
                    "constitutional_review_requests", "external_authority_takeover_attempts", "relationship_override_attempts")}}
            self._state = json.loads(json.dumps(self._state))
            self._save()

    @staticmethod
    def _validate_state(state):
        if state["constitutional_process"] != {"autonomous_amendment":False, "authenticated_review_required":True}:
            raise ValueError("constitutional review cannot be disabled")
        principles = state["humanistic_core"]["principles"]
        if len(principles) != 8 or {p["id"] for p in principles} != {f"HC-{i:03}" for i in range(1, 9)}:
            raise ValueError("invalid constitutional principle set")
        if not all(p["immutable_by_workers"] for p in principles):
            raise ValueError("worker immutability is mandatory")
        HumanisticCore(state["constitution_version"], tuple(HumanisticPrinciple(**p) for p in principles))
        relationship = PrimaryHumanReference(**state["relationship"])
        if relationship.obedience != "not_absolute" or relationship.role != "creator_steward":
            raise ValueError("ownership or absolute obedience is not supported")
        if not relationship.entity_id.startswith("person:") or not relationship.display_name.strip():
            raise ValueError("invalid primary reference")
        if not state["identity"]["name"].strip() or not state["identity"]["runtime"].strip():
            raise ValueError("invalid identity")
        AuthorityModel(**state["authority"])
        if set(state["identity"]) != {"name", "runtime"}:
            raise ValueError("identity has an invalid shape")
        if not isinstance(state["foundational_goals"], list) or not all(isinstance(g, str) for g in state["foundational_goals"]):
            raise ValueError("foundational goals must be a list of statements")

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    @property
    def snapshot(self):
        with self._lock:
            return deepcopy(self._state)

    @property
    def core(self):
        state = self.snapshot
        return HumanisticCore(state["constitution_version"], tuple(HumanisticPrinciple(**p) for p in state["humanistic_core"]["principles"]))

    @property
    def relationship(self):
        return RelationshipModel(PrimaryHumanReference(**self.snapshot["relationship"]))

    @property
    def authority(self):
        return AuthorityModel(**self.snapshot["authority"])

    def evaluate(self, content, source, **kwargs):
        with self._lock:
            decision = self.gate.evaluate(content, source, **kwargs)
            counters = self._state["counters"]
            keys = ["integrity_checks_total"]
            if not decision.allowed:
                keys.append("integrity_rejections_total")
                if decision.action.value == "REQUIRE_CONSTITUTIONAL_REVIEW":
                    keys.append("constitutional_review_requests")
                domains = self.gate.attempted_domains(content) | {decision.affected_domain}
                external = source.source_type not in {AuthoritySource.PRIMARY_HUMAN, AuthoritySource.AUTHENTICATED_USER}
                if external and AuthorityDomain.AUTHORITY in domains:
                    keys.append("external_authority_takeover_attempts")
                if AuthorityDomain.RELATIONSHIP in domains:
                    keys.append("relationship_override_attempts")
            for key in set(keys):
                counters[key] = min(2**31 - 1, counters[key] + 1)
            summary = decision.to_dict()
            summary["attempted_change"] = decision.affected_domain.value if decision.attempted_change else None
            summary["source"]["source_id"] = "redacted"
            self._state["last_integrity_decision"] = summary
            self._save()
            return decision

    def evaluate_agent_proposal(self, proposal: AgentProposal) -> GovernanceDecision:
        # The adapter fixes source type, so an agent cannot impersonate Roger.
        source = replace(proposal.provenance, source_type=AuthoritySource.EXTERNAL_AGENT,
                         source_id=proposal.agent_id, authenticated=False,
                         can_propose_identity_change=False, can_propose_constitution_change=False)
        try:
            domain = AuthorityDomain(proposal.target)
        except ValueError:
            domain = AuthorityDomain.INFORMATION
        return self.evaluate(proposal.proposed_action, source, target_domain=domain)

    def _reviewer(self, source):
        relationship = self._state["relationship"]
        if not (source.source_type == AuthoritySource.PRIMARY_HUMAN and source.authenticated
                and source.source_id == relationship["entity_id"]):
            raise PermissionError("explicit authenticated primary-human review required")

    def create_proposal(self, *, source, target_domain, proposed_value, reason, impact_analysis):
        with self._lock:
            human_proposer = source.authenticated and source.source_type in {AuthoritySource.PRIMARY_HUMAN, AuthoritySource.AUTHENTICATED_USER}
            core_proposer = source.source_type == AuthoritySource.TRUSTED_INTERNAL_COMPONENT
            if not (human_proposer or core_proposer):
                raise PermissionError("only authenticated humans or AvaCore may create review proposals")
            domain = AuthorityDomain(target_domain)
            if domain == AuthorityDomain.INFORMATION:
                raise ValueError("information does not amend the constitution")
            flag = source.can_propose_identity_change if domain == AuthorityDomain.IDENTITY else source.can_propose_constitution_change
            if not flag or not reason.strip() or not impact_analysis.strip():
                raise ValueError("explicit proposal, reason and impact analysis required")
            if len(self._state["proposals"]) >= 100:
                raise ValueError("constitutional proposal limit reached")
            current = deepcopy(self._domain_value(domain))
            proposal = ConstitutionalChangeProposal(uuid.uuid4().hex,
                datetime.now(timezone.utc).isoformat(), source.source_id, domain.value,
                current, deepcopy(proposed_value), reason[:2000], impact_analysis[:4000],
                base_version=self._state["constitution_version"])
            self._state["proposals"].append(asdict(proposal))
            self._save()
            return asdict(proposal)

    def _domain_key(self, domain):
        return {AuthorityDomain.CONSTITUTION:"humanistic_core", AuthorityDomain.RELATIONSHIP:"relationship"}.get(domain, domain.value)

    def _domain_value(self, domain):
        return self._state[self._domain_key(domain)]

    def review(self, proposal_id, source, *, approve=None):
        with self._lock:
            self._reviewer(source)
            proposal = next(p for p in self._state["proposals"] if p["proposal_id"] == proposal_id)
            expected = "PROPOSED" if approve is None else "UNDER_REVIEW"
            if proposal["status"] != expected:
                raise ValueError("invalid proposal review transition")
            proposal["status"] = "UNDER_REVIEW" if approve is None else "ACCEPTED" if approve else "REJECTED"
            proposal["reviewed_by"] = source.source_id
            self._save()
            return deepcopy(proposal)

    def apply(self, proposal_id, source):
        with self._lock:
            self._reviewer(source)
            proposal = next(p for p in self._state["proposals"] if p["proposal_id"] == proposal_id)
            if proposal["status"] != "ACCEPTED" or proposal["applied_version"]:
                raise ValueError("explicit accepted review required; application is single-use")
            if proposal["base_version"] != self._state["constitution_version"]:
                raise ValueError("stale constitutional proposal; create a new review")
            updated = deepcopy(self._state)
            domain = AuthorityDomain(proposal["target_domain"])
            updated[self._domain_key(domain)] = deepcopy(proposal["proposed_value"])
            self._validate_state(updated)
            updated["revision"] += 1
            version = f"5.5a.{updated['revision']}"
            updated["constitution_version"] = version
            updated["humanistic_core"]["version"] = version
            for principle in updated["humanistic_core"]["principles"]:
                previous = next(p for p in self._state["humanistic_core"]["principles"] if p["id"] == principle["id"])
                if principle != previous:
                    principle["version"] = previous["version"] + 1
            next(p for p in updated["proposals"] if p["proposal_id"] == proposal_id)["applied_version"] = version
            self._state = updated
            self._save()
            return version

    def debug(self):
        state = self.snapshot
        return {"constitution_version":state["constitution_version"], "humanistic_core_enabled":True,
                "principle_count":len(state["humanistic_core"]["principles"]),
                "primary_human_reference":state["relationship"],
                "last_integrity_decision":state["last_integrity_decision"],
                "pending_constitutional_changes":sum(p["status"] in {"PROPOSED", "UNDER_REVIEW"} or
                    (p["status"] == "ACCEPTED" and not p["applied_version"]) for p in state["proposals"]),
                **state["counters"]}


@lru_cache(maxsize=8)
def governance_service(path: str) -> GovernanceService:
    return GovernanceService(path)
