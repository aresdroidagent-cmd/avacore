"""Explicit review records; workers cannot accept or apply their own amendments."""
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProposalStatus(str, Enum):
    PROPOSED = "PROPOSED"
    UNDER_REVIEW = "UNDER_REVIEW"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ConstitutionalChangeProposal:
    proposal_id: str
    created_at: str
    proposed_by: str
    target_domain: str
    current_value: Any
    proposed_value: Any
    reason: str
    impact_analysis: str
    status: str = ProposalStatus.PROPOSED.value
    base_version: str = ""
    reviewed_by: str | None = None
    applied_version: str | None = None
