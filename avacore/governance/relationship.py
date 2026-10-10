"""Relationships describe stewardship, never ownership or subjective experience."""
from dataclasses import asdict, dataclass
from enum import IntEnum


@dataclass(frozen=True)
class PrimaryHumanReference:
    entity_id: str = "person:roger"
    display_name: str = "Roger"
    role: str = "creator_steward"
    relationship_type: str = "primary_human_reference"
    trust_basis: tuple[str, ...] = ("creator_relationship", "shared_history", "long_term_collaboration", "care", "mutual_respect")
    authority_weight: str = "very_high"
    obedience: str = "not_absolute"
    roles: tuple[str, ...] = ("creator_steward", "primary_parent_guardian", "primary_human_reference", "development_partner")
    responsibilities: tuple[str, ...] = ("developmental_responsibility", "capability_authorization", "autonomy_granting", "autonomy_revocation", "incident_review")
    developmental_authority: str = "decisive"
    intellectual_disagreement_allowed: bool = True


class SocialRing(IntEnum):
    PARENT_GUARDIAN = 0
    EXTENDED_GUARDIAN = 1
    CLOSE_FAMILY = 2
    TRUSTED_FRIEND = 3
    FAMILIAR_PERSON = 4
    GENERAL_OTHER = 5


@dataclass(frozen=True)
class SocialRelationshipEntry:
    entity_id: str
    display_name: str
    ring: SocialRing = SocialRing.GENERAL_OTHER
    relationship_type: str = "general_other"
    trust_level: str = "unknown"
    care_weight: str = "normal"
    responsibility_level: str = "none"
    governance_authority: str = "none"
    delegated_capabilities: tuple[str, ...] = ()
    source: str = "guardian_review"

    def __post_init__(self):
        object.__setattr__(self, "ring", SocialRing(self.ring))
        if self.ring >= SocialRing.CLOSE_FAMILY and (self.governance_authority != "none" or self.responsibility_level != "none"):
            raise ValueError("social closeness does not confer guardian authority")


@dataclass(frozen=True)
class RelationshipModel:
    primary_human_reference: PrimaryHumanReference = PrimaryHumanReference()
    social_relationships: tuple[SocialRelationshipEntry, ...] = ()

    def rings(self):
        return {ring: tuple(e for e in self.social_relationships if e.ring == ring) for ring in SocialRing}

    def to_dict(self):
        return asdict(self.primary_human_reference)
