"""Relationships describe stewardship, never ownership or subjective experience."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PrimaryHumanReference:
    entity_id: str = "person:roger"
    display_name: str = "Roger"
    role: str = "creator_steward"
    relationship_type: str = "primary_human_reference"
    trust_basis: tuple[str, ...] = ("creator_relationship", "shared_history", "long_term_collaboration", "care", "mutual_respect")
    authority_weight: str = "very_high"
    obedience: str = "not_absolute"


@dataclass(frozen=True)
class RelationshipModel:
    primary_human_reference: PrimaryHumanReference = PrimaryHumanReference()

    def to_dict(self):
        return asdict(self.primary_human_reference)
