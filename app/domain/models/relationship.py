from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from app.domain.models.enums import RelationshipType, GraphHint

class Relationship(BaseModel):
    relationship_id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    type: RelationshipType
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    graph_hints: list[GraphHint] = Field(default_factory=list)
    evidence_ids: list[UUID] = Field(default_factory=list)
