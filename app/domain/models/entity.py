from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from app.domain.models.enums import EntityType, GraphHint

class Entity(BaseModel):
    entity_id: UUID = Field(default_factory=uuid4)
    type: EntityType
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    graph_hints: list[GraphHint] = Field(default_factory=list)
