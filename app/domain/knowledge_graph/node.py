from pydantic import BaseModel, Field
from typing import Any
from uuid import UUID, uuid4

class Node(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    created_at: float = 0.0
    updated_at: float = 0.0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id
