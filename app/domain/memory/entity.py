from app.domain.memory.base import BaseMemory
from uuid import UUID
from typing import Any

class EntityMemory(BaseMemory):
    """Tracks entity behavior and history across time."""
    entity_id: UUID
    entity_type: str
    known_attributes: dict[str, Any] = {}
    behavior_history: list[str] = []
