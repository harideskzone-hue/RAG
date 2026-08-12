from app.domain.memory.base import BaseMemory
from uuid import UUID

class EpisodeMemory(BaseMemory):
    """Tracks cross-investigation episodic occurrences ("I have seen this entity in a previous investigation")."""
    episode_id: str
    entity_id: UUID
    summary: str
    investigation_id: str
    time_range: tuple[float, float] = (0.0, 0.0)
