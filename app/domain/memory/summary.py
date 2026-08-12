from app.domain.memory.base import BaseMemory
from uuid import UUID

class SummaryMemory(BaseMemory):
    """Rolls up detailed Knowledge Graph clusters or old conversation turns into dense semantic summaries."""
    source_memory_ids: list[UUID] = []
    summary_text: str
    token_count: int = 0
