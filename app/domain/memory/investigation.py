from app.domain.memory.base import BaseMemory
from typing import Any

class InvestigationMemory(BaseMemory):
    """Tracks active hypotheses, resolved questions, and evidence confidence at the investigation level."""
    investigation_id: str
    objective: str
    active_hypotheses: list[str] = []
    resolved_contradictions: list[str] = []
    global_state: dict[str, Any] = {}
