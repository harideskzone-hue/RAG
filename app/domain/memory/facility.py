from app.domain.memory.base import BaseMemory
from typing import Any

class FacilityMemory(BaseMemory):
    """Retains spatial and topological context (e.g., camera placement, facility layouts, known blind spots)."""
    facility_id: str
    layout_data: dict[str, Any] = {}
    camera_topology: dict[str, list[str]] = {}
    known_blind_spots: list[str] = []
