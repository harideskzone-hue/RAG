from enum import Enum
from app.domain.memory.base import BaseMemory

class MemoryUpdateAction(str, Enum):
    KEEP = "KEEP"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    DELETE = "DELETE"
    SUMMARIZE = "SUMMARIZE"

class MemoryUpdatePolicy:
    """Defines strict mutation policies governing how memories interact with new observations."""
    
    @staticmethod
    def evaluate(existing_memory: BaseMemory | None, new_observation: BaseMemory) -> MemoryUpdateAction:
        """
        Evaluate if we should keep the existing memory, update it with new facts,
        merge two memories into one, delete obsolete memory, or summarize a block of memories.
        """
        if not existing_memory:
            return MemoryUpdateAction.KEEP
        
        def _conf(obj):
            c = getattr(obj, "confidence", 1.0)
            return getattr(c, "overall", c) if hasattr(c, "overall") else (c if isinstance(c, (int, float)) else 1.0)
        
        if _conf(new_observation) > _conf(existing_memory):
            return MemoryUpdateAction.UPDATE
            
        return MemoryUpdateAction.KEEP
