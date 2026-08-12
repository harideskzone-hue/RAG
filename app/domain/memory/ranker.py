from app.domain.memory.base import BaseMemory

class MemoryRanker:
    """Truncates and ranks retrieved memories before passing them downstream."""
    
    @staticmethod
    def rank(memories: list[BaseMemory], max_items: int = 10) -> list[BaseMemory]:
        """
        Ranks memories based on recency, relevance, and confidence.
        For MVP, we sort by confidence and recency (updated_at).
        """
        # Sort descending by confidence, then by updated_at
        ranked = sorted(
            memories, 
            key=lambda m: (getattr(m, "confidence", 1.0), getattr(m, "updated_at", 0.0)), 
            reverse=True
        )
        return ranked[:max_items]
