from pydantic import BaseModel

class MemoryStatistics(BaseModel):
    """Tracks memory health metrics for future evaluation baselines."""
    entity_memories_count: int = 0
    conversation_memories_count: int = 0
    facility_memories_count: int = 0
    episode_memories_count: int = 0
    summaries_count: int = 0
    investigations_count: int = 0
    
    total_retrieval_requests: int = 0
    cache_hits: int = 0
    average_retrieval_latency_ms: float = 0.0
    compression_ratio: float = 1.0 # 1.0 means no compression
    
    @property
    def hit_rate(self) -> float:
        """Percentage of retrieval requests that found matching memories."""
        if self.total_retrieval_requests == 0:
            return 0.0
        return self.cache_hits / self.total_retrieval_requests
