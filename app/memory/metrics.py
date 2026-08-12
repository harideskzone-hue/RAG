from pydantic import BaseModel


class MemoryMetrics(BaseModel):
    """
    Tracks statistics and operational metrics for the Memory Manager.
    """
    summaries_created: int = 0
    tokens_removed: int = 0
    eviction_count: int = 0
    cache_hit_ratio: float = 0.0
    average_conversation_length: float = 0.0
    memory_size_bytes: int = 0
    
    def increment_eviction(self, count: int = 1):
        self.eviction_count += count
        
    def increment_summaries(self, count: int = 1):
        self.summaries_created += count
        
    def increment_tokens_removed(self, count: int):
        self.tokens_removed += count
