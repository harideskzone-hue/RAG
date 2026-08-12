
from pydantic import BaseModel


class MemoryPolicy(BaseModel):
    """
    Configuration for memory management and eviction.
    """
    # Time-to-Live settings (in hours). None means permanent.
    conversation_ttl_hours: int | None = 24
    evidence_ttl_hours: int | None = 12
    report_ttl_hours: int | None = None
    
    # Token or message count threshold before summarization triggers
    summary_threshold_tokens: int = 6000
    max_messages: int = 40
    
    # Enable/disable components
    enable_summarization: bool = True
    enable_eviction: bool = True
