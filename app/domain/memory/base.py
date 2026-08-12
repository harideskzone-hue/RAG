from pydantic import BaseModel, Field
from typing import Any
from uuid import UUID, uuid4
import time

class BaseMemory(BaseModel):
    """Base class for all domain-specific memory objects."""
    memory_id: UUID = Field(default_factory=uuid4)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    
    @property
    def memory_type(self) -> str:
        return self.__class__.__name__
