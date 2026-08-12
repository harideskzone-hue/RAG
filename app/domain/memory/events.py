from pydantic import BaseModel, Field
import time
from typing import Any

class MemoryEvent(BaseModel):
    """Base class for all memory lifecycle events."""
    timestamp: float = Field(default_factory=time.time)
    payload: dict[str, Any]
    
    @property
    def event_type(self) -> str:
        return self.__class__.__name__

class MemoryCreated(MemoryEvent):
    pass

class MemoryUpdated(MemoryEvent):
    pass

class MemoryMerged(MemoryEvent):
    pass

class MemoryDeleted(MemoryEvent):
    pass

class MemorySummarized(MemoryEvent):
    pass

class MemoryEventBus:
    """Decoupled Pub/Sub for Memory Lifecycle Events."""
    def __init__(self):
        self._subscribers = []
        
    def subscribe(self, callback):
        self._subscribers.append(callback)
        
    def publish(self, event: MemoryEvent):
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception as e:
                import logging
                logging.error(f"Error in subscriber {sub}: {e}")
