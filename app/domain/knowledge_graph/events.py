from pydantic import BaseModel, Field
from typing import Any
import time

class GraphEvent(BaseModel):
    """Base class for all graph mutation events."""
    event_type: str
    timestamp: float = Field(default_factory=time.time)
    payload: dict[str, Any]

class EntityAdded(GraphEvent):
    event_type: str = "EntityAdded"
    
class EntityMerged(GraphEvent):
    event_type: str = "EntityMerged"
    
class RelationshipCreated(GraphEvent):
    event_type: str = "RelationshipCreated"

class GraphEventBus:
    """Decoupled Pub/Sub for Graph Mutations."""
    def __init__(self):
        self._subscribers = []
        
    def subscribe(self, callback):
        self._subscribers.append(callback)
        
    def publish(self, event: GraphEvent):
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception as e:
                import logging
                logging.error(f"Error in subscriber {sub}: {e}")
