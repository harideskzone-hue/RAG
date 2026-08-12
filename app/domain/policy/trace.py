from pydantic import BaseModel, Field
import time

class TraceEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    rule_id: str
    action_taken: str
    details: str

class PolicyTrace(BaseModel):
    """Maintains an auditable log of which rules were matched and how the plan mutated."""
    execution_id: str
    events: list[TraceEvent] = []
    
    def add_event(self, rule_id: str, action_taken: str, details: str):
        self.events.append(TraceEvent(
            rule_id=rule_id,
            action_taken=action_taken,
            details=details
        ))
