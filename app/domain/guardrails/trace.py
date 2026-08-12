from pydantic import BaseModel, Field
import time

class GuardrailEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    guardrail_name: str
    status: str
    details: str

class GuardrailTrace(BaseModel):
    """Tracks which guardrails tripped and how many responses were blocked."""
    execution_id: str
    events: list[GuardrailEvent] = []
    blocked: bool = False
    
    def add_event(self, name: str, status: str, details: str):
        self.events.append(GuardrailEvent(guardrail_name=name, status=status, details=details))
