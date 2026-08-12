from typing import Any

from pydantic import BaseModel, Field


class AgentEvent(BaseModel):
    """
    Standard Telemetry/Event model shared across the system.
    """
    agent_name: str
    event_type: str  # e.g., 'START', 'COMPLETE', 'ERROR', 'TIMEOUT'
    start_time: float
    end_time: float | None = None
    status: str
    latency_ms: float | None = None
    tokens: int = 0
    tool_calls: list[str] = Field(default_factory=list)
    cost: float = 0.0
    errors: list[str] = Field(default_factory=list)
    trace_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
