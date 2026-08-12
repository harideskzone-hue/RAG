from typing import Any

from pydantic import Field

from app.domain.models import AgentResult


class EventResult(AgentResult):
    """
    Result returned by the Event Agent.
    """
    event_type: str = ""
    severity: str = "Low"
    correlations: list[str] = Field(default_factory=list)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
