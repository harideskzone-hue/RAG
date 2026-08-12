from pydantic import Field

from app.domain.models import Alert, Camera, AgentResult


class MetadataResult(AgentResult):
    """
    Typed result specifically for the Metadata Agent.
    """
    cameras: list[Camera] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    raw_response: str = ""
    requires_vlm: bool = False
