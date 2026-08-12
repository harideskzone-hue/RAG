from pydantic import Field

from app.domain.models import AgentResult


class VideoResult(AgentResult):
    """
    Result returned by the Video Agent.
    """
    scene_summary: str = ""
    objects: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    timeline: list[dict[str, str]] = Field(default_factory=list)
    frames_analyzed: int = 0
    reasoning: str = ""
