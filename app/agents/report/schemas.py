from typing import Any

from pydantic import Field

from app.domain.models import AgentResult


class ReportResult(AgentResult):
    """
    Result returned by the Report Agent.
    """
    report_uri: str = ""
    narrative: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
