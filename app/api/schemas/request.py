
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="The user query for VISTA AI")
    conversation_id: str | None = Field(None, description="Identifier for a continuing conversation")
    camera_ids: list[str] = Field(default_factory=list, description="Specific cameras to focus on")
    video_id: str | None = Field(None, description="Active video investigation filename or ID")

class ReportRequest(BaseModel):
    query: str = Field(..., description="Details of the report to generate")
    time_range_hours: int = Field(24, description="Time range for the report in hours")
