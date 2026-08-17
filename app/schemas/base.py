from typing import Any
from pydantic import BaseModel, Field
from app.domain.models.confidence import ConfidenceScore

class Evidence(BaseModel):
    type: str = ""
    camera_id: str | None = None
    timestamp: str | None = None
    metadata_id: str | None = None
    milvus_match_id: str | None = None
    video_uri: str | None = None
    description: str
    confidence: float = 1.0


class Citation(BaseModel):
    source_type: str
    source_id: str
    content: str
    relevance_score: float = 1.0


class BaseResult(BaseModel):
    success: bool
    evidence: list[Any] = Field(default_factory=list)
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore)
    error: str | None = None
