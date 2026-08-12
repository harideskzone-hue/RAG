
from pydantic import BaseModel, Field


class CitationModel(BaseModel):
    source: str
    content: str
    confidence: float

class EvidenceModel(BaseModel):
    evidence_id: str
    source: str
    camera_id: str | None = None
    timestamp: str | None = None
    description: str | None = None
    confidence: float

class ChatResponse(BaseModel):
    status: str
    answer: str | None
    confidence: float
    citations: list[CitationModel] = Field(default_factory=list)
    evidence: list[EvidenceModel] = Field(default_factory=list)
    processing_time_ms: int
    trace_id: str

class ReportResponse(BaseModel):
    job_id: str
    status: str
    message: str
