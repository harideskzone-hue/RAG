
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
    crop_url: str | None = None
    clip_url: str | None = None
    person_id: str | None = None
    track_id: str | None = None

class ExecutionStepModel(BaseModel):
    name: str
    status: str
    latency_ms: int = 0
    error: str | None = None

class ExecutionTelemetryModel(BaseModel):
    status: str = "completed"
    steps: list[ExecutionStepModel] = Field(default_factory=list)

class ChatResponse(BaseModel):
    status: str
    detection_status: str = "DETECTED"  # DETECTED | EMPTY | ABSTAINED | ERROR
    person_count: int = 0
    zone: str = "entrance"
    evaluation_window: str | None = None
    scene_clip: str | None = None
    scene_thumbnail: str | None = None
    thought: str | None = None
    thinking_process: str | None = None
    answer: str | None
    grounding_status: str = "PENDING"
    confidence: float
    citations: list[CitationModel] = Field(default_factory=list)
    evidence: list[EvidenceModel] = Field(default_factory=list)
    timeline: list[dict] = Field(default_factory=list)
    processing: dict = Field(default_factory=dict)
    execution: ExecutionTelemetryModel = Field(default_factory=ExecutionTelemetryModel)
    processing_time_ms: int
    trace_id: str

class ReportResponse(BaseModel):
    job_id: str
    status: str
    message: str
