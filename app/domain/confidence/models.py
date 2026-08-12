from pydantic import BaseModel, Field


class ConfidenceScore(BaseModel):
    overall: float = 0.0
    factors: list[float] = Field(default_factory=list)


class ConfidenceExplanation(BaseModel):
    factor: str
    score: float
    explanation: str


class ConfidenceReport(BaseModel):
    overall: float = 0.0
    metadata: float | None = None
    vector: float | None = None
    video: float | None = None
    temporal: float = 0.0
    completeness: float = 0.0
    agreement: float = 0.0
    explanations: list[ConfidenceExplanation] = Field(default_factory=list)


class ConfidencePolicy(BaseModel):
    answer: float = 0.90
    clarification: float = 0.60
    reject: float = 0.30


class ConfidenceResult(BaseModel):
    success: bool = True
    report: ConfidenceReport
    next_action: str
    requires_clarification: bool = False
    ready_for_response: bool = False
    error: str = ""
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore)