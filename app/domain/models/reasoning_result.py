from pydantic import BaseModel, Field

class ReasoningResult(BaseModel):
    conclusion: str
    confidence: float
    supporting_evidence: list[str] = Field(default_factory=list)
