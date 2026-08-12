from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
import pydantic

from app.domain.models.enums import AgentStatus

class AgentExecutionRequest(BaseModel):
    agent: str
    priority: str = "HIGH"
    reason: str
    required_entities: list[UUID] = Field(default_factory=list)
    expected_output: list[str] = Field(default_factory=list)

class EngineResult(BaseModel):
    success: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    partial_output: dict[str, Any] = Field(default_factory=dict)
    next_action: AgentExecutionRequest | None = None

class ConfidenceFactor(BaseModel):
    source: str
    score: float
    explanation: str

class Hypothesis(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    statement: str
    evidence_ids: list[str | UUID] = Field(default_factory=list)
    support_type: str = "direct"
    contradicting_evidence: list[str | UUID] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    confidence_factors: list[ConfidenceFactor] = Field(default_factory=list)
    evidence_weights: dict[str, float] = Field(default_factory=dict)
    score: float = 0.0

class Claim(BaseModel):
    statement: str = Field(description="The factual statement.")
    evidence_ids: list[str] = Field(description="List of evidence IDs that strictly support the claim. Can be empty if support_type is 'unknown' or 'abstention'.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0.")
    support_type: str = Field(description="How the evidence supports the claim (e.g., 'direct', 'inferred', 'unknown', 'abstention')")
    
    @pydantic.model_validator(mode='after')
    def validate_evidence(self):
        if self.support_type not in ['unknown', 'abstention'] and not self.evidence_ids:
            raise ValueError(f"evidence_ids cannot be empty unless support_type is 'unknown' or 'abstention'. Got {self.support_type}")
        return self

class ReasoningFailure(BaseModel):
    success: bool = False
    error: str = Field(description="Error message detailing the failure.")

class ReasoningResult(BaseModel):
    success: bool
    claims: list[Claim] = Field(description="List of verified claims.")
    uncertainties: list[str] = Field(description="List of contradictions or uncertainties.")
    answer: str = Field(description="The final natural language answer summarizing the claims.")
    
    # Legacy fields kept for compatibility with existing coordinator stages if needed
    confidence_factors: list[ConfidenceFactor] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    explanation: str = ""
    errors: list[str] = Field(default_factory=list)
    next_actions: list[AgentExecutionRequest] = Field(default_factory=list)

class Contradiction(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    description: str
    conflicting_evidence: list[UUID] = Field(default_factory=list)
    severity: str = "HIGH"

class InformationGap(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    description: str
    missing_entities: list[str] = Field(default_factory=list)
    recommended_actions: list[AgentExecutionRequest] = Field(default_factory=list)
