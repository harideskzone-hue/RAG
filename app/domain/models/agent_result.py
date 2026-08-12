from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

from app.domain.evidence import BaseEvidence
from app.domain.models.entity import Entity
from app.domain.models.relationship import Relationship
from app.domain.models.execution_metadata import ExecutionMetadata
from app.domain.models.enums import AgentType, SchemaVersion, AgentStatus
from app.domain.models.confidence import ConfidenceScore

class AgentResult(BaseModel):
    result_id: UUID = Field(default_factory=uuid4)
    execution_id: UUID
    trace_id: UUID | None = None
    
    version: SchemaVersion = SchemaVersion.V1_0
    agent_name: str
    agent_type: AgentType
    status: AgentStatus
    confidence: ConfidenceScore
    
    evidence: list[BaseEvidence] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    
    execution: ExecutionMetadata
    metadata: dict[str, Any] = Field(default_factory=dict)
