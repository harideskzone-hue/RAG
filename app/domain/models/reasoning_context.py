from typing import Any
from pydantic import BaseModel, Field, ConfigDict

from app.domain.models.entity import Entity
from app.domain.models.relationship import Relationship
from app.domain.models.agent_result import AgentResult
from app.domain.models.execution_metadata import ExecutionMetadata
from app.domain.models.reasoning_trace import ReasoningTrace
from app.domain.evidence import EvidenceBundle
from app.domain.models.blackboard import InvestigationBlackboard
from app.domain.knowledge_graph.query_engine import GraphQueryEngine

class ReasoningContext(BaseModel):
    """
    State payload passed between reasoning stages.
    We pass the GraphQueryEngine separately to engines, or keep it in context as an Any reference 
    if Pydantic doesn't serialize it well.
    """
    query: str
    query_intent: Any | None = None
    user: Any = None
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    
    # We will pass the GraphQueryEngine separately to engines, or keep it in context as an Any reference 
    # since it's an object with methods, but BaseModel doesn't handle arbitrary objects well unless arbitrary_types_allowed=True.
    # To keep it serializable, we just pass raw entities/relationships if needed, or allow arbitrary types.
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    
    investigation_state: str = "RUNNING"
    memory: dict[str, Any] = Field(default_factory=dict)
    execution_metadata: ExecutionMetadata | None = None
    
    trace: ReasoningTrace = Field(default_factory=ReasoningTrace)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # We inject the engine and bundle at runtime
    evidence_bundle: EvidenceBundle | None = None
    query_engine: GraphQueryEngine | None = None
    blackboard: InvestigationBlackboard = Field(default_factory=InvestigationBlackboard)
    investigation_goal: str = ""
    current_question: str = ""
    success_criteria: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
