from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.domain.models.confidence import ConfidenceScore, ConfidenceReport
from app.domain.evidence import EvidenceBundle
from app.domain.models import AgentResult
from app.domain.models.enums import ExecutionMode, ExecutionState
from app.domain.models.blackboard import InvestigationBlackboard
from app.domain.models.reasoning_context import ReasoningContext

# Re-export from focused modules for backward compatibility
from app.schemas.query import QueryIntent
from app.schemas.base import Evidence, Citation, BaseResult
from app.schemas.plan import ToolRequirement, ExecutionTask, ExecutionGroup, ExecutionPlan

class UserContext(BaseModel):
    user_id: str
    role: str
    allowed_cameras: list[str] | None = None


class VistaContext(BaseModel):
    user: UserContext
    evidence_bundle: EvidenceBundle | None = None
    execution_plan: ExecutionPlan | None = None
    confidence_report: ConfidenceReport | None = None
    confidence_score: float = 0.0
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    active_video_id: str | None = None
    current_query: str | None = None
    query_intent: QueryIntent | None = None
    execution_mode: ExecutionMode = ExecutionMode.SIMPLE
    results: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    agent_decisions: list[dict[str, Any]] = Field(default_factory=list)
    execution_ledger: list["AgentExecutionRecord"] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    messages: list[dict[str, Any]] = Field(default_factory=list)
    mcp_execution_count: int = 0


class AgentExecutionRecord(BaseModel):
    agent_name: str
    task_id: str
    output: dict[str, Any] = Field(default_factory=dict)
    status: str
    execution_time_ms: float
    confidence: float = 0.0
    retry_count: int = 0
    timestamp: float
    error: str | None = None