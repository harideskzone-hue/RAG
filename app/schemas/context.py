from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.domain.models.confidence import ConfidenceScore, ConfidenceReport
from app.domain.evidence import EvidenceBundle
from app.domain.models import AgentResult
from app.domain.models.enums import ExecutionMode, ExecutionState
from app.domain.models.blackboard import InvestigationBlackboard
from app.domain.models.reasoning_context import ReasoningContext

class QueryIntent(BaseModel):
    entities: list[str] = Field(default_factory=list)
    attributes: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    raw_query: str = ""

class UserContext(BaseModel):
    user_id: str
    role: str
    allowed_cameras: list[str] | None = None


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
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: ConfidenceScore = Field(default_factory=ConfidenceScore)
    error: str | None = None


class ToolRequirement(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    required: bool = True
    timeout: int = 5
    retry: int = 3


class ExecutionTask(BaseModel):
    task_id: str
    description: str
    agent_type: str
    dependencies: list[str] = Field(default_factory=list)


class ExecutionGroup(BaseModel):
    """A group of agents that can execute in parallel."""
    agents: list[str] = Field(default_factory=list)

    def __iter__(self):
        """Allow iterating directly over agent names for backward compatibility."""
        return iter(self.agents)

    def __len__(self):
        return len(self.agents)


class ExecutionPlan(BaseResult):
    intent: str = ""
    tasks: list[ExecutionTask] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
    tools: list[ToolRequirement] = Field(default_factory=list)
    execution_groups: list[ExecutionGroup] = Field(default_factory=list)
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    priority: str = "normal"
    risk_level: str = "LOW"
    requires_vlm: bool = False
    requires_confirmation: bool = False
    estimated_tokens: int = 0
    estimated_latency_ms: int = 0
    estimated_tools: int = 0
    estimated_llm_calls: int = 0
    success_rate: float = 1.0


class VistaContext(BaseModel):
    user: UserContext
    evidence_bundle: EvidenceBundle | None = None
    execution_plan: ExecutionPlan | None = None
    confidence_report: ConfidenceReport | None = None
    confidence_score: float = 0.0
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
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