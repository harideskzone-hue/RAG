from typing import Any
from pydantic import BaseModel, Field
from app.schemas.base import BaseResult

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
