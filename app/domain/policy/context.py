from pydantic import BaseModel
from typing import Any
from app.domain.policy.budget import ExecutionBudget

class PolicyContext(BaseModel):
    """Comprehensive state payload passed into the policy engine."""
    execution_mode: str
    memory_profile: str
    budget: ExecutionBudget
    confidence_threshold: float = 0.7
    query_type: str = "general"
    graph_statistics: dict[str, Any] = {}
    memory_statistics: dict[str, Any] = {}
    reasoning_result: Any | None = None
