from pydantic import BaseModel
from typing import Any
from app.domain.policy.decision import PolicyDecision

class PolicyExplanation(BaseModel):
    """Provides explicit reasoning for *why* a decision was made."""
    decision: PolicyDecision
    primary_rule_id: str | None = None
    reason: str
    affected_agents: list[str] = []
    estimated_cost: float = 0.0
    estimated_latency: float = 0.0
    original_plan: Any | None = None
    validated_plan: Any | None = None
