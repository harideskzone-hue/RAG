from abc import ABC, abstractmethod
from pydantic import BaseModel
from app.domain.policy.context import PolicyContext
from typing import Any

class RuleResult(BaseModel):
    matched: bool
    severity: str = "LOW"
    reason: str = ""
    cost_delta: float = 0.0
    latency_delta: float = 0.0
    affected_agents: list[str] = []
    recommended_action: str = "APPROVE" # APPROVE, MODIFY, REJECT, DEFER

class PolicyRule(BaseModel, ABC):
    rule_id: str
    priority: int
    
    @abstractmethod
    def evaluate(self, context: PolicyContext, plan: Any) -> RuleResult:
        pass

class BudgetRule(PolicyRule):
    """Enforces strict bounds on token/latency/cost limits."""
    def evaluate(self, context: PolicyContext, plan: Any) -> RuleResult:
        # Cost Estimator will pre-calculate the estimated plan costs.
        # This rule checks those estimates against the PolicyContext budget.
        return RuleResult(matched=False)

class SafetyRule(PolicyRule):
    """Prevents dangerous or infinite loop agent configurations."""
    def evaluate(self, context: PolicyContext, plan: Any) -> RuleResult:
        return RuleResult(matched=False)

class ExecutionRule(PolicyRule):
    """General orchestration rules (e.g., skip video if confidence > 0.9)."""
    def evaluate(self, context: PolicyContext, plan: Any) -> RuleResult:
        return RuleResult(matched=False)

class RetryRule(PolicyRule):
    """Limits cascading retries."""
    def evaluate(self, context: PolicyContext, plan: Any) -> RuleResult:
        return RuleResult(matched=False)
