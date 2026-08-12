from app.domain.policy.repository import InMemoryPolicyRepository
from app.domain.policy.rules import BudgetRule, ExecutionRule

def get_default_repository() -> InMemoryPolicyRepository:
    """Provides a repository pre-loaded with the default Investigation Policy."""
    repo = InMemoryPolicyRepository()
    
    default_rules = [
        BudgetRule(rule_id="default_budget_enforcement", priority=100),
        ExecutionRule(rule_id="skip_video_if_high_confidence", priority=50)
    ]
    
    repo.save_policy("default_investigation", default_rules)
    return repo
