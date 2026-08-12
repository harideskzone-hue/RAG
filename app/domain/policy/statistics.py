from pydantic import BaseModel

class PolicyStatistics(BaseModel):
    """Exposes metrics for the upcoming Evaluation Framework."""
    policies_executed: int = 0
    policies_matched: int = 0
    rejected_plans: int = 0
    modified_plans: int = 0
    deferred_plans: int = 0
    approved_plans: int = 0
    
    # Cost savings tracked through modifications/rejections
    execution_savings_tokens: int = 0
    execution_savings_usd: float = 0.0
