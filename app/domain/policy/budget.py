from pydantic import BaseModel

class ExecutionBudget(BaseModel):
    """Tracks allowed resource usage for an execution trace."""
    max_tokens: int = 100000
    max_latency_ms: int = 60000
    max_retries: int = 3
    max_cost_usd: float = 1.0
