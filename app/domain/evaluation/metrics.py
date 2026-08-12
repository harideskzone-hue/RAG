from pydantic import BaseModel

class EvaluationMetrics(BaseModel):
    """Raw numerical metrics from a benchmark run."""
    planner_accuracy: float = 0.0
    reasoning_accuracy: float = 0.0
    graph_precision: float = 0.0
    graph_recall: float = 0.0
    memory_hit_rate: float = 0.0
    policy_budget_adherence: float = 0.0
    hallucination_rate: float = 0.0
    average_latency_ms: float = 0.0
    total_token_cost: int = 0
