from pydantic import BaseModel
from app.domain.evaluation.metrics import EvaluationMetrics

class EvaluationScore(BaseModel):
    """Weighted aggregate results based on raw metrics."""
    overall_score: float = 0.0
    
    @staticmethod
    def compute(metrics: EvaluationMetrics) -> "EvaluationScore":
        # Simplified weighted average for MVP
        score = (
            (metrics.planner_accuracy * 0.3) +
            (metrics.reasoning_accuracy * 0.3) +
            (metrics.graph_precision * 0.2) +
            (metrics.policy_budget_adherence * 0.2)
        )
        # Penalize hallucination
        score -= (metrics.hallucination_rate * 0.5)
        return EvaluationScore(overall_score=max(0.0, score * 100))
