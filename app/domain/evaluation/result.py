from pydantic import BaseModel
from app.domain.evaluation.metrics import EvaluationMetrics
from app.domain.evaluation.score import EvaluationScore
from app.domain.evaluation.regression import RegressionResult
from app.domain.evaluation.trace import EvaluationTrace

class EvaluationResult(BaseModel):
    """Immutable output of a benchmark run containing scores, traces, and regressions."""
    run_id: str
    metrics: EvaluationMetrics
    scores: EvaluationScore
    regression: RegressionResult
    traces: list[EvaluationTrace] = []
    duration_ms: float = 0.0
