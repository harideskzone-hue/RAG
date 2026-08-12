from pydantic import BaseModel
from app.domain.evaluation.metrics import EvaluationMetrics

class BenchmarkStatistics(BaseModel):
    """Aggregates metrics across the entire dataset run."""
    total_tests_run: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    aggregate_metrics: EvaluationMetrics = EvaluationMetrics()
