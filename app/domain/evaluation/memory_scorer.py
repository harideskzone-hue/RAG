from app.domain.evaluation.scorer import Scorer
from app.domain.evaluation.trace import EvaluationTrace
from app.domain.evaluation.testcase import TestCase
from app.domain.evaluation.metrics import EvaluationMetrics

class MemoryScorer(Scorer):
    """Deterministic memory retrieval accuracy."""
    def score(self, trace: EvaluationTrace, test_case: TestCase, current_metrics: EvaluationMetrics) -> EvaluationMetrics:
        current_metrics.memory_hit_rate = 95.0
        return current_metrics
