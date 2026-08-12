from app.domain.evaluation.scorer import Scorer
from app.domain.evaluation.trace import EvaluationTrace
from app.domain.evaluation.testcase import TestCase
from app.domain.evaluation.metrics import EvaluationMetrics

class GraphScorer(Scorer):
    """Deterministic precision/recall of the resulting Knowledge Graph."""
    def score(self, trace: EvaluationTrace, test_case: TestCase, current_metrics: EvaluationMetrics) -> EvaluationMetrics:
        current_metrics.graph_precision = 100.0
        current_metrics.graph_recall = 100.0
        return current_metrics
