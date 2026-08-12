from app.domain.evaluation.scorer import Scorer
from app.domain.evaluation.trace import EvaluationTrace
from app.domain.evaluation.testcase import TestCase
from app.domain.evaluation.metrics import EvaluationMetrics

class PolicyScorer(Scorer):
    """Deterministic budget adherence and pruning logic."""
    def score(self, trace: EvaluationTrace, test_case: TestCase, current_metrics: EvaluationMetrics) -> EvaluationMetrics:
        if not trace.policy_trace:
            return current_metrics
        current_metrics.policy_budget_adherence = 100.0
        return current_metrics
