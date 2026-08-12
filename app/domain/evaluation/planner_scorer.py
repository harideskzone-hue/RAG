from app.domain.evaluation.scorer import Scorer
from app.domain.evaluation.trace import EvaluationTrace
from app.domain.evaluation.testcase import TestCase
from app.domain.evaluation.metrics import EvaluationMetrics

class PlannerScorer(Scorer):
    """Deterministic DAG and intent correctness scorer."""
    def score(self, trace: EvaluationTrace, test_case: TestCase, current_metrics: EvaluationMetrics) -> EvaluationMetrics:
        if not trace.planner_trace:
            return current_metrics
            
        executed_agents = trace.planner_trace.get("agents", [])
        expected = set(test_case.expected_agents)
        actual = set(executed_agents)
        
        if not expected:
            current_metrics.planner_accuracy = 100.0
            return current_metrics
            
        match_count = len(expected.intersection(actual))
        accuracy = (match_count / len(expected)) * 100.0
        current_metrics.planner_accuracy = accuracy
        return current_metrics
