from abc import ABC, abstractmethod
from app.domain.evaluation.trace import EvaluationTrace
from app.domain.evaluation.testcase import TestCase
from app.domain.evaluation.metrics import EvaluationMetrics

class Scorer(ABC):
    @abstractmethod
    def score(self, trace: EvaluationTrace, test_case: TestCase, current_metrics: EvaluationMetrics) -> EvaluationMetrics:
        pass
