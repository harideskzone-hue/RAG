from app.domain.evaluation.adapter import EvaluationAdapter
from app.domain.evaluation.testcase import TestCase
from app.domain.evaluation.trace import EvaluationTrace
from app.domain.evaluation.metrics import EvaluationMetrics
from app.domain.evaluation.scorer import Scorer

class Evaluator:
    """Runs a TestCase through the Adapter and scores the resulting trace."""
    
    def __init__(self, scorers: list[Scorer]):
        self.scorers = scorers
        self.adapter = EvaluationAdapter()
        
    def evaluate(self, test_case: TestCase) -> tuple[EvaluationTrace, EvaluationMetrics]:
        trace = self.adapter.execute(test_case)
        
        metrics = EvaluationMetrics()
        for scorer in self.scorers:
            metrics = scorer.score(trace, test_case, metrics)
            
        return trace, metrics
