from app.domain.evaluation.baseline import EvaluationBaseline
from app.domain.evaluation.score import EvaluationScore

class FakeBaselineRepository:
    """Mock repository for retrieving expected baseline scores in tests."""
    @staticmethod
    def get_baseline(version: str) -> EvaluationBaseline:
        return EvaluationBaseline(
            baseline_id="default_baseline",
            version=version,
            expected_score=EvaluationScore(overall_score=90.0) # Baseline is 90/100
        )
