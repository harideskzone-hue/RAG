from pydantic import BaseModel
from app.domain.evaluation.score import EvaluationScore

class EvaluationBaseline(BaseModel):
    """Historical high-water marks for scores to enable delta comparisons."""
    baseline_id: str
    version: str
    expected_score: EvaluationScore
    
class BaselineRepository:
    """Mock repository for retrieving expected baseline scores."""
    @staticmethod
    def get_baseline(version: str) -> EvaluationBaseline:
        return EvaluationBaseline(
            baseline_id="default_baseline",
            version=version,
            expected_score=EvaluationScore(overall_score=90.0) # Baseline is 90/100
        )
