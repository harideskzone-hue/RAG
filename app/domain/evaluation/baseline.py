from pydantic import BaseModel
from app.domain.evaluation.score import EvaluationScore

class EvaluationBaseline(BaseModel):
    """Historical high-water marks for scores to enable delta comparisons."""
    baseline_id: str
    version: str
    expected_score: EvaluationScore
    
class BaselineRepository:
    """Repository for retrieving expected baseline scores."""
    @staticmethod
    def get_baseline(version: str) -> EvaluationBaseline:
        raise NotImplementedError(
            "BaselineRepository requires a configured backend. "
            "Production fake implementations have been removed."
        )
