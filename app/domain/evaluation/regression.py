from pydantic import BaseModel
from app.domain.evaluation.score import EvaluationScore
from app.domain.evaluation.baseline import EvaluationBaseline

class RegressionResult(BaseModel):
    is_regression: bool
    delta: float
    message: str

class RegressionDetector:
    """Evaluates deltas and flags failures if a score drops below acceptable thresholds."""
    
    @staticmethod
    def check_regression(current_score: EvaluationScore, baseline: EvaluationBaseline, fail_threshold: float = 5.0) -> RegressionResult:
        delta = current_score.overall_score - baseline.expected_score.overall_score
        is_regression = delta < -fail_threshold
        
        status = "FAIL" if is_regression else "PASS"
        message = f"[{status}] Current: {current_score.overall_score:.1f}, Baseline: {baseline.expected_score.overall_score:.1f}, Delta: {delta:+.1f}"
        
        return RegressionResult(
            is_regression=is_regression,
            delta=delta,
            message=message
        )
