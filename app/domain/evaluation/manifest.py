from pydantic import BaseModel
from app.domain.evaluation.profile import EvaluationProfile

class EvaluationManifest(BaseModel):
    """Declarative configuration for a benchmark run."""
    enabled_scorers: list[str]
    required_profiles: list[EvaluationProfile]
    dataset_version: str = "v1"
    baseline_version: str = "v1"
    pass_threshold: float = 0.90
    fail_threshold: float = 0.80
    report_formats: list[str] = ["console", "markdown", "json", "junit"]
    ci_mode: bool = False
