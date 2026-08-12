from pydantic import BaseModel
from app.domain.evaluation.dataset import EvaluationDataset

class BenchmarkSuite(BaseModel):
    """Groups datasets into functional benchmark suites."""
    suite_id: str
    name: str # e.g. "Smoke Suite", "Regression Suite", "Performance Suite"
    datasets: list[EvaluationDataset] = []
