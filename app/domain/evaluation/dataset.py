from pydantic import BaseModel
from app.domain.evaluation.testcase import TestCase

class EvaluationDataset(BaseModel):
    dataset_id: str
    version: str # e.g., 'v1', 'v2'
    description: str
    test_cases: list[TestCase] = []
