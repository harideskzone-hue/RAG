from typing import Protocol
from app.domain.evaluation.dataset import EvaluationDataset
from app.domain.evaluation.trace import EvaluationTrace

class EvaluationRepository(Protocol):
    """Decoupled storage for benchmarks and traces."""
    def save_dataset(self, dataset: EvaluationDataset) -> None: ...
    def load_dataset(self, dataset_id: str, version: str) -> EvaluationDataset | None: ...
    def save_trace(self, trace: EvaluationTrace) -> None: ...
    def get_traces(self, test_id: str) -> list[EvaluationTrace]: ...

class InMemoryEvaluationRepository(EvaluationRepository):
    def __init__(self):
        self._datasets = {}
        self._traces = []
        
    def save_dataset(self, dataset: EvaluationDataset):
        self._datasets[(dataset.dataset_id, dataset.version)] = dataset
        
    def load_dataset(self, dataset_id: str, version: str) -> EvaluationDataset | None:
        return self._datasets.get((dataset_id, version))
        
    def save_trace(self, trace: EvaluationTrace):
        self._traces.append(trace)
        
    def get_traces(self, test_id: str) -> list[EvaluationTrace]:
        return [t for t in self._traces if t.test_id == test_id]
