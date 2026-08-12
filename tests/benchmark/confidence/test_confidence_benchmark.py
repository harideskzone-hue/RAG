from datetime import datetime
from uuid import uuid4

from app.agents.confidence.engine import ConfidenceEngine
from app.domain.models.confidence import ConfidencePolicy
from app.domain.evidence import (
    EvidenceBundle,
    MetadataEvidence,
    PersonEvidence,
    VideoEvidence,
)


def test_confidence_engine_latency(benchmark):
    """
    Benchmark confidence engine calculation time.
    """
    policy = ConfidencePolicy()
    engine = ConfidenceEngine(policy)
    
    bundle = EvidenceBundle(
        evidence=[
            MetadataEvidence(evidence_id=uuid4(), timestamp=datetime.utcnow(), trace_id=uuid4(), source="postgres_metadata", content={"status": "online"}, metadata={}, confidence=1.0, reasoning=""),
            PersonEvidence(evidence_id=uuid4(), timestamp=datetime.utcnow(), trace_id=uuid4(), source="milvus_vector", content={"match": True}, metadata={}, confidence=0.8, reasoning=""),
            VideoEvidence(evidence_id=uuid4(), timestamp=datetime.utcnow(), trace_id=uuid4(), source="s3_video", content={"action": "walking"}, metadata={}, confidence=0.9, reasoning=""),
        ]
    )
    
    def run_engine():
        return engine.evaluate(bundle, "person_search")
        
    result = benchmark(run_engine)
    assert result.report.overall > 0
