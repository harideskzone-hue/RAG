from datetime import datetime

from app.agents.confidence.engine import ConfidenceEngine
from app.agents.intent.enums import Intent
from app.domain.models.confidence import ConfidencePolicy
from app.domain.evidence import EvidenceBundle, MetadataEvidence, PersonEvidence


def test_confidence_engine_high_agreement():
    engine = ConfidenceEngine(ConfidencePolicy())
    bundle = EvidenceBundle()
    
    # Add metadata
    from uuid import uuid4
    from datetime import timedelta
    
    base_time = datetime.utcnow()
    
    bundle.add_evidence(MetadataEvidence(
        evidence_id=uuid4(), source="postgres_metadata", confidence=1.0, timestamp=base_time, trace_id=uuid4(),
        metadata={"camera_id": "cam_1"}
    ))
    # Add vector match on same camera (different timestamp to avoid deduplication)
    bundle.add_evidence(PersonEvidence(
        evidence_id=uuid4(), source="milvus_vector", confidence=0.9, timestamp=base_time + timedelta(seconds=5), trace_id=uuid4(),
        metadata={"camera_id": "cam_1"}
    ))
    
    result = engine.evaluate(bundle, Intent.PERSON_SEARCH.value)
    report = result.report
    
    assert report.metadata == 1.0
    assert report.vector == 0.9
    assert report.agreement == 0.95 # High agreement because same camera
    assert report.overall > 0.8
    assert result.next_action == "invoke_video" # Completeness is not 1.0 for Person Search without video
    assert result.requires_clarification == False
    
def test_confidence_engine_missing_evidence():
    engine = ConfidenceEngine(ConfidencePolicy())
    bundle = EvidenceBundle() # Empty bundle
    
    result = engine.evaluate(bundle, Intent.PERSON_SEARCH.value)
    assert result.report.completeness == 0.0
    assert result.next_action == "reject_query" # Overall score 0.0 < reject threshold (0.3)
    assert result.requires_clarification == True

def test_confidence_engine_low_confidence_clarification():
    policy = ConfidencePolicy(answer=0.9, clarification=0.8, reject=0.1)
    engine = ConfidenceEngine(policy)
    
    bundle = EvidenceBundle()
    from uuid import uuid4
    bundle.add_evidence(MetadataEvidence(
        evidence_id=uuid4(), source="postgres_metadata", confidence=0.2, timestamp=datetime.utcnow(), trace_id=uuid4()
    ))
    # Overall score will be low (e.g. ~0.3 - 0.5 depending on weights), which is < 0.8 but > 0.1
    
    result = engine.evaluate(bundle, Intent.CAMERA_STATUS.value)
    assert result.next_action == "ask_clarification"
    assert result.requires_clarification == True
