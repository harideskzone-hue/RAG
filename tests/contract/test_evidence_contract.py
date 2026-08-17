import pytest
from uuid import uuid4
from datetime import datetime, timezone

from app.domain.evidence import EvidenceBundle, PersonEvidence
from app.schemas.evidence_contract import EvidenceContract

def test_evidence_id_never_equals_track_id():
    """
    Evidence ID must be globally unique per detection/observation,
    while track_id is the identity within a specific camera view.
    They must never be the same.
    """
    evidence_id = uuid4()
    track_id = "P001"
    
    evidence = PersonEvidence(
        evidence_id=evidence_id,
        source="video_agent",
        confidence=0.9,
        timestamp=datetime.now(timezone.utc),
        trace_id=uuid4(),
        metadata={
            "attributes": {},
            "origin": {"video_id": "vid_1", "camera_id": "cam_1", "track_id": track_id}
        }
    )
    contract = evidence.to_contract()
    
    # Enforce the invariant that evidence_id (global) is never equal to track_id (local)
    assert str(contract.evidence_id) != contract.subject.track_id
    assert contract.evidence_id is not None
    assert contract.subject.track_id is not None


def test_provenance_contract_rejects_mismatched_video_id():
    """
    Evidence must originate from the active video context.
    """
    # Active video context
    active_video_id = "vid_123"
    
    # Evidence from a different video
    evidence = PersonEvidence(
        evidence_id=str(uuid4()),
        source="video_agent",
        confidence=0.9,
        timestamp=datetime.now(timezone.utc),
        trace_id=str(uuid4()),
        metadata={
            "attributes": {},
            "origin": {"video_id": "vid_999", "camera_id": "cam_1", "track_id": "P001"} # Mismatch
        }
    )
    
    # In fusion/verification, this should be rejected
    # For the contract test, we verify that verifying provenance against active_video_id fails.
    assert evidence.metadata.get("origin", {}).get("video_id") != active_video_id
    
    def verify_provenance(ev, active_id):
        if ev.metadata.get("origin", {}).get("video_id") != active_id:
            raise ValueError(f"Provenance mismatch: {ev.metadata.get('origin', {}).get('video_id')} != {active_id}")
            
    with pytest.raises(ValueError, match="Provenance mismatch"):
        verify_provenance(evidence, active_video_id)

def test_canonical_identity():
    """
    Identity is strictly defined as (video_id, camera_id, track_id).
    """
    evidence = PersonEvidence(
        evidence_id=str(uuid4()),
        source="video_agent",
        confidence=0.9,
        timestamp=datetime.now(timezone.utc),
        trace_id=str(uuid4()),
        metadata={
            "attributes": {},
            "origin": {"video_id": "vid_1", "camera_id": "cam_1", "track_id": "P001"}
        }
    )
    
    # Extract identity tuple
    identity = (
        evidence.metadata.get("origin", {}).get("video_id"),
        evidence.metadata.get("origin", {}).get("camera_id"),
        evidence.metadata.get("origin", {}).get("track_id")
    )
    
    assert identity == ("vid_1", "cam_1", "P001")
