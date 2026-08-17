import os
import pytest
import numpy as np
from pathlib import Path

from app.schemas.event_contract import DetectedEvent, VerifiedEventContract, IncidentEventType
from app.cv.events.evidence_builder import EventEvidenceBuilder
from app.cv.events.provenance_validator import EventProvenanceValidator
from app.cv.events.clip_slicer import EventClipSlicer
from app.domain.guardrails.validator import GuardrailValidator
from app.graph.nodes.grounding import GroundingValidatorNode


def test_evidence_builder_physical_measurements():
    builder = EventEvidenceBuilder()
    
    # Mock track observations with stationary presence
    mock_obs = [
        {"timestamp": 10.0, "bbox": [100, 100, 150, 200]},
        {"timestamp": 15.0, "bbox": [102, 101, 152, 201]},
        {"timestamp": 20.0, "bbox": [101, 100, 151, 200]},
        {"timestamp": 40.0, "bbox": [100, 102, 150, 202]},
    ]
    
    summaries = builder.build_track_summaries({"P014": mock_obs}, video_id="TEST_VID.mp4", camera_id="cam_01")
    assert len(summaries) == 1
    s = summaries[0]
    assert s["track_id"] == "P014"
    assert s["start_time"] == 10.0
    assert s["end_time"] == 40.0
    assert s["duration_sec"] == 30.0
    assert s["dispersion_radius_px"] < 10.0  # Stationary


def test_provenance_validator_rejects_abstain():
    validator = EventProvenanceValidator(min_confidence=0.70)
    
    abstain_event = DetectedEvent(
        event_type=IncidentEventType.ABSTAIN,
        confidence=0.40,
        start_time=10.0,
        end_time=20.0,
        track_ids=["P014"],
        reason="Normal walking movement",
        severity="LOW"
    )
    physical_summary = {
        "track_id": "P014",
        "camera_id": "cam_01",
        "video_id": "TEST_VID.mp4",
        "start_time": 10.0,
        "end_time": 20.0
    }
    contract = validator.validate_and_build_contract(abstain_event, physical_summary, ["PERSON_123"])
    assert contract is None


def test_provenance_validator_pins_cv_measurements():
    validator = EventProvenanceValidator(min_confidence=0.70)
    
    # Even if LLM proposed start_time=0.0 and track_ids=["P999"]
    proposed_event = DetectedEvent(
        event_type=IncidentEventType.LOITERING,
        confidence=0.92,
        start_time=0.0,
        end_time=99.0,
        track_ids=["P999"],
        reason="Person stayed stationary near entrance",
        severity="MEDIUM"
    )
    physical_summary = {
        "track_id": "P014",
        "camera_id": "cam_01",
        "video_id": "TEST_VID.mp4",
        "start_time": 10.0,
        "end_time": 40.0,
        "frame_count": 300
    }
    contract = validator.validate_and_build_contract(proposed_event, physical_summary, ["PERSON_A123"])
    assert contract is not None
    assert contract.event_type == "LOITERING"
    # Provenance pinned strictly from CV
    assert contract.start_time == 10.0
    assert contract.end_time == 40.0
    assert contract.track_ids == ["P014"]
    assert contract.canonical_person_ids == ["PERSON_A123"]
    assert contract.camera_id == "cam_01"


def test_clip_slicer_generation_and_sha256():
    source_video = "input/completed/VIDEO-2026-08-13-14-20-13.mp4"
    if not os.path.exists(source_video):
        pytest.skip("Test CCTV video not available")
        
    slicer = EventClipSlicer(output_root="dataset/events")
    event_id = "TEST_EVT_001"
    
    success, clip_path, thumb_path, sha256_hash = slicer.slice_event_clip(
        source_video_path=source_video,
        event_id=event_id,
        start_sec=10.0,
        end_sec=15.0,
        camera_id="cam_auto_01",
        video_id="VIDEO-2026-08-13-14-20-13.mp4"
    )
    
    assert success is True
    assert os.path.exists(clip_path)
    assert os.path.exists(thumb_path)
    assert len(sha256_hash) == 64
    assert (Path("dataset/events") / event_id / "event.json").exists()


@pytest.mark.asyncio
async def test_grounding_validator_rejects_fake_event_url():
    node = GroundingValidatorNode()
    
    class MockContract:
        verified_count = 1
        verified_persons = ["PERSON_A123"]
        cameras = ["CAM_01"]
        verified_events = [
            VerifiedEventContract(
                event_id="EVT_REAL_01",
                event_type="LOITERING",
                camera_id="CAM_01",
                video_id="VID.mp4",
                start_time=10.0,
                end_time=20.0,
                duration_sec=10.0,
                track_ids=["P001"],
                canonical_person_ids=["PERSON_A123"],
                confidence=0.95,
                severity="MEDIUM",
                clip_path="",
                clip_url="/media/events/EVT_REAL_01/clip.mp4",
                thumbnail_path="",
                thumbnail_url="",
                reason="Loitering detected",
                clip_sha256="",
                provenance={}
            )
        ]
        
    # State with fabricated event ID and URL
    malicious_state = {
        "final_response": "A robbery occurred: EVT_FAKE_99 at /media/events/EVT_FAKE_99/clip.mp4 on CAM_01.",
        "verified_contract": MockContract(),
        "abstain_reason": None
    }
    
    res = await node.execute(malicious_state)
    assert res["grounding_valid"] is False
    assert "rejected" in res["abstain_reason"]
