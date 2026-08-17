import pytest
import os
import cv2
import json
from pathlib import Path
from app.cv.pipeline.video_pipeline import VideoPipeline
from app.schemas.evidence_contract import EvidenceContract

TEST_VIDEO_PATH = "input/VIDEO-2026-08-13-14-20-13.mp4"
MODEL_DIR = "models"
DETECTOR_MODEL = "yolo26n.pt"

@pytest.fixture
def pipeline_env():
    os.environ["CV_MODEL_DIR"] = MODEL_DIR
    os.environ["CV_DETECTOR_MODEL"] = DETECTOR_MODEL
    
    # Use a temporary directory for crops
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipeline = VideoPipeline(crop_dir=tmpdirname)
        yield pipeline, tmpdirname

def test_cv_tracking_pipeline(pipeline_env):
    pipeline, crop_dir = pipeline_env
    
    contracts = pipeline.process_video(
        video_path=TEST_VIDEO_PATH,
        video_id="test_vid_2",
        camera_id="FRONT_LOBBY_CAM_01"
    )
    
    assert len(contracts) > 0, "No evidence contracts produced"
    
    evidence_ids = set()
    track_ids = set()
    
    # 7. Bounding-box validation & 8. Crop quality validation
    # 9. evidence_id uniqueness & 10. track_id != evidence_id invariant
    # 11. EvidenceContract validation
    for contract in contracts:
        # 11. EvidenceContract validation (instance check and data presence)
        assert isinstance(contract, EvidenceContract)
        assert contract.evidence_id is not None
        assert contract.subject.track_id is not None
        assert contract.provenance.video_id == "test_vid_2"
        assert contract.provenance.camera_id == "FRONT_LOBBY_CAM_01"
        assert contract.confidence > 0.0
        assert contract.provenance.video_timestamp_sec >= 0.0
        
        # 9. evidence_id uniqueness
        assert contract.evidence_id not in evidence_ids, f"Duplicate evidence_id found: {contract.evidence_id}"
        evidence_ids.add(contract.evidence_id)
        
        # 10. track_id != evidence_id invariant
        assert contract.subject.track_id != str(contract.evidence_id), "track_id and evidence_id are the same!"
        
        track_ids.add(contract.subject.track_id)
        
        # 7. Bounding box validation
        assert "bbox" in contract.observation
        x1, y1, x2, y2 = contract.observation["bbox"]
        assert x2 > x1
        assert y2 > y1
        
        # 8. Crop validation
        crop_path = Path(crop_dir) / contract.subject.track_id / "crops" / f"{contract.observation.get('original_evidence_id', contract.evidence_id)}.jpg"
        assert crop_path.exists(), f"Crop missing: {crop_path}"
        crop_img = cv2.imread(str(crop_path))
        assert crop_img is not None
        assert crop_img.shape[0] > 0 and crop_img.shape[1] > 0

    print(f"Total evidence contracts (observations): {len(contracts)}")
    print(f"Unique track IDs: {len(track_ids)}")
    
    # 5. ID-switch measurement & 6. Track fragmentation measurement
    # While exact metrics require ground truth, we can validate that the pipeline 
    # executes without extreme fragmentation (e.g. tracks aren't 1-frame long on average)
    avg_track_length = len(contracts) / len(track_ids)
    assert avg_track_length > 1.0, "Extreme fragmentation detected: tracks are on average <= 1 observation long."
    print(f"Average track length: {avg_track_length:.2f} observations")

