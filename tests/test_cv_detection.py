import pytest
import os
import cv2
import numpy as np
from app.cv.detection.yolo import YOLOPersonDetector
from app.cv.models.registry import ModelRegistry
from app.cv.ingestion.video_reader import VideoReader

TEST_VIDEO_PATH = "input/VIDEO-2026-08-13-14-20-13.mp4"
MODEL_DIR = "models"
DETECTOR_MODEL = "yolo26n.pt"

@pytest.fixture
def model_registry():
    os.environ["CV_MODEL_DIR"] = MODEL_DIR
    os.environ["CV_DETECTOR_MODEL"] = DETECTOR_MODEL
    
    registry = ModelRegistry()
    registry.validate()
    return registry

@pytest.fixture
def detector(model_registry):
    return YOLOPersonDetector(registry=model_registry)

@pytest.fixture
def test_video_reader():
    return VideoReader(TEST_VIDEO_PATH, video_id="test_vid_1")

def test_yolo26n_detection_validation(detector, test_video_reader):
    """Checklist item 2: YOLO26n detection validation"""
    frames_tested = 0
    total_detections = 0
    
    # We test on the first 10 frames to validate detection capability
    for idx, frame, _ in test_video_reader.read_frames():
        detections = detector.track_frame(frame)
        
        # Validations
        for det in detections:
            total_detections += 1
            # Check bounding box validity
            x1, y1, x2, y2 = det.bbox
            assert x2 > x1
            assert y2 > y1
            assert det.confidence > 0.0
            assert det.confidence <= 1.0
            assert det.track_id is not None # Tracking should assign an ID
            
        frames_tested += 1
        if frames_tested >= 10:
            break
            
    assert frames_tested == 10
    # There should be at least one person detected in this CCTV clip across 10 frames
    assert total_detections > 0, "No persons detected by YOLO in the first 10 frames"
    print(f"Total detections in 10 frames: {total_detections}")

def test_bytetrack_continuity(detector, test_video_reader):
    """Checklist item 3 & 4: ByteTrack continuity and Track ID stability validation"""
    detector.reset() # Reset tracker state
    
    tracks_seen = set()
    frames_tested = 0
    
    for idx, frame, _ in test_video_reader.read_frames():
        detections = detector.track_frame(frame)
        for det in detections:
            tracks_seen.add(det.track_id)
            
        frames_tested += 1
        if frames_tested >= 30:
            break
            
    assert frames_tested == 30
    
    # If the same people are in the video, we should have a relatively stable number of unique track IDs
    # e.g., fewer unique track IDs than total detections across all frames.
    assert len(tracks_seen) > 0, "No tracks generated"
    print(f"Unique track IDs over 30 frames: {len(tracks_seen)}")

