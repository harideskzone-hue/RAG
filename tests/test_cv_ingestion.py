import pytest
import os
import cv2
import numpy as np
from app.cv.ingestion.video_reader import VideoReader

TEST_VIDEO_PATH = "input/VIDEO-2026-08-13-14-20-13.mp4"

@pytest.fixture
def test_video_path():
    assert os.path.exists(TEST_VIDEO_PATH), f"Test video not found: {TEST_VIDEO_PATH}"
    return TEST_VIDEO_PATH

def test_video_reader_metadata(test_video_path):
    reader = VideoReader(test_video_path, video_id="test_vid_1")
    
    assert reader.metadata.video_id == "test_vid_1"
    assert reader.metadata.filename == "VIDEO-2026-08-13-14-20-13.mp4"
    assert reader.metadata.fps > 0
    assert reader.metadata.width > 0
    assert reader.metadata.height > 0
    assert reader.metadata.total_frames > 0
    assert reader.metadata.duration_sec > 0.0
    
    reader.close()

def test_video_reader_frames(test_video_path):
    reader = VideoReader(test_video_path, video_id="test_vid_1")
    
    frames_read = 0
    last_timestamp = -1.0
    
    # Read the first 50 frames to ensure it works
    for idx, frame, timestamp in reader.read_frames():
        assert idx == frames_read
        assert isinstance(frame, np.ndarray)
        assert frame.size > 0
        assert frame.shape[0] == reader.height
        assert frame.shape[1] == reader.width
        assert timestamp > last_timestamp
        
        last_timestamp = timestamp
        frames_read += 1
        
        if frames_read >= 50:
            break
            
    assert frames_read == 50
    reader.close()

def test_video_reader_invalid_path():
    with pytest.raises(IOError):
        VideoReader("invalid/path/to/video.mp4", video_id="invalid_1")
