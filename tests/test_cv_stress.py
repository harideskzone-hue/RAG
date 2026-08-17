import pytest
import os
import gc
import psutil
from app.cv.pipeline.video_pipeline import VideoPipeline

TEST_VIDEO_PATH = "input/VIDEO-2026-08-13-14-20-13.mp4"
MODEL_DIR = "models"
DETECTOR_MODEL = "yolo26n.pt"

@pytest.fixture
def pipeline_env():
    os.environ["CV_MODEL_DIR"] = MODEL_DIR
    os.environ["CV_DETECTOR_MODEL"] = DETECTOR_MODEL
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        pipeline = VideoPipeline(crop_dir=tmpdirname)
        yield pipeline

def test_cv_stress_pipeline(pipeline_env):
    """Checklist item 12: 5-minute video stress test (simulated by full processing of the clip).
    Validates that the pipeline does not OOM and successfully completes a full video.
    """
    pipeline = pipeline_env
    process = psutil.Process(os.getpid())
    
    import time
    start_time = time.time()
    
    initial_mem = process.memory_info().rss / (1024 * 1024)
    print(f"Initial Memory: {initial_mem:.2f} MB")
    
    total_processed_duration = 0.0
    total_frames = 0
    total_observations = 0
    unique_tracks = set()
    
    from app.cv.ingestion.video_reader import VideoReader
    
    # We loop the video to exceed 300 seconds
    loop_count = 0
    while total_processed_duration < 300.0:
        loop_count += 1
        vid_id = f"stress_vid_{loop_count}"
        
        # Determine video duration
        reader = VideoReader(TEST_VIDEO_PATH, vid_id)
        duration = reader.total_frames / reader.native_fps
        total_processed_duration += duration
        total_frames += reader.total_frames
        
        contracts = pipeline.process_video(
            video_path=TEST_VIDEO_PATH,
            video_id=vid_id,
            camera_id="FRONT_LOBBY_CAM_01"
        )
        
        total_observations += len(contracts)
        unique_tracks.update([c.subject.track_id for c in contracts if c.subject.track_id])
    
    gc.collect()
    final_mem = process.memory_info().rss / (1024 * 1024)
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("\\n--- 5-Minute Stress Test Telemetry ---")
    print(f"Input duration: 300+ seconds ({total_processed_duration:.2f} seconds)")
    print(f"Processed duration: {total_processed_duration:.2f} seconds")
    print(f"Frames processed: {total_frames}")
    print(f"Sample FPS: {os.environ.get('CV_SAMPLE_FPS', 5)}")
    print(f"Unique tracks: {len(unique_tracks)}")
    print(f"Observations: {total_observations}")
    print(f"Peak memory (final): {final_mem:.2f} MB")
    print(f"Processing time: {processing_time:.2f} seconds")
    print("--------------------------------------\\n")
    
    assert total_processed_duration >= 300.0, "Failed to process 300 seconds of video."
    assert total_observations > 0, "Stress test failed to extract observations"
    assert final_mem < initial_mem + 1000, f"Memory grew excessively to {final_mem:.2f} MB"
