import os
import sys
import logging

logging.basicConfig(level=logging.INFO)

# Configure environment before loading registry
os.environ["CV_MODEL_DIR"] = "/Users/hariharans/Documents/longgraph/models"
os.environ["CV_DETECTOR_MODEL"] = "yolo26n.pt"
os.environ["CV_DEVICE"] = "cpu"
os.environ["CV_SAMPLE_FPS"] = "5"

from app.cv.pipeline.video_pipeline import VideoPipeline

def test_pipeline():
    print("Testing Video Pipeline...")
    pipeline = VideoPipeline()
    video_path = "/Users/hariharans/Documents/longgraph/vista_agentic_ai/dataset/storage/vista-video-bucket/sample_cctv.mp4"
    
    contracts = pipeline.process_video(video_path=video_path, camera_id="CAM_TEST")
    
    print(f"\nExtracted {len(contracts)} contracts total.")
    
    # Collect unique tracks
    unique_tracks = set()
    for contract in contracts:
        if contract.subject.track_id:
            unique_tracks.add(contract.subject.track_id)
            
    print(f"Detected {len(unique_tracks)} unique tracks: {unique_tracks}")
    
    print("\nSample Contract:")
    if contracts:
        print(contracts[0].model_dump_json(indent=2))

if __name__ == "__main__":
    test_pipeline()
