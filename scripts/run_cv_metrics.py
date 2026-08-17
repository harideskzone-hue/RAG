import os
import sys
import argparse
import logging
from collections import defaultdict
from app.cv.pipeline.video_pipeline import VideoPipeline

def run_metrics(video_path: str, video_id: str, camera_id: str):
    print("========================================")
    print("      VISTA Phase 1 Quality Gate")
    print("========================================")
    
    pipeline = VideoPipeline()
    
    # Process video and capture output
    contracts = pipeline.process_video(video_path=video_path, video_id=video_id, camera_id=camera_id)
    
    # Collect metrics
    video_duration = pipeline.detector.registry.config.get("duration", "N/A") # Placeholder if we don't grab it directly
    
    # Since we can't easily retrieve duration after pipeline, let's just use the contracts to measure
    # the time spanned
    min_ts = min([c.observation["timestamp_sec"] for c in contracts]) if contracts else 0
    max_ts = max([c.observation["timestamp_sec"] for c in contracts]) if contracts else 0
    duration = max_ts - min_ts
    
    # Track statistics
    track_observations = defaultdict(list)
    for c in contracts:
        track_observations[c.subject.track_id].append(c.observation["timestamp_sec"])
        
    unique_tracks = len(track_observations)
    
    track_durations = []
    for tid, times in track_observations.items():
        if len(times) > 1:
            track_durations.append(max(times) - min(times))
        else:
            track_durations.append(0.0)
            
    avg_duration = sum(track_durations) / len(track_durations) if track_durations else 0.0
    min_duration = min(track_durations) if track_durations else 0.0
    max_duration = max(track_durations) if track_durations else 0.0
    
    evidence_ids = [str(c.evidence_id) for c in contracts]
    duplicate_evidence_ids = len(evidence_ids) - len(set(evidence_ids))

    print("\n[Video Properties]")
    print(f"Video:              {video_path}")
    print(f"Video ID:           {video_id}")
    print(f"Processed duration: {duration:.2f} seconds")
    print(f"Sample FPS Config:  {os.environ.get('CV_SAMPLE_FPS', 'Native')}")

    print("\n[Detection & Tracking]")
    print(f"Observations:          {len(contracts)}")
    print(f"Unique Tracker IDs:    {unique_tracks}")
    print(f"Average track length:  {avg_duration:.2f} seconds")
    print(f"Min track length:      {min_duration:.2f} seconds")
    print(f"Max track length:      {max_duration:.2f} seconds")
    
    print("\n[Quality Errors]")
    print(f"Duplicate Evidence IDs: {duplicate_evidence_ids}")
    print(f"Missing Crops:          NOT MEASURED")
    print(f"Invalid Bounding Boxes: NOT MEASURED")
    
    print("\n[Evaluation against Ground Truth]")
    print(f"Detection Precision:    NOT MEASURED")
    print(f"Detection Recall:       NOT MEASURED")
    print(f"ID Switches:            NOT MEASURED")
    print(f"Track Fragmentation:    NOT MEASURED")
    
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CV Metrics on a Video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--video_id", default="VID_001", help="Canonical video ID")
    parser.add_argument("--camera_id", default="CAM01", help="Canonical camera ID")
    
    args = parser.parse_args()
    run_metrics(args.video_path, args.video_id, args.camera_id)
