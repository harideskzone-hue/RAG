#!/usr/bin/env python3
"""
reprocess_video_metadata.py — Regenerates metadata and slices 10s incident clips for ingested videos.
"""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.cv.metadata.extractor import AutoVideoMetadataExtractor

def reprocess_all():
    tracks_root = PROJECT_ROOT / "dataset" / "tracks"
    if not tracks_root.exists():
        print("No tracks directory found.")
        return

    extractor = AutoVideoMetadataExtractor()

    for vid_dir in sorted(tracks_root.iterdir()):
        if vid_dir.is_dir():
            video_id = vid_dir.name
            print(f"▶ Reprocessing metadata & incident clip for: {video_id}")
            
            # Load track observations from track folders
            track_obs_map = {}
            resolved_pids = {}
            for tdir in sorted(vid_dir.iterdir()):
                if tdir.is_dir() and tdir.name.startswith("P"):
                    tjson = tdir / "track.json"
                    if tjson.exists():
                        try:
                            with open(tjson) as f:
                                tdata = json.load(f)
                            track_obs_map[tdir.name] = tdata.get("observations", [])
                            if tdata.get("canonical_person_id"):
                                resolved_pids[tdir.name] = tdata["canonical_person_id"]
                        except Exception:
                            pass

            if track_obs_map:
                meta = extractor.generate_video_metadata_json(
                    video_id=video_id,
                    camera_id="cam_auto_01",
                    track_obs_map=track_obs_map,
                    resolved_pids=resolved_pids
                )
                print(f"  ✓ Generated metadata with {len(meta.get('tracks', []))} tracks.")
                if meta.get("active_incident"):
                    print(f"  🚨 Active Incident: {meta['active_incident'].get('title')}")
                    print(f"     Clip URL: {meta['active_incident'].get('clip_url')}")

if __name__ == "__main__":
    reprocess_all()
