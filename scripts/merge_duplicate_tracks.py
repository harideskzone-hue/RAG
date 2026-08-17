#!/usr/bin/env python3
"""
merge_duplicate_tracks.py — Unified Tracklet Consolidation & Fusion

Identifies fragmented tracklets of the same person within a video segment,
merges their observations and crops into a single primary track folder,
purges duplicate fragmented track folders, and updates track.json and metadata.json.
"""
import os
import sys
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.cv.reid.osnet import OSNetExtractor
from app.cv.identity.quality import CropQualitySelector

def merge_tracks_for_video(video_id: str = "chain_robbery_cctv.mp4", sim_threshold: float = 0.82):
    tracks_dir = PROJECT_ROOT / "dataset" / "tracks" / video_id
    if not tracks_dir.exists():
        print(f"Tracks directory not found: {tracks_dir}")
        return

    print("═" * 74)
    print(f"   VISTA AI — TRACKLET FUSION & CONSOLIDATION: {video_id}")
    print(f"   Re-ID Match Threshold: {sim_threshold:.2f}")
    print("═" * 74)

    extractor = OSNetExtractor()
    quality_selector = CropQualitySelector()

    # 1. Scan and compute representative embeddings for each track
    track_dirs = sorted([d for d in tracks_dir.iterdir() if d.is_dir() and d.name.startswith("P")])
    print(f"▶ Scanning {len(track_dirs)} track folders...")

    track_data = {}
    for tdir in track_dirs:
        crops = sorted(list((tdir / "crops").glob("*.jpg")))
        if not crops:
            continue

        best_crop = None
        best_score = -1.0
        for cf in crops[:20]:
            img = cv2.imread(str(cf))
            if img is not None:
                q = quality_selector.assess_quality(img)
                sc = float(q.get("score", 0))
                if sc > best_score:
                    best_score = sc
                    best_crop = img

        if best_crop is not None:
            emb = np.array(extractor.extract(best_crop), dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
                track_data[tdir.name] = {
                    "dir": tdir,
                    "crops_dir": tdir / "crops",
                    "track_json": tdir / "track.json",
                    "emb": emb,
                    "crop_count": len(crops),
                    "best_score": best_score
                }

    print(f"  ✓ Computed normalized visual embeddings for {len(track_data)} valid tracks.")

    # 2. Cluster tracks using Disjoint Set (Union-Find)
    parent = {t: t for t in track_data.keys()}

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            # Prefer the one with more crops or earlier track ID as root
            if track_data[root_i]["crop_count"] >= track_data[root_j]["crop_count"]:
                parent[root_j] = root_i
            else:
                parent[root_i] = root_j

    # Compute pairwise similarities and union tracks above threshold
    t_names = sorted(list(track_data.keys()))
    merged_pairs = []

    for i in range(len(t_names)):
        t1 = t_names[i]
        for j in range(i + 1, len(t_names)):
            t2 = t_names[j]
            sim = float(np.dot(track_data[t1]["emb"], track_data[t2]["emb"]))
            if sim >= sim_threshold:
                union(t1, t2)
                merged_pairs.append((t1, t2, sim))

    # Group tracks by their cluster root
    clusters = defaultdict(list)
    for t in t_names:
        clusters[find(t)].append(t)

    print(f"\n▶ Formed {len(clusters)} consolidated track clusters from {len(t_names)} initial tracks.")
    multi_track_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    print(f"  • Clusters with multiple fragmented tracklets: {len(multi_track_clusters)}")

    # 3. Perform On-Disk Merge and Cleanup
    print("\n▶ Consolidating on-disk track folders & metadata...")
    total_folders_merged = 0
    total_crops_merged = 0

    consolidated_metadata_tracks = []

    for root_t, members in sorted(clusters.items()):
        primary_info = track_data[root_t]
        primary_dir = primary_info["dir"]
        primary_crops_dir = primary_info["crops_dir"]
        primary_json_path = primary_info["track_json"]

        # Load primary track.json
        primary_data = {
            "video_id": video_id,
            "camera_id": "cam_auto_01",
            "track_id": root_t,
            "merged_track_ids": [root_t],
            "observations": []
        }
        if primary_json_path.exists():
            try:
                with open(primary_json_path) as f:
                    primary_data = json.load(f)
                    if "merged_track_ids" not in primary_data:
                        primary_data["merged_track_ids"] = [root_t]
            except Exception:
                pass

        # Existing crop names in primary
        primary_crop_names = {p.name for p in primary_crops_dir.glob("*.jpg")}

        # Merge secondary tracks into primary
        secondary_tracks = [m for m in members if m != root_t]
        for sec_t in secondary_tracks:
            sec_info = track_data[sec_t]
            sec_dir = sec_info["dir"]
            sec_crops_dir = sec_info["crops_dir"]
            sec_json_path = sec_info["track_json"]

            if sec_t not in primary_data["merged_track_ids"]:
                primary_data["merged_track_ids"].append(sec_t)

            # Move unique crops from secondary to primary
            if sec_crops_dir.exists():
                for cf in sec_crops_dir.glob("*.jpg"):
                    if cf.name not in primary_crop_names:
                        dest_cf = primary_crops_dir / cf.name
                        shutil.move(str(cf), str(dest_cf))
                        primary_crop_names.add(cf.name)
                        total_crops_merged += 1

            # Append observations
            if sec_json_path.exists():
                try:
                    with open(sec_json_path) as f:
                        sec_json = json.load(f)
                    for obs in sec_json.get("observations", []):
                        obs["original_track_id"] = sec_t
                        primary_data["observations"].append(obs)
                except Exception:
                    pass

            # Remove secondary fragmented track directory
            try:
                shutil.rmtree(str(sec_dir))
                total_folders_merged += 1
            except Exception as e:
                print(f"  Warning deleting {sec_dir}: {e}")

        # Sort all observations by frame_index / timestamp
        primary_data["observations"].sort(key=lambda x: x.get("frame_index", 0))

        # Save consolidated primary track.json
        with open(primary_json_path, "w") as f:
            json.dump(primary_data, f, indent=2)

        # Record for metadata.json
        obs_count = len(primary_data["observations"])
        start_ts = primary_data["observations"][0].get("timestamp_sec", 0.0) if primary_data["observations"] else 0.0
        end_ts = primary_data["observations"][-1].get("timestamp_sec", 0.0) if primary_data["observations"] else 0.0
        consolidated_metadata_tracks.append({
            "track_id": root_t,
            "merged_track_ids": primary_data["merged_track_ids"],
            "observation_count": obs_count,
            "start_time_sec": start_ts,
            "end_time_sec": end_ts,
            "duration_sec": round(end_ts - start_ts, 2),
            "canonical_person_id": primary_data.get("canonical_person_id", root_t)
        })

    # 4. Update metadata.json for the video
    meta_json_path = tracks_dir / "metadata.json"
    meta_data = {
        "video_id": video_id,
        "camera_id": "cam_auto_01",
        "total_tracks": len(clusters),
        "unique_persons_count": len(clusters),
        "tracks": consolidated_metadata_tracks,
        "events": []
    }
    if meta_json_path.exists():
        try:
            with open(meta_json_path) as f:
                old_meta = json.load(f)
            meta_data["events"] = old_meta.get("events", [])
        except Exception:
            pass

    with open(meta_json_path, "w") as f:
        json.dump(meta_data, f, indent=2)

    print("─" * 74)
    print("📊 Tracklet Consolidation Summary:")
    print(f"  • Initial Fragmented Track Folders:     {len(track_dirs)}")
    print(f"  • Redundant Duplicate Folders Merged:   {total_folders_merged}")
    print(f"  • Unified Primary Tracks Retained:      {len(clusters)}")
    print(f"  • Crops Re-indexed into Primary Tracks: {total_crops_merged}")
    print("═" * 74 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="chain_robbery_cctv.mp4")
    parser.add_argument("--threshold", type=float, default=0.82)
    args = parser.parse_args()
    merge_tracks_for_video(video_id=args.video, sim_threshold=args.threshold)
