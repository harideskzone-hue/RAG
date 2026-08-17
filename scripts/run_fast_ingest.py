#!/usr/bin/env python3
"""
Fast Ingestion Executor for existing and new video segments.
"""
import os
import sys
import json
import asyncio
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cv.ingestion.manager import IngestionManager
from app.cv.evidence.builder import EvidenceBuilder
from app.api.dependencies.repositories import get_person_repository, get_vector_tool, get_event_bus
from app.infrastructure.db.mongodb.repository import MongoObservationRepository
from app.infrastructure.db.mongodb.client import mongo_client
from scripts.cross_store_audit import run_audit
from scripts.auto_ingest_daemon import update_segment_state, compute_file_hash, get_video_telemetry, COMPLETED_DIR, FAILED_DIR, PROCESSING_DIR


async def main():
    camera_id = "cam_auto_01"
    
    # Locate video file dynamically
    video_path = None
    if len(sys.argv) > 1:
        cand = Path(sys.argv[1])
        if cand.exists():
            video_path = cand

    if not video_path:
        for d in [Path("input/watch"), Path("input/processing"), Path("input/completed"), Path("input")]:
            mp4s = list(d.glob("*.mp4"))
            if mp4s:
                video_path = mp4s[0]
                break
            
    if not video_path:
        print("Error: No MP4 video found in input folders!")
        return

    video_filename = video_path.name

    print(f"1. Preparing video: {video_path}")
    proc_path = PROCESSING_DIR / video_filename
    if video_path != proc_path:
        shutil.move(str(video_path), str(proc_path))
    video_path = proc_path

    file_hash = compute_file_hash(video_path)
    telemetry = get_video_telemetry(video_path)

    # 2. Check if tracks already exist on disk
    track_dir = Path("dataset/tracks") / video_filename
    contracts = None
    if track_dir.exists():
        print(f"2. Found existing tracks in {track_dir}, loading pre-extracted observations...")
        all_obs = []
        for p_dir in sorted(track_dir.glob("P*")):
            track_json = p_dir / "track.json"
            if track_json.exists():
                try:
                    with open(track_json, "r") as f:
                        t_data = json.load(f)
                    t_vid = t_data.get("video_id", video_filename)
                    t_cam = t_data.get("camera_id", camera_id)
                    t_id = t_data.get("track_id", p_dir.name)
                    obs_list = t_data.get("observations", [])
                    for o in obs_list:
                        o["video_id"] = t_vid
                        o["camera_id"] = t_cam
                        o["track_id"] = t_id
                    all_obs.extend(obs_list)
                except Exception as e:
                    print(f"Error loading {track_json}: {e}")
        if all_obs:
            contracts = EvidenceBuilder.build_from_observations(all_obs)
            print(f"   Loaded {len(contracts)} pre-extracted observations for {len(list(track_dir.glob('P*')))} tracks.")

    # 3. Initialize IngestionManager
    event_bus = get_event_bus()
    vector_tool = get_vector_tool(event_bus)
    person_repo = get_person_repository(vector_tool)
    obs_repo = MongoObservationRepository(mongo_client)
    manager = IngestionManager(person_repo, obs_repo)

    print("3. Executing IngestionManager (Crop Quality -> OSNet Re-ID -> IdentityResolver -> Persistence)...")
    await update_segment_state(video_filename, status="PROCESSING", camera_id=camera_id, sha256=file_hash, telemetry=telemetry)

    success = await manager.process_and_persist(
        video_path=str(video_path),
        video_id=video_filename,
        camera_id=camera_id,
        contracts=contracts
    )

    if not success:
        print("✗ IngestionManager failed!")
        return

    print("4. Updating state to VERIFYING and running Cross-Store Referential Audit...")
    await update_segment_state(video_filename, status="VERIFYING", camera_id=camera_id, sha256=file_hash)

    audit = await run_audit(video_filename)
    audit.print_report()

    if not audit.passed:
        print("✗ Cross-store audit failed!")
        await update_segment_state(video_filename, status="FAILED", camera_id=camera_id, sha256=file_hash, error_message="Audit failed")
        shutil.move(str(video_path), str(FAILED_DIR / video_filename))
        return

    print("5. Audit PASSED! Moving video to completed directory and finalizing state...")
    await update_segment_state(video_filename, status="VERIFIED", camera_id=camera_id, sha256=file_hash)
    await update_segment_state(video_filename, status="CLEANUP_PENDING", camera_id=camera_id, sha256=file_hash)

    # In production with AUTO_DELETE, it would delete the raw video; here we move to completed for UI serving
    dest = COMPLETED_DIR / video_filename
    if video_path.exists() and video_path != dest:
        shutil.move(str(video_path), str(dest))

    await update_segment_state(video_filename, status="COMPLETED", cleanup_status="RETAINED", camera_id=camera_id, sha256=file_hash)
    print(f"✓ Video {video_filename} successfully ingested and verified!")


if __name__ == "__main__":
    asyncio.run(main())
