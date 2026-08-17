#!/usr/bin/env python3
"""
VISTA AI — 24/7 Continuous Ingestion Daemon (Phase 10 Production Specification)

Orchestrates continuous video segment ingestion through:
1. Atomic detection in input/watch/
2. Durable PostgreSQL state machine transitions:
   READY -> PROCESSING -> CV_COMPLETE -> PERSISTING -> VERIFYING -> VERIFIED -> CLEANUP_PENDING -> COMPLETED (or FAILED)
3. Existing VISTA CV Pipeline:
   YOLO26n -> ByteTrack -> Crop Quality -> OSNet MSMT17 512-D -> IdentityResolver (0.82/0.05)
4. Multi-Store Persistence:
   - PostgreSQL (Source of Truth)
   - MongoDB (Observation History)
   - Qdrant (512-D Isolated Person Embeddings)
   - Object Storage (Keyframe crops & Sliced Event Clips)
5. Conditional Cross-Store Referential Audit
6. Safe Video Cleanup (removes raw segment while retaining crops, event clips & metadata)
7. Startup Restart Recovery & Reconciliation
"""
import os
import shutil
import time
import logging
import hashlib
import asyncio
import sys
import cv2
from pathlib import Path
from datetime import datetime, timezone
from typing import Set, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CV_MODEL_DIR"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

from app.cv.ingestion.manager import IngestionManager
from app.api.dependencies.repositories import get_person_repository, get_vector_tool, get_event_bus
from app.infrastructure.db.mongodb.repository import MongoObservationRepository
from app.infrastructure.db.mongodb.client import mongo_client
from app.config.db import db_settings
from scripts.cross_store_audit import run_audit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("daemon_error.log"), logging.StreamHandler()]
)
logger = logging.getLogger("AutoIngestDaemon")

INPUT_DIR = Path("input")
RECORDING_DIR = INPUT_DIR / "recording"
WATCH_DIR = INPUT_DIR / "watch"
PROCESSING_DIR = INPUT_DIR / "processing"
COMPLETED_DIR = INPUT_DIR / "completed"
FAILED_DIR = INPUT_DIR / "failed"

AUTO_DELETE_AFTER_VERIFICATION = os.getenv("AUTO_DELETE_AFTER_VERIFICATION", "true").lower() == "true"
POLL_INTERVAL_SEC = 3


def ensure_directories():
    RECORDING_DIR.mkdir(parents=True, exist_ok=True)
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    COMPLETED_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 hash of video segment."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_video_telemetry(filepath: Path) -> dict:
    """Extracts frame count, duration, and fps using OpenCV."""
    cap = cv2.VideoCapture(str(filepath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_sec": round(duration_sec, 2),
        "expected_frames": frame_count,
        "received_frames": frame_count,
        "dropped_frames": 0
    }


async def update_segment_state(
    video_id: str,
    status: str,
    camera_id: str = "cam_auto_01",
    sha256: Optional[str] = None,
    telemetry: Optional[dict] = None,
    cleanup_status: Optional[str] = None,
    error_message: Optional[str] = None
):
    """Updates the durable state machine in PostgreSQL."""
    try:
        import asyncpg
        pg_uri = db_settings.POSTGRES_URI.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(pg_uri)
        
        telemetry = telemetry or {}
        fps = telemetry.get("fps")
        duration_sec = telemetry.get("duration_sec")
        frame_count = telemetry.get("frame_count", 0)
        
        await conn.execute("""
            INSERT INTO video_segments (
                video_id, camera_id, sha256, status, cleanup_status,
                duration_sec, fps, frame_count, error_message, processed_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            ON CONFLICT (video_id) DO UPDATE SET
                status = EXCLUDED.status,
                cleanup_status = COALESCE($5, video_segments.cleanup_status),
                error_message = EXCLUDED.error_message,
                processed_at = NOW()
        """, video_id, camera_id, sha256 or "", status, cleanup_status or "PENDING",
            duration_sec, fps, frame_count, error_message)
        
        await conn.close()
    except Exception as e:
        logger.warning(f"PostgreSQL state update ({status}) for {video_id}: {e}")


async def reconcile_interrupted_segments():
    """
    Restart Recovery: Reconciles interrupted segments from previous daemon crashes.
    """
    logger.info("Checking for interrupted segments to reconcile...")
    ensure_directories()
    
    # 1. Clean up lingering files in processing/ if previously completed or verified
    for proc_file in PROCESSING_DIR.glob("*.mp4"):
        video_id = proc_file.name
        try:
            import asyncpg
            pg_uri = db_settings.POSTGRES_URI.replace("postgresql+asyncpg://", "postgresql://")
            conn = await asyncpg.connect(pg_uri)
            row = await conn.fetchrow("SELECT status, cleanup_status FROM video_segments WHERE video_id = $1", video_id)
            await conn.close()
            
            if row:
                status = row.get("status")
                if status in ["VERIFIED", "CLEANUP_PENDING", "COMPLETED"]:
                    logger.info(f"Reconciliation: Safely removing verified lingering video {video_id}")
                    if proc_file.exists():
                        proc_file.unlink()
                    await update_segment_state(video_id, status="COMPLETED", cleanup_status="DELETED")
                    continue
        except Exception:
            pass

        # Otherwise move back to watch/ for clean retry
        logger.info(f"Reconciliation: Moving interrupted segment {proc_file.name} back to watch/ for processing")
        watch_target = WATCH_DIR / proc_file.name
        try:
            shutil.move(str(proc_file), str(watch_target))
        except Exception as e:
            logger.error(f"Failed to move {proc_file.name} to watch: {e}")


async def process_video_file(filepath: Path, processed_hashes: Set[str], manager: IngestionManager):
    """Handles the full lifecycle of a single video segment."""
    video_id = filepath.name
    camera_id = "cam_auto_01"
    
    # Parse camera ID from filename if formatted as <cam_id>_<timestamp>.mp4
    if "_" in video_id:
        prefix = video_id.split("_")[0]
        if prefix.startswith("cam") or prefix.startswith("entrance") or prefix.startswith("exit"):
            camera_id = prefix

    logger.info(f"Processing segment: {video_id} (Camera: {camera_id})")

    # 1. Compute SHA-256 and Telemetry
    file_hash = compute_file_hash(filepath)
    telemetry = get_video_telemetry(filepath)
    
    # Check idempotency
    if file_hash in processed_hashes:
        logger.warning(f"Duplicate SHA-256 detected for {video_id}. Finalizing cleanup.")
        if AUTO_DELETE_AFTER_VERIFICATION:
            filepath.unlink()
        else:
            shutil.move(str(filepath), str(COMPLETED_DIR / filepath.name))
        return

    # 2. State: PROCESSING
    await update_segment_state(video_id, status="PROCESSING", camera_id=camera_id, sha256=file_hash, telemetry=telemetry)

    try:
        # 3. Existing VISTA CV Pipeline Execution
        success = await manager.process_and_persist(
            video_path=str(filepath),
            video_id=video_id,
            camera_id=camera_id
        )

        if not success:
            raise Exception("CV Pipeline failed to process video segment.")

        await update_segment_state(video_id, status="CV_COMPLETE", camera_id=camera_id, sha256=file_hash)
        await update_segment_state(video_id, status="PERSISTING", camera_id=camera_id, sha256=file_hash)
        await update_segment_state(video_id, status="VERIFYING", camera_id=camera_id, sha256=file_hash)

        # 4. Conditional Cross-Store Referential Audit
        audit = await run_audit(video_id)
        if not audit.passed:
            raise Exception("Cross-Store Referential Audit failed.")

        # 5. State: VERIFIED -> CLEANUP_PENDING
        await update_segment_state(video_id, status="VERIFIED", camera_id=camera_id, sha256=file_hash)
        await update_segment_state(video_id, status="CLEANUP_PENDING", camera_id=camera_id, sha256=file_hash)
        processed_hashes.add(file_hash)

        # 6. Crash-Safe Deletion of Raw Source Video
        if AUTO_DELETE_AFTER_VERIFICATION:
            logger.info(f"✓ Post-Verification Safe Deletion: Removing raw video {filepath.name}")
            if filepath.exists():
                filepath.unlink()
            await update_segment_state(video_id, status="COMPLETED", cleanup_status="DELETED", camera_id=camera_id, sha256=file_hash)
        else:
            logger.info(f"Moving verified video to completed directory: {filepath.name}")
            shutil.move(str(filepath), str(COMPLETED_DIR / filepath.name))
            await update_segment_state(video_id, status="COMPLETED", cleanup_status="RETAINED", camera_id=camera_id, sha256=file_hash)

        logger.info(f"✓ Ingestion lifecycle successfully COMPLETED for {video_id}")

    except Exception as e:
        logger.error(f"✗ Ingestion failed for {video_id}: {e}")
        await update_segment_state(video_id, status="FAILED", camera_id=camera_id, sha256=file_hash, error_message=str(e))
        failed_path = FAILED_DIR / filepath.name
        if filepath.exists():
            shutil.move(str(filepath), str(failed_path))
            logger.info(f"Quarantined failed segment to: {failed_path}")


async def watch_directory():
    """Main 24/7 continuous ingestion watch loop."""
    logger.info("Initializing VISTA 24/7 Auto Ingestion Daemon...")
    ensure_directories()
    
    # Startup reconciliation
    await reconcile_interrupted_segments()
    
    event_bus = get_event_bus()
    vector_tool = get_vector_tool(event_bus)
    person_repo = get_person_repository(vector_tool)
    obs_repo = MongoObservationRepository(mongo_client)
    manager = IngestionManager(person_repo, obs_repo)

    processed_hashes = set()
    logger.info(f"Daemon actively watching folder: {WATCH_DIR.resolve()}")

    while True:
        try:
            # Find eligible segments in watch/
            mp4_files = sorted(list(WATCH_DIR.glob("*.mp4")), key=lambda p: p.stat().st_mtime)
            
            for filepath in mp4_files:
                # Check that file is not currently being written (size stability check)
                try:
                    s1 = filepath.stat().st_size
                    await asyncio.sleep(0.5)
                    s2 = filepath.stat().st_size
                    if s1 != s2 or s2 < 1024:
                        logger.info(f"File {filepath.name} is still being written. Skipping this cycle.")
                        continue
                except Exception:
                    continue

                # Move atomically to processing/
                processing_path = PROCESSING_DIR / filepath.name
                try:
                    shutil.move(str(filepath), str(processing_path))
                except Exception as e:
                    logger.error(f"Failed to move {filepath.name} to processing: {e}")
                    continue

                await process_video_file(processing_path, processed_hashes, manager)

        except Exception as loop_err:
            logger.error(f"Error in daemon watch loop: {loop_err}")

        await asyncio.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    try:
        asyncio.run(watch_directory())
    except KeyboardInterrupt:
        logger.info("Auto Ingestion Daemon stopped.")
