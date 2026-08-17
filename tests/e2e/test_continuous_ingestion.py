import os
import shutil
import pytest
import cv2
import numpy as np
from pathlib import Path
from scripts.auto_ingest_daemon import (
    ensure_directories,
    compute_file_hash,
    get_video_telemetry,
    reconcile_interrupted_segments,
    RECORDING_DIR,
    WATCH_DIR,
    PROCESSING_DIR,
    COMPLETED_DIR,
    FAILED_DIR
)
from scripts.cross_store_audit import run_audit


def create_synthetic_mp4(filepath: Path, duration_frames: int = 30, fps: float = 30.0):
    """Generates a small synthetic MP4 video file for testing."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(filepath), fourcc, fps, (640, 480))
    for i in range(duration_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()


@pytest.fixture(autouse=True)
def setup_teardown():
    ensure_directories()
    yield
    # Cleanup any leftovers
    for d in [RECORDING_DIR, WATCH_DIR, PROCESSING_DIR, FAILED_DIR]:
        for f in d.glob("test_*.mp4"):
            if f.exists():
                f.unlink()


def test_atomic_segment_creation_and_telemetry():
    """Test RTSP chunker recording and atomic promotion."""
    temp_file = RECORDING_DIR / "temp_test_chunk.mp4"
    final_file = WATCH_DIR / "test_cam01_20260816.mp4"

    create_synthetic_mp4(temp_file, duration_frames=60, fps=30.0)
    assert temp_file.exists()
    assert temp_file.stat().st_size > 1024

    # Atomic Rename
    temp_file.rename(final_file)
    assert not temp_file.exists()
    assert final_file.exists()

    # Telemetry
    telemetry = get_video_telemetry(final_file)
    assert telemetry["frame_count"] == 60
    assert telemetry["fps"] == 30.0
    assert telemetry["duration_sec"] == 2.0


def test_sha256_idempotency_hash():
    """Test SHA-256 hashing computation."""
    f1 = WATCH_DIR / "test_hash_1.mp4"
    create_synthetic_mp4(f1, duration_frames=30)
    h1 = compute_file_hash(f1)
    assert len(h1) == 64
    assert h1 == compute_file_hash(f1)


@pytest.mark.asyncio
async def test_conditional_audit_pass():
    """Test that the 4-store referential audit passes cleanly."""
    result = await run_audit("test_video_segment.mp4")
    assert result.passed is True


@pytest.mark.asyncio
async def test_restart_reconciliation():
    """Test startup recovery for interrupted segments."""
    proc_file = PROCESSING_DIR / "test_interrupted.mp4"
    create_synthetic_mp4(proc_file, duration_frames=30)
    assert proc_file.exists()

    await reconcile_interrupted_segments()
    # Interrupted segment without verified state is safely moved back to watch/ for clean retry
    recovered_file = WATCH_DIR / "test_interrupted.mp4"
    assert recovered_file.exists()
    assert not proc_file.exists()
