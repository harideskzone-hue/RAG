#!/usr/bin/env python3
"""
VISTA 24/7 RTSP Stream Chunker

Connects to live RTSP / IP Camera video streams and slices them into seamless
10-minute MP4 video segments with decoder telemetry.

Lifecycle:
1. Writes to `input/recording/temp_<camera_id>_<timestamp>.mp4`
2. Measures expected_frames, received_frames, dropped_frames, fps, and duration
3. Flushes & closes file descriptor
4. Atomically renames to `input/watch/<camera_id>_<timestamp>.mp4` for pipeline ingestion
"""
import os
import sys
import time
import signal
import logging
import cv2
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RTSPStreamChunker")

RECORDING_DIR = Path("input/recording")
WATCH_DIR = Path("input/watch")

RECORDING_DIR.mkdir(parents=True, exist_ok=True)
WATCH_DIR.mkdir(parents=True, exist_ok=True)


class RTSPStreamChunker:
    def __init__(
        self,
        rtsp_url: str,
        camera_id: str = "cam_auto_01",
        segment_duration_sec: int = 600,  # 10 minutes
        fps: float = 30.0,
        resolution: tuple = (1920, 1080)
    ):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.segment_duration_sec = segment_duration_sec
        self.fps = fps
        self.resolution = resolution
        self.running = True

    def record_segment(self, segment_idx: int) -> Optional[Path]:
        """Records a single 10-minute segment and atomically promotes it to watch/."""
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        temp_filename = f"temp_{self.camera_id}_{ts_str}.mp4"
        final_filename = f"{self.camera_id}_{ts_str}.mp4"

        temp_path = RECORDING_DIR / temp_filename
        final_path = WATCH_DIR / final_filename

        logger.info(f"Starting 10-minute recording: {temp_path}")

        # Open video capture
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logger.error(f"Failed to connect to RTSP stream: {self.rtsp_url}")
            return None

        # Prepare VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(temp_path), fourcc, self.fps, self.resolution)

        expected_frames = int(self.segment_duration_sec * self.fps)
        received_frames = 0
        dropped_frames = 0
        start_time = time.time()

        try:
            while self.running and (time.time() - start_time) < self.segment_duration_sec:
                ret, frame = cap.read()
                if not ret:
                    dropped_frames += 1
                    time.sleep(0.01)
                    continue

                if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
                    frame = cv2.resize(frame, self.resolution)

                out.write(frame)
                received_frames += 1

        finally:
            cap.release()
            out.release()

        elapsed = time.time() - start_time
        logger.info(
            f"Segment complete: {received_frames} frames received, {dropped_frames} dropped in {elapsed:.1f}s (fps={received_frames/max(elapsed, 0.1):.1f})"
        )

        # Atomic Rename Promotion into watch/
        if temp_path.exists() and temp_path.stat().st_size > 1024:
            temp_path.rename(final_path)
            logger.info(f"✓ Atomically promoted segment to watch folder: {final_path}")
            return final_path
        else:
            logger.error(f"Recording failed or file too small: {temp_path}")
            if temp_path.exists():
                temp_path.unlink()
            return None

    def start_24_7_loop(self):
        """Continuously records 10-minute segments in a 24/7 loop."""
        logger.info(f"Starting 24/7 RTSP Chunking for {self.camera_id} from {self.rtsp_url}")
        segment_idx = 1
        while self.running:
            try:
                self.record_segment(segment_idx)
                segment_idx += 1
            except Exception as e:
                logger.error(f"Error in recording loop: {e}")
                time.sleep(5)


def handle_sigint(sig, frame):
    logger.info("Stopping RTSP Chunker cleanly...")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    rtsp_source = sys.argv[1] if len(sys.argv) > 1 else "0"  # defaults to webcam or sample stream
    cam_id = sys.argv[2] if len(sys.argv) > 2 else "cam_auto_01"
    
    chunker = RTSPStreamChunker(rtsp_url=rtsp_source, camera_id=cam_id)
    chunker.start_24_7_loop()
