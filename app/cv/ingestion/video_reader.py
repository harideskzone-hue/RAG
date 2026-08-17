import cv2
import logging
from dataclasses import dataclass
from typing import Iterator, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class VideoMetadata:
    video_id: str
    filename: str
    fps: float
    duration_sec: float
    width: int
    height: int
    total_frames: int

class VideoReader:
    """Reads a video file and provides metadata and frames."""

    def __init__(self, video_path: str, video_id: str):
        self.video_path = str(video_path)
        self.filename = self.video_path.split("/")[-1]
        self.video_id = video_id
        self._cap = cv2.VideoCapture(self.video_path)
        
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
            
        self.native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.duration_sec = self.total_frames / self.native_fps if self.native_fps > 0 else 0.0

        self.metadata = VideoMetadata(
            video_id=self.video_id,
            filename=self.filename,
            fps=self.native_fps,
            duration_sec=self.duration_sec,
            width=self.width,
            height=self.height,
            total_frames=self.total_frames
        )
        logger.info(f"Video {self.video_id} (filename: {self.filename}) loaded: {self.width}x{self.height} @ {self.native_fps}fps, {self.duration_sec:.2f}s")

    def read_frames(self, start_time: Optional[datetime] = None) -> Iterator[Tuple[int, any, float]]:
        """Yields (frame_index, frame_bgr, timestamp_sec)."""
        read_idx = 0
        try:
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                    
                timestamp_sec = read_idx / self.native_fps
                yield read_idx, frame, timestamp_sec
                read_idx += 1
        finally:
            self.close()

    def close(self):
        if self._cap.isOpened():
            self._cap.release()
