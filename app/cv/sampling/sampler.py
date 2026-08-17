import os
import logging
from typing import Iterator, Tuple

logger = logging.getLogger(__name__)

class FrameSampler:
    """Samples frames from a stream to a target FPS."""
    
    def __init__(self, target_fps: float = None):
        if target_fps is None:
            # Check environment config
            env_fps = os.environ.get("CV_SAMPLE_FPS")
            self.target_fps = float(env_fps) if env_fps else None
        else:
            self.target_fps = target_fps
            
        logger.info(f"FrameSampler configured with target FPS: {self.target_fps or 'Native'}")

    def sample(self, native_fps: float, frame_iterator: Iterator[Tuple[int, any, float]]) -> Iterator[Tuple[int, any, float]]:
        """
        Takes an iterator of frames and yields only those matching the target FPS.
        Yields (frame_index, frame_bgr, timestamp_sec).
        """
        if not self.target_fps or self.target_fps >= native_fps:
            yield from frame_iterator
            return

        step = max(1, round(native_fps / self.target_fps))
        
        for frame_index, frame, timestamp_sec in frame_iterator:
            if frame_index % step == 0:
                yield frame_index, frame, timestamp_sec
