from enum import Enum


class SamplingPolicy(Enum):
    FAST = 1 # 1 FPS
    BALANCED = 2 # 2 FPS
    HIGH_PRECISION = 5 # 5 FPS

class FrameSampler:
    """
    Simulates extracting frames from a video clip at a specific sampling rate.
    """
    def sample_frames(self, video_uri: str, policy: SamplingPolicy, duration_seconds: int = 10) -> list[str]:
        # In production, this would use FFmpeg or OpenCV
        fps = policy.value
        total_frames = fps * duration_seconds
        
        # Mocking frame extraction
        frames = [f"{video_uri}_frame_{i}.jpg" for i in range(total_frames)]
        return frames
