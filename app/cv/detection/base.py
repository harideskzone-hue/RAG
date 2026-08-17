from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Detection:
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float

class BaseDetector(ABC):
    """Abstract base class for person detection and tracking."""

    @abstractmethod
    def track_frame(self, frame_bgr) -> List[Detection]:
        """Detect and track persons in a single frame, returning temporal track IDs."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the internal tracking state (e.g., when a new video segment starts)."""
        pass
