import logging
from typing import List
from app.cv.detection.base import Detection
from app.cv.crops.writer import TrackWriter

logger = logging.getLogger(__name__)

class TrackAggregator:
    """
    Receives per-frame detections, maps tracker_id to the PXXX video-local track format,
    and forwards them to the TrackWriter to be cropped and saved.
    """
    
    def __init__(self, video_id: str, camera_id: str, output_dir: str = "dataset/tracks"):
        self.video_id = video_id
        self.camera_id = camera_id
        self.writer = TrackWriter(output_dir=output_dir)
        self.generated_evidence = []
        
    def _format_track_id(self, tracker_id: int) -> str:
        """Converts raw integer tracker ID to P001 format."""
        return f"P{tracker_id:03d}"

    def process_frame(
        self, 
        frame_bgr, 
        frame_index: int, 
        timestamp_sec: float, 
        detections: List[Detection]
    ):
        """Processes detections for a single frame."""
        for det in detections:
            track_id = self._format_track_id(det.track_id)
            evidence_id = self.writer.write_observation(
                video_id=self.video_id,
                camera_id=self.camera_id,
                track_id=track_id,
                frame_bgr=frame_bgr,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                bbox=det.bbox,
                confidence=det.confidence
            )
            
            self.generated_evidence.append({
                "evidence_id": evidence_id,
                "track_id": track_id,
                "frame_index": frame_index,
                "timestamp_sec": timestamp_sec,
                "bbox": det.bbox,
                "confidence": det.confidence,
                "video_id": self.video_id,
                "camera_id": self.camera_id
            })
            
    def get_all_evidence(self) -> List[dict]:
        """Returns all evidence observations collected so far."""
        return self.generated_evidence
