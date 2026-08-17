import logging
from typing import List

from .base import BaseDetector, Detection
from app.cv.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

class YOLOPersonDetector(BaseDetector):
    """Detects and tracks the COCO 'person' class using Ultralytics YOLO & ByteTrack."""
    
    PERSON_CLASS_ID = 0

    def __init__(self, registry: ModelRegistry, conf_threshold: float = 0.5):
        from ultralytics import YOLO
        
        self.registry = registry
        self.conf_threshold = conf_threshold
        
        # Pull configurations from registry
        model_path = str(registry.get_detector_path())
        self.tracker_cfg = registry.get_tracker_config()
        self.device = registry.get_device()
        
        logger.info(f"Loading YOLO model from {model_path} onto {self.device} with tracker {self.tracker_cfg}")
        self.model = YOLO(model_path)

    def track_frame(self, frame_bgr) -> List[Detection]:
        """Run detection+tracking on a single frame, persisting tracking state."""
        results = self.model.track(
            source=frame_bgr,
            persist=True,
            classes=[self.PERSON_CLASS_ID],
            conf=self.conf_threshold,
            tracker=self.tracker_cfg,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            return detections

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for box, tid, conf in zip(boxes_xyxy, ids, confs):
            x1, y1, x2, y2 = (int(v) for v in box)
            detections.append(
                Detection(track_id=int(tid), bbox=(x1, y1, x2, y2), confidence=float(conf))
            )
        return detections

    def reset(self):
        """
        Clears the tracker state. 
        Ultralytics tracking persist=True manages state internally per model instance. 
        Re-instantiating or resetting the tracker objects may be required.
        For ultralytics, we can clear the internal predictor trackers.
        """
        if hasattr(self.model, "predictor") and self.model.predictor:
            for tracker in getattr(self.model.predictor, "trackers", []):
                tracker.reset()
        logger.info("YOLO tracker state reset for new segment.")
