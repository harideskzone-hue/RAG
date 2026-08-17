import os
import json
import logging
import cv2
import uuid
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class TrackWriter:
    """Handles cropping and persisting track metadata and crops to disk."""

    def __init__(self, output_dir: str = "dataset/tracks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_track_dir(self, video_id: str, track_id: str) -> Path:
        safe_vid = Path(video_id).name if video_id else "default"
        track_dir = self.output_dir / safe_vid / track_id
        track_dir.mkdir(parents=True, exist_ok=True)
        (track_dir / "crops").mkdir(parents=True, exist_ok=True)
        return track_dir

    def _clamp_bbox(self, bbox, width, height):
        x1, y1, x2, y2 = bbox
        return (
            max(0, int(x1)),
            max(0, int(y1)),
            min(width - 1, int(x2)),
            min(height - 1, int(y2))
        )

    def write_observation(
        self, 
        video_id: str, 
        camera_id: str, 
        track_id: str, 
        frame_bgr, 
        frame_index: int, 
        timestamp_sec: float, 
        bbox: tuple, 
        confidence: float
    ) -> str:
        """
        Saves a crop and updates the track.json metadata.
        Returns the generated evidence_id.
        """
        track_dir = self._get_track_dir(video_id, track_id)
        evidence_id = f"obs_{uuid.uuid4().hex[:8]}"
        
        # 1. Save Crop
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = self._clamp_bbox(bbox, w, h)
        
        # Ensure valid crop area (minimum 24x48 to avoid writing empty/degenerate artifacts)
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w >= 24 and crop_h >= 48:
            crop = frame_bgr[y1:y2, x1:x2]
            crop_rel_path = f"crops/{evidence_id}.jpg"
            crop_abs_path = track_dir / crop_rel_path
            try:
                cv2.imwrite(str(crop_abs_path), crop)
            except Exception as e:
                logger.warning(f"Failed to write crop {crop_abs_path}: {e}")
                crop_rel_path = ""
        else:
            # Degenerate or clipped bbox
            crop_rel_path = ""

        # 2. Update track.json
        track_json_path = track_dir / "track.json"
        track_data = None
        
        if track_json_path.exists():
            try:
                with open(track_json_path, "r") as f:
                    track_data = json.load(f)
            except Exception:
                pass
                
        if not track_data:
            track_data = {
                "video_id": Path(video_id).name if video_id else "default",
                "camera_id": camera_id,
                "track_id": track_id,
                "observations": []
            }
            
        track_data["observations"].append({
            "evidence_id": evidence_id,
            "frame_index": frame_index,
            "timestamp_sec": timestamp_sec,
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "crop": crop_rel_path
        })
        
        # Write track.json safely
        try:
            tmp_path = track_dir / f".track_{evidence_id}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(track_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_path), str(track_json_path))
        except Exception as write_err:
            try:
                with open(track_json_path, "w") as f:
                    json.dump(track_data, f, indent=2)
            except Exception as direct_err:
                logger.error(f"Failed to write track.json for {track_id}: {direct_err}")
        
        return evidence_id
