import os
import cv2
import json
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from app.schemas.event_contract import VerifiedEventContract
from app.cv.crops.enhancer import CropEnhancer

logger = logging.getLogger("EventClipSlicer")


class EventClipSlicer:
    """
    Slices exact video evidence windows, generates thumbnails, computes SHA-256 hashes,
    and writes auditable event manifests under dataset/events/{event_id}/.
    """

    def __init__(self, output_root: str = "dataset/events", padding_before_sec: float = 2.0, padding_after_sec: float = 3.0):
        self.output_root = Path(output_root)
        self.padding_before = padding_before_sec
        self.padding_after = padding_after_sec
        self.enhancer = CropEnhancer()

    def slice_event_clip(
        self,
        source_video_path: str,
        event_id: str,
        start_sec: float,
        end_sec: float,
        camera_id: str,
        video_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, str, str]:
        """
        Slices the video segment and returns (success, clip_path, thumbnail_path, sha256_hash).
        """
        source_path = Path(source_video_path)
        if not source_path.exists():
            logger.error(f"Source video {source_video_path} does not exist for clip slicing.")
            return False, "", "", ""

        event_dir = self.output_root / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        clip_path = event_dir / "clip.mp4"
        thumbnail_path = event_dir / "thumbnail.jpg"
        manifest_path = event_dir / "event.json"

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            logger.error(f"Could not open source video {source_video_path}")
            return False, "", "", ""

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        clip_start_sec = max(0.0, start_sec - self.padding_before)
        clip_end_sec = min(total_duration, end_sec + self.padding_after) if total_duration > 0 else (end_sec + self.padding_after)
        duration_sec = max(1.0, clip_end_sec - clip_start_sec)

        start_frame = int(clip_start_sec * fps)
        end_frame = max(start_frame + 1, int(clip_end_sec * fps))

        # Try slicing with ffmpeg for pristine H.264 browser compatibility
        import subprocess
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip_start_sec),
            "-i", str(source_path),
            "-t", str(duration_sec),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-movflags", "+faststart",
            "-an",
            str(clip_path)
        ]
        
        ffmpeg_success = False
        try:
            res = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            if res.returncode == 0 and clip_path.exists() and clip_path.stat().st_size > 1000:
                ffmpeg_success = True
        except Exception as e:
            logger.warning(f"ffmpeg slicing failed, falling back to OpenCV: {e}")

        best_thumbnail_frame = None

        if not ffmpeg_success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
            if not out.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))

            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                out.write(frame)
                if best_thumbnail_frame is None and current_frame >= int((start_sec + end_sec) / 2.0 * fps):
                    best_thumbnail_frame = frame.copy()
                current_frame += 1

            out.release()

        # Extract best thumbnail frame
        if best_thumbnail_frame is None:
            mid_frame = int((clip_start_sec + clip_end_sec) / 2.0 * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()
            if ret and frame is not None:
                best_thumbnail_frame = frame

        cap.release()

        # Generate thumbnail
        if best_thumbnail_frame is not None:
            enhanced_thumb = self.enhancer.enhance(best_thumbnail_frame)
            cv2.imwrite(str(thumbnail_path), enhanced_thumb)
        else:
            # Fallback thumbnail
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.imwrite(str(thumbnail_path), blank)

        # Verify clip exists and calculate SHA-256
        if not clip_path.exists() or clip_path.stat().st_size == 0:
            logger.error(f"Failed to generate valid clip at {clip_path}")
            return False, "", "", ""

        hasher = hashlib.sha256()
        with open(clip_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        clip_sha256 = hasher.hexdigest()

        # Write auditable event.json manifest
        manifest = {
            "event_id": event_id,
            "source_video_id": video_id,
            "source_video_path": str(source_path),
            "camera_id": camera_id,
            "source_start_frame": start_frame,
            "source_end_frame": end_frame,
            "source_start_time": clip_start_sec,
            "source_end_time": clip_end_sec,
            "target_event_start": start_sec,
            "target_event_end": end_sec,
            "clip_sha256": clip_sha256,
            "clip_size_bytes": clip_path.stat().st_size,
            "metadata": metadata or {}
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return True, str(clip_path), str(thumbnail_path), clip_sha256
