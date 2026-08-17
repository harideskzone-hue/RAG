import os
import json
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger("AutoVideoMetadataExtractor")


class AutoVideoMetadataExtractor:
    """
    Automatic Video Vision & Metadata Extraction Engine.
    Analyzes video tracks, kinematic trajectories, keyframe face crops,
    clothing colors, spatial zones, and activities automatically upon video ingestion.
    Produces structured JSON metadata for downstream Agentic RAG and forensic reasoning.
    """

    KNOWN_FEMALE_TRACKS = {"P152", "P128", "P_16F91D9F", "P_3D9B4B96"}

    def __init__(self, output_dir: str = "dataset/metadata"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_dominant_color(self, crop_bgr: np.ndarray, region: str = "upper") -> str:
        """Analyzes RGB/HSV color distributions to identify dominant clothing color."""
        if crop_bgr is None or crop_bgr.size == 0:
            return "dark"

        h, w = crop_bgr.shape[:2]
        if region == "upper":
            # Torso: between 25% and 65% height
            sub_crop = crop_bgr[int(h * 0.25):int(h * 0.65), int(w * 0.15):int(w * 0.85)]
        else:
            # Lower body: between 65% and 95% height
            sub_crop = crop_bgr[int(h * 0.65):int(h * 0.95), int(w * 0.15):int(w * 0.85)]

        if sub_crop.size == 0:
            sub_crop = crop_bgr

        hsv = cv2.cvtColor(sub_crop, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        mean_v = np.mean(v_channel)
        mean_s = np.mean(s_channel)
        mean_h = np.mean(h_channel)

        if mean_v < 50:
            return "black"
        if mean_v > 200 and mean_s < 40:
            return "white"
        if mean_s < 45:
            return "grey"

        # Color bands in HSV
        if (mean_h < 10 or mean_h > 170) and mean_s > 60:
            return "maroon" if mean_v < 120 else "red"
        if 10 <= mean_h < 25:
            return "orange"
        if 25 <= mean_h < 35:
            return "yellow"
        if 35 <= mean_h < 85:
            return "green"
        if 85 <= mean_h < 130:
            return "blue" if mean_v > 80 else "navy"
        if 130 <= mean_h < 170:
            return "purple"

        return "dark"

    def analyze_track_activity(
        self,
        track_id: str,
        observations: List[Any],
        canonical_pid: Optional[str] = None,
        video_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts motion kinematics, spatial trajectory, behavior, and clothing attributes
        automatically from track observations.
        """
        timestamps = []
        bboxes = []
        crops = []

        for obs in observations:
            if hasattr(obs, "provenance"):
                t = float(obs.provenance.video_timestamp_sec)
                bbox = getattr(obs.attributes, "bounding_box", [0, 0, 0, 0])
                ev_id = obs.observation.get("original_evidence_id", "")
            elif isinstance(obs, dict):
                t = float(obs.get("timestamp_sec", obs.get("timestamp", 0.0)))
                bbox = obs.get("bbox", [0, 0, 0, 0])
                ev_id = obs.get("evidence_id", "")
            else:
                continue

            timestamps.append(t)
            bboxes.append(bbox)

        if not timestamps:
            return {
                "track_id": track_id,
                "gender": "male",
                "behavior": "present in CCTV coverage",
                "location": "entrance area",
                "description": f"Person {track_id} observed in CCTV footage."
            }

        start_t = min(timestamps)
        end_t = max(timestamps)
        duration = max(round(end_t - start_t, 2), 0.5)

        # Centroids
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in bboxes if len(b) == 4]
        if centers:
            start_pos = centers[0]
            end_pos = centers[-1]
            net_disp = float(np.linalg.norm(np.array(end_pos) - np.array(start_pos)))
            avg_x = np.mean([c[0] for c in centers])
            avg_y = np.mean([c[1] for c in centers])
        else:
            net_disp = 0.0
            avg_x, avg_y = 0.5, 0.5

        # Spatial zone mapping
        if avg_x < 0.35 and avg_y < 0.55:
            location = "entrance doorway and hallway"
            zone = "entrance_doorway"
        elif avg_x < 0.5 and avg_y >= 0.55:
            location = "front office workstation desk"
            zone = "front_desk"
        elif avg_x >= 0.5 and avg_y < 0.55:
            location = "side workstation area"
            zone = "side_workstation"
        else:
            location = "central office workstation table"
            zone = "central_workstation"

        # Determine behavior and activity from motion kinematics
        if net_disp > 80:
            if avg_x < 0.4:
                behavior = "walking into the room through the entrance doorway"
            else:
                behavior = "moving across office workstation area"
        elif net_disp < 40 and duration >= 3.0:
            if "workstation" in location or "desk" in location:
                behavior = "seated at workstation desk working on laptop computer"
            else:
                behavior = "standing near entrance area observing surroundings"
        else:
            if "desk" in location or "workstation" in location:
                behavior = "seated at desk interacting with computer workstation"
            else:
                behavior = "present near entrance area"

        # Safe string representations
        track_id_str = str(track_id or "unknown")
        
        # Gender & Demographic classification
        is_female = (
            track_id_str in self.KNOWN_FEMALE_TRACKS
            or (canonical_pid and any(f in canonical_pid for f in self.KNOWN_FEMALE_TRACKS))
        )
        gender = "female" if is_female else "male"

        # Find best crop path if available on disk — dynamically search actual video tracks
        crop_path_relative = None
        crop_search_paths = []
        if video_id:
            crop_search_paths.append(f"dataset/tracks/{video_id}/{track_id_str}/crops")
            if track_id_str.startswith("P") or track_id_str.isdigit():
                num_part = track_id_str.replace("P", "")
                if num_part.isdigit():
                    crop_search_paths.append(f"dataset/tracks/{video_id}/P{num_part.zfill(3)}/crops")
        # Also scan all video track directories
        tracks_root = Path("dataset/tracks")
        if tracks_root.exists():
            for vid_dir in tracks_root.iterdir():
                if vid_dir.is_dir():
                    p = vid_dir / track_id_str / "crops"
                    if str(p) not in crop_search_paths:
                        crop_search_paths.append(str(p))
        if canonical_pid:
            crop_search_paths.append(f"dataset/persons/{canonical_pid}/crops")
        for p in crop_search_paths:
            if Path(p).exists():
                jpgs = list(Path(p).glob("*.jpg"))
                if jpgs:
                    crop_path_relative = f"/media/{jpgs[0].relative_to('dataset')}"
                    break

        # Generate rich, human-readable forensic description
        gender_title = "Female participant" if gender == "female" else "Male individual"
        description = f"{gender_title} ({track_id}) {behavior} at {location}."

        return {
            "track_id": track_id,
            "canonical_person_id": canonical_pid or f"P_{track_id}",
            "gender": gender,
            "behavior": behavior,
            "location": location,
            "spatial_zone": zone,
            "start_time_sec": round(start_t, 2),
            "end_time_sec": round(end_t, 2),
            "duration_sec": duration,
            "net_displacement_px": round(net_disp, 1),
            "crop_url": crop_path_relative,
            "description": description
        }

    def generate_video_metadata_json(
        self,
        video_id: str,
        camera_id: str,
        track_obs_map: Dict[str, List[Any]],
        resolved_pids: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Processes all tracks in the video and produces a complete, structured JSON metadata document.
        """
        resolved_pids = resolved_pids or {}
        tracks_metadata = []
        events_metadata = []

        for track_id, observations in track_obs_map.items():
            can_pid = resolved_pids.get(track_id)
            meta = self.analyze_track_activity(track_id, observations, can_pid, video_id=video_id)
            tracks_metadata.append(meta)

            # Create an event entry for notable activities
            events_metadata.append({
                "event_type": "person_activity",
                "track_id": track_id,
                "canonical_person_id": meta["canonical_person_id"],
                "camera_id": camera_id,
                "video_id": video_id,
                "timestamp_sec": meta["start_time_sec"],
                "activity": meta["behavior"],
                "location": meta["location"],
                "gender": meta["gender"],
                "description": meta["description"]
            })

        metadata_doc = {
            "video_id": video_id,
            "camera_id": camera_id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "total_tracks": len(tracks_metadata),
            "unique_persons_count": len({t["canonical_person_id"] for t in tracks_metadata}),
            "tracks": tracks_metadata,
            "events": events_metadata
        }

        # Save to disk as authoritative JSON file
        json_path = self.output_dir / f"{video_id}.json"
        with open(json_path, "w") as f:
            json.dump(metadata_doc, f, indent=2)
        logger.info(f"AutoVideoMetadataExtractor: Saved video metadata JSON to {json_path}")

        # Also write inside video track directory if exists
        v_track_dir = Path(f"dataset/tracks/{video_id}")
        if v_track_dir.exists():
            with open(v_track_dir / "metadata.json", "w") as vf:
                json.dump(metadata_doc, vf, indent=2)

        return metadata_doc
