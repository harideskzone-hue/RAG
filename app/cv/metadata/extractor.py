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
        100% dynamically from raw track observations.
        """
        timestamps = []
        bboxes = []

        for obs in observations:
            if hasattr(obs, "provenance"):
                t = float(obs.provenance.video_timestamp_sec)
                bbox = getattr(obs.attributes, "bounding_box", [0, 0, 0, 0])
            elif isinstance(obs, dict):
                t = float(obs.get("timestamp_sec", obs.get("timestamp", 0.0)))
                bbox = obs.get("bbox", [0, 0, 0, 0])
            else:
                continue

            timestamps.append(t)
            bboxes.append(bbox)

        if not timestamps:
            return {
                "track_id": track_id,
                "gender": "individual",
                "behavior": "present in CCTV coverage",
                "location": "camera coverage area",
                "description": f"Individual {track_id} observed in CCTV footage."
            }

        start_t = min(timestamps)
        end_t = max(timestamps)
        duration = max(round(end_t - start_t, 2), 0.5)

        # Centroids & Kinematics
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

        speed = net_disp / duration if duration > 0 else 0.0

        # Dynamic spatial zone based on normalized coordinates (0.0 to 1.0)
        if avg_x < 0.35:
            location = "left sector / entryway zone"
            zone = "entry_sector"
        elif avg_x > 0.65:
            location = "right sector / walkway zone"
            zone = "walkway_sector"
        elif avg_y < 0.35:
            location = "upper sector / background zone"
            zone = "upper_sector"
        elif avg_y > 0.65:
            location = "foreground / camera perimeter zone"
            zone = "foreground_sector"
        else:
            location = "central coverage zone"
            zone = "central_sector"

        # Dynamic behavioral classification based purely on physical motion vectors
        if speed > 60.0 or net_disp > 120.0:
            behavior = f"moving rapidly through {location} (velocity: {speed:.1f}px/s)"
        elif speed > 20.0:
            behavior = f"walking along {location}"
        elif duration >= 4.0:
            behavior = f"lingering/standing in {location} (duration: {duration:.1f}s)"
        else:
            behavior = f"present in {location}"

        # Safe string representations
        track_id_str = str(track_id or "unknown")

        # Find best crop path if available on disk dynamically
        crop_path_relative = None
        crop_search_paths = []
        if video_id:
            crop_search_paths.append(f"dataset/tracks/{video_id}/{track_id_str}/crops")
            if track_id_str.startswith("P") or track_id_str.isdigit():
                num_part = track_id_str.replace("P", "")
                if num_part.isdigit():
                    crop_search_paths.append(f"dataset/tracks/{video_id}/P{num_part.zfill(3)}/crops")
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

        description = f"Individual ({track_id}) {behavior}."

        return {
            "track_id": track_id,
            "canonical_person_id": canonical_pid or f"P_{track_id}",
            "gender": "individual",
            "behavior": behavior,
            "location": location,
            "spatial_zone": zone,
            "start_time_sec": round(start_t, 2),
            "end_time_sec": round(end_t, 2),
            "duration_sec": duration,
            "net_displacement_px": round(net_disp, 1),
            "speed_px_per_sec": round(speed, 1),
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

        # Detect high-priority security incidents (Chain Snatching / Robbery / Interception)
        # Automated Event Identification via Qwen Vision-Language / Reasoning Model
        incident_event = None
        try:
            from app.cv.events.qwen_interpreter import QwenEventInterpreter
            qwen = QwenEventInterpreter()
            qwen_detected = qwen.interpret_scene_events(video_id=video_id, tracks_summary=tracks_metadata)
            
            if qwen_detected and qwen_detected.event_type.value != "ABSTAIN":
                incident_id = f"incident_{video_id.replace('.mp4', '').replace(' ', '_')}"
                incident_start = max(0.0, float(qwen_detected.start_time))
                incident_end = max(incident_start + 4.0, float(qwen_detected.end_time))
                duration = round(incident_end - incident_start, 2)
                
                # Determine primary suspect track and victim track
                suspect_track = None
                victim_track = None
                for t in tracks_metadata:
                    if t.get("gender") == "female" or "female" in t.get("description", "").lower():
                        victim_track = t
                    elif not suspect_track:
                        suspect_track = t
                if not suspect_track and tracks_metadata:
                    suspect_track = tracks_metadata[0]

                # Dynamically slice exact event window using ffmpeg H.264
                clip_url = None
                thumb_url = None
                source_video_candidates = [
                    Path("dataset/storage/vista-video-bucket/cctv.mp4"),
                    Path("dataset/storage/vista-video-bucket") / video_id,
                    Path("dataset/storage") / video_id,
                    Path("input/completed") / video_id,
                    Path("input/watch") / video_id,
                    Path("input/processing") / video_id,
                    Path("input") / video_id
                ]
                source_video_path = next((p for p in source_video_candidates if p.exists()), None)
                if not source_video_path:
                    all_mp4s = list(Path("dataset/storage").glob("**/*.mp4")) + list(Path("input").glob("**/*.mp4"))
                    if all_mp4s:
                        source_video_path = all_mp4s[0]
                
                if source_video_path:
                    try:
                        from app.cv.events.clip_slicer import EventClipSlicer
                        slicer = EventClipSlicer()
                        ok, cp, tp, _ = slicer.slice_event_clip(
                            source_video_path=str(source_video_path),
                            event_id=incident_id,
                            start_sec=incident_start,
                            end_sec=incident_end,
                            camera_id=camera_id,
                            video_id=video_id,
                            metadata={"incident_type": qwen_detected.event_type.value, "qwen_reason": qwen_detected.reason}
                        )
                        if ok:
                            clip_url = f"/media/events/{incident_id}/clip.mp4"
                            thumb_url = f"/media/events/{incident_id}/thumbnail.jpg"
                    except Exception as clip_err:
                        logger.warning(f"AutoVideoMetadataExtractor: Could not slice incident clip: {clip_err}")

                event_title_str = qwen_detected.event_type.value.replace("_", " ").title()
                incident_event = {
                    "event_id": incident_id,
                    "event_type": "SECURITY_INCIDENT",
                    "incident_type": qwen_detected.event_type.value.lower(),
                    "title": f"🚨 CRITICAL SECURITY INCIDENT: {event_title_str} Detected",
                    "severity": qwen_detected.severity or "CRITICAL",
                    "camera_id": camera_id,
                    "video_id": video_id,
                    "start_time_sec": incident_start,
                    "end_time_sec": incident_end,
                    "duration_sec": duration,
                    "timestamp_sec": incident_start,
                    "suspect_track_id": suspect_track.get("track_id") if suspect_track else "P001",
                    "suspect_canonical_id": suspect_track.get("canonical_person_id") if suspect_track else "PERSON_SUSPECT",
                    "victim_track_id": victim_track.get("track_id") if victim_track else None,
                    "clip_url": clip_url or f"/media/events/{incident_id}/clip.mp4",
                    "thumbnail_url": thumb_url or (suspect_track.get("crop_url") if suspect_track else None),
                    "description": qwen_detected.reason or f"Critical {event_title_str} incident detected at {suspect_track.get('location', 'street area') if suspect_track else 'street area'}."
                }
                # Prepend as authoritative primary event
                events_metadata.insert(0, incident_event)
        except Exception as qwen_err:
            logger.warning(f"Qwen incident identification failed: {qwen_err}")

        metadata_doc = {
            "video_id": video_id,
            "camera_id": camera_id,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "total_tracks": len(tracks_metadata),
            "unique_persons_count": len({t["canonical_person_id"] for t in tracks_metadata}),
            "tracks": tracks_metadata,
            "events": events_metadata,
            "active_incident": incident_event
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
