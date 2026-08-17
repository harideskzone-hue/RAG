import numpy as np
from typing import List, Dict, Any


class EventEvidenceBuilder:
    """
    Extracts deterministic physical measurements and spatio-temporal trajectories
    from raw track observations for LLM semantic interpretation.
    """

    def build_track_summaries(self, observations_by_track: Dict[str, List[Any]], video_id: str, camera_id: str) -> List[Dict[str, Any]]:
        summaries = []

        for track_id, obs_list in observations_by_track.items():
            if not obs_list:
                continue

            # Sort by frame index / timestamp
            def get_obs_time(o):
                if hasattr(o, "provenance"):
                    return getattr(o.provenance, "video_timestamp_sec", 0.0)
                if isinstance(o, dict):
                    return o.get("timestamp_sec", o.get("timestamp", o.get("video_timestamp_sec", 0.0)))
                return 0.0

            sorted_obs = sorted(obs_list, key=get_obs_time)
            
            timestamps = []
            centers = []
            bboxes = []

            for obs in sorted_obs:
                if hasattr(obs, "provenance"):
                    t = float(obs.provenance.video_timestamp_sec)
                    bbox = obs.attributes.bounding_box
                elif isinstance(obs, dict):
                    t = float(obs.get("timestamp_sec", obs.get("timestamp", obs.get("video_timestamp_sec", 0.0))))
                    bbox = obs.get("bbox", [0, 0, 0, 0])
                else:
                    t = 0.0
                    bbox = [0, 0, 0, 0]

                timestamps.append(t)
                bboxes.append(bbox)
                
                # Bbox center (x_center, y_center)
                if len(bbox) == 4:
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    centers.append((cx, cy))

            if not timestamps or not centers:
                continue

            start_t = min(timestamps)
            end_t = max(timestamps)
            duration = end_t - start_t
            frame_count = len(timestamps)

            # Spatial movement metrics
            centers_arr = np.array(centers)
            net_displacement = float(np.linalg.norm(centers_arr[-1] - centers_arr[0])) if len(centers_arr) > 1 else 0.0
            
            # Total path length traveled
            step_distances = np.linalg.norm(np.diff(centers_arr, axis=0), axis=1) if len(centers_arr) > 1 else np.array([0.0])
            total_path_length = float(np.sum(step_distances))
            
            # Dispersion radius around mean centroid
            centroid = np.mean(centers_arr, axis=0)
            distances_from_centroid = np.linalg.norm(centers_arr - centroid, axis=1)
            dispersion_radius = float(np.max(distances_from_centroid)) if len(distances_from_centroid) > 0 else 0.0
            
            # Velocity metrics (pixels per second)
            avg_speed = (total_path_length / duration) if duration > 0 else 0.0
            max_speed = float(np.max(step_distances) / 0.1) if len(step_distances) > 0 else 0.0

            # Bounding box height variation
            heights = [abs(b[3] - b[1]) for b in bboxes if len(b) == 4]
            avg_height = float(np.mean(heights)) if heights else 0.0
            min_height = float(np.min(heights)) if heights else 0.0

            summaries.append({
                "track_id": track_id,
                "video_id": video_id,
                "camera_id": camera_id,
                "start_time": round(start_t, 2),
                "end_time": round(end_t, 2),
                "duration_sec": round(duration, 2),
                "frame_count": frame_count,
                "net_displacement_px": round(net_displacement, 1),
                "total_path_length_px": round(total_path_length, 1),
                "dispersion_radius_px": round(dispersion_radius, 1),
                "avg_speed_px_per_sec": round(avg_speed, 1),
                "max_speed_px_per_sec": round(max_speed, 1),
                "avg_height_px": round(avg_height, 1),
                "min_height_px": round(min_height, 1),
                "initial_bbox": bboxes[0] if bboxes else [],
                "final_bbox": bboxes[-1] if bboxes else []
            })

        return summaries
