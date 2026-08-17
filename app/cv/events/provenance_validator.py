from typing import Dict, Any, List, Optional
import uuid
from app.schemas.event_contract import DetectedEvent, VerifiedEventContract, IncidentEventType


class EventProvenanceValidator:
    """
    Deterministic validator: LLM proposes semantics; deterministic code owns provenance.
    Ensures no timestamps, track IDs, camera IDs, or video IDs are hallucinated.
    """

    def __init__(self, min_confidence: float = 0.70):
        self.min_confidence = min_confidence

    def validate_and_build_contract(
        self,
        detected_event: DetectedEvent,
        physical_summary: Dict[str, Any],
        canonical_person_ids: List[str],
        clip_path: str = "",
        clip_url: str = "",
        thumbnail_path: str = "",
        thumbnail_url: str = "",
        clip_sha256: str = ""
    ) -> Optional[VerifiedEventContract]:
        """
        Validates the proposed event against ground-truth physical track measurements.
        Returns a VerifiedEventContract if valid, or None if rejected/abstained.
        """
        # 1. Reject ABSTAIN or low confidence
        if detected_event.event_type == IncidentEventType.ABSTAIN:
            return None
        if detected_event.confidence < self.min_confidence:
            return None

        # 2. Pin provenance strictly from CV physical measurements
        track_id = physical_summary["track_id"]
        camera_id = physical_summary["camera_id"]
        video_id = physical_summary["video_id"]
        start_time = float(physical_summary["start_time"])
        end_time = float(physical_summary["end_time"])
        duration = max(0.1, end_time - start_time)

        event_id = f"EVT_{uuid.uuid4().hex[:8].upper()}"

        return VerifiedEventContract(
            event_id=event_id,
            event_type=detected_event.event_type.value,
            camera_id=camera_id,
            video_id=video_id,
            start_time=start_time,
            end_time=end_time,
            duration_sec=round(duration, 2),
            track_ids=[track_id],
            canonical_person_ids=canonical_person_ids,
            confidence=detected_event.confidence,
            severity=detected_event.severity,
            clip_path=clip_path,
            clip_url=clip_url,
            thumbnail_path=thumbnail_path,
            thumbnail_url=thumbnail_url,
            reason=detected_event.reason,
            clip_sha256=clip_sha256,
            provenance={
                "source_video_id": video_id,
                "camera_id": camera_id,
                "track_id": track_id,
                "frame_count": physical_summary.get("frame_count", 0),
                "total_path_length_px": physical_summary.get("total_path_length_px", 0.0),
                "dispersion_radius_px": physical_summary.get("dispersion_radius_px", 0.0)
            }
        )
