"""
cv_ingestor.py — CV-output -> Evidence Contract adapter for VISTA AI.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.domain.evidence import (
    EvidenceBundle,
    PersonEvidence,
    VehicleEvidence,
    MetadataEvidence,
    VideoEvidence
)
from app.domain.models.enums import EvidenceType


class CVIngestorError(ValueError):
    """Raised when raw CV input fails validation before it can become Evidence."""


def _parse_timestamp(v: str) -> datetime:
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


def load_mock_cv_inputs(directory: str | Path) -> EvidenceBundle:
    directory = Path(directory)

    def _load(name: str):
        path = directory / name
        if not path.exists():
            raise CVIngestorError(f"Missing required input file: {path}")
        with open(path) as f:
            return json.load(f)

    metadata = _load("metadata.json")
    raw_persons = _load("persons.json")
    raw_vehicles = _load("vehicles.json")
    raw_clips = _load("video_clips.json")

    bundle = EvidenceBundle()
    
    # Store global stats
    bundle.statistics["video_id"] = metadata.get("video_source", {}).get("video_id") or metadata.get("provenance", {}).get("video_id")
    bundle.statistics["camera_id"] = metadata["camera"]["camera_id"]

    try:
        # Load Video Clips
        for c in raw_clips:
            clip_ev = VideoEvidence(
                evidence_id=uuid.UUID(c["clip_id"]) if "-" in c["clip_id"] else uuid.uuid4(),
                evidence_type=EvidenceType.VIDEO,
                source=c["source"],
                source_id=c["clip_id"],
                confidence=1.0,
                timestamp=_parse_timestamp(c["start_timestamp"]),
                metadata={
                    "camera_id": c["camera_id"],
                    "end_timestamp": c["end_timestamp"],
                    "duration_seconds": c["duration_seconds"],
                    "description": c["caption"],
                    "involved_track_ids": c.get("involved_track_ids", [])
                },
                provenance=c["provenance"]
            )
            bundle.add_evidence(clip_ev)

        # Load Persons
        for p in raw_persons:
            person_ev = PersonEvidence(
                evidence_id=uuid.UUID(p["detection_id"]),
                evidence_type=EvidenceType.VECTOR,
                source=p["source"],
                source_id=p["detection_id"],
                confidence=p["confidence"],
                timestamp=_parse_timestamp(p["timestamp"]),
                metadata={
                    "camera_id": p["camera_id"],
                    "track_id": p["track_id"],
                    "bbox": p["bbox"],
                    "attributes": p["attributes"],
                    "description": p["description"],
                    "clip_id": p["clip_id"]
                },
                provenance=p["provenance"]
            )
            bundle.add_evidence(person_ev)

        # Load Vehicles
        for v in raw_vehicles:
            vehicle_ev = VehicleEvidence(
                evidence_id=uuid.UUID(v["detection_id"]),
                evidence_type=EvidenceType.VECTOR,
                source=v["source"],
                source_id=v["detection_id"],
                confidence=v["confidence"],
                timestamp=_parse_timestamp(v["timestamp"]),
                metadata={
                    "camera_id": v["camera_id"],
                    "track_id": v["track_id"],
                    "bbox": v["bbox"],
                    "attributes": v.get("attributes", {}),
                    "description": v.get("description", ""),
                    "clip_id": v["clip_id"],
                    "license_plate": v.get("license_plate")
                },
                provenance=v["provenance"]
            )
            bundle.add_evidence(vehicle_ev)

        # Load Events (Metadata)
        for e in metadata.get("scene_events", []):
            event_ev = MetadataEvidence(
                evidence_id=uuid.uuid4(),
                evidence_type=EvidenceType.METADATA,
                source="scene_events",
                source_id=e["event_id"],
                confidence=1.0,
                timestamp=_parse_timestamp(e["timestamp"]),
                metadata={
                    "event_type": e["type"],
                    "severity": e["severity"],
                    "description": e["description"],
                    "camera_id": e["camera_id"],
                    "related_track_ids": e.get("related_track_ids", [])
                }
            )
            bundle.add_evidence(event_ev)

    except (KeyError, ValueError) as exc:
        raise CVIngestorError(f"Validation failed while building EvidenceBundle: {exc}") from exc

    return bundle


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "dataset/mock_cv_inputs"
    bundle = load_mock_cv_inputs(target)
    print(f"OK: Built EvidenceBundle with {len(bundle.evidence)} evidence items.")
