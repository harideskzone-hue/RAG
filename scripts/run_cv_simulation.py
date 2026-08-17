#!/usr/bin/env python3
"""
run_cv_simulation.py — validates dataset/mock_cv_inputs/ end-to-end through
the cv_ingestor stage.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.domain.evidence import PersonEvidence, VehicleEvidence, MetadataEvidence, VideoEvidence
from app.tools.ingestors.cv_ingestor import load_mock_cv_inputs, CVIngestorError  # noqa: E402


def run_checks(fixtures_dir: str) -> int:
    print(f"Loading fixtures from: {fixtures_dir}\n")

    try:
        bundle = load_mock_cv_inputs(fixtures_dir)
    except CVIngestorError as exc:
        print(f"FAIL: ingestor rejected the fixtures: {exc}")
        return 1

    video_id = bundle.statistics.get("video_id")
    camera_id = bundle.statistics.get("camera_id")

    persons = [e for e in bundle.evidence if isinstance(e, PersonEvidence)]
    vehicles = [e for e in bundle.evidence if isinstance(e, VehicleEvidence)]
    events = [e for e in bundle.evidence if isinstance(e, MetadataEvidence)]
    clips = [e for e in bundle.evidence if isinstance(e, VideoEvidence)]

    print(
        f"OK: EvidenceBundle built for video_id={video_id!r} "
        f"camera_id={camera_id!r}"
    )
    person_tracks = {p.metadata.get("track_id") for p in persons if p.metadata.get("track_id")}
    print(f"  persons : {len(persons)} detections across {len(person_tracks)} tracks")
    print(f"  vehicles: {len(vehicles)} detections")
    print(f"  events  : {len(events)} scene events")
    print(f"  clips   : {len(clips)} segments\n")

    failures = []

    # 1. clip_id cross-reference
    known_clip_ids = {c.source_id for c in clips}
    for p in persons:
        clip_id = p.metadata.get("clip_id")
        if clip_id and clip_id not in known_clip_ids:
            failures.append(f"PersonEvidence {p.evidence_id} references unknown clip_id={clip_id!r}")
    
    if not any("unknown clip_id" in f for f in failures):
        print("[x] clip_id references resolve against video_clips.json")

    # 2. every person/vehicle timestamp should fall within the union of clip windows
    if clips:
        # For VideoEvidence, the timestamp is start_timestamp, and metadata['end_timestamp'] is end
        def _parse(ts):
            if isinstance(ts, datetime):
                return ts
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            
        window_start = min(_parse(c.timestamp) for c in clips)
        window_end = max(_parse(c.metadata["end_timestamp"]) for c in clips)
        for p in persons:
            p_ts = _parse(p.timestamp)
            if not (window_start <= p_ts <= window_end):
                failures.append(
                    f"person {p.evidence_id} timestamp {p_ts} outside "
                    f"recording window [{window_start}, {window_end}]"
                )
        if not failures:
            print(f"[x] all person timestamps fall within recording window "
                  f"[{window_start.time()}, {window_end.time()}]")

    # 3. confidence bounds
    bad_conf = [p.evidence_id for p in persons if not (0.0 <= p.confidence <= 1.0)]
    if bad_conf:
        failures.append(f"out-of-range confidence on: {bad_conf}")
    else:
        print("[x] all confidence scores in [0.0, 1.0]")

    # 4. every track referenced by a clip's involved_track_ids actually has detections
    for clip in clips:
        involved_tracks = set(clip.metadata.get("involved_track_ids", []))
        missing = involved_tracks - person_tracks
        if missing:
            failures.append(f"{clip.source_id} involved_track_ids references undetected tracks: {missing}")
    if not any("involved_track_ids" in f for f in failures):
        print("[x] every clip.involved_track_ids resolves to at least one person detection")

    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("All structural checks passed.")
    print("\nNext step (manual, once your Supervisor is wired to this bundle):")
    print("  Run the positive + negative query scenarios below and compare")
    print("  against the 'expected' column.\n")
    print_scenarios()
    return 0


def print_scenarios():
    scenarios = [
        ("Who was near the display case the longest?", "grounded -> track_02"),
        ("Find the person in the green outfit.", "grounded -> track_04"),
        ("What happened around the shopkeeper near the end?", "grounded -> evt_004 + clip_seg_04"),
        ("How many people were at the counter around 13:15:26?", "grounded -> ~6 tracks"),
        ("Find a person wearing a red shirt.", "ABSTAIN (attribute not present in fixture)"),
        ("What was the vehicle involved?", "ABSTAIN (vehicles.json is empty)"),
        ("Who committed the robbery?", "ABSTAIN (no ground-truth criminal attribution in data)"),
    ]
    width = max(len(s[0]) for s in scenarios)
    for query, expected in scenarios:
        print(f"  {query.ljust(width)}  -> {expected}")


if __name__ == "__main__":
    fixtures_dir = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(__file__).resolve().parent.parent / "dataset" / "mock_cv_inputs"
    )
    sys.exit(run_checks(fixtures_dir))
