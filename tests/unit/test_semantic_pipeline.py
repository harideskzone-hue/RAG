import pytest
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.schemas.context import VistaContext, UserContext, QueryIntent
from app.domain.evidence import EvidenceBundle, PersonEvidence
from app.domain.models.enums import EvidenceType
from app.graph.supervisor.response_coordinator import ResponseCoordinator, evaluate_structured_constraints


def test_evaluate_structured_constraints_generic():
    """Verify that generic constraint evaluation works dynamically without hardcoded values."""
    attrs_male = {"entity_type": "person", "gender": "male", "clothing_upper": "black jacket"}
    attrs_female = {"entity_type": "person", "gender": "female", "clothing_upper": "blue jacket"}

    assert evaluate_structured_constraints(attrs_male, ["gender=male"]) is True
    assert evaluate_structured_constraints(attrs_female, ["gender=male"]) is False

    assert evaluate_structured_constraints(attrs_female, ["gender=female"]) is True
    assert evaluate_structured_constraints(attrs_male, ["gender=female"]) is False

    assert evaluate_structured_constraints(attrs_male, ["clothing_upper=black jacket"]) is True
    assert evaluate_structured_constraints(attrs_female, ["clothing_upper=black jacket"]) is False


def test_provenance_integrity_gate():
    """Verify that evidence lacking valid origin/video_id is rejected when active_video_id is set."""
    coord = ResponseCoordinator()
    
    ctx = VistaContext(
        user=UserContext(user_id="u1", role="admin"),
        current_query="how many people?",
        active_video_id="VIDEO-2026-08-13-14-20-13.mp4"
    )
    
    now = datetime.now(timezone.utc)
    
    # Evidence with valid matching origin
    ev_valid = PersonEvidence(
        evidence_type=EvidenceType.VECTOR,
        source="video_ingestion",
        confidence=0.9,
        timestamp=now,
        trace_id=str(uuid.uuid4()),
        metadata={
            "camera_id": "cam_01",
            "description": "Man in black jacket",
            "origin": {
                "type": "video_ingestion",
                "video_id": "VIDEO-2026-08-13-14-20-13.mp4",
                "track_id": "P002",
                "frame_index": 260,
                "video_timestamp_sec": 20.0
            },
            "attributes": {"entity_type": "person", "gender": "male"}
        }
    )
    
    # Evidence with invalid/different video_id origin
    ev_invalid = PersonEvidence(
        evidence_type=EvidenceType.VECTOR,
        source="video_ingestion",
        confidence=0.9,
        timestamp=now,
        trace_id=str(uuid.uuid4()),
        metadata={
            "camera_id": "cam_01",
            "description": "Mock jewelry counter person",
            "origin": {
                "type": "video_ingestion",
                "video_id": "STALE_MOCK_VIDEO.mp4",
                "track_id": "P999"
            },
            "attributes": {"entity_type": "person", "gender": "male"}
        }
    )

    ctx.evidence_bundle = EvidenceBundle(evidence=[ev_valid, ev_invalid])
    
    intent_mock = MagicMock(
        intent="investigation",
        domain="investigation",
        requires_clarification=False,
        query_intent=QueryIntent(domain="investigation", operation="count", target_type="person")
    )
    ctx.results["intent_agent"] = intent_mock
    
    response = coord.generate_response(ctx)
    assert response["status"] == "success"
    
    # Assert only valid provenance evidence passed the gate
    assert len(response["evidence"]) == 1
    assert response["evidence"][0]["origin"]["track_id"] == "P002"
    assert "1" in response["final_answer"]


def test_mutated_evidence_response_changes():
    """Mutated-evidence regression test: prove response changes dynamically when evidence changes."""
    coord = ResponseCoordinator()
    
    intent_mock = MagicMock(
        intent="investigation",
        domain="investigation",
        requires_clarification=False,
        query_intent=QueryIntent(
            domain="investigation",
            operation="count",
            target_type="person",
            semantic_constraints=["gender=male"]
        )
    )

    now = datetime.now(timezone.utc)
    t_id = str(uuid.uuid4())

    # Dataset A: 2 males, 1 female
    ev_male_1 = PersonEvidence(
        evidence_type=EvidenceType.VECTOR,
        source="video_ingestion",
        confidence=0.9,
        timestamp=now,
        trace_id=t_id,
        metadata={
            "camera_id": "cam_01",
            "description": "Man in dark jacket",
            "origin": {"type": "video_ingestion", "video_id": "v1", "track_id": "P002"},
            "attributes": {"gender": "male"}
        }
    )
    ev_male_2 = PersonEvidence(
        evidence_type=EvidenceType.VECTOR,
        source="video_ingestion",
        confidence=0.9,
        timestamp=now,
        trace_id=t_id,
        metadata={
            "camera_id": "cam_01",
            "description": "Man in winter jacket",
            "origin": {"type": "video_ingestion", "video_id": "v1", "track_id": "P004"},
            "attributes": {"gender": "male"}
        }
    )
    ev_female_1 = PersonEvidence(
        evidence_type=EvidenceType.VECTOR,
        source="video_ingestion",
        confidence=0.9,
        timestamp=now,
        trace_id=t_id,
        metadata={
            "camera_id": "cam_01",
            "description": "Woman in blue fleece",
            "origin": {"type": "video_ingestion", "video_id": "v1", "track_id": "P001"},
            "attributes": {"gender": "female"}
        }
    )

    ctx_a = VistaContext(user=UserContext(user_id="u1", role="admin"), current_query="how many men?")
    ctx_a.evidence_bundle = EvidenceBundle(evidence=[ev_male_1, ev_male_2, ev_female_1])
    ctx_a.results["intent_agent"] = intent_mock
    res_a = coord.generate_response(ctx_a)

    # Dataset B: 1 male, 2 females
    ctx_b = VistaContext(user=UserContext(user_id="u1", role="admin"), current_query="how many men?")
    ctx_b.evidence_bundle = EvidenceBundle(evidence=[ev_male_1, ev_female_1])
    ctx_b.results["intent_agent"] = intent_mock
    res_b = coord.generate_response(ctx_b)

    # Assert responses change dynamically based on evidence
    assert "2" in res_a["final_answer"]
    assert "1" in res_b["final_answer"]
    assert res_a["final_answer"] != res_b["final_answer"]


def test_no_keyword_matching_in_response_coordinator():
    """Anti-hardcoding test: verify response_coordinator.py contains NO query string regex or keyword rules."""
    rc_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "app", "graph", "supervisor", "response_coordinator.py")
    with open(rc_path, "r") as f:
        code = f.read()

    prohibited_snippets = [
        "target_male",
        "target_female",
        "in query_lower",
        "in desc_lower",
        "kurta",
        "polka-dot",
        "display case",
        "checked shirt",
        "re.search("
    ]

    for snippet in prohibited_snippets:
        assert snippet not in code, f"Prohibited hardcoded/keyword snippet '{snippet}' found in response_coordinator.py!"


def test_behavioral_investigation_zero_evidence_abstention():
    """Verify that when candidate_items == 0, behavioral query abstains without fabricating people or track_xxxxx IDs."""
    coord = ResponseCoordinator()
    ctx = VistaContext(
        user=UserContext(user_id="u1", role="admin"),
        current_query="Is there any suspicious person in the CCTV?",
        active_video_id="VIDEO-2026-08-13-14-20-13.mp4"
    )
    
    intent_mock = MagicMock(
        intent="behavioral_investigation",
        domain="investigation",
        requires_clarification=False,
        query_intent=QueryIntent(
            domain="investigation",
            operation="behavioral_investigation",
            target_type="person"
        )
    )
    ctx.results["intent_agent"] = intent_mock
    ctx.evidence_bundle = EvidenceBundle(evidence=[])
    
    response = coord.generate_response(ctx)
    assert response["status"] == "success"
    assert ("couldn't verify" in response["final_answer"].lower() or "no" in response["final_answer"].lower())
    assert len(response["evidence"]) == 0
    assert "track_" not in response["final_answer"]
    assert "P001" not in response["final_answer"]


def test_behavioral_investigation_evidence_grounding():
    """Verify that behavioral investigation results depend strictly on verified event evidence, not person count."""
    coord = ResponseCoordinator()
    now = datetime.now(timezone.utc)
    t_id = str(uuid.uuid4())

    intent_mock = MagicMock(
        intent="behavioral_investigation",
        domain="investigation",
        requires_clarification=False,
        query_intent=QueryIntent(
            domain="investigation",
            operation="behavioral_investigation",
            target_type="person"
        )
    )

    # 6 Person evidence items (visual person metadata only, no event_type)
    persons = [
        PersonEvidence(
            evidence_type=EvidenceType.VECTOR,
            source="video_ingestion",
            confidence=0.9,
            timestamp=now,
            trace_id=t_id,
            metadata={"camera_id": "cam_01", "description": f"Person P{i:03d}", "origin": {"type": "video_ingestion", "video_id": "v1", "track_id": f"P{i:03d}"}, "attributes": {"gender": "male" if i % 2 == 0 else "female"}}
        )
        for i in range(1, 7)
    ]

    # Test A: 6 people exist, but 0 behavioral event evidence -> verified_count = 0, verified_tracks = []
    ctx_a = VistaContext(user=UserContext(user_id="u1", role="admin"), current_query="Is there any suspicious person in the CCTV?", active_video_id="v1")
    ctx_a.evidence_bundle = EvidenceBundle(evidence=persons)
    ctx_a.results["intent_agent"] = intent_mock
    res_a = coord.generate_response(ctx_a)

    contract_a = ctx_a.results.get("verified_contract", {})
    assert contract_a.get("verified_count") == 0
    assert contract_a.get("verified_tracks") == []
    assert "couldn't verify" in res_a["final_answer"].lower() or "no" in res_a["final_answer"].lower()

    # Test B: 1 structured behavioral event evidence belongs to P004 -> verified_count = 1, verified_tracks = ["P004"]
    ev_p004 = PersonEvidence(
        evidence_type=EvidenceType.EVENT,
        source="event_query",
        confidence=0.95,
        timestamp=now,
        trace_id=t_id,
        metadata={
            "camera_id": "cam_01",
            "event_type": "unattended_bag_drop",
            "description": "Suspicious unattended bag drop by P004",
            "origin": {"type": "event_query", "video_id": "v1", "track_id": "P004", "event_type": "unattended_bag_drop"}
        }
    )
    ctx_b = VistaContext(user=UserContext(user_id="u1", role="admin"), current_query="Is there any suspicious person in the CCTV?", active_video_id="v1")
    ctx_b.evidence_bundle = EvidenceBundle(evidence=persons + [ev_p004])
    ctx_b.results["intent_agent"] = intent_mock
    res_b = coord.generate_response(ctx_b)

    contract_b = ctx_b.results.get("verified_contract", {})
    assert contract_b.get("verified_count") == 1
    assert contract_b.get("verified_tracks") == ["P004"]
    assert "P004" in res_b["final_answer"]
    assert "unattended bag drop" in res_b["final_answer"].lower()

    # Test C: Structured behavioral event belongs to P002 -> verified_count = 1, verified_tracks = ["P002"]
    ev_p002 = PersonEvidence(
        evidence_type=EvidenceType.EVENT,
        source="event_query",
        confidence=0.95,
        timestamp=now,
        trace_id=t_id,
        metadata={
            "camera_id": "cam_01",
            "event_type": "loitering",
            "description": "Suspicious loitering by P002",
            "origin": {"type": "event_query", "video_id": "v1", "track_id": "P002", "event_type": "loitering"}
        }
    )
    ctx_c = VistaContext(user=UserContext(user_id="u1", role="admin"), current_query="Is there any suspicious person in the CCTV?", active_video_id="v1")
    ctx_c.evidence_bundle = EvidenceBundle(evidence=persons + [ev_p002])
    ctx_c.results["intent_agent"] = intent_mock
    res_c = coord.generate_response(ctx_c)

    contract_c = ctx_c.results.get("verified_contract", {})
    assert contract_c.get("verified_count") == 1
    assert contract_c.get("verified_tracks") == ["P002"]
    assert "P002" in res_c["final_answer"]
    assert "loitering" in res_c["final_answer"].lower()
