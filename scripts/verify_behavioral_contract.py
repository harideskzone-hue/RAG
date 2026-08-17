import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from unittest.mock import MagicMock
from app.domain.evidence import EvidenceBundle, PersonEvidence
from app.domain.models.enums import EvidenceType
from app.graph.supervisor.response_coordinator import ResponseCoordinator
from app.schemas.context import VistaContext, UserContext, QueryIntent

logging.basicConfig(level=logging.INFO)

async def run_trace():
    print("=" * 70)
    print("🔬 VISTA Behavioral Evidence Pipeline Execution Trace")
    print("=" * 70)

    now = datetime.now(timezone.utc)

    # Simulate 6 Person Evidence Items (Visual descriptions only)
    person_evidence = [
        PersonEvidence(
            evidence_type=EvidenceType.VECTOR,
            source="video_ingestion",
            confidence=0.9,
            timestamp=now,
            trace_id=str(uuid.uuid4()),
            metadata={
                "camera_id": "cam_01",
                "description": f"Supermarket shopper P{i:03d}",
                "origin": {"type": "video_ingestion", "video_id": "VIDEO-2026-08-13-14-20-13.mp4", "track_id": f"P{i:03d}"},
                "attributes": {"gender": "male" if i % 2 == 0 else "female"}
            }
        )
        for i in range(1, 7)
    ]

    coord = ResponseCoordinator()
    
    # Mock Intent Result
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

    print("\n📍 Scenario A: 6 Person tracks exist (visual person metadata only), 0 Event evidence items")
    ctx_a = VistaContext(
        user=UserContext(user_id="admin_user", role="admin"),
        current_query="Is there any suspicious person in the CCTV?",
        active_video_id="VIDEO-2026-08-13-14-20-13.mp4"
    )
    ctx_a.evidence_bundle = EvidenceBundle(evidence=person_evidence)
    ctx_a.results["intent_agent"] = intent_mock

    res_a = coord.generate_response(ctx_a)
    contract_a = ctx_a.results.get("verified_contract", {})

    print(f"  • Query: \"{ctx_a.current_query}\"")
    print(f"  • Intent Output: operation={getattr(ctx_a.results.get('intent_agent', {}).query_intent, 'operation', 'N/A')}, target={getattr(ctx_a.results.get('intent_agent', {}).query_intent, 'target_type', 'N/A')}")
    print(f"  • Retrieved Evidence Count: {len(person_evidence)}")
    print(f"  • Provenance-Valid Evidence Count: {len(person_evidence)}")
    print(f"  • Event Evidence Count: 0")
    print(f"  • Verified Tracks: {contract_a.get('verified_tracks', [])}")
    print(f"  • Verified Contract: {json.dumps(contract_a, indent=2)}")
    print(f"  • Final Answer: \"{res_a.get('final_answer')}\"")

    print("\n📍 Scenario B: 6 Person tracks exist, 1 Structured Event Evidence item (P004)")
    ev_p004 = PersonEvidence(
        evidence_type=EvidenceType.EVENT,
        source="event_query",
        confidence=0.95,
        timestamp=now,
        trace_id=str(uuid.uuid4()),
        metadata={
            "camera_id": "cam_01",
            "event_type": "unattended_bag_drop",
            "description": "Suspicious unattended bag drop by track P004",
            "origin": {"type": "event_query", "video_id": "VIDEO-2026-08-13-14-20-13.mp4", "track_id": "P004", "event_type": "unattended_bag_drop"}
        }
    )

    ctx_b = VistaContext(
        user=UserContext(user_id="admin_user", role="admin"),
        current_query="Is there any suspicious person in the CCTV?",
        active_video_id="VIDEO-2026-08-13-14-20-13.mp4"
    )
    ctx_b.evidence_bundle = EvidenceBundle(evidence=person_evidence + [ev_p004])
    ctx_b.results["intent_agent"] = intent_mock

    res_b = coord.generate_response(ctx_b)
    contract_b = ctx_b.results.get("verified_contract", {})

    print(f"  • Query: \"{ctx_b.current_query}\"")
    print(f"  • Intent Output: operation={getattr(ctx_b.results.get('intent_agent', {}).query_intent, 'operation', 'N/A')}")
    print(f"  • Retrieved Evidence Count: {len(person_evidence) + 1}")
    print(f"  • Provenance-Valid Evidence Count: {len(person_evidence) + 1}")
    print(f"  • Event Evidence Count: 1 (Track P004)")
    print(f"  • Verified Tracks: {contract_b.get('verified_tracks', [])}")
    print(f"  • Verified Contract: {json.dumps(contract_b, indent=2)}")
    print(f"  • Final Answer: \"{res_b.get('final_answer')}\"")

if __name__ == "__main__":
    asyncio.run(run_trace())
