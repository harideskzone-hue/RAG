import pytest
from app.graph.supervisor.response_coordinator import ResponseCoordinator
from app.schemas.context import VistaContext, UserContext

def test_llm_cannot_override_verified_count():
    """
    If the LLM hallucinates a count, the ResponseCoordinator must strictly
    use the verified_contract's count.
    """
    coordinator = ResponseCoordinator()
    
    context = VistaContext(user=UserContext(user_id="1", role="admin"))
    context.active_video_id = "vid_1"
    context.current_query = "How many people are there?"
    
    # 1. Provide a verified contract with exactly 2 people
    verified_contract = {
        "status": "verified",
        "operation": "count",
        "target": "person",
        "constraints": [],
        "verified_count": 2,
        "verified_tracks": ["P001", "P003"],
        "events": [],
        "video_id": "vid_1"
    }
    context.results["verified_contract"] = verified_contract
    
    from app.agents.intent.schemas import IntentResult
    from app.schemas.context import QueryIntent
    from app.agents.intent.schemas import Intent
    from app.domain.models.confidence import ConfidenceScore
    context.results["intent_agent"] = IntentResult(
        success=True,
        intent=Intent.GENERAL_QUERY,
        domain="investigation",
        operation="count",
        confidence=ConfidenceScore(overall=0.9),
        query_intent=QueryIntent(domain="investigation", operation="count", target_type="person", raw_query="")
    )
    
    from app.domain.evidence import EvidenceBundle, PersonEvidence
    from datetime import datetime, timezone
    context.evidence_bundle = EvidenceBundle(
        evidence=[
            PersonEvidence(source="mock_cam", confidence=1.0, timestamp=datetime.now(timezone.utc), metadata={"track_id": "P001", "camera_id": "CAM_1", "description": "person", "origin": {"type": "video_analysis", "video_id": "vid_1", "track_id": "P001", "camera_id": "CAM_1"}}),
            PersonEvidence(source="mock_cam", confidence=1.0, timestamp=datetime.now(timezone.utc), metadata={"track_id": "P003", "camera_id": "CAM_2", "description": "person", "origin": {"type": "video_analysis", "video_id": "vid_1", "track_id": "P003", "camera_id": "CAM_2"}})
        ]
    )
    
    # 2. Inject a hallucinated LLM answer saying there are 7 people
    # Let's assume the LLM output is stored in context.results['reasoning_agent']
    from app.domain.models.agent_result import AgentResult
    from app.domain.models.enums import AgentType, AgentStatus
    import uuid
    from app.domain.models.confidence import ConfidenceScore
    from app.domain.models.execution_metadata import ExecutionMetadata
    context.results["reasoning_agent"] = AgentResult(
        execution_id=uuid.uuid4(),
        agent_name="Reasoning Agent",
        agent_type=AgentType.REASONING,
        status=AgentStatus.SUCCESS,
        confidence=ConfidenceScore(overall=0.9),
        execution=ExecutionMetadata(duration_ms=10),
        metadata={"answer": "There are 7 people."}
    )
    
    # 3. Generate response
    response = coordinator.generate_response(context)
    
    # 4. Assert the response strictly uses the contract count (2), NOT the hallucinated 7
    assert "7" not in response["final_answer"], "Response must not contain the LLM hallucinated count"
    assert "2" in response["final_answer"] or "two" in response["final_answer"].lower(), "Response must reflect the verified count"

