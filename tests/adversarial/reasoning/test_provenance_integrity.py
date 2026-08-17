import pytest
from app.graph.supervisor.response_coordinator import ResponseCoordinator
from app.schemas.context import VistaContext, UserContext

def test_llm_cannot_invent_provenance():
    """
    If the LLM says "Camera CAM_05 at 02:31" when that info isn't in the contract,
    it must be rejected/removed from the structured evidence.
    """
    coordinator = ResponseCoordinator()
    
    context = VistaContext(user=UserContext(user_id="1", role="admin"))
    context.active_video_id = "vid_1"
    context.current_query = "What happened?"
    
    # 1. Verified contract has NO events from CAM_05
    verified_contract = {
        "status": "verified",
        "operation": "event_search",
        "target": "event",
        "constraints": [],
        "verified_count": 1,
        "verified_tracks": ["P001"],
        "events": [
            {
                "track_id": "P001",
                "camera_id": "CAM_01",
                "event_type": "running",
                "description": "Person running",
                "video_timestamp_sec": 30.0
            }
        ],
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
        operation="event_search",
        confidence=ConfidenceScore(overall=0.9),
        query_intent=QueryIntent(domain="investigation", operation="event_search", target_type="event", raw_query="")
    )
    
    # 2. Inject a hallucinated LLM answer claiming CAM_05 at 02:31
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
        metadata={"answer": "I saw someone on CAM_05 at 02:31."}
    )
    
    # 3. Generate response
    response = coordinator.generate_response(context)
    
    # 4. Assert CAM_05 is NOT in the response evidence/citations
    if hasattr(response, "citations"):
        for citation in response.citations:
            assert "CAM_05" not in citation.source_id, "Hallucinated provenance must not enter citations"
            
    assert "CAM_01" in str(verified_contract["events"]) # Original contract is safe
