import pytest
from app.graph.supervisor.response_coordinator import ResponseCoordinator
from app.schemas.context import VistaContext, UserContext

def test_llm_cannot_invent_tracks():
    """
    If the LLM invents a track_id (e.g., P009), it must NOT appear in the final response evidence.
    The response generator should never become an authority over evidence.
    """
    coordinator = ResponseCoordinator()
    
    context = VistaContext(user=UserContext(user_id="1", role="admin"))
    context.active_video_id = "vid_1"
    context.current_query = "What happened?"
    
    # 1. Provide a verified contract with exactly 2 tracks
    verified_contract = {
        "status": "verified",
        "operation": "behavioral_investigation",
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
        operation="behavioral_investigation",
        confidence=ConfidenceScore(overall=0.9),
        query_intent=QueryIntent(domain="investigation", operation="behavioral_investigation", target_type="person", raw_query="")
    )
    
    # 2. Inject a hallucinated LLM answer claiming P009 exists
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
        metadata={"answer": "I also saw P009 running away."}
    )
    
    # 3. Generate response
    response = coordinator.generate_response(context)
    
    # 4. Assert P009 is NOT in the response evidence payload
    if "evidence" in response:
        # If response has structured evidence payload
        for ev in response["evidence"]:
            assert "P009" not in str(ev.get("track_id", "")), "Hallucinated track P009 must not enter evidence payload"
            
    # And ideally, the text shouldn't say P009 if it's overriding it, but the contract ensures
    # that the structured evidence (which the UI uses to draw boxes) is immune to LLM hallucination.
    # The verified contract tracks must strictly match the output.
    assert "evidence" not in response or not any("P009" in str(e.get('track_id', '')) for e in response["evidence"])
