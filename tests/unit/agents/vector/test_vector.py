import pytest
from unittest.mock import AsyncMock, MagicMock
from app.agents.intent.enums import Intent
from app.agents.intent.schemas import IntentResult
from app.agents.vector.agent import VectorAgent
from app.graph.supervisor.event_bus import EventBus
from app.schemas.context import ExecutionPlan, UserContext, VistaContext
from app.domain.models import PersonMatch, ConfidenceScore
from app.domain.models.enums import AgentStatus


@pytest.mark.asyncio
async def test_vector_agent_end_to_end():
    # Mock Service
    mock_service = MagicMock()
    match = PersonMatch(
        id="P001",
        camera_id="cam_01",
        timestamp=10.0,
        score=0.92,
        description="Person in blue shirt"
    )
    mock_service.search_person = AsyncMock(return_value=[match])
    mock_service.search_vehicle = AsyncMock(return_value=[])

    # Agent
    agent = VectorAgent(mock_service)

    # Context setup
    context = VistaContext(user=UserContext(user_id="1", role="admin", allowed_cameras=["cam_01", "CAM_02"]), conversation_id="123")
    context.execution_plan = ExecutionPlan(success=True, agents=["vector_agent"], intent=Intent.PERSON_SEARCH.value)

    # Emulate intent agent result
    context.results["intent_agent"] = IntentResult(
        success=True,
        intent=Intent.PERSON_SEARCH,
        entities={"description": "Person in blue shirt"}
    )

    # Test Execute
    result = await agent.execute(context, None)

    assert result.status == AgentStatus.SUCCESS
    assert len(result.person_matches) > 0
    assert len(result.evidence) > 0
    assert agent.confidence(result) > 0

