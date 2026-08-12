
import pytest

from app.agents.intent.enums import Intent
from app.agents.intent.schemas import IntentResult
from app.agents.vector.agent import VectorAgent
from app.graph.supervisor.event_bus import EventBus
from app.schemas.context import ExecutionPlan, UserContext, VistaContext
from app.services.repositories.person_repository import PersonRepository
from app.services.repositories.vehicle_repository import VehicleRepository
from app.services.vector_service import VectorService
from app.tools.vector.milvus_tool import MilvusTool


@pytest.mark.asyncio
async def test_vector_agent_end_to_end():
    event_bus = EventBus()
    db_tool = MilvusTool(event_bus)
    
    # Repositories
    person_repo = PersonRepository(db_tool)
    vehicle_repo = VehicleRepository(db_tool)
    
    # Service
    service = VectorService(person_repo, vehicle_repo, event_bus)
    
    # Agent
    agent = VectorAgent(service)
    
    # Context setup
    context = VistaContext(user=UserContext(user_id="1", role="admin", allowed_cameras=["CAM_02"]), conversation_id="123")
    context.execution_plan = ExecutionPlan(success=True, agents=["vector_agent"], intent=Intent.PERSON_SEARCH.value)
    
    # Emulate intent agent result
    context.results["intent_agent"] = IntentResult(
        success=True,
        intent=Intent.PERSON_SEARCH,
        entities={"description": "Person in blue shirt"}
    )
    
    # Test Execute
    result = await agent.execute(context, None)
    
    from app.domain.models.enums import AgentStatus
    assert result.status == AgentStatus.SUCCESS
    assert len(result.person_matches) > 0
    assert len(result.evidence) > 0
    
    # Check that highest score is returned as confidence
    assert agent.confidence(result) > 0
