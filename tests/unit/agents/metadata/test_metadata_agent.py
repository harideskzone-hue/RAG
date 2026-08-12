
import pytest

from app.agents.intent.enums import Intent
from app.agents.intent.schemas import IntentResult
from app.agents.metadata.agent import MetadataAgent
from app.graph.supervisor.event_bus import EventBus
from app.schemas.context import ExecutionPlan, UserContext, VistaContext
from app.services.metadata_service import MetadataService
from app.services.repositories.alert_repository import AlertRepository
from app.services.repositories.camera_repository import CameraRepository
from app.tools.metadata.postgres_tool import PostgresTool
from app.domain.models.enums import AgentStatus


@pytest.mark.asyncio
async def test_metadata_agent_end_to_end():
    event_bus = EventBus()
    db_tool = PostgresTool(event_bus)
    
    # Repositories
    camera_repo = CameraRepository(db_tool)
    alert_repo = AlertRepository(db_tool)
    
    # Service
    service = MetadataService(camera_repo, alert_repo, event_bus)
    
    # Agent
    agent = MetadataAgent(service)
    
    # Context setup
    context = VistaContext(user=UserContext(user_id="1", role="admin", allowed_cameras=["CAM_02"]), conversation_id="123")
    context.execution_plan = ExecutionPlan(success=True, agents=["metadata_agent"], intent=Intent.CAMERA_STATUS.value)
    
    # Emulate intent agent result
    context.results["intent_agent"] = IntentResult(
        success=True,
        intent=Intent.CAMERA_STATUS,
        entities={"camera_id": "CAM_02"}
    )
    
    # Insert dummy data for isolation
    await db_tool.execute(context, query="INSERT INTO cameras (id, location, status, firmware_version) VALUES ('CAM_02', 'test', 'online', '1.0')")
    
    # Test Execute
    result = await agent.execute(context, None)
    
    assert result.status == AgentStatus.SUCCESS
    assert len(result.cameras) == 1
    assert result.cameras[0].id == "CAM_02"
    assert len(result.evidence) == 1
    
    # Test Cache Hit on second run
    result2 = await agent.execute(context, None)
    assert result2.status == AgentStatus.SUCCESS
    assert len(result2.cameras) == 1
    
    # Verify events
    cache_hit_events = [e for e in event_bus.history if e.event_type == "CACHE_HIT"]
    assert len(cache_hit_events) == 1
