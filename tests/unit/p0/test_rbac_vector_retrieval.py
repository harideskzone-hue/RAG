import pytest
import uuid
import os
import sys

from app.schemas.context import VistaContext, UserContext, ExecutionPlan, ExecutionGroup
from app.agents.vector.agent import VectorAgent
from app.services.vector_service import VectorService
from app.services.repositories.person_repository import PersonRepository
from app.services.repositories.vehicle_repository import VehicleRepository
from app.tools.vector.milvus_tool import MilvusTool
from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.result_collector import ResultCollector
from app.domain.evidence import EvidenceBundle
from app.agents.intent.enums import Intent

pytestmark = pytest.mark.asyncio

@pytest.fixture
def vector_agent_setup():
    event_bus = EventBus()
    milvus_tool = MilvusTool(event_bus)
    person_repo = PersonRepository(milvus_tool)
    vehicle_repo = VehicleRepository(milvus_tool)
    vector_service = VectorService(person_repo, vehicle_repo, event_bus)
    agent = VectorAgent(vector_service)
    
    def create_context(allowed_cameras):
        # We need to simulate the environment correctly
        context = VistaContext(
            conversation_id=str(uuid.uuid4()), 
            execution_id=str(uuid.uuid4()),
            current_query="person in blue shirt", 
            user=UserContext(user_id="u1", role="admin", allowed_cameras=allowed_cameras)
        )
        plan = ExecutionPlan(
            success=True,
            intent=Intent.PERSON_SEARCH.value,
            agents=["vector_agent"],
            tools=[],
            execution_groups=[ExecutionGroup(agents=["vector_agent"])],
            dependencies={}
        )
        context.execution_plan = plan
        context.results = {}
        context.evidence_bundle = EvidenceBundle()
        return context

    return agent, create_context

async def test_rbac_unrestricted_access(vector_agent_setup):
    agent, create_context = vector_agent_setup
    context = create_context(allowed_cameras=None)
    
    result = await agent.execute(context, None)
    
    # We expect 5 matches from the dataset (they are all CAM_02)
    assert len(result.person_matches) > 0, "Expected matches for unrestricted access"

async def test_rbac_zero_access(vector_agent_setup):
    agent, create_context = vector_agent_setup
    context = create_context(allowed_cameras=[])
    
    result = await agent.execute(context, None)
    
    assert len(result.person_matches) == 0, "Expected 0 matches when allowed_cameras=[]"

async def test_rbac_specific_camera_match(vector_agent_setup):
    agent, create_context = vector_agent_setup
    context = create_context(allowed_cameras=["CAM_02"])
    
    result = await agent.execute(context, None)
    
    assert len(result.person_matches) > 0, "Expected matches for CAM_02"
    for match in result.person_matches:
        assert match.camera_id == "CAM_02"

async def test_rbac_specific_camera_no_match(vector_agent_setup):
    agent, create_context = vector_agent_setup
    context = create_context(allowed_cameras=["CAM_01"])
    
    result = await agent.execute(context, None)
    
    # The dataset only contains CAM_02 for this query
    assert len(result.person_matches) == 0, "Expected 0 matches for CAM_01"

async def test_rbac_multiple_cameras(vector_agent_setup):
    agent, create_context = vector_agent_setup
    context = create_context(allowed_cameras=["CAM_01", "CAM_02"])
    
    result = await agent.execute(context, None)
    
    assert len(result.person_matches) > 0, "Expected matches when multiple cameras provided"
    for match in result.person_matches:
        assert match.camera_id in ["CAM_01", "CAM_02"]

async def test_rbac_collector_evidence_bundle(vector_agent_setup):
    agent, create_context = vector_agent_setup
    context = create_context(allowed_cameras=["CAM_01"])
    
    result = await agent.execute(context, None)
    
    # CAM_01 should return no matches, ensuring they don't enter EvidenceBundle
    collector = ResultCollector()
    collector.collect("vector_agent", result, context)
    
    for ev in context.evidence_bundle.evidence:
        assert ev.metadata.get("camera_id") != "CAM_02", "CAM_02 MUST NEVER enter EvidenceBundle when authorized only for CAM_01"
