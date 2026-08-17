
import pytest

from app.agents.confidence.agent import ConfidenceAgent
from app.agents.confidence.engine import ConfidenceEngine
from app.agents.event.agent import EventAgent
from app.agents.evidence.agent import EvidenceAgent

# Agents
from app.agents.metadata.agent import MetadataAgent
from app.agents.registry import agent_registry
from app.agents.report.agent import ReportAgent
from app.agents.vector.agent import VectorAgent
from app.agents.video.agent import VideoAgent
from app.domain.models.confidence import ConfidencePolicy
from app.graph.supervisor.event_bus import EventBus
from app.graph.supervisor.supervisor import Supervisor
from app.platform.tracing.context import set_conversation_id, set_execution_id
from app.schemas.context import (
    ExecutionPlan,
    UserContext,
    VistaContext,
)
from app.services.event_service.service import EventService

# Services
from app.services.metadata_service import MetadataService
from app.services.report_service.service import ReportService
from app.services.repositories.alert_repository import AlertRepository

# Repositories
from app.services.repositories.camera_repository import CameraRepository
from app.services.repositories.person_repository import PersonRepository
from app.services.repositories.vehicle_repository import VehicleRepository
from app.services.vector_service import VectorService
from app.services.video_service.service import VideoService


# Tools
from app.tools.metadata.postgres_tool import PostgresTool
from app.tools.registry import tool_registry
from app.tools.vector.milvus_tool import MilvusTool
from app.tools.video.s3_tool import S3Tool


@pytest.fixture
def supervisor():
    # 1. Core Infrastructure
    event_bus = EventBus()
    
    # 2. Tools
    postgres_tool = PostgresTool(event_bus)
    milvus_tool = MilvusTool(event_bus)
    s3_tool = S3Tool(event_bus)
    
    # 3. Repositories
    camera_repo = CameraRepository(postgres_tool)
    alert_repo = AlertRepository(postgres_tool)
    person_repo = PersonRepository(milvus_tool)
    vehicle_repo = VehicleRepository(milvus_tool)
    
    # 4. Services
    metadata_service = MetadataService(camera_repo, alert_repo, event_bus)
    vector_service = VectorService(person_repo, vehicle_repo, event_bus)
    from app.infrastructure.llm.model_registry import ModelRegistry
    video_service = VideoService(s3_tool, ModelRegistry.get_client(), event_bus)
    event_service = EventService(event_bus)
    report_service = ReportService(event_bus)
    
    from app.api.dependencies.supervisor import get_supervisor
    
    # Ensure registries are clean before initializing to avoid test cross-talk
    from app.agents.registry import agent_registry
    from app.tools.registry import tool_registry
    import app.api.dependencies.supervisor as deps
    deps._initialized = False
    agent_registry._agents.clear()
    tool_registry._tools.clear()
    tool_registry._descriptions.clear()
    
    sup = get_supervisor(
        event_bus=event_bus,
        postgres_tool=postgres_tool,
        milvus_tool=milvus_tool,
        s3_tool=s3_tool,
        metadata_service=metadata_service,
        vector_service=vector_service,
        video_service=video_service,
        event_service=event_service,
        report_service=report_service
    )
    
    # Mocking event_bus in supervisor to share state for assertions
    sup.event_bus = event_bus 
    sup.scheduler.event_bus = event_bus
    
    return sup

@pytest.fixture
def base_context():
    return VistaContext(
        user=UserContext(user_id="user_123", role="operator"),
        conversation_id="conv_1",
        current_query="Find the person wearing a blue shirt near Gate A."
    )

@pytest.mark.asyncio
async def test_person_search_pipeline(supervisor, base_context):
    """
    Scenario: User asks to find a person.
    Expected: Metadata -> Vector -> Evidence -> Confidence -> Video -> Response
    """
    set_execution_id("exec_test_1")
    set_conversation_id("conv_test_1")
    
    # Mock Planner Output
    base_context.execution_plan = ExecutionPlan(
        success=True,
        intent="PERSON_SEARCH",
        agents=["metadata_agent", "vector_agent", "evidence_agent", "confidence_agent", "video_agent"],
        execution_groups=[
            {"agents": ["metadata_agent", "vector_agent"]},
            {"agents": ["evidence_agent"]},
            {"agents": ["confidence_agent"]},
            {"agents": ["video_agent"]}
        ]
    )
    response = await supervisor.run(base_context)
    if response.get("status") != "success":
        print(f"\nFAILED! Response: {response}\n")
    
    assert response["status"] == "success"
    # Ensure all agents executed
    executed_agents = [d["agent"] for d in base_context.agent_decisions]
    assert "metadata_agent" in executed_agents
    assert "vector_agent" in executed_agents
    assert "evidence_agent" in executed_agents
    assert "confidence_agent" in executed_agents
    assert "video_agent" in executed_agents
    
    # Ensure evidence bundle was populated and sorted
    assert base_context.evidence_bundle is not None
    assert len(base_context.evidence_bundle.evidence) > 0
    
    # Ensure confidence score is present
    assert base_context.confidence_score > 0
    assert base_context.confidence_report is not None

@pytest.mark.asyncio
async def test_camera_status_pipeline(supervisor, base_context):
    """
    Scenario: User asks for camera status.
    Expected: Metadata -> Evidence -> Confidence -> Response (Video agent skipped)
    """
    set_execution_id("exec_test_2")
    set_conversation_id("conv_test_2")
    
    base_context.current_query = "Is camera 5 online?"
    base_context.execution_plan = ExecutionPlan(
        success=True,
        intent="CAMERA_STATUS",
        agents=["metadata_agent", "evidence_agent", "confidence_agent"],
        execution_groups=[
            {"agents": ["metadata_agent"]},
            {"agents": ["evidence_agent"]},
            {"agents": ["confidence_agent"]}
        ]
    )
    response = await supervisor.run(base_context)
    if response.get("status") != "success":
        print(f"\nFAILED! Response: {response}\n")
    
    assert response["status"] == "success"
    executed_agents = [d["agent"] for d in base_context.agent_decisions]
    assert "metadata_agent" in executed_agents
    assert "evidence_agent" in executed_agents
    assert "confidence_agent" in executed_agents
    assert "video_agent" not in executed_agents # Should not execute
    assert "vector_agent" not in executed_agents # Should not execute

@pytest.mark.asyncio
async def test_report_generation_pipeline(supervisor, base_context):
    """
    Scenario: User asks for a weekly report.
    Expected: Metadata -> Evidence -> Report -> Response
    """
    set_execution_id("exec_test_3")
    set_conversation_id("conv_test_3")
    
    base_context.current_query = "Generate a weekly JSON report."
    base_context.execution_plan = ExecutionPlan(
        success=True,
        intent="REPORT",
        agents=["metadata_agent", "evidence_agent", "report_agent"],
        execution_groups=[
            {"agents": ["metadata_agent"]},
            {"agents": ["evidence_agent"]},
            {"agents": ["report_agent"]}
        ]
    )
    response = await supervisor.run(base_context)
    if response.get("status") != "success":
        print(f"\nFAILED! Response: {response}\n")
    
    assert response["status"] == "success"
    executed_agents = [d["agent"] for d in base_context.agent_decisions]
    assert "report_agent" in executed_agents
    
    assert "report_agent" in base_context.results
    report_res = base_context.results["report_agent"]
    assert report_res.report_uri != ""
    assert "analytics" in report_res.data
