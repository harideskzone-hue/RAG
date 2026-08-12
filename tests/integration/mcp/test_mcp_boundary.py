import pytest
import asyncio
from uuid import uuid4

from app.mcp.adapter import MCPToolAdapter
from app.mcp.registry import ToolRegistry
from app.mcp.client import LocalMCPClient
from app.schemas.context import VistaContext, UserContext, ToolRequirement, ExecutionPlan, ExecutionGroup

@pytest.fixture
def mcp_adapter():
    return MCPToolAdapter()

@pytest.fixture
def base_context():
    ctx = VistaContext(
        conversation_id=str(uuid4()), 
        current_query="Find person",
        user=UserContext(user_id="user123", role="investigator", allowed_cameras=["CAM_01"])
    )
    ctx.execution_plan = ExecutionPlan(
        success=True,
        intent="PERSON_SEARCH",
        agents=[],
        tools=[
            ToolRequirement(name="search_person_occurrences", arguments={"description": "red shirt"})
        ],
        execution_groups=[ExecutionGroup(agents=["search_person_occurrences"])]
    )
    return ctx

@pytest.mark.asyncio
async def test_mcp_unknown_tool(mcp_adapter, base_context):
    """Test 1: Unknown tool name -> BLOCK"""
    with pytest.raises(ValueError, match="Unknown MCP tool: fake_tool"):
        await mcp_adapter.execute_tool("fake_tool", {}, base_context)

@pytest.mark.asyncio
async def test_mcp_missing_required_arg(mcp_adapter, base_context):
    """Test 2: Missing required argument -> BLOCK"""
    with pytest.raises(ValueError, match="Invalid arguments for search_person_occurrences:"):
        await mcp_adapter.execute_tool("search_person_occurrences", {}, base_context) # missing description

@pytest.mark.asyncio
async def test_mcp_wrong_arg_type(mcp_adapter, base_context):
    """Test 3: Wrong argument type -> BLOCK"""
    with pytest.raises(ValueError, match="Invalid arguments for search_person_occurrences:"):
        await mcp_adapter.execute_tool("search_person_occurrences", {"description": {"complex": "object"}}, base_context)

@pytest.mark.asyncio
async def test_mcp_pre_execution_rbac(mcp_adapter, base_context):
    """Test 4: Unauthorized camera requested -> BLOCK"""
    # Valid schema, but explicitly asking for unauthorized camera
    with pytest.raises(PermissionError, match="Pre-execution RBAC blocked access to camera CAM_02"):
        await mcp_adapter.execute_tool("search_person_occurrences", {"description": "red shirt", "camera_id": "CAM_02"}, base_context)

@pytest.mark.asyncio
async def test_mcp_post_execution_rbac(mcp_adapter, base_context, monkeypatch):
    """Test 5: MCP returns unauthorized camera -> DROP"""
    # Mock the MCP server to return CAM_99 (unauthorized) and CAM_01 (authorized)
    async def mock_call(*args, **kwargs):
        return {
            "status": "success",
            "results": [
                {"evidence_id": str(uuid4()), "camera_id": "CAM_99", "timestamp": "2026-08-08T10:45:00Z", "description": "unauth"},
                {"evidence_id": str(uuid4()), "camera_id": "CAM_01", "timestamp": "2026-08-08T10:45:00Z", "description": "auth"}
            ]
        }
    monkeypatch.setattr(mcp_adapter.client, "call_tool", mock_call)
    
    await mcp_adapter.execute_tool("search_person_occurrences", {"description": "test"}, base_context)
    
    bundle = base_context.evidence_bundle
    assert len(bundle.evidence) == 1
    assert bundle.evidence[0].metadata["camera_id"] == "CAM_01"

@pytest.mark.asyncio
async def test_mcp_fake_evidence_id_and_missing_provenance(mcp_adapter, base_context, monkeypatch):
    """Test 6 & 8: Fake evidence ID & missing provenance -> DROP"""
    async def mock_call(*args, **kwargs):
        return {
            "status": "success",
            "results": [
                {"evidence_id": "not-a-uuid", "camera_id": "CAM_01", "timestamp": "2026-08-08T10:45:00Z", "description": "fake id"},
                {"evidence_id": str(uuid4()), "camera_id": "CAM_01", "description": "missing timestamp"}, # missing timestamp
                {"evidence_id": str(uuid4()), "camera_id": "CAM_01", "timestamp": "invalid-time", "description": "invalid time"},
                {"evidence_id": str(uuid4()), "camera_id": "CAM_01", "timestamp": "2026-08-08T10:45:00Z", "description": "valid"}
            ]
        }
    monkeypatch.setattr(mcp_adapter.client, "call_tool", mock_call)
    
    await mcp_adapter.execute_tool("search_person_occurrences", {"description": "test"}, base_context)
    
    bundle = base_context.evidence_bundle
    assert len(bundle.evidence) == 1
    assert bundle.evidence[0].metadata["description"] == "valid"

@pytest.mark.asyncio
async def test_mcp_malformed_json(mcp_adapter, base_context, monkeypatch):
    """Test 7: Malformed JSON/Response -> BLOCK"""
    async def mock_call(*args, **kwargs):
        return {"status": "success", "results": "not a list"}
    monkeypatch.setattr(mcp_adapter.client, "call_tool", mock_call)
    
    with pytest.raises(RuntimeError, match="MCP Evidence Normalization failed"):
        await mcp_adapter.execute_tool("search_person_occurrences", {"description": "test"}, base_context)

@pytest.mark.asyncio
async def test_mcp_e2e_pipeline(monkeypatch):
    """
    Test 8: E2E Pipeline
    Query -> MCP -> RBAC -> normalized Evidence -> EvidenceBundle -> Reasoning -> Guardrail -> Response
    """
    from app.graph.supervisor.supervisor import Supervisor
    from tests.integration.pipeline.test_llm_workflows import MockReasoningClient
    from app.domain.models.enums import ExecutionMode
    
    # Setup Context
    ctx = VistaContext(
        conversation_id=str(uuid4()), 
        current_query="Find the guy in the red shirt",
        user=UserContext(user_id="test", role="admin", allowed_cameras=["CAM_01"])
    )
    
    # 1. Provide an ExecutionPlan that invokes MCP tool then reasoning
    ctx.execution_plan = ExecutionPlan(
        success=True,
        intent="PERSON_SEARCH",
        agents=[],
        tools=[
            ToolRequirement(name="search_person_occurrences", arguments={"description": "red shirt"})
        ],
        execution_groups=[
            ExecutionGroup(agents=["search_person_occurrences"]),
            ExecutionGroup(agents=["reasoning_agent"])
        ]
    )
    
    # 2. Mock MCP Tool Response
    # The MCP tool adapter is created internally by the Dispatcher.
    # We can mock the LocalMCPClient.call_tool globally or just intercept the Dispatcher.
    from app.mcp.client import LocalMCPClient
    async def mock_mcp_call(self, tool_name, arguments):
        assert tool_name == "search_person_occurrences"
        return {
            "status": "success",
            "results": [
                {
                    "evidence_id": str(uuid4()), 
                    "camera_id": "CAM_01",
                    "timestamp": "2026-08-08T10:42:15Z",
                    "description": "Red shirt on a bike",
                    "confidence": 0.95
                }
            ]
        }
    monkeypatch.setattr(LocalMCPClient, "call_tool", mock_mcp_call)
    
    # 3. Setup Supervisor and Mock Reasoning LLM
    # Reasoning mock will receive E1 and we will map it to a claim
    reasoning_client = MockReasoningClient(claims=[
        {
            "statement": "I found a person in a red shirt on a bike.",
            "evidence_ids": ["E1"],
            "confidence": 0.95,
            "support_type": "direct"
        }
    ], answer="Red shirt on a bike")
    
    from app.api.dependencies.supervisor import get_supervisor
    from app.api.dependencies.repositories import get_event_bus, get_postgres_tool, get_milvus_tool, get_camera_repository, get_alert_repository, get_person_repository, get_vehicle_repository
    from app.api.dependencies.services import get_metadata_service, get_vector_service, get_video_service, get_event_service, get_report_service, get_s3_tool
    from app.agents.registry import agent_registry
    
    event_bus = get_event_bus()
    pg_tool = get_postgres_tool(event_bus)
    milvus_tool = get_milvus_tool(event_bus)
    s3_tool = get_s3_tool(event_bus)
    cam_repo = get_camera_repository(pg_tool)
    alert_repo = get_alert_repository(pg_tool)
    person_repo = get_person_repository(milvus_tool)
    vehicle_repo = get_vehicle_repository(milvus_tool)
    metadata_service = get_metadata_service(cam_repo, alert_repo, event_bus)
    vector_service = get_vector_service(person_repo, vehicle_repo, event_bus)
    video_service = get_video_service(s3_tool, event_bus)
    event_service = get_event_service(event_bus)
    report_service = get_report_service(event_bus)
    
    supervisor = get_supervisor(
        event_bus=event_bus,
        postgres_tool=pg_tool,
        milvus_tool=milvus_tool,
        s3_tool=s3_tool,
        metadata_service=metadata_service,
        vector_service=vector_service,
        video_service=video_service,
        event_service=event_service,
        report_service=report_service
    )
    
    reasoning_agent = agent_registry.get_agent("reasoning_agent")
    monkeypatch.setattr(reasoning_agent.coordinator.pipeline.hypothesis_generator, "llm", reasoning_client)
    
    # 4. Run!
    result = await supervisor.run(ctx)
    
    # 5. Assertions
    assert result["status"] == "success", f"Pipeline failed: {result.get('final_answer')}"
    assert "red shirt on a bike" in result["final_answer"].lower()
    
    # Verify the EvidenceBundle was populated correctly
    bundle = ctx.evidence_bundle
    assert bundle is not None
    assert len(bundle.evidence) == 1
    ev = bundle.evidence[0]
    assert ev.metadata["camera_id"] == "CAM_01"
    assert ev.provenance["mcp_tool"] == "search_person_occurrences"
    
    # Verify trace matches
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["camera_id"] == "CAM_01"


@pytest.mark.asyncio
async def test_mcp_video_clip_e2e_pipeline(monkeypatch):
    """
    Test 9: E2E Pipeline with Video Clip Extraction
    Query -> MCP Search -> EvidenceBundle -> Reasoning -> MCP Video Clip -> Reasoning -> Guardrail -> Response
    """
    from app.graph.supervisor.supervisor import Supervisor
    from tests.integration.pipeline.test_llm_workflows import MockReasoningClient
    
    # 1. Provide an ExecutionPlan that invokes search, reasoning, clip, then reasoning again
    person_ev_uuid = str(uuid4())
    ctx = VistaContext(
        conversation_id=str(uuid4()), 
        current_query="Find the guy in the red shirt and give me a video clip",
        user=UserContext(user_id="test", role="admin", allowed_cameras=["CAM_01"])
    )
    
    ctx.execution_plan = ExecutionPlan(
        success=True,
        intent="PERSON_SEARCH",
        agents=[],
        tools=[
            ToolRequirement(name="search_person_occurrences", arguments={"description": "red shirt"}),
            ToolRequirement(name="get_video_clip", arguments={
                "camera_id": "CAM_01",
                "start_time": "2026-08-08T10:42:00Z",
                "end_time": "2026-08-08T10:43:00Z",
                "evidence_id": person_ev_uuid
            })
        ],
        execution_groups=[
            ExecutionGroup(agents=["search_person_occurrences"]),
            ExecutionGroup(agents=["reasoning_agent"]),
            ExecutionGroup(agents=["get_video_clip"]),
            ExecutionGroup(agents=["reasoning_agent"])
        ]
    )
    
    # 2. Mock MCP Tool Responses
    from app.mcp.client import LocalMCPClient
    async def mock_mcp_call(self, tool_name, arguments):
        if tool_name == "search_person_occurrences":
            return {
                "status": "success",
                "results": [
                    {
                        "evidence_id": person_ev_uuid, 
                        "camera_id": "CAM_01",
                        "timestamp": "2026-08-08T10:42:15Z",
                        "description": "Red shirt on a bike",
                        "confidence": 0.95
                    }
                ]
            }
        elif tool_name == "get_video_clip":
            return {
                "status": "success",
                "result": {
                    "camera_id": arguments["camera_id"],
                    "start_time": arguments["start_time"],
                    "end_time": arguments["end_time"],
                    "evidence_id": arguments["evidence_id"],
                    "clip_uri": f"s3://vista-storage/clips/{arguments['camera_id']}_clip.mp4",
                    "description": "Extracted clip of red shirt on a bike"
                }
            }
    monkeypatch.setattr(LocalMCPClient, "call_tool", mock_mcp_call)
    
    # 3. Setup Supervisor and Mock Reasoning LLM
    class StatefulMockReasoningClient(MockReasoningClient):
        def __init__(self):
            super().__init__()
            self.call_count = 0
            
        async def ainvoke(self, messages):
            import json
            class DummyResponse:
                def __init__(self, content):
                    self.content = content
            import re
            prompt = str(messages)
            uuids = re.findall(r"UUID: ([0-9a-fA-F\-]+)", prompt)
            
            self.call_count += 1
            if self.call_count == 1:
                # First time: Only person occurrence UUID exists
                claims = [
                    {
                        "statement": "I found a person in a red shirt on a bike.",
                        "evidence_ids": [uuids[0]] if uuids else [],
                        "confidence": 0.95,
                        "support_type": "direct"
                    }
                ]
                answer = "I found the person."
            else:
                # Second time: Both occurrence and video clip UUIDs exist
                claims = [
                    {
                        "statement": "I found a person in a red shirt on a bike, and extracted a video clip.",
                        "evidence_ids": uuids[:2] if len(uuids) >= 2 else uuids,
                        "confidence": 0.95,
                        "support_type": "direct"
                    }
                ]
                answer = "I found the person in a red shirt on a bike. I have extracted a video clip for you."
                
            response_dict = {
                "success": True,
                "answer": answer,
                "claims": claims,
                "uncertainties": []
            }
            return DummyResponse(json.dumps(response_dict))

    reasoning_client = StatefulMockReasoningClient()
    
    from app.api.dependencies.supervisor import get_supervisor
    from app.api.dependencies.repositories import get_event_bus, get_postgres_tool, get_milvus_tool, get_camera_repository, get_alert_repository, get_person_repository, get_vehicle_repository
    from app.api.dependencies.services import get_metadata_service, get_vector_service, get_video_service, get_event_service, get_report_service, get_s3_tool
    from app.agents.registry import agent_registry
    
    event_bus = get_event_bus()
    pg_tool = get_postgres_tool(event_bus)
    milvus_tool = get_milvus_tool(event_bus)
    s3_tool = get_s3_tool(event_bus)
    cam_repo = get_camera_repository(pg_tool)
    alert_repo = get_alert_repository(pg_tool)
    person_repo = get_person_repository(milvus_tool)
    vehicle_repo = get_vehicle_repository(milvus_tool)
    metadata_service = get_metadata_service(cam_repo, alert_repo, event_bus)
    vector_service = get_vector_service(person_repo, vehicle_repo, event_bus)
    video_service = get_video_service(s3_tool, event_bus)
    event_service = get_event_service(event_bus)
    report_service = get_report_service(event_bus)
    
    supervisor = get_supervisor(
        event_bus=event_bus,
        postgres_tool=pg_tool,
        milvus_tool=milvus_tool,
        s3_tool=s3_tool,
        metadata_service=metadata_service,
        vector_service=vector_service,
        video_service=video_service,
        event_service=event_service,
        report_service=report_service
    )
    
    reasoning_agent = agent_registry.get_agent("reasoning_agent")
    monkeypatch.setattr(reasoning_agent.coordinator.pipeline.hypothesis_generator, "llm", reasoning_client)
    
    # 4. Run!
    result = await supervisor.run(ctx)
    
    # 5. Assertions
    assert result["status"] == "success", f"Pipeline failed: {result.get('final_answer')}"
    assert "video clip" in result["final_answer"]
    
    # Verify the EvidenceBundle was populated correctly with BOTH pieces of evidence
    bundle = ctx.evidence_bundle
    assert bundle is not None
    assert len(bundle.evidence) == 2 # 1 occurrence, 1 video clip
    
    video_ev = next(ev for ev in bundle.evidence if ev.source == "mcp_get_video_clip")
    assert video_ev.metadata["camera_id"] == "CAM_01"
    assert "s3://vista-storage" in video_ev.metadata["clip_uri"]
    assert video_ev.provenance["occurrence_evidence_id"] == person_ev_uuid
