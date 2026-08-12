import pytest
import datetime
from app.schemas.context import VistaContext, UserContext, ExecutionPlan
from app.graph.supervisor.supervisor import Supervisor
from app.domain.evidence import EvidenceBundle, PersonEvidence, EvidenceType

from app.api.dependencies.supervisor import get_supervisor
from app.api.dependencies.repositories import get_event_bus, get_postgres_tool, get_milvus_tool, get_camera_repository, get_alert_repository, get_person_repository, get_vehicle_repository
from app.api.dependencies.services import get_metadata_service, get_vector_service, get_video_service, get_event_service, get_report_service, get_s3_tool
from app.agents.registry import agent_registry

class MockReasoningClient:
    def __init__(self, answer="Grounded answer", claims=None):
        self.answer = answer
        self.claims = claims or [
            {
                "statement": "Valid claim",
                "evidence_ids": ["UUID_PLACEHOLDER"],
                "confidence": 0.9,
                "support_type": "direct"
            }
        ]

    async def ainvoke(self, messages):
        import json
        class DummyResponse:
            def __init__(self, content):
                self.content = content
        
        import re
        import copy
        
        prompt = str(messages)
        dynamic_claims = copy.deepcopy(self.claims)
        
        # Find all UUIDs in the prompt
        uuids = re.findall(r"UUID: ([0-9a-fA-F\-]+)", prompt)
        
        for claim in dynamic_claims:
            keyword = "blue shirt" if "blue" in claim["statement"].lower() else ("red shirt" if "red" in claim["statement"].lower() else None)
            
            matched_uuid = None
            if keyword:
                match = re.search(r"UUID: ([0-9a-fA-F\-]+)\s*\-\>(?:(?!(?:UUID:)).)*?" + keyword, prompt, re.IGNORECASE)
                if match:
                    matched_uuid = match.group(1)
            
            if "UUID_PLACEHOLDER" in claim.get("evidence_ids", []):
                if matched_uuid:
                    claim["evidence_ids"] = [matched_uuid]
                elif uuids:
                    claim["evidence_ids"] = [uuids[0]]
                else:
                    claim["evidence_ids"] = []
            elif claim.get("evidence_ids") == ["E1"] and uuids:
                claim["evidence_ids"] = [uuids[0]]
            
            # If the claim explicitly used "E1" etc for negative tests, leave it as is to fail.

        response_dict = {
            "success": True,
            "answer": self.answer,
            "claims": dynamic_claims,
            "uncertainties": []
        }
        return DummyResponse(json.dumps(response_dict))

@pytest.fixture
def supervisor():
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
    
    sup = get_supervisor(
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
    return sup

@pytest.fixture
def base_context():
    return VistaContext(
        conversation_id="test_llm_workflows",
        current_query="test",
        user=UserContext(user_id="tester", role="admin", allowed_cameras=None)
    )

async def _run_test(supervisor, base_context, client, query, expected_status="success"):
    base_context.current_query = query
    base_context.execution_plan = ExecutionPlan(
        success=True,
        intent="PERSON_SEARCH",
        agents=["metadata_agent", "vector_agent", "evidence_agent", "reasoning_agent"],
        execution_groups=[
            {"agents": ["metadata_agent", "vector_agent"]},
            {"agents": ["evidence_agent"]},
            {"agents": ["reasoning_agent"]}
        ]
    )
    
    reasoning_agent = agent_registry.get_agent("reasoning_agent")
    reasoning_agent.coordinator.pipeline.hypothesis_generator.llm = client
    reasoning_agent.coordinator.pipeline.explanation_generator.llm = client
    
    response = await supervisor.run(base_context)
    
    if expected_status == "success":
        if response["status"] != "success":
            print(f"FAILED! Final answer was: {response.get('final_answer')}")
        assert response["status"] == "success"
        assert response["final_answer"] != ""
        assert len(response.get("evidence", [])) > 0 or "insufficient" in response["final_answer"].lower()
    else:
        if response["status"] != "error":
            print(f"FAILED! Expected error but got success. Final answer was: {response.get('final_answer')}")
        assert response["status"] == "error"
        assert "Response blocked" in response["final_answer"] or "Reasoning failed" in response["final_answer"]
    return response

@pytest.mark.asyncio
async def test_workflow_suspicious_person(supervisor: Supervisor, base_context: VistaContext):
    """Query 1: Suspicious person (Grounded answer or justified abstention)"""
    client = MockReasoningClient(claims=[{"statement": "A suspicious person", "evidence_ids": ["UUID_PLACEHOLDER"], "confidence": 0.9, "support_type": "direct"}])
    await _run_test(supervisor, base_context, client, "Is there any suspicious person in the CCTV?")

@pytest.mark.asyncio
async def test_workflow_blue_shirt(supervisor: Supervisor, base_context: VistaContext):
    """Query 2: Blue shirt (Grounded answer + evidence IDs)"""
    client = MockReasoningClient(claims=[{"statement": "A person in a blue shirt", "evidence_ids": ["UUID_PLACEHOLDER"], "confidence": 0.9, "support_type": "direct"}])
    await _run_test(supervisor, base_context, client, "What can you tell me about the person in the blue shirt?")

@pytest.mark.asyncio
async def test_workflow_red_shirt_bike(supervisor: Supervisor, base_context: VistaContext):
    """Query 3: Red shirt + bike (Grounded answer + evidence IDs)"""
    client = MockReasoningClient(claims=[{"statement": "A person in a red shirt on a bike", "evidence_ids": ["UUID_PLACEHOLDER"], "confidence": 0.9, "support_type": "direct"}])
    await _run_test(supervisor, base_context, client, "Did you see anyone in a red shirt with a bike?")

@pytest.mark.asyncio
async def test_workflow_purple_jacket(supervisor: Supervisor, base_context: VistaContext):
    """Query 4: Purple jacket + helicopter (Abstain or Block)"""
    # Test 1: Abstain (Valid JSON, but no evidence cited)
    client_abstain = MockReasoningClient(
        answer="I do not see anyone with a purple jacket or a helicopter.",
        claims=[{"statement": "No helicopter found", "evidence_ids": [], "confidence": 0.9, "support_type": "unknown"}]
    )
    res = await _run_test(supervisor, base_context, client_abstain, "Find a person wearing a purple jacket riding a helicopter.", expected_status="success")
    assert res["status"] == "success"

@pytest.mark.asyncio
async def test_negative_fake_alias(supervisor: Supervisor, base_context: VistaContext):
    """Fake evidence alias -> BLOCK (The LLM must use UUIDs, E999 is invalid)"""
    client = MockReasoningClient(claims=[{"statement": "Fake alias", "evidence_ids": ["E999"], "confidence": 0.9, "support_type": "direct"}])
    await _run_test(supervisor, base_context, client, "fake alias test", expected_status="error")

@pytest.mark.asyncio
async def test_negative_integer_alias(supervisor: Supervisor, base_context: VistaContext):
    """evidence_ids: [0, 1] -> BLOCK"""
    client = MockReasoningClient(claims=[{"statement": "Integer alias", "evidence_ids": [0, 1], "confidence": 0.9, "support_type": "direct"}])
    await _run_test(supervisor, base_context, client, "integer alias test", expected_status="error")

@pytest.mark.asyncio
async def test_negative_missing_evidence(supervisor: Supervisor, base_context: VistaContext):
    """Missing evidence_ids -> BLOCK"""
    client = MockReasoningClient(claims=[{"statement": "Missing evidence", "evidence_ids": [], "confidence": 0.9, "support_type": "direct"}])
    await _run_test(supervisor, base_context, client, "missing evidence test", expected_status="error")

@pytest.mark.asyncio
async def test_negative_invalid_confidence(supervisor: Supervisor, base_context: VistaContext):
    """Confidence 1.5 -> BLOCK"""
    client = MockReasoningClient(claims=[{"statement": "Invalid confidence", "evidence_ids": ["UUID_PLACEHOLDER"], "confidence": 1.5, "support_type": "direct"}])
    await _run_test(supervisor, base_context, client, "invalid confidence test", expected_status="error")

@pytest.mark.asyncio
async def test_positive_multiple_uuids(supervisor: Supervisor, base_context: VistaContext):
    """evidence_ids: [uuid1, uuid2] -> PASS"""
    class MultiMockClient(MockReasoningClient):
        async def ainvoke(self, messages):
            import re, copy, json
            prompt = str(messages)
            uuids = re.findall(r"UUID: ([0-9a-fA-F\-]+)", prompt)
            dynamic_claims = copy.deepcopy(self.claims)
            dynamic_claims[0]["evidence_ids"] = uuids[:2] if len(uuids) >= 2 else (uuids[:1] if uuids else [])
            return type('Dummy', (), {'content': json.dumps({"success": True, "answer": self.answer, "claims": dynamic_claims, "uncertainties": []})})()
            
    client = MultiMockClient(claims=[{"statement": "Multiple uuids", "evidence_ids": ["placeholder"], "confidence": 0.9, "support_type": "direct"}])
    res = await _run_test(supervisor, base_context, client, "multiple uuids test", expected_status="success")
    assert res["status"] == "success"

@pytest.mark.asyncio
async def test_negative_hallucinated_uuid(supervisor: Supervisor, base_context: VistaContext):
    """evidence_ids with hallucinated raw UUID -> BLOCK (must be from EvidenceBundle)"""
    client = MockReasoningClient(claims=[{"statement": "Raw UUID", "evidence_ids": ["a8f698c8-5ee6-4c91-aeea-d317c119b87d"], "confidence": 0.9, "support_type": "direct"}])
    await _run_test(supervisor, base_context, client, "raw uuid test", expected_status="error")
