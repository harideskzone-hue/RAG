from datetime import datetime, timedelta

import pytest

from app.agents.evidence.agent import EvidenceAgent
from app.agents.metadata.schemas import MetadataResult
from app.agents.vector.schemas import VectorResult
from app.domain.models import Alert, Camera, PersonMatch
from app.graph.supervisor.event_bus import EventBus
from app.schemas.context import ExecutionPlan, UserContext, VistaContext


@pytest.mark.asyncio
async def test_evidence_collector_deduplication_and_ordering():
    from unittest.mock import Mock
    from app.services.metadata_service import MetadataService
    from app.services.vector_service import VectorService
    mock_meta = Mock(spec=MetadataService)
    mock_vec = Mock(spec=VectorService)
    agent = EvidenceAgent()
    
    context = VistaContext(user=UserContext(user_id="1", role="admin"), conversation_id="123")
    context.execution_plan = ExecutionPlan(success=True, agents=["evidence_agent"], intent="test")
    
    # Mock Metadata Result
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    cam_1 = Camera(id="cam_1", location="Lobby", status="online")
    alert_1 = Alert(id="a_1", type="fight", camera_id="cam_1", timestamp=now + timedelta(minutes=2))
    from uuid import uuid4
    from app.domain.models import ConfidenceScore, ConfidenceFactor
    context.results["metadata_agent"] = MetadataResult(
        result_id=uuid4(),
        execution_id=uuid4(),
        trace_id=uuid4(),
        agent_name="metadata_agent",
        agent_type="metadata",
        status="success",
        confidence=ConfidenceScore(overall=1.0, factors=[]),
        execution={"duration_ms": 10},
        cameras=[cam_1],
        alerts=[alert_1]
    )
    
    # Mock Vector Result
    person_1 = PersonMatch(id="p_1", camera_id="cam_1", timestamp=now + timedelta(minutes=1), score=0.9, description="Person running")
    context.results["vector_agent"] = VectorResult(
        result_id=uuid4(),
        execution_id=uuid4(),
        trace_id=uuid4(),
        agent_name="vector_agent",
        agent_type="vector",
        status="success",
        confidence=ConfidenceScore(overall=0.9, factors=[]),
        execution={"duration_ms": 10},
        person_matches=[person_1]
    )
    
    # Execute Agent
    result = await agent.execute(context, None)
    
    assert result.success == True
    bundle = result.bundle
    
    # Check Evidence length (1 camera + 1 alert + 1 person)
    assert len(bundle.evidence) == 3
    
    # Check ordering
    timeline = bundle.get_timeline()
    # Should be Camera (now) -> Person (now + 1m) -> Alert (now + 2m)
    assert "Camera" in timeline[0]["summary"]
    assert "Person" in timeline[1]["summary"]
    assert "alert" in timeline[2]["summary"].lower()
    
    # Verify provenance and relationships
    assert bundle.evidence[2].provenance["agent"] == "metadata_agent"
    assert bundle.evidence[1].relationships[0]["target_id"] == "cam_1"
    
    assert len(bundle.evidence) == 3
