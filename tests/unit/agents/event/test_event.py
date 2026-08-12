from datetime import datetime, timezone

import pytest

from app.agents.event.agent import EventAgent
from app.domain.event_types import EventType
from app.domain.evidence import EvidenceBundle, MetadataEvidence, VideoEvidence
from app.graph.supervisor.event_bus import EventBus
from app.schemas.context import ExecutionPlan, UserContext, VistaContext
from app.services.event_service.service import EventService


@pytest.mark.asyncio
async def test_event_agent_end_to_end():
    event_bus = EventBus()
    service = EventService(event_bus)
    agent = EventAgent(service)
    
    context = VistaContext(user=UserContext(user_id="1", role="admin"), conversation_id="123")
    context.execution_plan = ExecutionPlan(success=True, agents=["event_agent"], intent="fight")
    
    bundle = EvidenceBundle()
    from uuid import uuid4
    bundle.add_evidence(MetadataEvidence(
        evidence_id=uuid4(), source="postgres_metadata", confidence=1.0, timestamp=datetime.now(timezone.utc), trace_id=uuid4(), metadata={"camera_id": "cam_1", "description": "Alert: fight detected."}
    ))
    bundle.add_evidence(VideoEvidence(
        evidence_id=uuid4(), source="vlm_gemini", confidence=0.9, timestamp=datetime.now(timezone.utc), trace_id=uuid4(), metadata={"camera_id": "cam_1", "summary": "Two people are fighting."}
    ))
    context.evidence_bundle = bundle
    
    # Test Execute
    result = await agent.execute(context, None)
    
    from app.domain.models.enums import AgentStatus
    print(f"ERROR: {result.metadata.get('error')}")
    assert result.status == AgentStatus.SUCCESS
    assert result.event_type == EventType.FIGHT.value
    assert result.severity == "HIGH"
    assert len(result.correlations) > 0
    assert len(result.timeline) == 2
    assert len(result.recommendations) > 0
