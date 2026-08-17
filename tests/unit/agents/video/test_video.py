from datetime import datetime, timezone

import pytest

from app.agents.video.agent import VideoAgent
from app.domain.models.confidence import ConfidenceReport
from app.domain.evidence import EvidenceBundle, MetadataEvidence
from app.graph.supervisor.event_bus import EventBus
from app.schemas.context import ExecutionPlan, UserContext, VistaContext
from app.services.video_service.service import VideoService
from app.infrastructure.llm.model_registry import ModelRegistry
from app.tools.video.s3_tool import S3Tool


@pytest.mark.asyncio
async def test_video_agent_end_to_end():
    event_bus = EventBus()
    s3_tool = S3Tool(event_bus)
    vlm = ModelRegistry.get_client()
    
    # Service
    service = VideoService(s3_tool, vlm, event_bus)
    
    # Mock VLM response
    from app.services.video_service.service import VideoAnalysisResult
    from unittest.mock import AsyncMock
    mock_vlm_response = VideoAnalysisResult(
        scene_summary="A person running through the lobby.",
        objects=["person", "blue shirt", "backpack"],
        activities=["running", "entering"],
        confidence=0.95,
        frames_analyzed=10,
        timeline=[{"timestamp": "0:02", "description": "Person appears"}],
        reasoning="The visual evidence clearly shows a person matching the description running."
    )
    vlm.generate_structured = AsyncMock(return_value=mock_vlm_response)
    
    # Agent
    agent = VideoAgent(service)
    
    # Context setup
    context = VistaContext(user=UserContext(user_id="1", role="admin"), conversation_id="123")
    context.execution_plan = ExecutionPlan(success=True, agents=["video_agent"])
    
    # Needs evidence to target a camera
    bundle = EvidenceBundle()
    from uuid import uuid4
    bundle.add_evidence(MetadataEvidence(
        evidence_id=uuid4(), source="mock", confidence=1.0, timestamp=datetime.now(timezone.utc), trace_id=uuid4(), metadata={"camera_id": "cam_5"}
    ))
    context.evidence_bundle = bundle
    
    # Needs confidence recommendation
    context.confidence_report = ConfidenceReport()
    
    # Test validate
    assert agent.validate(context) == True
    
    # Test Execute
    result = await agent.execute(context, None)
    
    from app.domain.models.enums import AgentStatus
    assert result.status == AgentStatus.SUCCESS, result.metadata.get("error", getattr(result, "error", ""))
    assert result.confidence.overall == 0.95
    assert len(result.objects) > 0
    assert result.frames_analyzed > 0
    
    # Verify cache hit on second run
    result2 = await agent.execute(context, None)
    assert result2.status == AgentStatus.SUCCESS
    
    cache_hit_events = [e for e in event_bus.history if e.event_type == "CACHE_HIT"]
    assert len(cache_hit_events) == 1
    
    vlm_complete_events = [e for e in event_bus.history if e.event_type == "VLM_COMPLETE"]
    assert len(vlm_complete_events) == 1 # Only one VLM call because of cache
