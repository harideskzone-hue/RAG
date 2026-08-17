from datetime import datetime, timezone

import pytest

from app.agents.report.agent import ReportAgent
from app.domain.evidence import EvidenceBundle, MetadataEvidence
from app.graph.supervisor.event_bus import EventBus
from app.schemas.context import ExecutionPlan, UserContext, VistaContext
from app.services.report_service.service import ReportService


@pytest.mark.asyncio
async def test_report_agent_end_to_end():
    event_bus = EventBus()
    service = ReportService(event_bus)
    from tests.fakes.report import FakeReportExporter
    service.exporter = FakeReportExporter()
    agent = ReportAgent(service)
    
    context = VistaContext(user=UserContext(user_id="1", role="admin"), conversation_id="123")
    context.execution_plan = ExecutionPlan(success=True, agents=["report_agent"], intent="generate_report")
    context.current_query = "Generate json report"
    
    bundle = EvidenceBundle()
    from uuid import uuid4
    bundle.add_evidence(MetadataEvidence(
        evidence_id=uuid4(), source="postgres_metadata", confidence=1.0, timestamp=datetime.now(timezone.utc), trace_id=uuid4(), metadata={"camera_id": "cam_1"}
    ))
    context.evidence_bundle = bundle
    
    # Test Execute
    result = await agent.execute(context, None)
    
    from app.domain.models.enums import AgentStatus
    assert result.status == AgentStatus.SUCCESS, result.metadata.get("error")
    assert result.report_uri is None
    assert "1 incidents" in result.narrative
    assert "postgres_metadata" in result.data["statistics"]["sources"]
