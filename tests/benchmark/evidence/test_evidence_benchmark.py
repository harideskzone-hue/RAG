import asyncio
from uuid import uuid4

from app.agents.evidence.agent import EvidenceAgent
from app.agents.metadata.schemas import MetadataResult
from app.domain.models import Camera
from app.graph.supervisor.event_bus import EventBus
from app.schemas.context import UserContext, VistaContext
from app.domain.models import ConfidenceScore, ConfidenceFactor


def test_evidence_merge_latency(benchmark):
    """
    Benchmark evidence agent deduplication and merging latency.
    """
    event_bus = EventBus()
    agent = EvidenceAgent()
    
    # Pre-populate context results
    context = VistaContext(user=UserContext(user_id="bench", role="operator"), conversation_id="bench-1", current_query="merge")
    cameras = [Camera(id=f"cam_{i}", location="Loc", status="online") for i in range(100)]
    context.results["metadata_agent"] = MetadataResult(
        execution_id=uuid4(),
        trace_id=uuid4(),
        agent_name="metadata_agent",
        agent_type="metadata",
        status="success",
        confidence=ConfidenceScore(overall=1.0, factors=[]),
        execution={"duration_ms": 10},
        cameras=cameras,
        alerts=[]
    )
    
    def run_merge():
        return asyncio.run(agent.execute(context, None))
        
    result = benchmark(run_merge)
    assert len(result.bundle.evidence) == 100
