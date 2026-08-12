import asyncio

from app.graph.supervisor.event_bus import EventBus
from app.schemas.context import UserContext, VistaContext
from app.tools.metadata.postgres_tool import PostgresTool


def test_metadata_query_latency(benchmark):
    """
    Benchmark metadata query execution latency.
    """
    event_bus = EventBus()
    tool = PostgresTool(event_bus)
    context = VistaContext(user=UserContext(user_id="bench", role="operator"), conversation_id="1", current_query="bench")
    
    def run_query():
        # In a real environment, this hits the DB. We're benchmarking the abstraction overhead & mock latency.
        return asyncio.run(tool.execute(context, query="SELECT status FROM cameras WHERE id='cam-5'"))
        
    result = benchmark(run_query)
    assert result is not None
