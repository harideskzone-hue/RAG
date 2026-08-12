import asyncio

from app.graph.supervisor.event_bus import EventBus
from app.schemas.context import UserContext, VistaContext
from app.tools.vector.milvus_tool import MilvusTool


def test_milvus_search_latency(benchmark):
    """
    Benchmark vector similarity search latency.
    """
    event_bus = EventBus()
    tool = MilvusTool(event_bus)
    context = VistaContext(user=UserContext(user_id="bench", role="operator"), conversation_id="1", current_query="bench")
    
    def run_search():
        return asyncio.run(tool.execute(context, collection="person_collection", vector=[0.1]*128))
        
    result = benchmark(run_search)
    assert result is not None
