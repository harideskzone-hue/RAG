import asyncio

from app.memory.manager import MemoryManager
from app.memory.policy import MemoryPolicy
from app.schemas.context import UserContext, VistaContext


def test_memory_optimization_latency(benchmark):
    """
    Benchmark the memory manager's state eviction and summarization logic.
    """
    def run_optimization():
        policy = MemoryPolicy(enable_summarization=False)
        manager = MemoryManager(policy)
        context = VistaContext(user=UserContext(user_id="bench", role="operator"), conversation_id="1", current_query="bench")
        return asyncio.run(manager.run(context))
        
    result = benchmark(run_optimization)
    assert result is not None
