import asyncio

from app.graph.supervisor.supervisor import Supervisor
from app.schemas.context import UserContext, VistaContext


def test_supervisor_orchestration_latency(benchmark):
    """
    Benchmark the full supervisor graph orchestration overhead.
    This executes a fast intent (Metadata) to measure graph traversal time rather than tool time.
    """
    def run_supervisor():
        supervisor = Supervisor()
        context = VistaContext(
            user=UserContext(user_id="bench_user", role="operator"),
            conversation_id="bench-1",
            current_query="Is camera 5 online?"
        )
        return asyncio.run(supervisor.run(context))
        
    result = benchmark(run_supervisor)
    assert isinstance(result, dict)
