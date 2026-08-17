import asyncio

from app.infrastructure.llm.model_registry import ModelRegistry


def test_vlm_adapter_latency(benchmark):
    """
    Benchmark the latency of the VLM API call (mocked).
    In a real CI setup, this would test the actual preprocessing and API wrapper overhead.
    """
    adapter = ModelRegistry.get_client()
    
    def run_vlm():
        # This currently returns a hardcoded mock response
        return asyncio.run(adapter.generate([{"role": "user", "content": "Find person"}]))
        
    result = benchmark(run_vlm)
    assert result is not None
