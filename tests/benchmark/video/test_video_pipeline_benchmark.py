import asyncio

from app.services.video_service.vlm_adapter import GeminiAdapter


def test_vlm_adapter_latency(benchmark):
    """
    Benchmark the latency of the VLM API call (mocked).
    In a real CI setup, this would test the actual preprocessing and API wrapper overhead.
    """
    adapter = GeminiAdapter()
    
    def run_vlm():
        # This currently returns a hardcoded mock response
        return asyncio.run(adapter.analyze(["frame1", "frame2"], "Find person"))
        
    result = benchmark(run_vlm)
    assert result is not None
