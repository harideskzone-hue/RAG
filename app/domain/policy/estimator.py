from pydantic import BaseModel
from typing import Any

class EstimatedCosts(BaseModel):
    tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    gpu_required: bool = False
    memory_mb: int = 0

class CostEstimator:
    """Projects the estimated costs for a given ExecutionPlan."""
    
    # Mock lookup table for agent costs
    AGENT_COSTS = {
        "metadata_agent": {"tokens": 500, "cost": 0.001, "latency": 500, "gpu": False},
        "vector_agent": {"tokens": 2000, "cost": 0.005, "latency": 1500, "gpu": True},
        "video_agent": {"tokens": 5000, "cost": 0.05, "latency": 10000, "gpu": True},
        "ocr_agent": {"tokens": 1000, "cost": 0.002, "latency": 2000, "gpu": False},
        "reasoning_agent": {"tokens": 8000, "cost": 0.02, "latency": 5000, "gpu": False},
    }
    
    @classmethod
    def estimate(cls, execution_plan: Any) -> EstimatedCosts:
        # Assuming execution_plan is a list of agent names for the MVP
        total_tokens = 0
        total_cost = 0.0
        total_latency = 0
        gpu = False
        
        agents = execution_plan if isinstance(execution_plan, list) else []
        for agent in agents:
            costs = cls.AGENT_COSTS.get(agent, {"tokens": 1000, "cost": 0.01, "latency": 1000, "gpu": False})
            total_tokens += costs["tokens"]
            total_cost += costs["cost"]
            total_latency += costs["latency"]
            if costs["gpu"]:
                gpu = True
                
        return EstimatedCosts(
            tokens=total_tokens,
            cost_usd=total_cost,
            latency_ms=total_latency,
            gpu_required=gpu,
            memory_mb=1024 # Baseline
        )
