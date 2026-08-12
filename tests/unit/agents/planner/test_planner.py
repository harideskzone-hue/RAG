import pytest

from app.agents.intent.enums import Intent
from app.agents.intent.schemas import IntentResult
from app.agents.planner.optimizer import CostOptimizer
from app.agents.planner.planner import ExecutionPlanner

# Golden Dataset for Planner
# Intent, expected required agents, expected requires_vlm flag
GOLDEN_PLANNER_DATASET = [
    (Intent.PERSON_SEARCH, ["metadata_agent", "vector_agent", "video_agent"], True),
    (Intent.REPORT, ["metadata_agent", "report_agent"], False),
    (Intent.CAMERA_STATUS, ["metadata_agent"], False),
]

@pytest.mark.asyncio
async def test_deterministic_planner():
    planner = ExecutionPlanner()
    
    for intent, expected_agents, expected_vlm in GOLDEN_PLANNER_DATASET:
        intent_result = IntentResult(intent=intent, success=True)
        plan = await planner.plan(intent_result)
        
        assert plan is not None
        for agent in expected_agents:
            assert agent in plan.agents
        
        # Test basic deterministic assignment
        if intent == Intent.PERSON_SEARCH:
            assert plan.requires_vlm == True
            assert "milvus" in [t.name for t in plan.tools]
        elif intent == Intent.REPORT:
            assert plan.requires_vlm == False

def test_execution_optimizer():
    optimizer = CostOptimizer()
    
    from app.schemas.context import ExecutionPlan, ToolRequirement, ExecutionGroup
    from app.agents.registry import AgentRegistry
    from app.domain.models import AgentManifest, AgentCapability

    class MockAgent:
        def __init__(self, name, deps):
            self.name = name
            self.manifest = AgentManifest(
                name=name, description="", cost="low", latency="fast",
                capabilities=AgentCapability(supported_intents=[], supported_entities=[], supported_modalities=[], supported_operations=[]),
                dependencies=deps
            )

    registry = AgentRegistry()
    registry.register(MockAgent("video_agent", ["vlm"]))
    registry.register(MockAgent("metadata_agent", ["postgres"]))
    registry.register(MockAgent("vector_agent", ["milvus"]))

    optimizer = CostOptimizer(registry)
    
    plan = ExecutionPlan(
        success=True,
        intent=Intent.CAMERA_STATUS.value,
        agents=["metadata_agent", "vector_agent", "video_agent"],
        tools=[ToolRequirement(name="postgres"), ToolRequirement(name="s3")],
        execution_groups=[ExecutionGroup(agents=["metadata_agent"]), ExecutionGroup(agents=["vector_agent", "video_agent"])],
        dependencies={"video_agent": ["vector_agent"]},
        requires_vlm=True
    )
    
    optimized = optimizer.optimize(plan)
    
    assert "video_agent" not in optimized.agents
    assert optimized.requires_vlm == False
    assert "s3" not in [t.name for t in optimized.tools]
